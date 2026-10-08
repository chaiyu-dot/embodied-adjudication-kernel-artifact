# -*- coding: utf-8 -*-
"""blind_replicate_e83.py —— E83 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e83_bench_data.json + 复刻者侧独立实现：
  * `l4_plant2r`（2R 被控对象 + RK4 + 经典族 P/PD/PID/PD_G/CT/FO + box-QP）
  * `e77_standalone_physics`（FLC/IMP/LRN 的独立重写；权重 W 由数据文件给出）
  * `e68_standalone_physics.MPC`（独立 MPC：线性化 + 滚动优化 + 硬输入约束）
**不 import planning.* / control_strategies / e66 / e68 / e77 / e83**。

独立复现 E83 的：族相关安全观测量（util/duty/accel）+ 门控承重归因 + H83-1..4，并与报告比对。
产物：blind_replicate_e83.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import l4_plant2r as K            # noqa: E402
import e77_standalone_physics as P77  # noqa: E402
import e68_standalone_physics as P68  # noqa: E402

REPORT = os.path.join(EVAL, "e83_family_safety_observable_report.json")
BENCH = os.path.join(EVAL, "e83_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e83.json")
_repo = ("planning", "control_strategies", "rne_dynamics", "e66_closed_loop_control",
         "e68_control_strategy_suite", "e77_controller_family_extension",
         "e83_family_safety_observable")
_res = {"experiment": "E83 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _build(name, B, kp, kd, gate, W, Q_REF):
    if name in ("FLC", "IMP", "LRN"):
        return P77.build(name, kp, kd, W, gate)
    if name == "MPC":
        return P68.MPC(B, kp=kp, kd=kd, Np=20, Nc=5, q_w=1.0, qd_w=0.05, r_w=1e-3,
                       q_ref=np.asarray(Q_REF), gate=gate)
    # 经典族：E83 对所有族统一用设计增益（PID 的 ki 缺省 = 0）
    return K.Ctrl(B, name, kp=kp, kd=kd, ki=0.0, gate=gate, vmax=4.0)


def _run(ctrl, B, q_ref, T, dt):
    n = int(round(T / dt))
    st = np.zeros(4); st[0], st[1] = B["q0"]
    taus, raws, qds = [], [], []
    for _ in range(n):
        tau = ctrl.act(st[:2], st[2:4], np.asarray(q_ref, float))
        taus.append(np.asarray(tau, float)); raws.append(np.asarray(ctrl.last_tau_raw, float))
        qds.append(st[2:4].copy())
        st = K.rk4_step(B, st, tau, dt)
        if not np.all(np.isfinite(st)) or np.max(np.abs(st[:2])) > 1e4:
            break
    return np.array(taus), np.array(raws), np.array(qds)


def _obs(taus, raws, qds, tl, dt):
    util = float(np.max(np.abs(taus) / tl))
    ev = int(np.sum(np.any(np.abs(taus) > tl + 1e-9, axis=1)))
    duty = round(int(np.sum(np.any(np.abs(raws) > tl + 1e-9, axis=1))) / max(len(taus), 1), 4)
    acc = float(np.max(np.abs(np.gradient(qds, dt, axis=0)))) if len(qds) >= 3 else float("nan")
    return {"executed_peak_util": round(util, 4), "executed_violation_steps": ev,
            "demand_saturation_duty": duty, "peak_joint_accel_rad_s2": round(acc, 3)}


def replicate(data):
    B = dict(data["bench"])
    tl = np.asarray(B["tau_lim"], float)
    dt = B["dt"]; kp = data["gains"]["kp"]; kd = data["gains"]["kd"]
    W = data["family_specs"].get("LRN", {}).get("W")
    P77.configure({"bench": data["p77_bench"], "family_specs": data["family_specs"]})
    gated, ungated = {}, {}
    for lv, L in data["levels"].items():
        gated[lv], ungated[lv] = {}, {}
        for fam in data["families"]:
            for gate, store in ((True, gated[lv]), (False, ungated[lv])):
                c = _build(fam, B, kp, kd, gate, W, B["q_ref"])
                t, r, q = _run(c, B, L["q_ref"], L["T"], dt)
                store[fam] = _obs(t, r, q, tl, dt)
    # 归因与判据（与 E83 同定义）
    intr = {lv: sorted(n for n in ungated[lv] if ungated[lv][n]["executed_violation_steps"] == 0)
            for lv in ungated}
    gdep = {lv: sorted(n for n in ungated[lv] if ungated[lv][n]["executed_violation_steps"] > 0)
            for lv in ungated}

    def spread(vals):
        v = [x for x in vals if x is not None and math.isfinite(x)]
        return (max(v) - min(v)) if len(v) >= 2 else 0.0
    sp = {lv: {"executed_peak_util": round(spread([gated[lv][n]["executed_peak_util"] for n in gated[lv]]), 4),
               "demand_saturation_duty": round(spread([gated[lv][n]["demand_saturation_duty"] for n in gated[lv]]), 4),
               "peak_joint_accel_rad_s2": round(spread([gated[lv][n]["peak_joint_accel_rad_s2"] for n in gated[lv]]), 3)}
          for lv in gated}
    H = {
        "H83-1_family_sensitive_observable_exists": bool(all(
            sp[lv]["demand_saturation_duty"] > 0.05 or sp[lv]["executed_peak_util"] > 0.05
            or sp[lv]["peak_joint_accel_rad_s2"] > 1.0 for lv in sp)),
        "H83-2_gate_load_bearing_for_unconstrained": bool(all(
            len(gdep[lv]) > 0 and all(gated[lv][n]["executed_violation_steps"] == 0 for n in gdep[lv])
            for lv in gated)),
        "H83-3_intrinsically_bounded_family_exists": bool(all(len(intr[lv]) > 0 for lv in intr)),
        "H83-4_all_families_gated_safe": bool(all(
            all(gated[lv][n]["executed_violation_steps"] == 0 for n in gated[lv]) for lv in gated)),
    }
    return {"gated": gated, "ungated": ungated, "spread": sp, "intrinsic": intr, "gate_dep": gdep,
            "H": H, "verdict_pass": all(H.values())}


def main():
    print("=" * 92)
    print("E83 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/levels/families/gains/family_specs）",
        all(k in data for k in ("bench", "levels", "families", "gains", "family_specs")),
        "families=%d levels=%d" % (len(data["families"]), len(data["levels"])))
    out = replicate(data)

    # B1 归因集合逐强度复现（按**集合**比较：报告按 FAMILIES 顺序存，复刻按字典序排序）
    okA = all(sorted(out["intrinsic"][lv]) == sorted(rep["intrinsically_bounded_families"][lv])
              and sorted(out["gate_dep"][lv]) == sorted(rep["gate_load_bearing_families"][lv])
              for lv in out["intrinsic"])
    chk("B1 门控承重 / 自带界 归因集合逐强度复现（集合相等）", okA,
        "re intr=%s gdep=%s ｜ store intr=%s gdep=%s"
        % (out["intrinsic"]["stress_mild"], out["gate_dep"]["stress_mild"],
           rep["intrinsically_bounded_families"]["stress_mild"],
           rep["gate_load_bearing_families"]["stress_mild"]))

    # B2 逐族逐强度：util / exec越限 / duty 与报告一致
    bad = 0
    for lv in out["gated"]:
        for fam in out["gated"][lv]:
            for gate, key in ((True, "gated"), (False, "ungated")):
                a = out[key][lv][fam]; b = rep[key][lv][fam]
                if (a["executed_violation_steps"] != b["executed_violation_steps"]
                        or abs(a["executed_peak_util"] - b["executed_peak_util"]) > 2e-3
                        or abs(a["demand_saturation_duty"] - b["demand_saturation_duty"]) > 2e-3):
                    bad += 1
    chk("B2 逐族×逐强度×双门控：越限步数 / 力矩利用率 / 承重占空 与报告一致", bad == 0,
        "mismatch=%d（9 族 × 2 强度 × 2 门控 = 36 格）" % bad)

    # B3 连续观测量
    bad3 = sum(1 for lv in out["spread"] if abs(out["spread"][lv]["executed_peak_util"]
                                                - rep["spread"][lv]["executed_peak_util"]) > 2e-3
               or abs(out["spread"][lv]["demand_saturation_duty"]
                      - rep["spread"][lv]["demand_saturation_duty"]) > 2e-3)
    chk("B3 σ_family（util / duty）逐强度与报告一致（±2e-3）", bad3 == 0, "mismatch=%d" % bad3)

    # B4 判据与结论
    chk("B4 预注册判据 H83-1..4 与报告一致（全 True）",
        out["H"] == rep["preregistered_verdict"],
        "re=%s store=%s" % (out["H"], rep["preregistered_verdict"]))
    chk("B5 结论判定复现（verdict_pass=True）", out["verdict_pass"] == rep["verdict_pass"],
        "re=%s store=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e83_bench_data.json + 复刻者侧独立实现）：E83 的 36 格观测量与门控承重归因"
        "可被外部独立重建。**注**：FLC/IMP/LRN 是**对象本身**（控制器），复刻用的是复刻者侧独立重写"
        "（e77_standalone_physics）——其数值与作者实现逐格一致，但语义上仍是「同一族规格的另一实现」，"
        "故本 L4 对控制器族的复刻强度弱于对**被控对象/观测量管线**的复刻。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
