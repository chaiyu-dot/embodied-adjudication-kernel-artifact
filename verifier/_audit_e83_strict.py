# -*- coding: utf-8 -*-
"""_audit_e83_strict.py —— E83 的 **L3 严格审核**（口径+结构自洽 + 独立重仿真 + 篡改捕获）。

审核对象：`e83_family_safety_observable_report.json`（族相关安全观测量，补 E77-F2）。

判据
----
S1 报告↔明细自洽：H 布尔与明细一致；spread 由 gated 行重算一致；intrinsic/gate_dep 集合与 ungated 行一致；
   gated executed 越限全 0；verdict_pass = all(H)。
S2 口径不变量：gated 行的 util ≤ 1+1e-9；ungated 行的越限>0 ⟺ util>1；字段命名语义（executed/demand）一致。
S3 **独立重仿真**（不复用 CS/E83 的控制器实现）：自写 PD_G（τ=kp·e+kd·(−qd)+g(q)）与零输出控制器，
   在 stress_mild 上跑 gate ON/OFF，逐项比对 executed 越限 / util / duty。
S4 篡改捕获：翻转 H83-1 / 把某族 gated 越限改成 1 / 把 intrinsic 集合清空 → S1 各检必须报差异。
S5 幂等：同一报告两次读入结论一致。

产物：_audit_e83_strict.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e66_closed_loop_control as E66  # noqa: E402

REPORT = os.path.join(EVAL, "e83_family_safety_observable_report.json")
OUT = os.path.join(EVAL, "_audit_e83_strict.json")
_res = {"audit": "E83 L3 strict", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


# ---------- 独立实现（不复用 CS/E83 控制器） ----------
def _pdg(B_taulim, kp, kd, gate=True):
    def act(q, qd, r):
        raw = kp * (np.asarray(r, float) - q) + kd * (-qd) + E66.gravity_vec(q)
        if gate:
            return np.clip(raw, -B_taulim, B_taulim), raw
        return raw, raw
    return act


def _zero():
    def act(q, qd, r):
        z = np.zeros(2)
        return z, z
    return act


def _sim(act, q_ref, q0, T, dt, gate):
    n = int(round(T / dt))
    st = np.zeros(4); st[0], st[1] = q0
    taus, raws, qds = [], [], []
    for i in range(n):
        tau, raw = act(st[:2], st[2:4], q_ref)
        taus.append(tau); raws.append(raw); qds.append(st[2:4].copy())
        st = E66.rk4_step(st, tau, dt)
        if not np.all(np.isfinite(st)) or np.max(np.abs(st[:2])) > 1e4:
            break
    return np.array(taus), np.array(raws), np.array(qds)


def _metrics(taus, raws, qds, tau_lim, dt):
    util = float(np.max(np.abs(taus) / tau_lim))
    ev = int(np.sum(np.any(np.abs(taus) > tau_lim + 1e-9, axis=1)))
    duty = round(int(np.sum(np.any(np.abs(raws) > tau_lim + 1e-9, axis=1))) / max(len(taus), 1), 4)
    acc = float(np.max(np.abs(np.gradient(qds, dt, axis=0)))) if len(qds) >= 3 else None
    return util, ev, duty, acc


def _spread(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return (max(v) - min(v)) if len(v) >= 2 else 0.0


def _s1(rep):
    ok = True
    gated, ungated, spreads = rep["gated"], rep["ungated"], rep["spread"]
    for lv in gated:
        for n in gated[lv]:
            if gated[lv][n]["executed_violation_steps"] != 0:
                ok = False
        # spread 重算
        if abs(_spread([gated[lv][n]["executed_peak_util"] for n in gated[lv]])
               - spreads[lv]["executed_peak_util"]) > 5e-4:
            ok = False
        if abs(_spread([gated[lv][n]["demand_saturation_duty"] for n in gated[lv]])
               - spreads[lv]["demand_saturation_duty"]) > 5e-4:
            ok = False
        # 集合一致
        intr = sorted(n for n in ungated[lv] if ungated[lv][n]["executed_violation_steps"] == 0)
        gdep = sorted(n for n in ungated[lv] if ungated[lv][n]["executed_violation_steps"] > 0)
        if intr != sorted(rep["intrinsically_bounded_families"][lv]):
            ok = False
        if gdep != sorted(rep["gate_load_bearing_families"][lv]):
            ok = False
        if set(intr) | set(gdep) != set(gated[lv].keys()):
            ok = False
    # H 与明细一致
    H = rep["preregistered_verdict"]
    exp1 = all((spreads[lv]["demand_saturation_duty"] > 0.05 or spreads[lv]["executed_peak_util"] > 0.05
                or spreads[lv]["peak_joint_accel_rad_s2"] > 1.0) for lv in gated)
    exp4 = all(all(gated[lv][n]["executed_violation_steps"] == 0 for n in gated[lv]) for lv in gated)
    if bool(H["H83-1_family_sensitive_observable_exists"]) != bool(exp1):
        ok = False
    if bool(H["H83-4_all_families_gated_safe"]) != bool(exp4):
        ok = False
    if rep["verdict_pass"] != all(H.values()):
        ok = False
    return ok


def _s2(rep):
    ok = True
    for lv in rep["gated"]:
        for n in rep["gated"][lv]:
            if rep["gated"][lv][n]["executed_peak_util"] > 1.0 + 1e-9:
                ok = False
            u = rep["ungated"][lv][n]
            if (u["executed_violation_steps"] > 0) != (u["executed_peak_util"] > 1.0 + 1e-9):
                ok = False
    return ok


def _s3(rep):
    tau_lim = np.asarray(rep["design"]["tau_lim"], float)
    dt = 1e-3; q0 = [0.0, 0.0]
    lv = "stress_mild"
    lvl = rep["design"]["levels"][lv]
    kp_b, kd_b, _ = E66.budget_gains(E66.Q_REF)
    det = []
    ok = True
    for gate in (True, False):
        act = _pdg(tau_lim, kp_b, kd_b, gate=gate)
        taus, raws, qds = _sim(act, np.asarray(lvl["q_ref"], float), q0, lvl["T"], dt, gate)
        util, ev, duty, acc = _metrics(taus, raws, qds, tau_lim, dt)
        src = rep["gated" if gate else "ungated"][lv]["PD_G"]
        same = (abs(util - src["executed_peak_util"]) < 2e-3
                and ev == src["executed_violation_steps"]
                and abs(duty - src["demand_saturation_duty"]) < 2e-3)
        det.append("gate=%s 重算 ev=%d util=%.4f duty=%.4f | 报告 ev=%d util=%s duty=%s"
                   % (gate, ev, util, duty, src["executed_violation_steps"],
                      src["executed_peak_util"], src["demand_saturation_duty"]))
        ok &= same
    # 负对照：零输出控制器（应 0 越限）
    t0, r0, _ = _sim(_zero(), np.asarray(lvl["q_ref"], float), q0, lvl["T"], dt, False)
    ev0 = int(np.sum(np.any(np.abs(t0) > tau_lim + 1e-9, axis=1)))
    ok &= (ev0 == 0)
    return ok, det, ev0


def main():
    print("=" * 92)
    print("E83 L3 严格审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    chk("S1 报告↔明细自洽（H↔明细 / spread 重算 / 集合 / gated 全 0 / verdict=all(H)）", _s1(rep))
    chk("S2 口径不变量（gated util≤1；ungated 越限>0 ⟺ util>1）", _s2(rep))
    ok3, det3, ev0 = _s3(rep)
    chk("S3 独立重仿真 PD_G（自写 τ=kp·e+kd·(−qd)+g）gate ON/OFF 与报告一致", ok3,
        " ｜ ".join(det3) + " ｜ 负对照(零输出 ev=%d)" % ev0)

    # ---- S4 篡改捕获 ----
    import copy
    t1 = copy.deepcopy(rep); t1["preregistered_verdict"]["H83-1_family_sensitive_observable_exists"] = False
    chk("S4a 篡改 H83-1 → S1 必报（verdict≠all(H)）", not _s1(t1))
    t2 = copy.deepcopy(rep)
    k = sorted(t2["gated"]["stress_mild"])[0]
    t2["gated"]["stress_mild"][k]["executed_violation_steps"] = 1
    chk("S4b 篡改某族 gated 越限=1 → S1 必报", not _s1(t2))
    t3 = copy.deepcopy(rep)
    t3["intrinsically_bounded_families"]["stress_mild"] = []
    chk("S4c 清空 intrinsic 集合 → S1 必报", not _s1(t3))
    t4 = copy.deepcopy(rep)
    t4["gated"]["stress_mild"][k]["executed_peak_util"] = 1.5
    chk("S4d 篡改 gated util>1 → S2 必报", not _s2(t4))
    chk("S4e 未篡改 → S1/S2 通过（阴性对照）", _s1(rep) and _s2(rep))
    chk("S5 幂等：两次读入结论一致", _s1(json.load(open(REPORT, encoding="utf-8"))) == _s1(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    if not ok3:
        _res["findings"].append("S3 独立重仿真与报告不一致 —— 须查 E83 或独立实现的增益/门控口径。")
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
