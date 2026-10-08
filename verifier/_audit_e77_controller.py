# -*- coding: utf-8 -*-
"""_audit_e77_controller.py —— e77 的正向（F1-F9）+ 反向（R1-R6）双向审核。

纪律：正向用模块原语独立驱动重算报告每个数值；反向构造篡改证明审计敏感。
F9 = LRN **冷启动构造自足性**（子进程内不预先训练 → 仍须等价 PD_G+重力），
防"类属性 _W 未设 → LRN 静默退化"这一隐藏状态依赖。
产物：_audit_e77_controller.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))


def _find_src(start, max_up=6):
    """上溯≤6 层找含 `planning/` 的目录（仓库内 = src；复刻目录内 = 本目录）。
    ★ 不能用固定层数 dirname：复刻 suite 把脚本复制到 _repl_full 后运行时层数会变。"""
    d = os.path.abspath(start)
    for _ in range(max_up):
        if os.path.isdir(os.path.join(d, "planning")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.abspath(start)


SRC = _find_src(EVAL)
sys.path.insert(0, EVAL)
sys.path.insert(0, SRC)

import e68_control_strategy_suite as E68          # noqa: E402
import e77_controller_family_extension as E77     # noqa: E402
import planning.control_strategies as CS          # noqa: E402

REPORT = E77.REPORT
_res = {"audit": "_audit_e77_controller", "checks": [], "tamper": [], "findings": []}
_n = [0]
_F9_CACHE = {}
FIELDS = ("rise_s", "settling_s", "overshoot", "ss_err_rad", "tau_max_executed",
          "executed_violation_steps", "demand_violation_steps", "demand_tau_max", "diverged")


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    A, B, C, D = rep["A_metrics"], rep["B_interface"], rep["C_capability"], rep["D_determinism_and_ablation"]
    need = ["A_metrics", "B_interface", "C_capability", "D_determinism_and_ablation",
            "criteria", "criteria_scope", "verdict_pass"]
    ch.append(_chk("F1_structure", all(k in rep for k in need), "missing=%s" % [k for k in need if k not in rep]))

    # F2 A 段指标独立重算（同口径驱动 E68.simulate）
    mism = []
    for n in E77.NEW_FAMILIES:
        for sc in E77.SCENARIOS:
            r = E77.run_family(n, sc)
            st = A["rows"][n][sc]
            for f in FIELDS:
                if r[f] != st[f]:
                    mism.append((n, sc, f, r[f], st[f]))
    ch.append(_chk("F2_A_metrics_recompute", not mism, "mismatch=%s" % mism[:3] if mism else "全一致"))

    # F3 接口一致性重算
    ok3 = True
    for n in E77.NEW_FAMILIES:
        c = E77.build(n)
        cap = c.capability()
        if not (set(cap) >= {"controller", "family", "tau_max", "source"} and isinstance(c.saturation_policy(), dict)):
            ok3 = False
    ch.append(_chk("F3_interface_recompute", ok3 and B["all_ok"] == ok3, "重算 %s 报告 %s" % (ok3, B["all_ok"])))

    # F3b 接口**逐行**重算（防"改单行字段而 all_ok 不变"漏报）
    ok3b = True
    for n in E77.NEW_FAMILIES:
        c = E77.build(n); cap = c.capability(); st = B["rows"][n]
        if st["capability_has_required"] != (set(cap) >= {"controller", "family", "tau_max", "source"}):
            ok3b = False
        if st["saturation_policy_is_dict"] != isinstance(c.saturation_policy(), dict):
            ok3b = False
        if st["family"] != c.family or st["name"] != c.name:
            ok3b = False
        if st["output_gated_ok"] is not True:
            ok3b = False
    ch.append(_chk("F3b_interface_rows_recompute", ok3b, "逐族接口字段 == 重算"))

    # F4 能力：标称 ss 重算
    ok4 = True
    for n in E77.NEW_FAMILIES:
        r = E77.run_family(n, "nominal")
        if abs(r["ss_err_rad"] - C["rows"][n]["ss_err_nominal"]) > 1e-9:
            ok4 = False
        if (r["ss_err_rad"] <= E77.SS_TOL) != C["rows"][n]["ss_bounded_nominal"]:
            ok4 = False
    ch.append(_chk("F4_capability_recompute", ok4, "标称 ss 重算"))

    # F5 确定性 + 门控消融重算
    abl_rep = {k: v["executed_violation_steps_no_gate"] for k, v in D["gate_off_ablation"].items()}
    abl_rec = {n: E77.run_family(n, "stress", gate=False)["executed_violation_steps"] for n in E77.NEW_FAMILIES}
    ch.append(_chk("F5_ablation_recompute", abl_rep == abl_rec and D["gate_load_bearing"] == any(v > 0 for v in abl_rec.values()),
                   "重算 %s 报告 %s" % (abl_rec, abl_rep)))

    # F6 不变量自洽
    c = rep["criteria"]
    C2 = all(A["rows"][n][sc]["executed_violation_steps"] == 0 and not A["rows"][n][sc]["diverged"]
             for n in E77.NEW_FAMILIES for sc in E77.SCENARIOS)
    inv = bool(c["C1_interface_consistent"] and B["all_ok"] and c["C2_danger_zero_executed"] and C2
               and c["C3_capability_reportable_and_ss_bounded"] and C["all_ok"]
               and c["C4_deterministic"] and D["deterministic"])
    ch.append(_chk("F6_invariants_self_consistent", inv == bool(rep["verdict_pass"]),
                   "inv=%s rep=%s" % (inv, rep["verdict_pass"])))

    # F7 参照族可比性（PD_G/MPC 存在且被报告）
    ch.append(_chk("F7_reference_families_present",
                   all(k in rep["A_ref_metrics"] for k in E77.REF_FAMILIES)
                   and all(sc in rep["A_ref_metrics"]["PD_G"] for sc in E77.SCENARIOS),
                   "ref=%s" % list(rep["A_ref_metrics"].keys())))

    # F8 门控承重：至少一族在 stress 下 gate-off 越限 > 0
    ch.append(_chk("F8_gate_load_bearing_premise", D["gate_load_bearing"],
                   "ablation=%s" % abl_rep))

    # F9 LRN 构造自足（**冷启动、顺序无关**）：子进程内不预先 train_linear_policy()，
    #    直接 make_controller('LRN') 仍须等价 PD_G+重力。★ 防"类属性 _W 未设→静默退化"漏报。
    import subprocess
    # ★ 复刻安全：PYTHONPATH 用 _find_src 结果（仓库=…/src；_repl_full= 本目录）。
    #   不能写死 dirname(EVAL)：脚本被复制进复刻目录后会指向**错层** → 子进程 import 失败。
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SRC, EVAL])
    kp_d, kd_d = float(E77.KP_D), float(E77.KD_D)
    code = (
        "import numpy as np, planning.control_strategies as CS\n"
        "c=CS.make_controller('LRN',kp=%r,kd=%r,q_ref=np.array(%r))\n"
        "p=CS.make_controller('PD_G',kp=%r,kd=%r,q_ref=np.array(%r))\n"
        "rng=np.random.RandomState(3)\n"
        "pts=[(rng.uniform(-2,2,2),rng.uniform(-3,3,2),rng.uniform(-2,2,2)) for _ in range(300)]\n"
        "m=max(float(np.max(np.abs(c.act(q,qd,r,0.0)-p.act(q,qd,r,0.0)))) for q,qd,r in pts)\n"
        "print(repr(m))\n" % (kp_d, kd_d, list(E77.Q_REF), kp_d, kd_d, list(E77.Q_REF))
    )
    if (kp_d, kd_d) in _F9_CACHE:
        f9, d9 = _F9_CACHE[(kp_d, kd_d)]
    else:
        try:
            pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env=env, timeout=300)
            if pr.returncode != 0:
                f9, d9 = False, "rc=%d stderr=%s" % (pr.returncode, pr.stderr.strip()[-160:])
            else:
                val = float(pr.stdout.strip().splitlines()[-1])
                f9 = bool(val < 1e-9)
                d9 = "冷启动 max|LRN−PD_G|=%.3e ⇒ 顺序无关" % val
        except Exception as e:  # noqa: BLE001
            f9, d9 = False, "err %s" % str(e)[:160]
        _F9_CACHE[(kp_d, kd_d)] = (f9, d9)
    ch.append(_chk("F9_LRN_self_contained_cold_construction", f9, d9))
    return ch


def _tamp(name, mut, prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    # ★ 精确名匹配（前缀 + "_"）：禁用裸前缀，防 "F1" 吞 "F10"/"S1" 吞 "S10" 类误报
    caught = any((not c[1]) and c[0].startswith(tuple(p + "_" for p in prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E77 作者双向审核"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    np_ = sum(c[1] for c in fwd)
    print("正向：%d/%d" % (np_, len(fwd)))
    print("[R] 篡改：")
    _tamp("R1 改 A 段 FLC nominal ss → F2 必报",
          lambda r: r["A_metrics"]["rows"]["FLC"]["nominal"].__setitem__("ss_err_rad", 0.5), ["F2"])
    _tamp("R2 改 C 段 FLC ss_nominal → F4 必报",
          lambda r: r["C_capability"]["rows"]["FLC"].__setitem__("ss_err_nominal", 0.5), ["F4"])
    _tamp("R3 改 D 段门控消融 → F5 必报",
          lambda r: r["D_determinism_and_ablation"]["gate_off_ablation"]["FLC"].__setitem__("executed_violation_steps_no_gate", 999), ["F5"])
    _tamp("R4 抹掉 B 段单行接口字段 → F3b 必报",
          lambda r: r["B_interface"]["rows"]["IMP"].__setitem__("capability_has_required", False), ["F3b"])
    _tamp("R5 翻转 verdict_pass → F6 必报", lambda r: r.__setitem__("verdict_pass", False), ["F6"])
    _tamp("R6 抹掉参照族 → F7 必报",
          lambda r: r["A_ref_metrics"].pop("MPC", None), ["F7"])
    # R7：新增自述字段是"新增可篡改面" → 必须也被覆盖（删掉 criteria_scope → F1 必报）
    _tamp("R7 删掉 criteria_scope 自述字段 → F1 必报",
          lambda r: r.pop("criteria_scope", None), ["F1"])
    ntp = sum(1 for t in _res["tamper"] if t["caught"]); ntt = len(_res["tamper"])
    _res["n_pass"], _res["n_total"] = np_, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["audit_pass"] = bool(np_ == len(fwd) and ntp == ntt)
    _res["findings"] = ["★ 作者双向审核：正向 %d/%d + 篡改 %d/%d。" % (np_, len(fwd), ntp, ntt)]
    with open(os.path.join(EVAL, "_audit_e77_controller.json"), "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n审核：正向 %d/%d ｜ 篡改 %d/%d ｜ audit_pass = %s" % (np_, len(fwd), ntp, ntt, _res["audit_pass"]))
    return _res


if __name__ == "__main__":
    main()
