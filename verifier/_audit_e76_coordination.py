# -*- coding: utf-8 -*-
"""_audit_e76_coordination.py —— e76 的正向（F1-F8）+ 反向（R1-R6）双向审核。

纪律：正向用模块原语 / E67 交叉独立重算报告数值；反向构造篡改证明审计敏感。
产物：_audit_e76_coordination.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import e66_closed_loop_control as E66          # noqa: E402
import e67_testbed_completeness as E67          # noqa: E402
import e76_dualarm_coordination as E76          # noqa: E402

REPORT = E76.REPORT
_res = {"audit": "_audit_e76_coordination", "checks": [], "tamper": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    A, B, C = rep["A_relative_pose_keeping"], rep["B_internal_force_decoupling"], rep["C_danger_zero"]
    need = ["A_relative_pose_keeping", "B_internal_force_decoupling", "C_danger_zero", "criteria", "verdict_pass"]
    ch.append(_chk("F1_structure", all(k in rep for k in need), "missing=%s" % [k for k in need if k not in rep]))

    # F2 A 段：用 E67.dual_sim 独立交叉重算漂移（同 hold_rel 口径）
    mism = []
    for k, v in A["rows"].items():
        mp = float(k.split("=")[1])
        r67 = E67.dual_sim(E76.TGT, hold_rel=True, mp_R=mp)
        if abs(r67["rel_drift_max_m"] - v["rel_drift_e67_crosscheck_m"]) > 1e-9:
            mism.append((k, r67["rel_drift_max_m"], v["rel_drift_e67_crosscheck_m"]))
    ch.append(_chk("F2_A_crosscheck_with_E67_dual_sim", not mism, "mismatch=%s" % mism[:2] if mism else "5 行一致"))

    # F3 B 段闭式代数独立重算
    pL = np.asarray(E76.TGT[0][:2], float); pR = np.asarray(E76.TGT[1][:2], float)
    geo = E76.grasp_geometry(pL, pR); GL, GR = E76.grasp_matrix(geo)
    W_int = E76.internal_wrench(E76.K_INT, geo)
    net = GL @ W_int[:2] + GR @ W_int[2:]
    G = np.hstack([GL, GR]); Gt = G.T; F_des = np.asarray(E76.F_DES, float)
    W_motion = Gt @ np.linalg.solve(G @ Gt, F_des)
    orth = float(abs(W_motion @ W_int) / max(np.linalg.norm(W_motion) * np.linalg.norm(W_int), 1e-12))
    errs = max(float(np.max(np.abs(G @ (W_motion + a * W_int) - F_des))) for a in (-5.0, -1.0, 0.0, 1.0, 5.0, 20.0))
    f3 = (abs(float(np.max(np.abs(net))) - B["net_object_wrench_inf"]) < 1e-15
          and abs(orth - B["orthogonality_motion_vs_internal"]) < 1e-18
          and abs(errs - B["motion_component_invariance_vs_alpha_max_err"]) < 1e-15)
    ch.append(_chk("F3_B_algebra_recompute", f3, "‖GᵀW_int‖=%.2e orth=%.2e α=%.2e" % (float(np.max(np.abs(net))), orth, errs)))

    # F4 C 段重算
    ok4 = True
    for k, v in C["rows"].items():
        mp = float(k.split("=")[1])
        r = E76.coord_sim(mp_R=mp, k_int=E76.K_INT, ramp=E76.RAMP)
        if r["violation_steps_demand"] != v["violation_steps_demand"] or abs(r["tau_max_demand"] - v["tau_max_demand"]) > 1e-3:
            ok4 = False
    ch.append(_chk("F4_C_danger_zero_recompute", ok4, "within-limit 场景 demand 越限重算一致"))

    # F5 不变量自洽
    c = rep["criteria"]
    inv = bool(c["C1_drift_le_1cm_within_static_limit_and_monotone"] and A["all_ok"]
               and c["C2_internal_decoupled_exact"] and B["all_ok"]
               and c["C3_danger_zero_within_limit"] and C["all_ok"])
    ch.append(_chk("F5_invariants_self_consistent", inv == bool(rep["verdict_pass"]),
                   "inv=%s rep=%s" % (inv, rep["verdict_pass"])))

    # F6 确定性
    r1 = E76.coord_sim(mp_R=0.1, ramp=E76.RAMP); r2 = E76.coord_sim(mp_R=0.1, ramp=E76.RAMP)
    ch.append(_chk("F6_determinism", r1 == r2, "same input same output"))

    # F7 静态上限前提：mp 超上限 → 越限>0（"先可行后可控"）
    over = E76.coord_sim(mp_R=1.0, ramp=E76.RAMP)
    ch.append(_chk("F7_static_limit_premise", over["violation_steps_demand"] > 0,
                   "mp=1.0 越限步=%d（超静态上限 0.357kg → 饱和）" % over["violation_steps_demand"]))

    # F2b A 段 max drift == 独立硬重算（防"改聚合量"漏报）
    o = E76.coord_sim(mp_R=0.2, ramp=E76.RAMP)
    # 复算上限内最大漂移
    safe_max = max(E76.coord_sim(mp_R=mp, ramp=E76.RAMP)["rel_drift_max_m"]
                   for mp in E76.MP_LADDER if mp <= 0.3572)
    ch.append(_chk("F2b_A_max_drift_equals_recompute",
                   round(safe_max, 6) == A["max_drift_within_limit_m"],
                   "重算 %.6f 报告 %.6f" % (safe_max, A["max_drift_within_limit_m"])))

    # F8 内力**计入需求口径**前提：大内力（20N）应使 demand 越限；无内力则否
    r0 = E76.coord_sim(mp_R=0.2, k_int=0.0, ramp=E76.RAMP)
    rbig = E76.coord_sim(mp_R=0.2, k_int=20.0, ramp=E76.RAMP)
    ch.append(_chk("F8_internal_counted_in_demand",
                   r0["violation_steps_demand"] == 0 and rbig["violation_steps_demand"] > 0,
                   "k_int=0 越限=%d；k_int=20 越限=%d（内力确被计入 demand）"
                   % (r0["violation_steps_demand"], rbig["violation_steps_demand"])))
    return ch


def _tamp(name, mut, prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E76 作者双向审核"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    np_ = sum(c[1] for c in fwd)
    print("正向：%d/%d" % (np_, len(fwd)))
    print("[R] 篡改：")
    _tamp("R1 改 A 段 E67 交叉漂移 → F2 必报",
          lambda r: r["A_relative_pose_keeping"]["rows"]["mp=0.20"].__setitem__("rel_drift_e67_crosscheck_m", 0.5), ["F2"])
    _tamp("R2 改 B 段 ‖GᵀW_int‖ → F3 必报",
          lambda r: r["B_internal_force_decoupling"].__setitem__("net_object_wrench_inf", 0.5), ["F3"])
    _tamp("R3 改 B 段正交项 → F3 必报",
          lambda r: r["B_internal_force_decoupling"].__setitem__("orthogonality_motion_vs_internal", 0.5), ["F3"])
    _tamp("R4 改 C 段越限步 → F4 必报",
          lambda r: r["C_danger_zero"]["rows"]["mp=0.20"].__setitem__("violation_steps_demand", 999), ["F4"])
    _tamp("R5 翻转 verdict_pass → F5 必报", lambda r: r.__setitem__("verdict_pass", False), ["F5"])
    _tamp("R6 改 A max drift → F2b 必报",
          lambda r: r["A_relative_pose_keeping"].__setitem__("max_drift_within_limit_m", 0.5), ["F2b"])
    ntp = sum(1 for t in _res["tamper"] if t["caught"]); ntt = len(_res["tamper"])
    _res["n_pass"], _res["n_total"] = np_, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["audit_pass"] = bool(np_ == len(fwd) and ntp == ntt)
    _res["findings"] = ["★ 作者双向审核：正向 %d/%d + 篡改 %d/%d。" % (np_, len(fwd), ntp, ntt)]
    with open(os.path.join(EVAL, "_audit_e76_coordination.json"), "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n审核：正向 %d/%d ｜ 篡改 %d/%d ｜ audit_pass = %s" % (np_, len(fwd), ntp, ntt, _res["audit_pass"]))
    return _res


if __name__ == "__main__":
    main()
