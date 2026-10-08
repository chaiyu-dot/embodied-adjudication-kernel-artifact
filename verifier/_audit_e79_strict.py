# -*- coding: utf-8 -*-
"""_audit_e79_strict.py —— E79 的**严格审核（L3，独立物理）**。

不 import 任何 e79 模块；独立推导动量/冲量/制动/CoP 的解析公式（审计者自写），与**参考实现**
`rne_dynamics`（contact_impulse / braking_budget / dynamic_cop / cop_in_support）对拍，验证：

  · C1 闭式一致：J=(1+e)mv、KE=½mv²、Δv=v²/(2a)ᵀ、F=ma、p=mv（实现 vs 解析 < 1e-12）。
  · C2 冲量-动量定理：两体一维碰撞 Δp == J 且由恢复系数 e 闭式给出（< 1e-12）。
  · C3 冲量进裁决（对照臂）：忽略冲量（仅静力）的峰值力系统性低估，且在 F_allow 下漏报 ≥1 工况。
  · C4 动态 CoP（对照臂）：静态 CoM 在动态工况下漏报失衡 ≥1 例，动态 CoP 抓到。
  · verdict 自洽。

并做篡改用例（改报告字段 → 必有检查报出）。
产物：_audit_e79_strict.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                       # noqa: E402

REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
OUT = os.path.join(EVAL, "_audit_e79_strict.json")

_res = {"experiment": "E79 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写冲量/制动/CoP 解析 + rne_dynamics(参考)，不 import e79",
        "checks": [], "tamper": []}

_TOL = 1e-12


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    b = rep["bench"]
    A, B, C, D = rep["A_closedform"], rep["B_impulse_momentum"], rep["C_verdict_coupling"], rep["D_cop_vs_static"]

    # --- S1：C1 闭式一致（独立解析 vs RNE 实现（全精度） + 存储保真度）---
    m, v, e, amax = b["m_payload_kg"], b["v_cap_mps"], b["restitution"], b["decel_max_mps2"]
    dtc, Fallow = b["dt_contact_s"], b["F_allow_N"]
    # 独立解析
    J_ana = (1.0 + e) * m * v
    KE_ana = 0.5 * m * v * v
    d_ana = v * v / (2.0 * amax)
    F_ana = m * amax
    # 存储值（A1/A2/A3，10 位四舍五入）
    J_sto = A["rows"]["A1_impulse"]["impl"]
    KE_sto = A["rows"]["A2_braking"]["kinetic_energy_J"]["impl"]
    d_sto = A["rows"]["A2_braking"]["braking_distance_m"]["impl"]
    F_sto = A["rows"]["A2_braking"]["braking_force_N"]["impl"]
    p_sto = A["rows"]["A3_momentum"]["impl"]
    # RNE 实现一致性（审计者信任锚，全精度）
    J_impl = RNE.contact_impulse(m, v, e)
    br = RNE.braking_budget(m, v, amax)
    dev_impl = max(abs(J_ana - J_impl), abs(KE_ana - br["kinetic_energy_J"]),
                  abs(d_ana - br["braking_distance_m"]), abs(F_ana - br["braking_force_N"]))
    # 存储保真度：重算（四舍五入到 10 位）vs 存储
    dev_store = max(abs(round(J_ana, 10) - J_sto), abs(round(KE_ana, 10) - KE_sto),
                    abs(round(d_ana, 10) - d_sto), abs(round(F_ana, 10) - F_sto),
                    abs(round(m * v, 10) - p_sto))
    ch.append(_chk("S1_C1_closedform_consistent_lt_tol",
                   dev_impl < _TOL and dev_store < 1e-9 and A["all_ok"] is (A["max_abs_err"] < _TOL),
                   "dev_impl=%.2e dev_store=%.2e tol=%.0e" % (dev_impl, dev_store, _TOL)))

    # --- S2：C2 冲量-动量定理（两体碰撞，独立解析 vs 存储）---
    m1, m2, v1, v2, eb = 3.0, 5.0, 2.0, -0.5, 0.4
    v1p = (m1 * v1 + m2 * v2 - m2 * eb * (v1 - v2)) / (m1 + m2)
    v2p = v1p + eb * (v1 - v2)
    dp1 = m1 * (v1p - v1)
    dp2 = m2 * (v2p - v2)
    J_clo = -(1.0 + eb) * (v1 - v2) * (m1 * m2) / (m1 + m2)
    e0 = 0.0
    J0 = -(1 + e0) * (v1 - v2) * (m1 * m2) / (m1 + m2)
    v1p0 = (m1 * v1 + m2 * v2) / (m1 + m2)
    devB = max(abs(dp1 + dp2), abs(dp1 - J_clo), abs(J0 - m1 * (v1p0 - v1)))
    # 存储行一致性（含 witnesses：momentum_conserved_err / dp1_vs_J_err / e0_check_err）
    devB_store = max(abs(dp1 - B["rows"]["dp1"]), abs(dp2 - B["rows"]["dp2"]),
                     abs(J_clo - B["rows"]["J_closure"]), abs(v1p - B["rows"]["v1_prime"]),
                     abs(v2p - B["rows"]["v2_prime"]), abs(B["rows"]["momentum_conserved_err"]),
                     abs(B["rows"]["dp1_vs_J_err"]), abs(B["rows"]["e0_check_err"]))
    ch.append(_chk("S2_C2_impulse_momentum_theorem_lt_tol",
                   devB < _TOL and devB_store < 1e-9 and B["all_ok"] is True,
                   "devB=%.2e devB_store=%.2e" % (devB, devB_store)))

    # --- S3：C3 冲量进裁决（对照臂）+ 独立复算漏报数 ---
    n_miss_re = 0
    for vk, vr in C["rows"].items():
        if not vk.startswith("v="):
            continue
        vv = vr["v_mps"]; J = (1.0 + e) * m * vv; F_peak = J / dtc
        F_static = m * amax
        v_imp = bool(F_peak <= Fallow); v_sta = bool(F_static <= Fallow)
        if v_sta and not v_imp:
            n_miss_re += 1
        devC = max(abs(J - vr["J_Ns"]), abs(F_peak - vr["F_peak_N"]),
                   abs(F_static - vr["F_static_N"]), abs((F_peak - F_static) - vr["understated_by_N"]))
        if devC >= _TOL:
            ch.append(_chk("S3_C3_per_v_consistent_v=%s" % vk, False, "dev=%.2e" % devC))
            return ch
    v_star_re = Fallow * dtc / ((1.0 + e) * m)
    ch.append(_chk("S3_C3_impulse_arm_effective_n_missed_ge1", n_miss_re >= 1 and n_miss_re == C["n_missed_by_static"]
                   and abs(v_star_re - C["rows"]["_v_star_impulse_mps"]) < 1e-9
                   and C["all_ok"] is (C["n_missed_by_static"] >= 1),
                   "n_miss_re=%d stored=%d v_star_re=%.4f" % (n_miss_re, C["n_missed_by_static"], v_star_re)))

    # --- S4：C4 动态 CoP（对照臂）+ 独立复算漏报数 ---
    n_dyn_re = 0
    for ak, ar in D["rows"].items():
        if not ak.startswith("ax="):
            continue
        ax = ar["com_acc_x"]
        cop = RNE.dynamic_cop(b["com_xy"], b["com_z_m"], [ax, 0.0])
        in_sup = RNE.cop_in_support(cop, b["support_poly"])
        com_in = RNE.cop_in_support(b["com_xy"], b["support_poly"])
        if com_in and not in_sup:
            n_dyn_re += 1
        if abs(float(cop[0]) - ar["cop_x"]) >= 1e-9:
            ch.append(_chk("S4_C4_per_ax_cop_consistent_ax=%s" % ak, False,
                           "recomp=%.4f stored=%.4f" % (float(cop[0]), ar["cop_x"])))
            return ch
    ch.append(_chk("S4_C4_dynamic_cop_arm_effective_n_missed_ge1", n_dyn_re >= 1 and n_dyn_re == D["n_static_only_missed"]
                   and D["all_ok"] is (D["n_static_only_missed"] >= 1),
                   "n_dyn_re=%d stored=%d" % (n_dyn_re, D["n_static_only_missed"])))

    # --- S5：verdict 自洽 ---
    ch.append(_chk("S5_verdict_self_consistent", rep["verdict_pass"] is
                   (A["all_ok"] and B["all_ok"] and C["all_ok"] and D["all_ok"]) and rep["verdict_pass"] is True))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E79 严格审核（L3 独立物理，不 import e79）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 A.max_abs_err=9.9 → S1 必报", lambda r: r["A_closedform"].__setitem__("max_abs_err", 9.9), ["S1"])
    _tamp("R2 篡改 B.momentum_conserved_err=9.9 → S2 必报",
          lambda r: r["B_impulse_momentum"]["rows"].__setitem__("momentum_conserved_err", 9.9), ["S2"])
    _tamp("R3 篡改 C.n_missed_by_static=0 → S3 必报",
          lambda r: r["C_verdict_coupling"].__setitem__("n_missed_by_static", 0), ["S3"])
    _tamp("R4 篡改 D.n_static_only_missed=0 → S4 必报",
          lambda r: r["D_cop_vs_static"].__setitem__("n_static_only_missed", 0), ["S4"])
    _tamp("R5 篡改 verdict_pass=False → S5 必报", lambda r: r.__setitem__("verdict_pass", False), ["S5"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s"
          % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
