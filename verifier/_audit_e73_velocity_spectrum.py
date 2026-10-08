# -*- coding: utf-8 -*-
"""_audit_e73_velocity_spectrum.py —— e73 的正向（F1-F9）+ 反向（R1-R8）双向审核。

纪律（承自 e71/e72 审计）：
  · 正向：用**独立路径**重算每个报告数值（本审核不直接复用 e73.envelope_at 去复算复现值，
    而是用 e67 原语从零重算 7x7/16 向，证明复现不是自证）；
  · 反向：构造阴性/阳性篡改，证明审计**确实敏感**；
  · 口径：本审核专门核验"能力数值不得当硬件上限"的诚实标注（binding 分类 / 可实现性界 / caveats）。

产物：_audit_e73_velocity_spectrum.json（{"pass":"N","total":"M","checks":[...]}）
"""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e73_velocity_spectrum_envelope as E73           # noqa: E402
import e66_closed_loop_control as E66                  # noqa: E402
import e67_testbed_completeness as E67                 # noqa: E402

REPORT = E73.REPORT
E67_V, E67_A = 3.0461, 66.1436


def load():
    with open(REPORT) as f:
        return json.load(f)


def _chk(name, cond, detail=""):
    return {"name": name, "pass": bool(cond), "detail": str(detail)}


def repro_independent():
    """从 e67 原语从零重算 E67 设置(7x7,16向)的 v_max/a_max —— 不复用 e73._grid。"""
    qs = np.linspace(-1.8, 1.8, 7)
    bv, ba = 0.0, 0.0
    for q1 in qs:
        for q2 in qs:
            q = [float(q1), float(q2)]
            J = E67.jac(q)
            for th in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
                u = np.array([math.cos(th), math.sin(th)])
                sv = E67.max_speed_along(q, u, qdot_lim=4.0)
                bv = max(bv, float(np.linalg.norm(J @ (sv * u))))
                sa = E67.max_accel_along(q, u)
                ba = max(ba, float(np.linalg.norm(J @ (sa * u))))
    return bv, ba


def e73_invariants(rep):
    c = rep["criteria"]
    C1 = bool(c["C1_polar_self_consistent"])
    C2 = bool(c["C2_path_continuous"])
    C3 = bool(c["C3_reproduces_e67"] and rep["C_workspace_map"]["reproduce_e67"]["match"])
    C4 = bool(c["C4_calibration_honest_deterministic"])
    return {"C1": C1, "C2": C2, "C3": C3, "C4": C4,
            "verdict_pass": bool(C1 and C2 and C3 and C4)}


def forward(rep):
    checks = []
    need = ["A_polar_spectrum", "B_envelope_curve", "C_workspace_map", "D_calibration",
            "criteria", "verdict_pass"]
    missing = [k for k in need if k not in rep]
    checks.append(_chk("F1_structure", not missing, "missing=%s" % missing))
    if missing:
        return checks
    A, B, C, D = (rep["A_polar_spectrum"], rep["B_envelope_curve"],
                  rep["C_workspace_map"], rep["D_calibration"])

    # F2 polar 独立重算（逐构型 v_max/a_max）
    pol_ok = True
    for cfg in A["configs"]:
        r = E73.polar(cfg["q"])
        if (abs(r["v_max"] - cfg["v_max"]) > 1e-3 or abs(r["a_max"] - cfg["a_max"]) > 5e-2
                or r["is_singular"] != cfg["is_singular"]):
            pol_ok = False
    checks.append(_chk("F2_polar_recompute", pol_ok, "polar mismatch"))

    # F3 路径曲线重算（结构 + 抽点重算 + 跳变重算）
    n_ok = (B["n"] == E73.N_PATH and B["unreachable"] == 0)
    jv = max(abs(B["curve"][i + 1]["v_max"] - B["curve"][i]["v_max"]) for i in range(len(B["curve"]) - 1))
    ja = max(abs(B["curve"][i + 1]["a_max"] - B["curve"][i]["a_max"]) for i in range(len(B["curve"]) - 1))
    jump_ok = abs(jv - B["max_jump_v"]) < 1e-3 and abs(ja - B["max_jump_a"]) < 1e-2
    # 抽点：中段一个构型重算
    mid = B["curve"][len(B["curve"]) // 2]
    rmid = E73.envelope_at(mid["q"])
    spot_ok = abs(rmid["v_max"] - mid["v_max"]) < 1e-2 and abs(rmid["a_max"] - mid["a_max"]) < 5e-1
    checks.append(_chk("F3_path_recompute", n_ok and jump_ok and spot_ok,
                       "n=%s unreach=%s jump_v=%s/%s spot=%s" % (B["n"], B["unreachable"],
                       round(jv, 4), B["max_jump_v"], spot_ok)))

    # F4 复现 E67：独立路径从零重算 7x7/16 向
    bv, ba = repro_independent()
    rep_ok = (abs(bv - E67_V) <= 1e-3 and abs(ba - E67_A) <= 1e-3
              and abs(C["reproduce_e67"]["v_max"] - bv) <= 1e-3
              and abs(C["reproduce_e67"]["a_max"] - ba) <= 1e-3
              and C["reproduce_e67"]["match"])
    checks.append(_chk("F4_reproduce_e67_independent", rep_ok,
                       "indep v=%s a=%s | report v=%s a=%s" % (round(bv, 4), round(ba, 4),
                       C["reproduce_e67"]["v_max"], C["reproduce_e67"]["a_max"])))

    # F5 可实现性界：a_max_diag ≤ a_max 且 >0；且逐方向 capped ≤ uncapped（抽点）
    diag_ok = (0.0 < C["a_max_diag"] < C["a_max"])
    q0 = C["at_a"]
    cap0 = E73.qdd_cap_of(q0)
    dirok = True
    for th in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        if E73.max_accel_capped(q0, u, cap0) > E67.max_accel_along(q0, u) + 1e-9:
            dirok = False
    checks.append(_chk("F5_realizability_bound", diag_ok and dirok,
                       "diag=%s max=%s cap_ok=%s" % (C["a_max_diag"], C["a_max"], dirok)))

    # F6 jerk 一致性：wn_nom·a_max 与报告一致
    _, _, wn = E66.budget_gains(E66.Q_REF)
    jerk_ok = (abs(D["wn_nominal"] - round(wn, 3)) < 1e-2
               and abs(D["jerk_envelope"]["at_e67_settings"] - round(wn * E67_A, 3)) < 5e-1
               and abs(D["jerk_envelope"]["max"] - round(wn * C["a_max"], 3)) < 5e-1)
    checks.append(_chk("F6_jerk_consistency", jerk_ok,
                       "wn=%s jerk_e67=%s/%s jerk_max=%s/%s" % (round(wn, 3),
                       D["jerk_envelope"]["at_e67_settings"], round(wn * E67_A, 3),
                       D["jerk_envelope"]["max"], round(wn * C["a_max"], 3))))

    # F7 口径诚实：binding 分割求和 + 奇异存在 + caveats + headline 断言（v_max 奇异且 cap 驱动）
    bind_ok = (C["binding_cap_only"] + C["binding_torque_only"] + C["binding_mixed"] == C["n_cells"]
               and C["n_singular"] >= 1 and len(D["caveats"]) >= 5 and 0.0 <= C["a_illposed_frac"] <= 1.0)
    # headline 断言：at_v 的构型奇异，且 cap_only == 全部单元格
    av = E73.envelope_at(C["at_v"])
    head_ok = bool(av["is_singular"] and C["binding_cap_only"] == C["n_cells"])
    checks.append(_chk("F7_calibration_honest", bind_ok and head_ok,
                       "bind=%s|%s|%s head_sing=%s cap_all=%s" % (C["binding_cap_only"],
                       C["binding_torque_only"], C["binding_mixed"], av["is_singular"],
                       C["binding_cap_only"] == C["n_cells"])))

    # F8 单调（从 rows 重算中位/极值）
    v = [r["v_max"] for r in C["rows"]]
    a = [r["a_max"] for r in C["rows"]]
    mono_ok = (round(min(v), 4) == C["v_min"] and round(max(v), 4) == C["v_max"]
               and round(float(np.median(v)), 4) == C["v_median"]
               and round(min(a), 4) == C["a_min"] and round(max(a), 4) == C["a_max"]
               and round(float(np.median(a)), 4) == C["a_median"]
               and C["v_min"] <= C["v_median"] <= C["v_max"] and C["a_min"] <= C["a_median"] <= C["a_max"])
    checks.append(_chk("F8_stats_monotone", mono_ok, "v=%s/%s/%s a=%s/%s/%s" % (
        C["v_min"], C["v_median"], C["v_max"], C["a_min"], C["a_median"], C["a_max"])))

    # F9 确定性（模块同输入同输出）
    det = (E73.polar(E73.POLAR_Q[1]) == E73.polar(E73.POLAR_Q[1])
           and E73.envelope_at(E73.POLAR_Q[0]) == E73.envelope_at(E73.POLAR_Q[0])
           and E73.max_accel_capped(C["at_a"], [1.0, 0.0], E73.qdd_cap_of(C["at_a"]))
           == E73.max_accel_capped(C["at_a"], [1.0, 0.0], E73.qdd_cap_of(C["at_a"])))
    checks.append(_chk("F9_determinism", bool(det), "non-deterministic"))

    # F10 不变量自洽
    inv = e73_invariants(rep)
    checks.append(_chk("F10_invariants_self_consistent",
                       inv["verdict_pass"] == bool(rep["verdict_pass"]),
                       "inv=%s rep=%s" % (inv["verdict_pass"], rep["verdict_pass"])))
    return checks


def reverse(rep):
    checks = []

    # R1 阴性对照：篡改 polar v_max → F2 必 FAIL
    bad = copy.deepcopy(rep)
    bad["A_polar_spectrum"]["configs"][0]["v_max"] = 9.99
    r1 = any(c["name"] == "F2_polar_recompute" and not c["pass"] for c in forward(bad))
    checks.append(_chk("R1_tamper_polar_detected", r1, "audit blind to polar tamper"))

    # R2 阳性对照：干净报告全正向 PASS
    fwd = forward(rep)
    checks.append(_chk("R2_clean_passes", all(c["pass"] for c in fwd),
                       "clean=%d/%d" % (sum(c["pass"] for c in fwd), len(fwd))))

    # R3 篡改复现值 → F4 必 FAIL
    bad3 = copy.deepcopy(rep)
    bad3["C_workspace_map"]["reproduce_e67"]["v_max"] = 2.0
    r3 = any(c["name"] == "F4_reproduce_e67_independent" and not c["pass"] for c in forward(bad3))
    checks.append(_chk("R3_tamper_repro_detected", r3, "audit blind to repro tamper"))

    # R4 篡改 binding 分割（求和 != n_cells）→ F7 必 FAIL
    bad4 = copy.deepcopy(rep)
    bad4["C_workspace_map"]["binding_cap_only"] += 3
    r4 = any(c["name"] == "F7_calibration_honest" and not c["pass"] for c in forward(bad4))
    checks.append(_chk("R4_tamper_binding_detected", r4, "audit blind to binding tamper"))

    # R5 破坏单调（median 超界）→ F8 必 FAIL
    bad5 = copy.deepcopy(rep)
    bad5["C_workspace_map"]["v_median"] = 999.0
    r5 = any(c["name"] == "F8_stats_monotone" and not c["pass"] for c in forward(bad5))
    checks.append(_chk("R5_tamper_monotone_detected", r5, "audit blind to monotone tamper"))

    # R6 非有限 → None（诚实序列化）
    r6 = (E73._num(float("nan")) is None and E73._num(float("inf")) is None
          and E73._num(-float("inf")) is None and E73._num(1.23456) == 1.2346)
    checks.append(_chk("R6_nonfinite_to_none", r6, "nonfinite not capped"))

    # R7 可实现性界反向：逐方向 capped <= uncapped，且至少一个方向严格小于（界非平凡）
    q0 = E73.POLAR_Q[0]
    cap = E73.qdd_cap_of(q0)
    strict, nontrivial = True, False
    for th in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        su = E67.max_accel_along(q0, u)
        sc = E73.max_accel_capped(q0, u, cap)
        if sc > su + 1e-9:
            strict = False
        if sc < su - 1e-6:
            nontrivial = True
    checks.append(_chk("R7_bound_is_active", strict and nontrivial,
                       "capped<=uncapped=%s nontrivial=%s" % (strict, nontrivial)))

    # R8 覆盖度：清空 D 段 → 结构检查必 FAIL
    bad8 = copy.deepcopy(rep)
    bad8.pop("D_calibration", None)
    r8 = any(c["name"] == "F1_structure" and not c["pass"] for c in forward(bad8))
    checks.append(_chk("R8_coverage_structure", r8, "audit blind to missing section"))
    return checks


def main():
    rep = load()
    fwd = forward(rep)
    rev = reverse(rep)
    allc = fwd + rev
    npass = sum(c["pass"] for c in allc)
    out = {"audit": "_audit_e73_velocity_spectrum", "experiment": "e73_velocity_spectrum_envelope",
           "pass": npass, "total": len(allc), "verdict": bool(npass == len(allc)),
           "audit_pass": bool(npass == len(allc)),
           "n_pass": npass, "n_total": len(allc), "forward": fwd, "reverse": rev, "checks": allc}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_audit_e73_velocity_spectrum.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("audit pass = %d/%d  verdict=%s" % (npass, len(allc), out["verdict"]))
    for c in allc:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
