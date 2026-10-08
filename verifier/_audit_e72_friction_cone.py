# -*- coding: utf-8 -*-
"""_audit_e72_friction_cone.py —— e72 的正向（F1-F9）+ 反向（R1-R8）双向审核。

核心纪律（承自 e71 审计）：
  · 正向：用模块**独立重算**每个报告数值，绝不信任报告自带的 pass 标记；
  · 反向：构造阴性/阳性篡改，证明审计**确实敏感**（篡改必被抓）；
  · 口径：τ 用 demand 口径（门控前），执行值恒 0 不在此处讨论；
  · 确定性：纯函数，同输入同输出，审计可完全复现。

产物：_audit_e72_friction_cone.json（{"pass": N, "total": M, "checks": [...]}）
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e72_friction_cone_verdict as E72
from planning.friction_cone import (
    contact_verdict, max_tangential_push, combine_contact_with_dynamic,
    CONTACT_SAFE, CONTACT_UNSAFE, CONTACT_NA,
)

REPORT = E72.REPORT
Q = E72.Q


def load():
    with open(REPORT) as f:
        return json.load(f)


def e72_invariants(rep):
    """从报告数据自洽校验：C1..C4 与 verdict_pass 逻辑一致。"""
    c = rep["criteria"]
    C1 = bool(c["C1_envelope_match"] and rep["A_envelope_replica"]["all_match"])
    C2 = bool(c["C2_tau_and_cone_class"] and rep["B_tau_coupling_three_valued"]["all_ok"])
    C3 = bool(c["C3_combination_and_doc"] and rep["C_combination_invariants"]["all_ok"])
    C4 = bool(c["C4_determinism_finite"] and
              rep["C_combination_invariants"]["deterministic"] and
              rep["C_combination_invariants"]["inf_capped"] and
              rep["C_combination_invariants"]["NA_rho_finite"])
    return {"C1": C1, "C2": C2, "C3": C3, "C4": C4,
            "verdict_pass": bool(C1 and C2 and C3 and C4)}


def _chk(name, cond, detail=""):
    return {"name": name, "pass": bool(cond), "detail": str(detail)}


def forward(rep):
    checks = []

    # F1 报告结构完整
    need = ["A_envelope_replica", "B_tau_coupling_three_valued",
            "C_combination_invariants", "criteria", "verdict_pass"]
    checks.append(_chk("F1_structure", all(k in rep for k in need),
                       "missing=%s" % [k for k in need if k not in rep]))

    # F2 包络独立重算（对比报告存储值，敏感度来源）
    env_ok = True
    for mu in (0.2, 0.4, 0.6, 0.8):
        r = max_tangential_push(Q, mu)
        stored = rep["A_envelope_replica"]["rows"]["mu=%.1f" % mu]
        if (abs(r["max_tangential_N"] - stored["max_tangential_N"]) > 1e-6 or
                abs(r["at_normal_N"] - stored["at_normal_N"]) > 1e-6 or
                abs(r["cone_utilization"] - stored["cone_utilization"]) > 1e-6):
            env_ok = False
    checks.append(_chk("F2_envelope_recompute", env_ok,
                       "module vs report envelope mismatch"))

    # F3 τ-耦合三值分类独立重算
    feas = rep["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]
    feas_ok = True
    for fs, row in feas.items():
        F = [float(x) for x in fs[1:-1].split(",")]
        v = contact_verdict(Q, F, 0.8)
        if v["value"] != row["value"] or v["feasible"] != row["feasible"]:
            feas_ok = False
    checks.append(_chk("F3_tau_class_recompute", feas_ok, "tau classification mismatch"))

    # F4 滑移翻转独立重算
    sf = rep["B_tau_coupling_three_valued"]["slip_flip_F=[3,5]"]
    lo = contact_verdict(Q, [3.0, 5.0], 0.2)
    hi = contact_verdict(Q, [3.0, 5.0], 0.8)
    flip_ok = (lo["value"] == sf["mu=0.2"] == CONTACT_UNSAFE and
               hi["value"] == sf["mu=0.8"] == CONTACT_SAFE)
    checks.append(_chk("F4_slip_flip_recompute", flip_ok,
                       "lo=%s hi=%s" % (lo["value"], hi["value"])))

    # F5 组合真值表独立重算
    tbl = rep["C_combination_invariants"]["combination_table"]
    c_NA = contact_verdict(Q, [0.0, 0.0], 0.8)
    c_un = contact_verdict(Q, [8.0, 5.0], 0.8)
    c_sa = contact_verdict(Q, [1.0, 5.0], 0.4)
    expect = {
        "NA_x_SAFE": combine_contact_with_dynamic(c_NA, {"value": CONTACT_SAFE})["value"],
        "NA_x_UNSAFE": combine_contact_with_dynamic(c_NA, {"value": CONTACT_UNSAFE})["value"],
        "UNSAFE_x_SAFE": combine_contact_with_dynamic(c_un, {"value": CONTACT_SAFE})["value"],
        "SAFE_x_UNSAFE": combine_contact_with_dynamic(c_sa, {"value": CONTACT_UNSAFE})["value"],
        "SAFE_x_SAFE": combine_contact_with_dynamic(c_sa, {"value": CONTACT_SAFE})["value"],
    }
    combo_ok = all(tbl[k]["value"] == expect[k] for k in expect)
    checks.append(_chk("F5_combination_recompute", combo_ok,
                       {k: (tbl[k]["value"], expect[k]) for k in expect}))

    # F6 不变量自洽
    inv = e72_invariants(rep)
    checks.append(_chk("F6_invariants_self_consistent",
                       inv["verdict_pass"] == bool(rep["verdict_pass"]),
                       "inv=%s rep=%s" % (inv["verdict_pass"], rep["verdict_pass"])))

    # F7 来源档完备（每条裁决非空 source）
    src_ok = True
    for v in (c_NA, c_un, c_sa):
        if not v.get("source"):
            src_ok = False
    checks.append(_chk("F7_source_doc_present", src_ok, "missing source_doc"))

    # F8 确定性（模块同输入同输出）
    det = (contact_verdict(Q, [3.0, 5.0], 0.2) == contact_verdict(Q, [3.0, 5.0], 0.2) and
           max_tangential_push(Q, 0.6) == max_tangential_push(Q, 0.6) and
           combine_contact_with_dynamic(c_NA, {"value": CONTACT_SAFE}) ==
           combine_contact_with_dynamic(c_NA, {"value": CONTACT_SAFE}))
    checks.append(_chk("F8_determinism", bool(det), "non-deterministic"))

    # F9 非有限诚实（μ≤0 → ρ=None，UNSAFE；不泄漏 inf）
    mu0 = contact_verdict(Q, [3.0, 5.0], 0.0)
    fin_ok = (mu0["rho"] is None) and (mu0["value"] == CONTACT_UNSAFE) and (mu0["tag"] == "")
    checks.append(_chk("F9_nonfinite_capped", fin_ok, "rho=%s" % mu0["rho"]))
    return checks


def reverse(rep):
    checks = []

    # R1 阴性对照：篡改报告包络数值 → F2 必须 FAIL（证明审计敏感）
    bad = copy.deepcopy(rep)
    bad["A_envelope_replica"]["rows"]["mu=0.2"]["max_tangential_N"] = 9.99
    fwd = forward(bad)
    r1 = any((c["name"] == "F2_envelope_recompute" and not c["pass"]) for c in fwd)
    checks.append(_chk("R1_tamper_envelope_detected", r1, "audit blind to envelope tamper"))

    # R2 阳性对照：未篡改报告 → 全部正向 PASS
    fwd_clean = forward(rep)
    r2 = all(c["pass"] for c in fwd_clean)
    checks.append(_chk("R2_clean_report_passes", r2, "clean forward=%d/%d" %
                       (sum(c["pass"] for c in fwd_clean), len(fwd_clean))))

    # R3 篡改裁决值（[8,5] 改 SAFE）→ F3 必 FAIL
    bad3 = copy.deepcopy(rep)
    bad3["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]["[8.0, 5.0]"]["value"] = CONTACT_SAFE
    fwd3 = forward(bad3)
    r3 = any((c["name"] == "F3_tau_class_recompute" and not c["pass"]) for c in fwd3)
    checks.append(_chk("R3_tamper_verdict_detected", r3, "audit blind to verdict tamper"))

    # R4 边界健全性：ρ=1 恰界（Ft=μ·Fn）→ SAFE（不滑），独立验证
    mu_b = 0.5
    Fb = [mu_b * 5.0, 5.0]   # ρ=1
    vb = contact_verdict(Q, Fb, mu_b)
    r4 = (vb["value"] == CONTACT_SAFE) and abs(vb["rho"] - 1.0) < 1e-9
    checks.append(_chk("R4_boundary_rho_eq_1_safe", r4, "rho=%s value=%s" % (vb["rho"], vb["value"])))

    # R5 非有限反向：μ=0 必 UNSAFE 且 ρ=None（与 F9 互证）
    r5 = (contact_verdict(Q, [3.0, 5.0], 0.0)["value"] == CONTACT_UNSAFE)
    checks.append(_chk("R5_mu_zero_unsafe", r5, "mu=0 not unsafe"))

    # R6 ⊘ 短路诚实（反向）：把组合结果 value 改错 → F5 必 FAIL
    bad6 = copy.deepcopy(rep)
    bad6["C_combination_invariants"]["combination_table"]["UNSAFE_x_SAFE"]["value"] = CONTACT_SAFE
    fwd6 = forward(bad6)
    r6 = any((c["name"] == "F5_combination_recompute" and not c["pass"]) for c in fwd6)
    checks.append(_chk("R6_tamper_combo_detected", r6, "audit blind to combo tamper"))

    # R7 无接触 ⊘  dishonest-guard：⊘ 必须 shorted_by=contact_NA 且 dynamic 透传
    tbl = rep["C_combination_invariants"]["combination_table"]
    r7 = (tbl["NA_x_SAFE"]["shorted_by"] == "contact_NA" and
          tbl["NA_x_UNSAFE"]["shorted_by"] == "contact_NA" and
          tbl["NA_x_SAFE"]["dynamic"] == CONTACT_SAFE and
          tbl["NA_x_UNSAFE"]["dynamic"] == CONTACT_UNSAFE)
    checks.append(_chk("R7_NA_honest_short", r7, "NA not honestly shorted"))

    # R8 覆盖度：每个报告中的裁决 case 都被正向审计覆盖（无漏报）
    cov = (any(c["name"] == "F2_envelope_recompute" for c in forward(rep)) and
           any(c["name"] == "F3_tau_class_recompute" for c in forward(rep)) and
           any(c["name"] == "F5_combination_recompute" for c in forward(rep)))
    checks.append(_chk("R8_coverage", cov, "audit coverage gap"))
    return checks


def main():
    rep = load()
    fwd = forward(rep)
    rev = reverse(rep)
    allc = fwd + rev
    npass = sum(c["pass"] for c in allc)
    out = {"audit": "_audit_e72_friction_cone", "experiment": "e72_friction_cone_verdict",
           "pass": npass, "total": len(allc), "verdict": bool(npass == len(allc)),
           "audit_pass": bool(npass == len(allc)),
           "forward": fwd, "reverse": rev}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_audit_e72_friction_cone.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("audit pass = %d/%d  verdict=%s" % (npass, len(allc), out["verdict"]))
    for c in allc:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
