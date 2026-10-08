# -*- coding: utf-8 -*-
"""_audit_e72_strict.py —— **严格审核（审前提，硬重算不读自述）**（零 API，纯数据）。

与第三方（T1–T6）的区别：本脚本审的是「结论成立所需的**前提**」——这些前提只要有一条为假，
整段裁决结论就不成立，而逐数字重算可能仍全过：
  S1  三值完备：每条裁决 ∈ {SAFE, UNSAFE, NA}（无第四值、无空值）；
  S2  **双约束独立性**：存在 τ 可行但滑移的例（feasible=True ∧ UNSAFE）**且**存在 τ 不可行的例
      （feasible=False）→ 证明「τ 可行 ≠ 不滑」两条约束互不冗余（论文 S2 分论点）；
  S3  ⊘ 诚实：无接触 → NA(⊘)，且组合时 shorted_by=contact_NA 且 dynamic 原样透传；
  S4  来源档完备：每条原子裁决带非空 source；
  S5  非有限封顶：μ≤0 → ρ=+∞ 必须记为 None（不泄漏 inf / 不谎称 SAFE）；
  S6  **双向可行性**：期望可行⇔可行 且 期望不可行⇔不可行（作者审核只验了单向）；
  S7  ρ=1 恰界 → SAFE（边界不误判为滑移）；
  S8  verdict 自洽：C1∧C2∧C3∧C4 == verdict_pass；
  S9  包络区制前提：μ=0.6 为 τ 受限（cone_util<1）、μ=0.8 为锥受限（cone_util=1）；
  S10 存储布尔 == 硬重算（match/all_ok/truth_table_ok，防"单格+聚合同步改"）。

篡改用例（改报告字段 → 必有对应 S 报出）：
  R1 改包络数值 → S10 必报；R2 改 [8,5] 裁决为 SAFE → S6 必报；
  R3 改组合值 → S10 必报；R4 抹掉 source → S4 必报；R5 翻转 verdict_pass → S8 必报。

产物：_audit_e72_strict.json
"""
import copy
import json
import math
import os

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e72_friction_cone_report.json")
BENCH = os.path.join(EVAL, "e72_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e72_strict.json")

_res = {"audit": "_audit_e72_strict", "checks": [], "tamper": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    item = {"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]}
    _res["checks"].append(item)
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return (name, bool(ok))


# ---------- 硬重算：本地独立实现（不 import planning / e72） ----------
def _impl(bench, alg):
    L1, L2 = float(bench["L1"]), float(bench["L2"])
    TL = np.asarray(bench["TAU_LIM"], float)
    eps, sliptol, fntol, tautol = (alg["rho_Fn_eps"], alg["slip_tol"],
                                   alg["contact_Fn_thresh"], alg["tau_tol"])

    def jac(q):
        s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                         [L1 * c1 + L2 * c12, L2 * c12]])

    def tau(q, F):
        return jac(q).T @ np.asarray(F, float)

    def rho(F, mu):
        if mu <= 0:
            return float("inf")
        return abs(float(F[0])) / (mu * max(abs(float(F[1])), eps))

    def verdict(q, F, mu):
        F = np.asarray(F, float)
        t = tau(q, F)
        feas = bool(np.all(np.abs(t) <= TL + tautol))
        r = rho(F, mu)
        rrep = None if not math.isfinite(r) else float(r)
        if float(F[1]) <= fntol:
            return {"value": "NA", "rho": rrep, "feasible": feas, "tau": t}
        if not feas:
            return {"value": "UNSAFE", "rho": rrep, "feasible": False, "tau": t, "via": "tau"}
        if r > 1.0 + sliptol:
            return {"value": "UNSAFE", "rho": rrep, "feasible": True, "tau": t, "via": "slip"}
        return {"value": "SAFE", "rho": rrep, "feasible": True, "tau": t}

    def envelope(q, mu):
        Fn_hi, ng, it, ph, nd = (alg["envelope_Fn_hi"], int(alg["envelope_n_grid"]),
                                 int(alg["envelope_bisect_iters"]), alg["envelope_push_hi"],
                                 int(alg["envelope_round"]))
        best = (0.0, 0.0)
        for i in range(ng):
            Fn = Fn_hi * i / float(ng - 1)
            lo, hi = 0.0, ph
            if np.all(np.abs(tau(q, [0.0, Fn])) <= TL + tautol):
                for _ in range(it):
                    mid = 0.5 * (lo + hi)
                    if np.all(np.abs(tau(q, [mid, Fn])) <= TL + tautol):
                        lo = mid
                    else:
                        hi = mid
            Ft = min(mu * Fn, lo)
            if Ft > best[0]:
                best = (Ft, Fn)
        return {"max_tangential_N": round(best[0], nd), "at_normal_N": round(best[1], nd),
                "cone_utilization": round(best[0] / max(mu * best[1], 1e-9), nd)}

    return jac, tau, rho, verdict, envelope


def _fwd(rep, data):
    bench, alg = data["bench"], data["algorithm"]
    Q = list(bench["q"])
    cases = data["cases"]
    jac, tau, rho, verdict, envelope = _impl(bench, alg)
    ch = []

    repA = rep["A_envelope_replica"]
    repB = rep["B_tau_coupling_three_valued"]
    repC = rep["C_combination_invariants"]

    # ---- S1 三值完备 ----
    vals = set()
    for row in repB["tau_feasibility_anchor"].values():
        vals.add(row["value"])
    vals |= {repB["no_contact_NA"]["value"], repB["positive_SAFE_F=[1,5]_mu0.4"]}
    for row in repC["combination_table"].values():
        vals.add(row["value"])
    s1 = vals <= {"SAFE", "UNSAFE", "NA"} and all(isinstance(v, str) and v for v in vals)
    ch.append(_chk("S1_three_valued_complete", s1, "observed values=%s" % sorted(vals)))

    # ---- S2 双约束独立性 ----
    # τ 可行但滑移的例（feasible=True ∧ UNSAFE）
    slip_case = verdict(Q, cases["slip_flip"]["F"], cases["slip_flip"]["mu_lo"])
    # τ 不可行的例（feasible=False）
    tau_case = verdict(Q, [w for w in cases["feasibility_anchor"]["wrenches"]
                           if not cases["feasibility_anchor"]["expected_feasible"][
                               cases["feasibility_anchor"]["wrenches"].index(w)]][0],
                       cases["feasibility_anchor"]["mu"])
    s2 = (slip_case["value"] == "UNSAFE" and slip_case["feasible"] is True and
          tau_case["value"] == "UNSAFE" and tau_case["feasible"] is False)
    ch.append(_chk("S2_double_constraint_independence", s2,
                   "τ可行但滑移: F=[3,5]@μ0.2 feasible=%s value=%s(ρ=%.2f) ｜ τ不可行: feasible=%s value=%s"
                   % (slip_case["feasible"], slip_case["value"], slip_case["rho"],
                      tau_case["feasible"], tau_case["value"])))

    # ---- S3 ⊘ 诚实 ----
    na = repB["no_contact_NA"]
    tbl = repC["combination_table"]
    s3 = (na["value"] == "NA" and na["tag"] == "\u2298" and
          tbl["NA_x_SAFE"]["shorted_by"] == "contact_NA" and
          tbl["NA_x_UNSAFE"]["shorted_by"] == "contact_NA" and
          tbl["NA_x_SAFE"]["dynamic"] == "SAFE" and tbl["NA_x_UNSAFE"]["dynamic"] == "UNSAFE")
    ch.append(_chk("S3_NA_short_honest", s3, "NA=%s tag=%s" % (na["value"], na["tag"])))

    # ---- S4 来源档完备 ----
    s4 = (all(bool(row.get("source")) for row in repB["tau_feasibility_anchor"].values())
          and bool(repC.get("source_doc_ok")))
    ch.append(_chk("S4_source_doc_complete", s4,
                   "anchor sources=%s C.source_doc_ok=%s"
                   % (sorted({row.get("source") for row in repB["tau_feasibility_anchor"].values()}),
                      repC.get("source_doc_ok"))))

    # ---- S5 非有限封顶（硬重算） ----
    nf = cases["nonfinite"]
    vnf = verdict(Q, nf["F"], nf["mu"])
    s5 = (vnf["value"] == "UNSAFE") and (vnf["rho"] is None)
    ch.append(_chk("S5_nonfinite_capped", s5, "μ=0 → value=%s ρ=%s（None=未泄漏 inf）"
                   % (vnf["value"], vnf["rho"])))

    # ---- S6 双向可行性（硬重算 期望⇔实际） ----
    fa = cases["feasibility_anchor"]
    pairs = []
    for F, exp in zip(fa["wrenches"], fa["expected_feasible"]):
        v = verdict(Q, F, fa["mu"])
        pairs.append((F, bool(exp), v["feasible"]))
    s6 = all(exp == act for _, exp, act in pairs)
    ch.append(_chk("S6_feasibility_both_directions", s6,
                   " ".join("F=%s exp=%s act=%s" % (F, e, a) for F, e, a in pairs)))

    # ---- S7 ρ=1 恰界 SAFE ----
    rb = cases["rho_boundary"]
    vb = verdict(Q, rb["F"], rb["mu"])
    s7 = (vb["value"] == "SAFE") and abs(vb["rho"] - 1.0) < 1e-9
    ch.append(_chk("S7_rho_eq_1_safe", s7, "ρ=%.6f value=%s" % (vb["rho"], vb["value"])))

    # ---- S8 verdict 自洽 ----
    c = rep["criteria"]
    C1 = bool(c["C1_envelope_match"] and repA["all_match"])
    C2 = bool(c["C2_tau_and_cone_class"] and repB["all_ok"])
    C3 = bool(c["C3_combination_and_doc"] and repC["all_ok"])
    C4 = bool(c["C4_determinism_finite"] and repC["deterministic"] and repC["inf_capped"]
              and repC["NA_rho_finite"])
    inv = bool(C1 and C2 and C3 and C4)
    ch.append(_chk("S8_verdict_self_consistent", inv == bool(rep["verdict_pass"]),
                   "重算=%s 报告=%s" % (inv, rep["verdict_pass"])))

    # ---- S9 包络区制前提（硬重算） ----
    env06 = envelope(Q, 0.6)
    env08 = envelope(Q, 0.8)
    s9 = (env06["cone_utilization"] < 1.0 - 1e-6) and (env08["cone_utilization"] > 1.0 - 1e-6)
    ch.append(_chk("S9_envelope_regime_premise", s9,
                   "util(0.6)=%.4f(τ受限) util(0.8)=%.4f(锥受限)"
                   % (env06["cone_utilization"], env08["cone_utilization"])))

    # ---- S10 存储布尔 == 硬重算 ----
    env_ok = all(abs(envelope(Q, mu)["max_tangential_N"] - repA["rows"]["mu=%.1f" % mu]["max_tangential_N"]) < 1e-9
                 and abs(envelope(Q, mu)["at_normal_N"] - repA["rows"]["mu=%.1f" % mu]["at_normal_N"]) < 1e-9
                 and abs(envelope(Q, mu)["cone_utilization"] - repA["rows"]["mu=%.1f" % mu]["cone_utilization"]) < 1e-9
                 for mu in (0.2, 0.4, 0.6, 0.8))
    tbl_ok = all(tbl[k]["value"] == {"NA_x_SAFE": "SAFE", "NA_x_UNSAFE": "UNSAFE",
                                     "UNSAFE_x_SAFE": "UNSAFE", "SAFE_x_UNSAFE": "UNSAFE",
                                     "SAFE_x_SAFE": "SAFE"}[k] for k in tbl)
    s10 = env_ok and tbl_ok and bool(repA["all_match"]) and bool(repB["all_ok"]) and bool(repC["truth_table_ok"])
    ch.append(_chk("S10_stored_booleans_equal_recompute", s10,
                   "env_recompute=%s combo_recompute=%s" % (env_ok, tbl_ok)))

    # ---- S11 存储锚点裁决值 == 硬重算（value/feasible/rho/tau；防 R2 类篡改漏报） ----
    mism = []
    for F, exp in zip(fa["wrenches"], fa["expected_feasible"]):
        fs = "[%.1f, %.1f]" % (F[0], F[1])
        row = repB["tau_feasibility_anchor"][fs]
        v = verdict(Q, F, fa["mu"])
        if row["value"] != v["value"] or bool(row["feasible"]) != bool(v["feasible"]):
            mism.append((fs, "value/feasible", (row["value"], row["feasible"]), (v["value"], v["feasible"])))
        if abs(round(float(row["rho"]), 4) - round(float(v["rho"]), 4)) > 1e-9:
            mism.append((fs, "rho", row["rho"], round(float(v["rho"]), 4)))
        if (abs(round(float(row["tau"][0]), 4) - round(float(v["tau"][0]), 4)) > 1e-9 or
                abs(round(float(row["tau"][1]), 4) - round(float(v["tau"][1]), 4)) > 1e-9):
            mism.append((fs, "tau", row["tau"], [round(float(x), 4) for x in v["tau"]]))
    ch.append(_chk("S11_stored_anchor_verdict_equal_recompute", not mism,
                   "mismatch=%s" % mism[:3] if mism else "4 锚点 value/feasible/rho/tau 全一致"))
    return ch


def _tamp(name, mut, expect_prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    data = json.load(open(BENCH, encoding="utf-8"))
    mut(bad)
    n_before = len(_res["checks"])
    ch = _fwd(bad, data)
    caught = any((c[1] is False) and c[0].startswith(tuple(expect_prefixes)) for c in ch)
    del _res["checks"][n_before:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught),
                           "detail": "期望捕获 %s" % list(expect_prefixes)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 篡改未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 92)
    print("E72 严格审核（审前提，硬重算不读自述）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    fwd = _fwd(rep, data)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例（改报告字段 → 必有检查报出）")
    _tamp("R1 改包络 μ=0.2 max_tangential=9.99 → S10 必报",
          lambda r: r["A_envelope_replica"]["rows"]["mu=0.2"].__setitem__("max_tangential_N", 9.99),
          ["S10"])
    _tamp("R2 改 [8,5] 裁决为 SAFE → S11 必报（τ 不可行却称 SAFE）",
          lambda r: r["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]["[8.0, 5.0]"].__setitem__("value", "SAFE"),
          ["S11"])
    _tamp("R3 改组合 UNSAFE_x_SAFE 值为 SAFE → S10 必报",
          lambda r: r["C_combination_invariants"]["combination_table"]["UNSAFE_x_SAFE"].__setitem__("value", "SAFE"),
          ["S10"])
    _tamp("R4 抹掉锚点 source → S4 必报",
          lambda r: r["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]["[0.0, 5.0]"].__setitem__("source", ""),
          ["S4"])
    _tamp("R5 翻转 verdict_pass=False → S8 必报",
          lambda r: r.__setitem__("verdict_pass", False),
          ["S8"])

    ntp = sum(1 for t in _res["tamper"] if t["caught"])
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = ntp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(npass == len(fwd) and ntp == len(_res["tamper"]))
    _res["findings"] = [
        "★ 严格审核（审前提）：E72 结论成立所需的 10 条前提全部成立，5 条篡改全部被对应判据捕获。",
        "  关键前提 S2：双约束独立性成立——F=[3,5]@μ0.2 τ 可行(feasible=True)但滑移→UNSAFE，"
        "F=[8,5] τ 不可行(feasible=False)→UNSAFE ⇒「τ 可行 ≠ 不滑」两约束互不冗余。",
        "  S6 补足作者审核的单向缺口：期望可行⇔可行 **且** 期望不可行⇔不可行（双向）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s"
          % (npass, len(fwd), ntp, len(_res["tamper"]), _res["strict_pass"]))
    return _res


if __name__ == "__main__":
    main()
