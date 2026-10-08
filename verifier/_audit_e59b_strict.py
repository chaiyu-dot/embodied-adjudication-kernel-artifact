# -*- coding: utf-8 -*-
"""_audit_e59b_strict.py —— E59b 严格审核（L3 聚合自洽 + 审前提 + 覆盖/单调重算 + 数据自足性检查）。

E59b 用真实测量原语（solve_mass_from_weigh / solve_mu_m_from_pushes）；其**报告未落盘逐行
rows**（只落聚合）→ 无法逐项独立重算。本审核：
  · 守恒/一致性：n_total == 决定 + ⊘；n_nonterminated==0；wrong_side ⊆ decided
  · 覆盖重算：coverage_by_sigma_ratio 档 [0.8,1.25] ≥0.90 且随 σ 比单调不增（H59b-4）
  · 轮数单调分箱（H59b-3）、Spearman(rounds,R*)>0（H59b-2）
  · 判据→verdict 自洽；prereg 对齐
  · **自足性 finding**：报告缺 rows → 第三方无法逐项复算（记为 D 根因）
  · 篡改必报
产物：_audit_e59b_strict.json
"""
import copy
import hashlib
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e59b_termination_real_loop_report.json")
PREREG = os.path.join(EVAL, "e59b_termination_real_loop_prereg.json")
OUT = os.path.join(EVAL, "_audit_e59b_strict.json")

_res = {"experiment": "E59b 严格审核(L3 聚合自洽+审前提)", "independence_scope":
        "不 import e59b；只读 JSON（报告未落 rows → 只做聚合自洽）", "checks": [], "findings": [], "tamper": []}


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    R = rep["results"]
    # S1 守恒
    ch.append(_chk("S1_conservation", R["n_decided"] + R["n_inapplicable"] + R["n_nonterminated"] == R["n_total"]
                   and R["n_nonterminated"] == 0 and R["n_wrong_side"] <= R["n_decided"],
                   "dec=%d NA=%d nonterm=%d total=%d" % (R["n_decided"], R["n_inapplicable"], R["n_nonterminated"], R["n_total"])))
    # S2 p95 ≤ r_max
    ch.append(_chk("S2_p95_within_rmax", R["p95_rounds"] <= rep["design"]["r_max"],
                   "p95=%s rmax=%s" % (R["p95_rounds"], rep["design"]["r_max"])))
    # S3 覆盖 σ 比（H59b-4）
    cov = R["coverage_by_sigma_ratio"]
    seq = [cov["<0.8"]["coverage"], cov["0.8-1.25"]["coverage"], cov[">1.25"]["coverage"]]
    mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(2))
    ch.append(_chk("S3_coverage_in_spec_and_monotone", cov["0.8-1.25"]["coverage"] >= 0.90 and mono,
                   "cov=%s mono=%s" % (seq, mono)))
    # S4 Spearman
    sp = R["rounds_vs_rstar_spearman"]
    p_lt = (sp["p"] == "<1e-16") or (isinstance(sp["p"], (int, float)) and sp["p"] < 0.001)
    ch.append(_chk("S4_spearman_rounds_vs_rstar", sp["rho"] > 0 and p_lt, "rho=%.4f p=%s" % (sp["rho"], sp["p"])))
    # S5 轮数单调分箱（中位不减）
    med = [b[1] for b in R["rounds_monotone_bins"]]
    ch.append(_chk("S5_rounds_monotone_bins", all(med[i] <= med[i + 1] + 1e-9 for i in range(len(med) - 1)), "med=%s" % med))
    # S6 判据 → verdict
    H = rep["verdict"]
    ch.append(_chk("S6_criteria_verdict_selfconsistent",
                   rep["verdict_pass"] == all(H.values()), "verdict=%s" % rep["verdict_pass"]))
    # S7 prereg 对齐
    pa = rep.get("prereg_alignment", {})
    ch.append(_chk("S7_prereg_alignment", bool(pa.get("sha256_match")) and bool(pa.get("hypotheses_match")), str(pa)))
    # 自足性 finding
    _res["findings"].append({"id": "E59b-D1_no_rows", "severity": "D(覆盖/自足)",
                             "pass": ("rows" in rep),
                             "detail": "报告未落盘逐行 rows（只聚合）→ 第三方无法逐项独立复算；"
                                       "建议落盘 rows（d、R*、rounds、coverage hit）以支持 L4 盲复刻。"})
    print("  [%s] S8_rows_self_sufficiency%s" % ("PASS" if ("rows" in rep) else "FAIL",
          "" if ("rows" in rep) else "  —— 报告缺 rows"))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name, "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E59b 严格审核（L3 聚合自洽，不 import e59b）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 n_decided 破坏守恒 → S1 必报",
          lambda r: r["results"].__setitem__("n_decided", 1), ["S1"])
    _tamp("R2 篡改覆盖（0.8-1.25 → 0.5）→ S3 必报",
          lambda r: r["results"]["coverage_by_sigma_ratio"]["0.8-1.25"].__setitem__("coverage", 0.5), ["S3"])
    _tamp("R3 篡改 Spearman rho 负 → S4 必报",
          lambda r: r["results"]["rounds_vs_rstar_spearman"].__setitem__("rho", -0.5), ["S4"])
    _tamp("R4 篡改 verdict_pass → S6 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["S6"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass; _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp; _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
