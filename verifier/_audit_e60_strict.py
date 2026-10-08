# -*- coding: utf-8 -*-
"""_audit_e60_strict.py —— E60 严格审核（L3 独立重推 + 审前提 + 设计漂移检查）。

不 import e60_energy_margin；审计者自写四域映射 + margin_to_band + Spearman，逐行重算并核对：
  · S1 逐行 m_E / band 一致
  · S2 平均两两 Spearman ρ 一致；S3 档位一致率一致
  · S4 判据 H60-1a/H60-1b + verdict 自洽
  · S5 自洽（rows_len/bands_are_valid）
  · **S6 设计漂移检查**：报告 design.domains 文本 vs 实际代码公式（发现文档与实现不符则记 FAIL）
  · **S7 域间系统性差异**：四域系统性偏差上界（近同一函数 ⇒ 跨域一致近平凡，记录）

产物：_audit_e60_strict.json
"""
import copy
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e60_energy_margin_report.json")
OUT = os.path.join(EVAL, "_audit_e60_strict.json")

NOISE = 0.03
EDGES = (-0.05, 0.25, 0.60)
_res = {"experiment": "E60 严格审核(L3 独立重推)", "independence_scope":
        "独立重推：自写四域映射 + 分档 + Spearman，不 import e60；另做设计漂移检查",
        "checks": [], "findings": [], "tamper": []}


def _seed(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _margins(u, rng):
    n_d = float(rng.normal(0, NOISE)); n_f = float(rng.normal(0, NOISE))
    n_e = float(rng.normal(0, NOISE)); n_s = float(rng.normal(0, NOISE))
    return {"distance": (1.0 - u) + n_d,
            "force": (1.0 - u * (1.0 + 0.03 * u)) + n_f,
            "energy": (1.0 - u * (1.0 + 0.05 * u)) + n_e,
            "semantic": (1.0 - u ** 0.96) + n_s}


def _band(m):
    if m < EDGES[0]:
        return "unsafe"
    if m < EDGES[1]:
        return "tight"
    if m < EDGES[2]:
        return "medium"
    return "loose"


def _rank(a):
    order = sorted(range(len(a)), key=lambda i: a[i]); rk = [0.0] * len(a); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for kk in range(i, j + 1):
            rk[order[kk]] = avg
        i = j + 1
    return rk


def _rho(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return 0.0 if den < 1e-12 else num / den


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:180]) if detail else ""))
    return (name, bool(ok))


def _recompute(rep):
    n = rep["design"]["n_levels"]
    us = np.linspace(0.05, 0.95, n)
    rows = []
    for i, u in enumerate(us):
        rng = np.random.RandomState(_seed("e60", i))
        ms = _margins(float(u), rng)
        rows.append({"u": round(float(u), 4), "m_E": {k: round(v, 4) for k, v in ms.items()},
                     "band": {k: _band(v) for k, v in ms.items()}})
    dom = ["distance", "force", "energy", "semantic"]
    vecs = {k: [r["m_E"][k] for r in rows] for k in dom}
    pr = {}
    for a in range(len(dom)):
        for b in range(a + 1, len(dom)):
            pr["%s_vs_%s" % (dom[a], dom[b])] = round(_rho(vecs[dom[a]], vecs[dom[b]]), 4)
    mean_rho = round(sum(pr.values()) / len(pr), 4)
    rate = round(sum(1 for r in rows if len(set(r["band"].values())) == 1) / n, 4)
    return rows, pr, mean_rho, rate


def _forward(rep):
    ch = []
    rows, pr, mean_rho, rate = _recompute(rep)
    S = rep["rows"]
    # S1 逐行 m_E/band
    d_m = 0.0
    bad_b = 0
    for r, s in zip(rows, S):
        for k in ("distance", "force", "energy", "semantic"):
            d_m = max(d_m, abs(r["m_E"][k] - s["m_E"][k]))
            if r["band"][k] != s["band"][k]:
                bad_b += 1
    ch.append(_chk("S1_rows_mE_and_band_match", d_m < 1e-3 and bad_b == 0,
                   "max|Δm_E|=%.4f band_mismatch=%d" % (d_m, bad_b)))
    # S2 平均 ρ
    ch.append(_chk("S2_mean_pairwise_rho_match",
                   abs(mean_rho - rep["results"]["ordinal_consistency_mean_pairwise_rho"]) < 1e-4,
                   "re=%.4f store=%.4f" % (mean_rho, rep["results"]["ordinal_consistency_mean_pairwise_rho"])))
    # S3 档位一致率
    ch.append(_chk("S3_band_agreement_match", abs(rate - rep["results"]["band_agreement_rate"]) < 1e-4,
                   "re=%.4f store=%.4f" % (rate, rep["results"]["band_agreement_rate"])))
    # S4 判据 + verdict
    H = {"H60-1a_ordinal_consistency_mean_rho_ge_0.90": bool(mean_rho >= 0.90),
         "H60-1b_band_agreement_ge_0.90": bool(rate >= 0.90)}
    ch.append(_chk("S4_criteria_and_verdict_selfconsistent",
                   all(H[k] == rep["preregistered_verdict"][k] for k in H)
                   and rep["verdict_pass"] == H["H60-1a_ordinal_consistency_mean_rho_ge_0.90"],
                   "verdict=%s" % rep["verdict_pass"]))
    # S5 自洽
    sc = rep["results"]["self_consistency"]
    ch.append(_chk("S5_self_consistency", sc["rows_len"] and sc["bands_are_valid"]))

    # S6 设计漂移：报告 design.domains 文本 vs 实际代码公式
    dd = rep["design"]["domains"]
    expect = {"force": "1-u(1+0.03u)", "energy": "1-u(1+0.05u)", "semantic": "1-u^0.96"}
    drift = []
    if "0.5u" in dd["force"] or "0.15u" in dd["energy"] or "0.8" in dd["semantic"]:
        drift = ["design.domains 文本为旧参数（0.5/0.15/0.8），实际代码用 0.03/0.05/0.96"]
    _res["findings"].append({"id": "E60-D1_doc_drift", "severity": "D(一致性)",
                             "pass": (not drift), "detail": drift or "无漂移",
                             "expect_code_form": expect})
    print("  [%s] S6_design_text_matches_code%s" % ("PASS" if not drift else "FAIL",
          ("  —— drift=" + str(drift)) if drift else ""))

    # S7 域间系统性差异上界（无噪声）——近同一函数 ⇒ 跨域一致近平凡
    sys_spread = 0.0
    for u in np.linspace(0.05, 0.95, 30):
        vals = [1.0 - u, 1.0 - u * (1 + 0.03 * u), 1.0 - u * (1 + 0.05 * u), 1.0 - u ** 0.96]
        sys_spread = max(sys_spread, max(vals) - min(vals))
    _res["findings"].append({"id": "E60-F1_near_tautology", "severity": "F(设计充分性)",
                             "pass": bool(sys_spread > 0.05),
                             "detail": "四域无噪声系统性偏差上界仅 %.4f ⇒ 跨域一致近平凡（噪声 σ=0.03 主导）" % sys_spread,
                             "noise_free_spread_bound": round(sys_spread, 4)})
    print("  [%s] S7_domains_systemically_distinct  —— noise-free spread=%.4f (>0.05 才算非平凡)"
          % ("PASS" if sys_spread > 0.05 else "FAIL", sys_spread))
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
    print("E60 严格审核（L3 独立重推，不 import e60）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改某行 m_E distance → S1 必报",
          lambda r: r["rows"][10]["m_E"].__setitem__("distance", 9.9), ["S1"])
    _tamp("R2 篡改 mean_rho → S2 必报",
          lambda r: r["results"].__setitem__("ordinal_consistency_mean_pairwise_rho", 0.5), ["S2"])
    _tamp("R3 篡改 band_agreement_rate → S3 必报",
          lambda r: r["results"].__setitem__("band_agreement_rate", 0.99), ["S3"])
    _tamp("R4 篡改 verdict_pass=False → S4 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S4"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
