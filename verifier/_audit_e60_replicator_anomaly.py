# -*- coding: utf-8 -*-
"""_audit_e60_replicator_anomaly.py —— E60 的第三方/盲复刻（L2/L4，纯数据）。

不 import 任何实验模块；只读 e60_energy_margin_report.json，验证内部算术自洽、从存储 rows
重算档位一致率与两两 Spearman ρ、判据/verdict 自洽；篡改必有检查报出。

产物：_audit_e60_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e60_energy_margin_report.json")
OUT = os.path.join(EVAL, "_audit_e60_replicator_anomaly.json")

_res = {"experiment": "E60 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


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


def _forward(rep):
    ch = []
    d = rep["design"]
    rows = rep["rows"]
    # D1 design sane
    ch.append(chk("D1_design_sane", d["n_levels"] == len(rows) and len(d["band_edges"]) == 3
                  and d["noise_sigma"] > 0, "n_levels=%d" % d["n_levels"]))
    # D2 u sorted ascending + count
    us = [r["u"] for r in rows]
    ch.append(chk("D2_rows_count_and_sorted", len(rows) == d["n_levels"] and us == sorted(us),
                  "n=%d" % len(rows)))
    # D3 band valid + consistent flag == set(band)==1
    ok3 = all(all(b in ("unsafe", "tight", "medium", "loose") for b in r["band"].values())
              and r["consistent"] == (len(set(r["band"].values())) == 1) for r in rows)
    ch.append(chk("D3_band_valid_and_consistent_flag", ok3))
    # D4 band agreement rate recompute
    rate = round(sum(1 for r in rows if r["consistent"]) / len(rows), 4)
    ch.append(chk("D4_band_agreement_rate_recompute", abs(rate - rep["results"]["band_agreement_rate"]) < 1e-9,
                  "re=%.4f store=%.4f" % (rate, rep["results"]["band_agreement_rate"])))
    ch.append(chk("D4b_n_consistent_recompute", sum(1 for r in rows if r["consistent"]) == rep["results"]["n_consistent"]))
    # D5 pairwise rho recompute
    dom = ["distance", "force", "energy", "semantic"]
    vecs = {k: [r["m_E"][k] for r in rows] for k in dom}
    bad = 0
    for i in range(len(dom)):
        for j in range(i + 1, len(dom)):
            key = "%s_vs_%s" % (dom[i], dom[j])
            rv = round(_rho(vecs[dom[i]], vecs[dom[j]]), 4)
            if abs(rv - rep["results"]["pairwise_rho"][key]) > 1e-4:
                bad += 1
    ch.append(chk("D5_pairwise_rho_recompute", bad == 0, "mismatch=%d" % bad))
    # D6 mean rho
    mean_rho = round(sum(rep["results"]["pairwise_rho"].values()) / len(rep["results"]["pairwise_rho"]), 4)
    ch.append(chk("D6_mean_rho_selfconsistent",
                  abs(mean_rho - rep["results"]["ordinal_consistency_mean_pairwise_rho"]) < 1e-4,
                  "re=%.4f store=%.4f" % (mean_rho, rep["results"]["ordinal_consistency_mean_pairwise_rho"])))
    # D7 criteria + verdict
    H = rep["preregistered_verdict"]
    verd = H["H60-1a_ordinal_consistency_mean_rho_ge_0.90"]
    ch.append(chk("D7_criteria_and_verdict_selfconsistent",
                  H["H60-1a_ordinal_consistency_mean_rho_ge_0.90"]
                  == (rep["results"]["ordinal_consistency_mean_pairwise_rho"] >= 0.90)
                  and H["H60-1b_band_agreement_ge_0.90"] == (rep["results"]["band_agreement_rate"] >= 0.90)
                  and rep["verdict_pass"] == verd, "verdict=%s" % rep["verdict_pass"]))
    # D8 self_consistency
    sc = rep["results"]["self_consistency"]
    ch.append(chk("D8_self_consistency", sc["rows_len"] and sc["bands_are_valid"]))
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
    print("E60 第三方/盲复刻审核（L2/L4 纯数据）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 consistent 标记 → D3 必报",
          lambda r: r["rows"][0].__setitem__("consistent", not r["rows"][0]["consistent"]), ["D3"])
    _tamp("R2 篡改 band_agreement_rate → D4 必报",
          lambda r: r["results"].__setitem__("band_agreement_rate", 0.99), ["D4"])
    _tamp("R3 篡改某 pairwise_rho → D5 必报",
          lambda r: r["results"]["pairwise_rho"].__setitem__("distance_vs_force", 0.5), ["D5"])
    _tamp("R4 篡改某行 m_E → D5 必报",
          lambda r: r["rows"][5]["m_E"].__setitem__("semantic", 0.0), ["D5"])
    _tamp("R5 篡改 verdict_pass=False → D7 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["D7"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["replicator_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n第三方审核：算术 %d/%d ｜ 篡改 %d/%d ｜ pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
