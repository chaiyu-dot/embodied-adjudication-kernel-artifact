# -*- coding: utf-8 -*-
"""_audit_e59_strict.py —— E59 严格审核（L3 独立重推 + 审前提）。

不 import e59_termination_bound；审计者按声明规则独立重写：可分辨距离 d 生成（48×5 对数均匀）、
半宽收缩（r=0.35）、终止判据（K_max=6 / floor=0.012）、逐 eps 统计、p95、Spearman(rounds,ε)，
逐项与存储核对；篡改必报。另做**文档漂移检查**（docstring/design 文本 vs 实际常量）。

产物：_audit_e59_strict.json
"""
import copy
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e59_termination_report.json")
OUT = os.path.join(EVAL, "_audit_e59_strict.json")

N_OBJ, N_SEED = 48, 5
K_MAX = 6
SHRINK = 0.35
FLOOR = 0.012
EPS_LIST = [0.0, 0.05, 0.1, 0.2, 0.4]

_res = {"experiment": "E59 严格审核(L3 独立重推)", "independence_scope":
        "独立重推：自写 d 生成 + 终止判据 + 统计，不 import e59", "checks": [], "findings": [], "tamper": []}


def _seed(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _hw0(eps):
    return 0.02 if eps <= 0 else 0.5 * eps


def _decide(d, hw0):
    if d < FLOOR:
        return K_MAX, "⊘"
    hw = hw0
    for k in range(1, K_MAX + 1):
        if hw < d:
            return k, "decided"
        hw *= SHRINK
    return None, "nonterminated"


def _spearman_p(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i]); rk = [0.0] * len(a); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den < 1e-12:
        return 0.0, None
    rho = num / den
    if abs(1 - rho * rho) < 1e-12:
        return rho, None
    t = rho * math.sqrt((len(xs) - 2) / (1 - rho * rho))
    return rho, math.erfc(abs(t) / math.sqrt(2))


def _recompute():
    rows = []
    for o in range(N_OBJ):
        for s in range(N_SEED):
            rng = np.random.RandomState(_seed("obj", o, "seed", s))
            d = float(np.exp(rng.uniform(np.log(0.006), np.log(0.4))))
            rows.append({"obj": o, "seed": s, "d": round(d, 5)})
    per_eps = {}
    all_rounds, all_eps = [], []
    nonterm = 0
    inap = 0
    for eps in EPS_LIST:
        hw0 = _hw0(eps)
        rounds = []
        outs = []
        for r in rows:
            k, oc = _decide(r["d"], hw0)
            if oc == "nonterminated":
                nonterm += 1
            if oc == "⊘":
                inap += 1
            if oc == "decided":
                rounds.append(k); all_rounds.append(k); all_eps.append(eps)
            outs.append(oc)
        per_eps[str(eps)] = {"halfwidth0": round(hw0, 4), "n": len(rows), "n_decided": len(rounds),
                             "rounds_p50": float(np.percentile(rounds, 50)) if rounds else None,
                             "rounds_p95": float(np.percentile(rounds, 95)) if rounds else None,
                             "rounds_max": int(max(rounds)) if rounds else None,
                             "inapplicable_rate": round(outs.count("⊘") / len(rows), 4),
                             "nonterminated": outs.count("nonterminated")}
    p95 = float(np.percentile(all_rounds, 95)) if all_rounds else None
    rho, pval = _spearman_p(all_eps, all_rounds) if len(set(all_eps)) > 1 else (None, None)
    return rows, per_eps, nonterm, inap, p95, rho, pval


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:180]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    rows, per_eps, nonterm, inap, p95, rho, pval = _recompute()
    R = rep["results"]
    # S1 行 d 一致
    dmax = max(abs(a["d"] - b["d"]) for a, b in zip(rows, rep["rows"]))
    ch.append(_chk("S1_rows_d_match", dmax < 1e-4 and len(rows) == len(rep["rows"]), "max|Δd|=%.5f n=%d" % (dmax, len(rows))))
    # S2 nonterminated==0 一致
    ch.append(_chk("S2_nonterminated_zero", nonterm == 0 and R["nonterminated"] == 0, "re=%d store=%d" % (nonterm, R["nonterminated"])))
    # S3 per_eps 一致
    bad = 0
    for k, v in per_eps.items():
        sv = R["per_eps"][k]
        for f in ("n", "n_decided", "nonterminated"):
            if v[f] != sv[f]:
                bad += 1
        for f in ("rounds_p50", "rounds_p95", "rounds_max", "inapplicable_rate", "halfwidth0"):
            if v[f] is None or sv[f] is None:
                bad += (v[f] != sv[f])
            elif abs(v[f] - sv[f]) > 1e-3:
                bad += 1
    ch.append(_chk("S3_per_eps_stats_match", bad == 0, "mismatch=%d" % bad))
    # S4 p95 一致
    ch.append(_chk("S4_rounds_p95_match", abs(p95 - R["rounds_p95"]) < 1e-3, "re=%.4f store=%.4f" % (p95, R["rounds_p95"])))
    # S5 spearman rho 一致
    ch.append(_chk("S5_spearman_rho_match", abs(rho - R["spearman_rounds_vs_eps"]["rho"]) < 1e-4,
                   "re=%.4f store=%.4f" % (rho, R["spearman_rounds_vs_eps"]["rho"])))
    # S6 判据 + verdict
    H = {"H59-1_nonterminated_eq_0": nonterm == 0,
         "H59-2_rounds_p95_le_4": bool(p95 is not None and p95 <= 4.0),
         "H59-3_rounds_positive_with_eps": bool(rho is not None and rho > 0 and pval is not None and pval < 0.05)}
    ch.append(_chk("S6_criteria_and_verdict_selfconsistent",
                   all(H[k] == rep["preregistered_verdict"][k] for k in H)
                   and rep["verdict_pass"] == all(rep["preregistered_verdict"].values()),
                   "verdict=%s" % rep["verdict_pass"]))
    # S7 自洽
    sc = R["self_consistency"]
    ch.append(_chk("S7_self_consistency", sc["cells_conserved"] and sc["nonterminated_matches"]))
    # 文档漂移检查
    dg = rep["design"]
    drift = []
    if dg["shrink"] != SHRINK:
        drift.append("design.shrink")
    if dg["floor"] != FLOOR:
        drift.append("design.floor")
    _res["findings"].append({"id": "E59-D1_doc_drift", "severity": "D(一致性)", "pass": not drift,
                             "detail": drift or "无漂移", "code_shrink": SHRINK, "code_floor": FLOOR})
    print("  [%s] S8_doc_constants_match_code%s" % ("PASS" if not drift else "FAIL",
          ("  —— " + str(drift)) if drift else ""))
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
    print("E59 严格审核（L3 独立重推，不 import e59）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 nonterminated=5 → S2 必报",
          lambda r: (r["results"].__setitem__("nonterminated", 5),
                     r["results"]["per_eps"]["0.0"].__setitem__("nonterminated", 5)), ["S2"])
    _tamp("R2 篡改 rounds_p95 → S4 必报",
          lambda r: r["results"].__setitem__("rounds_p95", 9.9), ["S4"])
    _tamp("R3 篡改 spearman rho → S5 必报",
          lambda r: r["results"]["spearman_rounds_vs_eps"].__setitem__("rho", -0.9), ["S5"])
    _tamp("R4 篡改某行 d → S1 必报",
          lambda r: r["rows"][7].__setitem__("d", 9.9), ["S1"])
    _tamp("R5 篡改 verdict_pass → S6 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["S6"])

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
