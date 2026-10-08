# -*- coding: utf-8 -*-
"""_audit_e60b_strict.py —— E60b 严格审核（L3 独立重推）+ L5 隔离自检。

不 import e60b；审计者自写单调域 + 非单调负控 + Spearman + 噪声扫描，独立重算：
定理检验（σ=0 ⇒ ρ≡1）、噪声单调性、负控端点 <0.90；篡改必报。

产物：_audit_e60b_strict.json
"""
import copy
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e60b_ordinal_consistency_report.json")
PREREG = os.path.join(EVAL, "e60b_ordinal_consistency_prereg.json")
OUT = os.path.join(EVAL, "_audit_e60b_strict.json")

N_LEVELS = 30
U_RANGE = (0.05, 0.95)
SIGMA_GRID = [0.0, 0.01, 0.03, 0.08, 0.20]
K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED_TAG = "e60b"
MONO = {"distance": lambda u: 1.0 - u, "force": lambda u: 1.0 - np.sqrt(u),
        "energy": lambda u: (1.0 - u) ** 1.5, "semantic": lambda u: 1.0 / (1.0 + np.exp(8.0 * (u - 0.5)))}
_res = {"experiment": "E60b 严格审核(L3 独立重推 + L5 隔离)", "independence_scope":
        "独立重推：自写单调域/负控/Spearman，不 import e60b", "checks": [], "tamper": []}


def _seed(*p):
    return zlib.crc32(("|".join(str(x) for x in p)).encode()) % (2 ** 31)


def _spearman(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float); n = len(xs)

    def rank(a):
        order = np.argsort(a, kind="mergesort"); rk = np.empty(n, float); i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            rk[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys); rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return 0.0 if den < 1e-12 else float((rx * ry).sum()) / den


def _mean_rho(levels, fns, sigma, rng):
    vecs = {nm: np.array([fn(u) for u in levels]) + (rng.normal(0, sigma, size=len(levels)) if sigma > 0 else 0.0)
            for nm, fn in fns.items()}
    names = list(fns); rs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rs["%s_vs_%s" % (names[i], names[j])] = _spearman(vecs[names[i]], vecs[names[j]])
    return float(np.mean(list(rs.values()))), rs


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    # S0 隔离自检（审核侧不应 import e60b）
    ch.append(_chk("S0_isolation_e60b_not_imported",
                   "e60b_ordinal_consistency_negative_control" not in sys.modules,
                   "sys.modules 含 e60b=%s" % ("e60b_ordinal_consistency_negative_control" in sys.modules)))
    levels = np.linspace(U_RANGE[0], U_RANGE[1], N_LEVELS)
    R = rep["results"]
    # S1 定理检验 σ=0
    rng = np.random.RandomState(_seed("theorem"))
    mean0, rhos0 = _mean_rho(levels, MONO, 0.0, rng)
    allone = all(abs(v - 1.0) < 1e-12 for v in rhos0.values())
    ch.append(_chk("S1_theorem_check_sigma0_rho_exactly_1",
                   allone and abs(mean0 - R["theorem_check_sigma0"]["mean_rho"]) < 1e-9,
                   "re_mean=%.8f all1=%s" % (mean0, allone)))
    # S2 噪声曲线
    bad = 0
    for s in SIGMA_GRID:
        rg = np.random.RandomState(_seed("noise", s))
        m, _ = _mean_rho(levels, MONO, s, rg)
        if abs(round(m, 6) - R["noise_curve_mean_rho"][str(s)]) > 1e-6:
            bad += 1
    ch.append(_chk("S2_noise_curve_recompute", bad == 0, "mismatch=%d" % bad))
    # S3 负控扫描
    bad3 = 0
    kseq = []
    for k in K_GRID:
        a = np.array([1.0 - u + k * np.sin(2.0 * np.pi * u) for u in levels])
        rs = [_spearman(a, np.array([fn(u) for u in levels])) for fn in MONO.values()]
        mk = float(np.mean(rs)); kseq.append(mk)
        if abs(round(mk, 6) - R["negative_control"]["mean_rho_vs_monotone"][str(k)]) > 1e-6:
            bad3 += 1
    ch.append(_chk("S3_negative_control_recompute_and_endpoint_lt_090",
                   bad3 == 0 and abs(R["negative_control"]["endpoint_k1_rho"] - kseq[-1]) < 1e-5
                   and kseq[-1] < 0.90,
                   "mismatch=%d endpoint_re=%.6f store=%.6f" % (bad3, kseq[-1], R["negative_control"]["endpoint_k1_rho"])))
    # S4 判据 + verdict
    H = {"H60b-1_monotone_domains_rho_exactly_1_at_sigma_0": bool(allone),
         "H60b-2_mean_rho_monotone_nonincreasing_in_sigma":
             bool(all([R["noise_curve_mean_rho"][str(SIGMA_GRID[i])] >= R["noise_curve_mean_rho"][str(SIGMA_GRID[i + 1])] - 1e-9
                       for i in range(len(SIGMA_GRID) - 1)])),
         "H60b-3_negative_control_rho_monotone_nonincreasing_in_k_and_endpoint_lt_0.90":
             bool(R["negative_control"]["monotone_nonincreasing_in_k"] and R["negative_control"]["endpoint_k1_rho"] < 0.90)}
    ch.append(_chk("S4_criteria_and_verdict_selfconsistent",
                   all(H[k] == rep["verdict"][k] for k in H) and rep["verdict_pass"] == all(H.values()),
                   "verdict=%s" % rep["verdict_pass"]))
    # S5 prereg 对齐
    pa = rep.get("prereg_alignment", {})
    ch.append(_chk("S5_prereg_alignment", bool(pa.get("sha256_match")) and bool(pa.get("hypotheses_match")),
                   "align=%s" % pa))
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
    print("E60b 严格审核（L3 独立重推，不 import e60b）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 theorem mean_rho → S1 必报",
          lambda r: r["results"]["theorem_check_sigma0"].__setitem__("mean_rho", 0.5), ["S1"])
    _tamp("R2 篡改噪声曲线上一点 → S2 必报",
          lambda r: r["results"]["noise_curve_mean_rho"].__setitem__("0.0", 0.5), ["S2"])
    _tamp("R3 篡改负控端点 → S3 必报",
          lambda r: r["results"]["negative_control"].__setitem__("endpoint_k1_rho", 0.99), ["S3"])
    _tamp("R4 篡改 verdict_pass → S4 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["S4"])

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
