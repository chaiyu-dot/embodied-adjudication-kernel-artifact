# -*- coding: utf-8 -*-
"""blind_reverse_e60.py —— E60 的**逆向盲测（L5，隔离 + 不信声明）**。

独立性：隔离（运行期自检 sys.modules 未加载 e60_energy_margin）。审计者自写四域映射与分档，
以**负对照**证明 ρ 判据有分辨力（打乱一个域 → ρ 崩）与**噪声敏感性**（σ↑ → 档位一致率↓，
说明『一致』由噪声/档宽决定，而非域间真实同一）。

产物：blind_reverse_e60.json
"""
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e60_energy_margin_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e60.json")

NOISE = 0.03
EDGES = (-0.05, 0.25, 0.60)
DOMS = ["distance", "force", "energy", "semantic"]

_res = {"experiment": "E60 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e60_energy_margin", "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _seed(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _margins(u, rng, noise):
    n = [float(rng.normal(0, noise)) for _ in range(4)]
    return {"distance": (1.0 - u) + n[0], "force": (1.0 - u * (1.0 + 0.03 * u)) + n[1],
            "energy": (1.0 - u * (1.0 + 0.05 * u)) + n[2], "semantic": (1.0 - u ** 0.96) + n[3]}


def _band(m):
    return "unsafe" if m < EDGES[0] else ("tight" if m < EDGES[1] else ("medium" if m < EDGES[2] else "loose"))


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


def _mean_rho(vecs):
    rs = [_rho(vecs[DOMS[i]], vecs[DOMS[j]]) for i in range(4) for j in range(i + 1, 4)]
    return sum(rs) / len(rs)


def _gen(noise, n=30):
    us = np.linspace(0.05, 0.95, n)
    vecs = {k: [] for k in DOMS}
    rows = []
    for i, u in enumerate(us):
        rng = np.random.RandomState(_seed("e60", i))
        ms = _margins(float(u), rng, noise)
        for k in DOMS:
            vecs[k].append(ms[k])
        rows.append({k: _band(ms[k]) for k in DOMS})
    rate = sum(1 for r in rows if len(set(r.values())) == 1) / n
    return vecs, rate


def main():
    print("=" * 88)
    print("E60 逆向盲测（L5 隔离，自写四域 + 负对照 + 噪声敏感性）")
    print("=" * 88)
    chk("R0_isolation_e60_not_imported",
        "e60_energy_margin" not in sys.modules,
        "sys.modules 含 e60=%s" % ("e60_energy_margin" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    store_mean = rep["results"]["ordinal_consistency_mean_pairwise_rho"]

    # R1：独立重算 mean ρ 与档位一致率 == 存储
    vecs, rate = _gen(NOISE)
    mean_rho = _mean_rho(vecs)
    chk("R1_recompute_mean_rho_and_band_rate",
        abs(mean_rho - store_mean) < 1e-3 and abs(rate - rep["results"]["band_agreement_rate"]) < 1e-3,
        "rho_re=%.4f store=%.4f | rate_re=%.4f store=%.4f"
        % (mean_rho, store_mean, rate, rep["results"]["band_agreement_rate"]))

    # R2：负对照——打乱一个域的顺序 → ρ 崩（证 ρ 判据有分辨力，非恒 ≥0.90）
    rng = np.random.RandomState(7)
    perm = rng.permutation(30)
    vecs_bad = {k: list(v) for k, v in vecs.items()}
    vecs_bad["semantic"] = [vecs["semantic"][i] for i in perm]
    mean_bad = _mean_rho(vecs_bad)
    chk("R2_negative_control_shuffled_domain_breaks_rho",
        mean_bad < 0.90 and abs(mean_rho - store_mean) < 1e-3,
        "shuffled mean_rho=%.4f (should <0.90) vs clean=%.4f" % (mean_bad, mean_rho))

    # R3：噪声敏感性——σ↑ → 档位一致率↓（『一致』由噪声/档宽决定）
    _, rate_hi = _gen(0.10)
    chk("R3_noise_sensitivity_band_rate_drops",
        rate_hi < rate, "rate(σ=0.03)=%.3f rate(σ=0.10)=%.3f" % (rate, rate_hi))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
