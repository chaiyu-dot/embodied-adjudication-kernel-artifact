# -*- coding: utf-8 -*-
"""blind_reverse_e60b.py —— E60b 的 L5 逆向盲测（隔离 + 负控外推 + 参数反推）。

独立性：隔离（自检 sys.modules 不含 e60b）。与 L3 正交的探针：
  · 负控外推：把非单调强度 k 扩到存储网格之外（k=1.25/1.5/2.0）→ ρ 继续下降且 <0.90；
  · 定理稳健：把电平数从 30 换到 60 → σ=0 时单调域两两 ρ 仍**精确=1.0**；
  · 参数反推：独立求 ρ 跌破 0.90 的临界 k*（二分）→ 给出"ρ<0.90 的门槛在 k*≈?"的定量边界。
产物：blind_reverse_e60b.json
"""
import json
import math
import os
import sys

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e60b_ordinal_consistency_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e60b.json")

MONO = {"distance": lambda u: 1.0 - u, "force": lambda u: 1.0 - np.sqrt(u),
        "energy": lambda u: (1.0 - u) ** 1.5, "semantic": lambda u: 1.0 / (1.0 + np.exp(8.0 * (u - 0.5)))}
_res = {"experiment": "E60b 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e60b；负控外推/定理稳健/临界 k 反推", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


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


def _mean_rho_vs_bumpy(levels, k):
    a = np.array([1.0 - u + k * np.sin(2.0 * np.pi * u) for u in levels])
    return float(np.mean([_spearman(a, np.array([fn(u) for u in levels])) for fn in MONO.values()]))


def main():
    print("=" * 88)
    print("E60b 逆向盲测（L5 隔离，负控外推 + 定理稳健 + 临界 k 反推）")
    print("=" * 88)
    chk("R0_isolation_e60b_not_imported",
        "e60b_ordinal_consistency_negative_control" not in sys.modules,
        "sys.modules 含 e60b=%s" % ("e60b_ordinal_consistency_negative_control" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    levels = np.linspace(0.05, 0.95, 30)

    # R1：负控外推（存储网格只到 k=1.0）
    ext = {str(k): round(_mean_rho_vs_bumpy(levels, k), 6) for k in (1.25, 1.5, 2.0)}
    stored = rep["results"]["negative_control"]["mean_rho_vs_monotone"]
    seq = [stored[str(k)] for k in (0.0, 0.25, 0.5, 0.75, 1.0)] + [ext["1.25"], ext["1.5"], ext["2.0"]]
    mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
    chk("R1_negative_control_extrapolation", mono and all(v < 0.90 for v in seq[4:]),
        "ext=%s" % ext)

    # R2：定理稳健（换电平数 60）
    lv60 = np.linspace(0.05, 0.95, 60)
    rhos = []
    for i, (a, fn) in enumerate(MONO.items()):
        for b, fn2 in list(MONO.items())[i + 1:]:
            rhos.append(_spearman(np.array([fn(u) for u in lv60]), np.array([fn2(u) for u in lv60])))
    chk("R2_theorem_robust_at_n60", all(abs(v - 1.0) < 1e-12 for v in rhos), "n=60 all rho=1.0")

    # R3：临界 k 反推（二分 ρ(k)=0.90）
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _mean_rho_vs_bumpy(levels, mid) > 0.90:
            lo = mid
        else:
            hi = mid
    kstar = lo
    chk("R3_recover_critical_k_star",
        0.3 <= kstar <= 1.0 and rep["results"]["negative_control"]["endpoint_k1_rho"] < 0.90
        and _mean_rho_vs_bumpy(levels, kstar + 0.02) < 0.90 <= _mean_rho_vs_bumpy(levels, 0.0) + 1e-12,
        "k*(rho=0.90)=%.4f ; endpoint_k1=%.4f" % (kstar, rep["results"]["negative_control"]["endpoint_k1_rho"]))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass; _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["recovered"] = {"critical_k_star": round(kstar, 4), "extrapolated_rho": ext}
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
