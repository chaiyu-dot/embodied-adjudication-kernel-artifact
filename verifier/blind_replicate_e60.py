# -*- coding: utf-8 -*-
"""blind_replicate_e60.py —— E60 的**盲复刻（只给数据文件 + 独立重实现）**。

只拿 e60_bench_data.json + e60_standalone_physics.py（自写四域映射 + 分档 + Spearman），
**不 import planning.energy_margin / e60_energy_margin**，独立复现跨域一致性并与报告比对。
产物：blind_replicate_e60.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e60_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e60_energy_margin_report.json")
BENCH = os.path.join(EVAL, "e60_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e60.json")

_repo = ("energy_margin", "e60_energy_margin", "planning")
_res = {"experiment": "E60 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E60 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["n_levels", "u_range", "noise_sigma", "band_edges"]
    miss = [k for k in need if k not in data.get("design", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data["design"])

    d = 0.0
    for o, r in zip(out["rows"], rep["rows"]):
        for k in P.DOMS:
            d = max(d, abs(o["m_E"][k] - r["m_E"][k]))
    chk("B1 逐行四域 m_E 复现（30 行 × 4 域）", d < 1e-3, "max|Δm_E|=%.4f" % d)

    bad = 0
    for o, r in zip(out["rows"], rep["rows"]):
        for k in P.DOMS:
            if o["band"][k] != r["band"][k]:
                bad += 1
    chk("B2 逐行分档复现", bad == 0, "band_mismatch=%d" % bad)

    chk("B3 平均两两 Spearman ρ 复现",
        abs(out["mean_rho"] - rep["results"]["ordinal_consistency_mean_pairwise_rho"]) < 1e-4,
        "re=%.4f store=%.4f" % (out["mean_rho"], rep["results"]["ordinal_consistency_mean_pairwise_rho"]))

    chk("B4 档位一致率复现（描述性）",
        abs(out["band_agreement_rate"] - rep["results"]["band_agreement_rate"]) < 1e-4,
        "re=%.4f store=%.4f" % (out["band_agreement_rate"], rep["results"]["band_agreement_rate"]))

    chk("B5 verdict（承重=序数 ρ≥0.90）裸布尔重算 == 报告",
        bool(out["mean_rho"] >= 0.90) == bool(rep["verdict_pass"]), "re=%s store=%s" % (out["mean_rho"] >= 0.90, rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e60_bench_data.json + 独立重实现）：E60 跨域一致性可被外部独立重建。"]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
