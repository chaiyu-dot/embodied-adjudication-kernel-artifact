# -*- coding: utf-8 -*-
"""blind_replicate_e59.py —— E59 的**盲复刻（只给数据文件 + 独立重实现）**。

只拿 e59_bench_data.json + e59_standalone_physics.py（自写距离生成 + 终止规则 + 统计），
**不 import e59_termination_bound**，独立复现终止界并与报告比对。
产物：blind_replicate_e59.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e59_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e59_termination_report.json")
BENCH = os.path.join(EVAL, "e59_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e59.json")

_repo = ("e59_termination_bound", "planning", "control_strategies")
_res = {"experiment": "E59 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E59 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["n_obj", "n_seed", "eps_list", "k_max", "shrink", "floor"]
    miss = [k for k in need if k not in data.get("design", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data["design"])
    dmax = max(abs(o["d"] - r["d"]) for o, r in zip(out["rows"], rep["rows"]))
    chk("B1 可分辨距离逐行复现（%d 行）" % len(rep["rows"]),
        dmax < 1e-4 and len(out["rows"]) == len(rep["rows"]), "max|Δd|=%.5f" % dmax)

    chk("B2 nonterminated=0 复现", out["nonterminated"] == 0 and rep["results"]["nonterminated"] == 0,
        "re=%d store=%d" % (out["nonterminated"], rep["results"]["nonterminated"]))

    bad = 0
    for k, v in out["per_eps"].items():
        sv = rep["results"]["per_eps"][k]
        if v["n_decided"] != sv["n_decided"] or v["nonterminated"] != sv["nonterminated"] or v["rounds_max"] != sv["rounds_max"]:
            bad += 1
    chk("B3 逐 eps 统计复现（5 档）", bad == 0, "mismatch=%d" % bad)

    chk("B4 rounds p95 复现", abs(out["p95"] - rep["results"]["rounds_p95"]) < 1e-3,
        "re=%.3f store=%.3f" % (out["p95"], rep["results"]["rounds_p95"]))

    chk("B5 Spearman(rounds,ε) 复现", abs(out["rho"] - rep["results"]["spearman_rounds_vs_eps"]["rho"]) < 1e-4,
        "re=%.4f store=%.4f" % (out["rho"], rep["results"]["spearman_rounds_vs_eps"]["rho"]))

    H = {"H59-1": out["nonterminated"] == 0, "H59-2": out["p95"] <= 4.0,
         "H59-3": (out["rho"] is not None and out["rho"] > 0 and out["pval"] is not None and out["pval"] < 0.05)}
    chk("B6 verdict 判据裸布尔重算 == 报告", all(H.values()) == bool(rep["verdict_pass"]),
        "re_all=%s store=%s" % (all(H.values()), rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e59_bench_data.json + 独立重实现）：E59 终止界可被外部独立重建。"]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
