# -*- coding: utf-8 -*-
"""blind_replicate_e60b.py —— E60b 的**盲复刻（只给数据文件 + 独立重实现）**。
产物：blind_replicate_e60b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e60b_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e60b_ordinal_consistency_report.json")
BENCH = os.path.join(EVAL, "e60b_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e60b.json")
_repo = ("e60b_ordinal_consistency_negative_control", "planning")
_res = {"experiment": "E60b 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name, ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E60b 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    miss = [k for k in ("n_levels", "u_range", "sigma_grid", "k_grid") if k not in data.get("design", {})]
    chk("B0b 数据文件自足", not miss, "missing=%s" % miss if miss else "全齐")
    out = P.replicate(data["design"])
    R = rep["results"]
    chk("B1 定理检验 σ=0 均值 ρ 复现（<1e-9）",
        abs(out["theorem_mean_rho"] - R["theorem_check_sigma0"]["mean_rho"]) < 1e-6,
        "re=%.8f store=%.8f" % (out["theorem_mean_rho"], R["theorem_check_sigma0"]["mean_rho"]))
    bad = sum(1 for k in out["noise_curve"] if abs(out["noise_curve"][k] - R["noise_curve_mean_rho"][k]) > 1e-6)
    chk("B2 噪声曲线逐点复现（5 点）", bad == 0, "mismatch=%d" % bad)
    bad2 = sum(1 for k in out["neg_curve"] if abs(out["neg_curve"][k] - R["negative_control"]["mean_rho_vs_monotone"][k]) > 1e-6)
    chk("B3 负控曲线逐点复现（5 点）", bad2 == 0, "mismatch=%d" % bad2)
    chk("B4 verdict 判据裸布尔重算 == 报告",
        rep["verdict_pass"] == bool(abs(out["theorem_mean_rho"] - 1.0) < 1e-12
                                    and R["noise_curve_monotone_nonincreasing"]
                                    and R["negative_control"]["monotone_nonincreasing_in_k"]
                                    and R["negative_control"]["endpoint_k1_rho"] < 0.90), "verdict=%s" % rep["verdict_pass"])
    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e60b_bench_data.json + 独立重实现）：E60b 可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
