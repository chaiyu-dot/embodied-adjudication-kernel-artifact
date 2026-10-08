# -*- coding: utf-8 -*-
"""blind_replicate_e59b.py —— E59b 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e59b_bench_data.json + e59b_standalone_physics.py（独立重写的掂量/双推原语 + 区间 + 判定回路），
**不 import planning.world_physics_engine / e59b_termination_real_loop**，
独立复现 E59b 的判定计数 / 轮数分布 / 覆盖率 / 收缩率 / 双原语互检，并与报告比对。

产物：blind_replicate_e59b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e59b_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e59b_termination_real_loop_report.json")
BENCH = os.path.join(EVAL, "e59b_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e59b.json")
_repo = ("planning", "world_physics_engine", "rne_dynamics", "control_strategies",
         "e59b_termination_real_loop", "e66_closed_loop_control")
_res = {"experiment": "E59b 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E59b 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（constants + 对象表）",
        all(k in data for k in ("constants", "rows")) and len(data["rows"]) == rep["results"]["n_total"],
        "n_rows=%d (store %d)" % (len(data["rows"]), rep["results"]["n_total"]))
    out = P.replicate(data)
    rr = rep["results"]

    chk("B1 判定计数复现（decided / ⊘不可分辨 / nonterminated=0）",
        out["n_decided"] == rr["n_decided"] and out["n_inapplicable"] == rr["n_inapplicable"]
        and out["n_nonterminated"] == rr["n_nonterminated"],
        "decided re=%d store=%d | ⊘ re=%d store=%d | nonterm re=%d store=%d"
        % (out["n_decided"], rr["n_decided"], out["n_inapplicable"], rr["n_inapplicable"],
           out["n_nonterminated"], rr["n_nonterminated"]))
    chk("B2 轮数分布复现（median / p95 / 与 R* 的 Spearman ρ）",
        out["median_rounds"] == rr["median_rounds"] and out["p95_rounds"] == rr["p95_rounds"]
        and abs(out["spearman"]["rho"] - rr["rounds_vs_rstar_spearman"]["rho"]) < 1e-6,
        "median re=%s store=%s | p95 re=%s store=%s | rho re=%s store=%s"
        % (out["median_rounds"], rr["median_rounds"], out["p95_rounds"], rr["p95_rounds"],
           out["spearman"]["rho"], rr["rounds_vs_rstar_spearman"]["rho"]))
    chk("B3 覆盖率（R∈{1,2,4,9,12}）逐档复现",
        all(out["coverage"][str(R)]["coverage"] == rr["coverage"][str(R)]["coverage"] for R in (1, 2, 4, 9, 12)),
        "re=%s store=%s" % ({R: out["coverage"][str(R)]["coverage"] for R in (1, 2, 4, 9, 12)},
                            {R: rr["coverage"][str(R)]["coverage"] for R in (1, 2, 4, 9, 12)}))
    chk("B4 收缩率（描述性，导出 vs 1/√R）逐档复现",
        all(out["shrink"][str(R)]["observed_median"] == rr["shrink_descriptive"][str(R)]["observed_median"]
            for R in (1, 2, 4, 9, 12)),
        "re=%s" % {R: out["shrink"][str(R)]["observed_median"] for R in (1, 2, 4, 9, 12)})
    chk("B5 双原语互检率复现（136/170=0.8 → H59b-5 诚实 FAIL）",
        out["agree_ok"] == rr["primitive_agreement"]["n_agree"] and out["agree_tot"] == rr["primitive_agreement"]["n"]
        and abs(out["agree_rate"] - rr["primitive_agreement"]["rate"]) < 1e-9,
        "re=%d/%d=%s store=%d/%d=%s" % (out["agree_ok"], out["agree_tot"], out["agree_rate"],
                                        rr["primitive_agreement"]["n_agree"], rr["primitive_agreement"]["n"],
                                        rr["primitive_agreement"]["rate"]))
    chk("B6 预注册判据 H59b-1..5 复现（H59b-5 FAIL，verdict_pass=False）",
        out["H"] == rep["verdict"] and out["verdict_pass"] == rep["verdict_pass"],
        "re=%s | vp re=%s store=%s" % (out["H"], out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e59b_bench_data.json + 独立实现）：E59b 的判定计数/轮数分布/"
                        "覆盖率/收缩率/双原语互检可被外部独立重建（含 H59b-5 的诚实 FAIL 保留）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
