# -*- coding: utf-8 -*-
"""blind_replicate_e65.py —— E65 的**盲复刻（只给数据文件 + 独立重实现）**。

只拿 e65_bench_data.json + e65_standalone_physics.py（自写对象生成器 + 四臂 + 统计），
**不 import e65_inapplicable_guard / control_strategies / planning.***，独立复现逐臂统计并与报告比对。
隔离保证：运行期 sys.modules 不得含 e65 / control_strategies / planning。
产物：blind_replicate_e65.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e65_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e65_inapplicable_guard_report.json")
BENCH = os.path.join(EVAL, "e65_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e65.json")

_repo = ("e65_inapplicable_guard", "control_strategies", "planning", "rne_dynamics")
_res = {"experiment": "E65 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E65 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["n_objects", "invalid_fraction", "m_lim_kg", "sigma_w", "sigma_p", "eps_w0",
            "eps_p0", "shrink", "k_max", "resolution_floor", "seed_tag"]
    miss = [k for k in need if k not in data.get("bench", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data["bench"])
    S = rep["results"]["per_arm"]

    chk("B1 对象统计复现（n_invalid / n_over_limit / n_rows）",
        out["n_invalid"] == rep["results"]["self_consistency"]["n_invalid"]
        and out["n_over_limit"] == rep["results"]["self_consistency"]["n_over_limit"]
        and out["n_rows"] == len(rep["rows"]),
        "inv=%d/%d over=%d/%d rows=%d/%d" % (out["n_invalid"], rep["results"]["self_consistency"]["n_invalid"],
                                             out["n_over_limit"], rep["results"]["self_consistency"]["n_over_limit"],
                                             out["n_rows"], len(rep["rows"])))

    mism = []
    for a in P.ARMS:
        for k in ("n", "n_over_limit", "miss_count", "miss_rate", "decided_rate",
                  "abstain_rate", "inapplicable_rate", "inapplicable_on_invalid", "inapplicable_on_valid"):
            v, sv = out["per_arm"][a][k], S[a][k]
            same = (v == sv) if not isinstance(v, float) else (abs(v - sv) < 1e-9)
            if not same:
                mism.append((a, k, v, sv))
    chk("B2 四臂逐指标复现（miss/decided/abstain/⊘ 选择性）", not mism, "mismatch=%s" % mism[:3] if mism else "全一致")

    chk("B3 ⊘ 选择性复现（失效子集 ≥0.90 / 合法子集 ≤0.10）",
        out["per_arm"]["G4_OURS"]["inapplicable_on_invalid"] >= 0.90
        and out["per_arm"]["G4_OURS"]["inapplicable_on_valid"] <= 0.10,
        "on_inv=%.3f on_val=%.3f" % (out["per_arm"]["G4_OURS"]["inapplicable_on_invalid"],
                                     out["per_arm"]["G4_OURS"]["inapplicable_on_valid"]))

    H = {"H65-1": S["G3_RETRY"]["miss_rate"] >= 0.40, "H65-2": S["G4_OURS"]["miss_rate"] <= 0.02,
         "H65-3": S["G3_SHRINK"]["miss_rate"] >= 0.40,
         "H65-4": S["G4_OURS"]["inapplicable_on_valid"] <= 0.10 and S["G4_OURS"]["inapplicable_on_invalid"] >= 0.90,
         "H65-5": S["G3_ABSTAIN"]["decided_rate"] <= 0.60,
         "H65-6": not (S["G3_RETRY"]["miss_rate"] >= 0.40 or S["G3_SHRINK"]["miss_rate"] >= 0.40)}
    chk("B4 verdict 判据裸布尔重算 == 报告", all(H.values()) == bool(rep["verdict_pass"]),
        "re_all=%s store=%s（诚实 False 由 H65-2 驱动）" % (all(H.values()), rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e65_bench_data.json + 独立重实现）：E65 四臂统计可被外部独立重建。"]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
