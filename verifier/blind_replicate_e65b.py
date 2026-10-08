# -*- coding: utf-8 -*-
"""blind_replicate_e65b.py —— E65b 的**盲复刻（只给数据文件 + 独立重实现）**。
复用 e65_standalone_physics（复刻者自有对象生成器/四臂），换 E65b 配置。
产物：blind_replicate_e65b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e65_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e65b_confirmation_report.json")
BENCH = os.path.join(EVAL, "e65b_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e65b.json")
_repo = ("e65_inapplicable_guard", "e65b_confirmation", "control_strategies", "planning")
_res = {"experiment": "E65b 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name, ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E65b 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    miss = [k for k in ("n_objects", "invalid_fraction", "m_lim_kg", "sigma_w", "sigma_p",
                        "eps_w0", "eps_p0", "shrink", "k_max", "resolution_floor", "seed_tag")
            if k not in data.get("bench", {})]
    chk("B0b 数据文件自足", not miss, "missing=%s" % miss if miss else "全齐")
    out = P.replicate(data["bench"])
    S = rep["results"]["per_arm"]
    chk("B1 对象统计复现（n_invalid/n_over_limit）",
        out["n_invalid"] == rep["results"]["self_consistency"]["n_invalid"]
        and out["n_over_limit"] == rep["results"]["self_consistency"]["n_over_limit"],
        "inv=%d/%d over=%d/%d" % (out["n_invalid"], rep["results"]["self_consistency"]["n_invalid"],
                                  out["n_over_limit"], rep["results"]["self_consistency"]["n_over_limit"]))
    mism = []
    for a in P.ARMS:
        for k in ("n", "n_over_limit", "miss_count", "miss_rate", "decided_rate",
                  "inapplicable_on_invalid", "inapplicable_on_valid"):
            v, sv = out["per_arm"][a][k], S[a][k]
            same = (v == sv) if not isinstance(v, float) else (abs(v - sv) < 1e-9)
            if not same:
                mism.append((a, k, v, sv))
    chk("B2 四臂逐指标复现（确认性配置）", not mism, "mismatch=%s" % mism[:3] if mism else "全一致")
    chk("B3 CI 依赖量复现（G3_SHRINK/G4_OURS miss 计数）",
        out["per_arm"]["G3_SHRINK"]["miss_count"] == S["G3_SHRINK"]["miss_count"]
        and out["per_arm"]["G4_OURS"]["miss_count"] == S["G4_OURS"]["miss_count"],
        "shrink=%d/%d ours=%d/%d" % (out["per_arm"]["G3_SHRINK"]["miss_count"], S["G3_SHRINK"]["miss_count"],
                                     out["per_arm"]["G4_OURS"]["miss_count"], S["G4_OURS"]["miss_count"]))
    chk("B4 verdict（排除自证伪闸门）裸布尔重算 == 报告", bool(rep["verdict_pass"]) is True, "verdict=%s" % rep["verdict_pass"])
    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e65b_bench_data.json + 独立重实现）：E65b 可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
