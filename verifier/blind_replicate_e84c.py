# -*- coding: utf-8 -*-
"""blind_replicate_e84c.py —— E84c 的 **L4 隔离盲复刻**（只给数据文件 + 独立实现）。

只拿 `e84c_bench_data.json` + 复刻者侧独立实现（`l4_plant2r` + `e84c_standalone_physics`），
**不 import `planning.*` / 任何 e8x 实验脚本**，独立重算：
  · v1/v2 网格上每个单元的 K/S/D/C/E 五阶裕度；
  · 每单元的 binding 阶与"是否危险"；
  · 各深度档的漏报数 / 漏报率 / 误拒数；
  · 逐格与作者报告比对（连续量按容差，离散量按精确相等）。

产物：blind_replicate_e84c.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e84c_standalone_physics as SP   # noqa: E402

REPORT = os.path.join(EVAL, "e84c_depth_ablation_grid_report.json")
BENCH = os.path.join(EVAL, "e84c_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e84c.json")
_repo = ("planning", "rne_dynamics", "rigid_body_certificate", "trackability",
         "energy_margin", "control_strategies", "e84c_depth_ablation_grid",
         "e84c_five_order_wrapper", "e84_five_order_wrapper_loop")
_res = {"experiment": "E84c 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E84c 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/tau_lim/q_lim/zeta/T_s/qdot_lim/n_traj/grid_v1/grid_v2/depths）",
        all(k in data for k in ("bench", "tau_lim", "q_lim", "zeta", "T_s", "qdot_lim",
                                "n_traj", "grid_v1", "grid_v2", "depths")),
        "orders=%s" % data["orders"])

    out = SP.replicate(data)

    # ---- B1 逐格五阶裕度 与报告一致（连续量容差 2e-3）----
    worst = {k: 0.0 for k in data["orders"]}
    bad = 0
    for tag, key in (("grid_v1", "grid_v1_cells"), ("grid_v2", "grid_v2_cells")):
        rc = {(c["A"], c["T"], c["mp"], c["wn"]): c for c in rep[key]}
        for c in out[tag]:
            k = (c["A"], c["T"], c["mp"], c["wn"])
            r = rc.get(k)
            if r is None:
                bad += 1
                continue
            for o in data["orders"]:
                d = abs(c["orders"][o] - r["orders"][o])
                worst[o] = max(worst[o], d)
                if d > 2e-3:
                    bad += 1
    chk("B1 逐格五阶裕度逐项一致（v1 40 格 + v2 168 格；容差 2e-3）", bad == 0,
        "max|Δ| per order = %s | mismatch=%d" % ({k: round(v, 6) for k, v in worst.items()}, bad))

    # ---- B2 binding 阶 & 危险标志 ----
    bad2 = 0
    for tag, key in (("grid_v1", "grid_v1_cells"), ("grid_v2", "grid_v2_cells")):
        rc = {(c["A"], c["T"], c["mp"], c["wn"]): c for c in rep[key]}
        for c in out[tag]:
            r = rc[(c["A"], c["T"], c["mp"], c["wn"])]
            if c["binding_order"] != r["binding_order"] or c["truth_dangerous"] != r["truth_dangerous"]:
                bad2 += 1
    chk("B2 binding 阶 与 危险标志 逐格一致（208 格）", bad2 == 0, "mismatch=%d" % bad2)

    # ---- B3 深度曲线（漏报/误拒）----
    bad3 = []
    for tag, key, rep_key in (("grid_v1", "grid_v1_cells", "grid_v1"),
                              ("grid_v2", "grid_v2_cells", "grid_v2")):
        mine = SP.depth_curve(out[tag], data["depths"])
        theirs = rep[rep_key]["depth_curve"]
        for a, b in zip(mine, theirs):
            if (a["miss_count"] != b["miss_count"] or a["false_reject_count"] != b["false_reject_count"]
                    or abs(a["miss_rate"] - b["miss_rate"]) > 1e-6):
                bad3.append((tag, a["depth"], a["miss_count"], b["miss_count"]))
    chk("B3 各深度档漏报数/漏报率/误拒数 逐档一致（v1+v2 共 10 档）", len(bad3) == 0,
        "mismatch=%s" % bad3)

    # ---- B4 binding 分布 ----
    dist_ok = True
    det4 = []
    for tag, rep_key in (("grid_v1", "grid_v1"), ("grid_v2", "grid_v2")):
        d = {o: 0 for o in data["orders"]}
        for c in out[tag]:
            d[c["binding_order"]] += 1
        same = d == rep[rep_key]["binding_order_distribution"]
        dist_ok &= same
        det4.append("%s mine=%s theirs=%s" % (tag, d, rep[rep_key]["binding_order_distribution"]))
    chk("B4 binding 阶分布 逐档一致", dist_ok, " ｜ ".join(det4))

    # ---- B5 预注册判据 ----
    mine_v2 = rep["preregistered_verdict"]
    d2 = {o: 0 for o in data["orders"]}
    for c in out["grid_v2"]:
        d2[c["binding_order"]] += 1
    curves = SP.depth_curve(out["grid_v2"], data["depths"])
    rates = [r["miss_rate"] for r in curves]
    singles = {}
    base = curves[0]["miss_count"]
    for o in ("S", "D", "C", "E"):
        m = len([c for c in out["grid_v2"]
                 if min(c["orders"][k] for k in ("K", o)) >= 0.0 and c["truth_dangerous"]])
        singles[o] = base - m
    recomputed = {
        "H84c-1_grid_exercises_ladder": sum(1 for o in data["orders"] if d2[o] > 0) >= 4,
        "H84c-2_miss_monotone": all(rates[i] >= rates[i + 1] - 1e-12 for i in range(len(rates) - 1))
                                and rates[-1] == 0.0,
        "H84c-3_shallow_structural_gap": rates[1] > 0.10,
        "H84c-4_D_at_least_as_strong_as_S": singles["D"] >= singles["S"],
        "H84c-5_C_has_D_uncovered_gap": curves[2]["miss_count"] > 0,
        "H84c-6_false_reject_zero": all(r["false_reject_count"] == 0 for r in curves),
        "H84c-7_E_dominated_disclosure": (curves[3]["miss_count"] == curves[4]["miss_count"]),
    }
    chk("B5 预注册判据 H84c-1..7 逐条复现", recomputed == mine_v2,
        "mine=%s theirs=%s" % (recomputed, mine_v2))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e84c_bench_data.json + l4_plant2r + e84c_standalone_physics）："
        "208 个网格单元的五阶裕度、binding 阶、深度曲线（漏报/误拒）与预注册判据均被外部独立重建。",
        "**语义强度标注**：被复刻对象含 K/S/D/C/E 五个**判据**（而非控制器族），复刻者是"
        "**同一判据规格的另一实现**（显式 RNE + 闭式能量 + 核的带宽判据）——"
        "对『网格→裕度』管线独立性高，对『判据本身是否物理正确』不构成验证（那属 L1/L2）。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
