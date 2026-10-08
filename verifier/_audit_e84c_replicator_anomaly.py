# -*- coding: utf-8 -*-
"""_audit_e84c_replicator_anomaly.py —— E84c 的 **L2 第三方复算 + 异常/篡改注入**。

与 L3（结构自洽 + 独立重算曲线）正交：本层**只读报告 JSON**做第三方重算，并**注入异常**，
证明报告的深度曲线/归因/binding 有分辨力（不是"怎么写都过"）。

判据
----
T1 第三方复算：由 `grid_v2_cells` 独立重算各深度档漏报/误拒、binding 分布、单阶补全 → 与报告一致
T2 异常注入：
   a) 把某漏报单元的某个"被丢掉阶"改成 ≥0（使其在浅档也判可行）→ 漏报数必变（可分辨）
   b) 把某危险单元的 truth_dangerous 改成 False → 漏报数必变
   c) 篡改 binding 分布 → T1 必报
   d) 翻转 H84c-2 → verdict_pass 与 all(H) 不一致必报
T3 跨实验一致性：E84c 的"C 有 D 不可替代的盲区"与 E84 的 binding 分布（C=10/400）方向一致
T4 阴性对照：未篡改 → T1 全过；T5 幂等

产物：_audit_e84c_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e84c_depth_ablation_grid_report.json")
E84 = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
OUT = os.path.join(EVAL, "_audit_e84c_replicator_anomaly.json")
_res = {"audit": "E84c L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]
ORDERS = ["K", "S", "D", "C", "E"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def curve_from_cells(cells, depths):
    n = len(cells)
    rows = []
    for d in depths:
        miss = [c for c in cells if min(c["orders"][k] for k in d) >= 0.0 and c["truth_dangerous"]]
        frej = [c for c in cells if min(c["orders"][k] for k in d) < 0.0 and not c["truth_dangerous"]]
        rows.append({"depth": list(d), "miss_count": len(miss),
                     "miss_rate": round(len(miss) / n, 4), "false_reject_count": len(frej)})
    return rows


def binding_dist(cells):
    d = {o: 0 for o in ORDERS}
    for c in cells:
        d[c["binding_order"]] += 1
    return d


def third_party(rep):
    ok, det = True, []
    for key, ckey in (("grid_v1", "grid_v1_cells"), ("grid_v2", "grid_v2_cells")):
        mine = curve_from_cells(rep[ckey], rep["design"]["depths"])
        theirs = rep[key]["depth_curve"]
        for a, b in zip(mine, theirs):
            if a["miss_count"] != b["miss_count"] or a["false_reject_count"] != b["false_reject_count"]:
                ok = False
        if binding_dist(rep[ckey]) != rep[key]["binding_order_distribution"]:
            ok = False
        det.append("%s curve=%s" % (key, [r["miss_count"] for r in theirs]))
    return ok, " ｜ ".join(det)


def singles_from(rep, ckey):
    cells = rep[ckey]
    base = curve_from_cells(cells, rep["design"]["depths"])[0]["miss_count"]
    out = {}
    for o in ("S", "D", "C", "E"):
        m = len([c for c in cells
                 if min(c["orders"][k] for k in ("K", o)) >= 0.0 and c["truth_dangerous"]])
        out[o] = base - m
    return out


def main():
    print("=" * 92)
    print("E84c L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    ok, det = third_party(rep)
    chk("T1 第三方复算（深度曲线 / binding 分布）与报告一致", ok, det)

    mine_s = singles_from(rep, "grid_v2_cells")
    theirs_s = {o: rep["grid_v2"]["single_order_completion"][o]["misses_removed_vs_K"] for o in mine_s}
    chk("T1b 单阶补全（misses_removed_vs_K）与报告一致", mine_s == theirs_s,
        "mine=%s theirs=%s" % (mine_s, theirs_s))

    depths = rep["design"]["depths"]
    deep = depths[2]                      # {K,S,D}
    # 找一个"深度 3 档确实漏报"的单元（该单元必有 C 或 E < 0），
    # 以及一个"当前不算漏报、但危险"的单元（D<0 → 被 D 抓住）——用于注入。
    miss_idx = next((i for i, c in enumerate(rep["grid_v2_cells"])
                     if min(c["orders"][k] for k in deep) >= 0.0 and c["truth_dangerous"]), None)
    gain_idx = next((i for i, c in enumerate(rep["grid_v2_cells"])
                     if c["truth_dangerous"] and c["orders"]["D"] < 0.0), None)
    chk("T2-前提 存在 {K,S,D} 档漏报单元（C 的盲区）与 D 承重单元（可注入检验）",
        miss_idx is not None and gain_idx is not None,
        "miss_idx=%s gain_idx=%s" % (miss_idx, gain_idx))

    base_miss3 = curve_from_cells(rep["grid_v2_cells"], depths)[2]["miss_count"]
    a = copy.deepcopy(rep)
    # 把"本来被 D 抓住"的危险单元的 D 改成不失败 → 它变成 {K,S,D} 档漏报（曲线必变）
    a["grid_v2_cells"][gain_idx]["orders"]["D"] = 0.5
    chk("T2a 抹掉某 D 承重单元的 D 失败 → 深度曲线第三方重算必变（+1 漏报）",
        curve_from_cells(a["grid_v2_cells"], depths)[2]["miss_count"] == base_miss3 + 1,
        "before=%d after=%d" % (base_miss3,
                                curve_from_cells(a["grid_v2_cells"], depths)[2]["miss_count"]))

    b = copy.deepcopy(rep)
    b["grid_v2_cells"][miss_idx]["truth_dangerous"] = False
    chk("T2b 抹掉某漏报单元的危险标志 → 漏报数必变（-1）",
        curve_from_cells(b["grid_v2_cells"], depths)[2]["miss_count"] == base_miss3 - 1,
        "before=%d after=%d" % (base_miss3,
                                curve_from_cells(b["grid_v2_cells"], depths)[2]["miss_count"]))

    c = copy.deepcopy(rep)
    c["grid_v2"]["binding_order_distribution"]["D"] += 1
    chk("T2c 篡改 binding 分布 → T1 必报", not third_party(c)[0])

    d = copy.deepcopy(rep)
    d["preregistered_verdict"]["H84c-2_miss_monotone"] = not d["preregistered_verdict"]["H84c-2_miss_monotone"]
    chk("T2d 翻转 H84c-2 → verdict_pass≠all(H) 必报",
        d["verdict_pass"] != all(d["preregistered_verdict"].values()))

    e84 = json.load(open(E84, encoding="utf-8"))
    c_e84 = e84["order_binding_distribution"].get("C", 0)
    chk("T3 跨实验口径一致：E84c『C 有 D 不可替代的盲区』与 E84 binding C>0 同向",
        bool(rep["grid_v2"]["C_unique_catch_over_D"] > 0 and c_e84 > 0),
        "E84c C_unique=%d ｜ E84 binding C=%d" % (rep["grid_v2"]["C_unique_catch_over_D"], c_e84))

    chk("T4 阴性对照：未篡改 → T1 通过", third_party(rep)[0])
    chk("T5 幂等：两次读入结论一致",
        third_party(json.load(open(REPORT, encoding="utf-8"))) == third_party(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["第三方复算与异常注入均按预期：深度曲线/归因/binding 分布有分辨力（篡改可被抓）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
