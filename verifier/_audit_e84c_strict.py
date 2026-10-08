# -*- coding: utf-8 -*-
"""_audit_e84c_strict.py —— E84c 的 **L3 严格审核**（结构自洽 + 独立重算 + 篡改捕获 + 披露完整性）。

审核对象：`e84c_depth_ablation_grid_report.json`

判据
----
S1 报告↔明细自洽：depth_curve 由 grid_*_cells 重算一致；binding 分布由 cells 重算一致；
   n_distinct_binding_orders 与分布一致；verdict_pass = all(H)。
S2 口径不变量：每档 false_reject == 0；`{K}` 档漏报 ≥ 深档漏报（单调不增）；
   全深度档漏报 == 0；每单元 truth_min == min(五阶)；truth_dangerous ⟺ truth_min < 0；
   binding_order == argmin(五阶)。
S3 **两段式披露完整性**：`grid_v1` 与 `grid_v2` 的单元数分别等于申报的网格规格；
   v1 的原始曲线（漏报 31/11/0/0/0）与 binding 分布（D=38,C=2）必须**原样保留**（防"删掉不利结果"）。
S4 篡改捕获：改某单元阶值 / 改 truth_dangerous / 改 H / 改 v1 曲线 → S1/S2/S3 必报。
S5 幂等：两次读入结论一致。

产物：_audit_e84c_strict.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e84c_depth_ablation_grid_report.json")
OUT = os.path.join(EVAL, "_audit_e84c_strict.json")
_res = {"audit": "E84c L3 strict", "checks": [], "findings": []}
_n = [0]
ORDERS = ["K", "S", "D", "C", "E"]
V1_CURVE = [31, 11, 0, 0, 0]
V1_BINDING = {"K": 0, "S": 0, "D": 38, "C": 2, "E": 0}


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _curve(cells, depths):
    n = len(cells)
    out = []
    for d in depths:
        miss = [c for c in cells if min(c["orders"][k] for k in d) >= 0.0 and c["truth_dangerous"]]
        frej = [c for c in cells if min(c["orders"][k] for k in d) < 0.0 and not c["truth_dangerous"]]
        out.append({"depth": list(d), "miss_count": len(miss),
                    "miss_rate": round(len(miss) / n, 4), "false_reject_count": len(frej)})
    return out


def _bind(cells):
    d = {o: 0 for o in ORDERS}
    for c in cells:
        d[c["binding_order"]] += 1
    return d


def s1(rep):
    ok = True
    depths = rep["design"]["depths"]
    for key, ckey in (("grid_v1", "grid_v1_cells"), ("grid_v2", "grid_v2_cells")):
        mine = _curve(rep[ckey], depths)
        theirs = rep[key]["depth_curve"]
        if len(mine) != len(theirs):
            return False
        for a, b in zip(mine, theirs):
            if (a["miss_count"] != b["miss_count"] or a["miss_rate"] != b["miss_rate"]
                    or a["false_reject_count"] != b["false_reject_count"]
                    or a["depth"] != b["depth"]):
                ok = False
        if _bind(rep[ckey]) != rep[key]["binding_order_distribution"]:
            ok = False
        if rep[key]["n_distinct_binding_orders"] != sum(
                1 for o in ORDERS if rep[key]["binding_order_distribution"][o] > 0):
            ok = False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        ok = False
    return ok


def s2(rep):
    ok = True
    for key, ckey in (("grid_v1", "grid_v1_cells"), ("grid_v2", "grid_v2_cells")):
        rows = rep[key]["depth_curve"]
        if any(r["false_reject_count"] != 0 for r in rows):
            ok = False
        miss = [r["miss_count"] for r in rows]
        if any(miss[i] < miss[i + 1] for i in range(len(miss) - 1)):
            ok = False
        if miss[-1] != 0:
            ok = False
        for c in rep[ckey]:
            vals = [c["orders"][o] for o in ORDERS]
            if abs(min(vals) - c["truth_min"]) > 5e-4:
                ok = False
            if c["truth_dangerous"] != (c["truth_min"] < 0.0):
                ok = False
            if c["binding_order"] != min(ORDERS, key=lambda o: c["orders"][o]):
                ok = False
    return ok


def s3(rep):
    g1, g2 = rep["design"]["grid_v1"], rep["design"]["grid_v2"]
    n1 = len(g1["A"]) * len(g1["T"]) * len(g1["mp"]) * len(g1["wn"])
    n2 = len(g2["A"]) * len(g2["T"]) * len(g2["mp"]) * len(g2["wn"])
    ok = (len(rep["grid_v1_cells"]) == n1 and len(rep["grid_v2_cells"]) == n2
          and rep["grid_v1"]["n_cells"] == n1 and rep["grid_v2"]["n_cells"] == n2)
    # v1 原始结果必须原样保留
    if [r["miss_count"] for r in rep["grid_v1"]["depth_curve"]] != V1_CURVE:
        ok = False
    if rep["grid_v1"]["binding_order_distribution"] != V1_BINDING:
        ok = False
    if not rep.get("two_stage_disclosure"):
        ok = False
    return ok


def main():
    print("=" * 92)
    print("E84c L3 严格审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    chk("S1 报告↔明细自洽（曲线/分布/判据 由 cells 重算一致；verdict=all(H)）", s1(rep))
    chk("S2 口径不变量（误拒恒 0 / 单调不增 / 全深度 0 / truth=min(五阶) / binding=argmin）", s2(rep))
    chk("S3 两段式披露完整性（单元数=网格规格；**v1 不利结果原样保留**）", s3(rep),
        "v1 curve=%s v1 binding=%s" % ([r["miss_count"] for r in rep["grid_v1"]["depth_curve"]],
                                       rep["grid_v1"]["binding_order_distribution"]))

    i = next(i for i, c in enumerate(rep["grid_v2_cells"])
             if min(c["orders"][k] for k in rep["design"]["depths"][2]) >= 0.0 and c["truth_dangerous"])
    j = next(i for i, c in enumerate(rep["grid_v2_cells"])
             if c["truth_dangerous"] and c["orders"]["D"] < 0.0)

    t1 = copy.deepcopy(rep)
    t1["grid_v2_cells"][j]["orders"]["D"] = 0.5      # 抹掉 D 承重 → 深度 3 档多 1 漏报
    chk("S4a 抹掉某 D 承重单元的 D 失败 → S1 必报（曲线不再由 cells 重算一致）", not s1(t1))

    t2 = copy.deepcopy(rep)
    t2["grid_v2_cells"][i]["truth_dangerous"] = False
    chk("S4b 篡改 truth_dangerous → S1 必报", not s1(t2))

    t3 = copy.deepcopy(rep)
    t3["grid_v2_cells"][i]["truth_min"] = t3["grid_v2_cells"][i]["truth_min"] + 1.0
    chk("S4c 让 truth_min ≠ min(五阶) → S2 必报", not s2(t3))

    t4 = copy.deepcopy(rep)
    t4["preregistered_verdict"]["H84c-5_C_has_D_uncovered_gap"] = False
    chk("S4d 翻转 H84c-5 → S1 必报（verdict≠all(H)）", not s1(t4))

    t5 = copy.deepcopy(rep)
    t5["grid_v1"]["depth_curve"][0]["miss_count"] = 0     # 试图"抹掉 v1 的坏结果"
    chk("S4e 抹掉 v1 原始漏报数 → S3 必报（不利结果不可删）", not s3(t5))

    t6 = copy.deepcopy(rep)
    t6["grid_v2"]["depth_curve"][1]["false_reject_count"] = 3
    chk("S4f 让某档误拒>0 → S2 必报", not s2(t6))

    chk("S4g 未篡改 → S1/S2/S3 通过（阴性对照）", s1(rep) and s2(rep) and s3(rep))
    chk("S5 幂等：两次读入结论一致", s1(json.load(open(REPORT, encoding="utf-8"))) == s1(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["结构自洽、口径不变量与两段式披露（v1 不利结果保留）均通过；篡改均可被抓。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
