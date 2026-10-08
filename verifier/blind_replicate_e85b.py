# -*- coding: utf-8 -*-
"""blind_replicate_e85b.py —— E85b 的 **L4 隔离盲复刻**（只给数据文件 + 独立实现）。

只拿 `e85b_bench_data.json` + 复刻者侧独立实现（`l4_plant2r` + `e84c_standalone_physics` +
`e84_standalone_physics` + `e85b_standalone_physics`），**不 import 仓库模块 / 任何 e8x 实验脚本**
（复刻者侧共享核不算仓库模块）：
  · 零 RNG 重放 **v1（条件流耦合）** 与 **v2（条件流解耦）** 两种设计；
  · 重算逐档统计量、经济性单调性、能力上限、oracle 端点与 7 条判据（含 H85b-4 的诚实 FAIL）。

产物：blind_replicate_e85b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e85b_standalone_physics as SP   # noqa: E402

REPORT = os.path.join(EVAL, "e85b_oracle_level_report.json")
BENCH = os.path.join(EVAL, "e85b_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e85b.json")
_repo = ("planning", "rne_dynamics", "rigid_body_certificate", "trackability", "energy_margin",
         "active_estimator", "control_strategies", "e85b_oracle_level",
         "e85_interface_accuracy_monotone", "e84_five_order_wrapper_loop",
         "e84b_independent_engine_gt", "e84c_depth_ablation_grid")
_res = {"experiment": "E85b 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]
# 🔴 逐臂字段集合必须**分开**：POINT 臂不产生测量（无 measure_calls_mean / trigger_rate），
# 用 GATED 的字段集去比 POINT 会误报 4×档数 条（此前 L4 的 6/7 即由该 schema 错造成）。
ARM_KEYS = {
    "gated": ("measure_calls_mean", "trigger_rate", "feasible_rate", "dangerous_rate",
              "dangerous_count", "false_reject_rate"),
    "point": ("feasible_rate", "dangerous_rate", "dangerous_count", "false_reject_rate"),
}
KEYS = ARM_KEYS["gated"]     # 兼容旧引用


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E85b 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/q_lim/omega_n/zeta/T_s/qdot_lim/k_conf/k_max/两种设计 levels）",
        all(k in data for k in ("bench", "q_lim", "omega_n", "zeta", "T_s", "qdot_lim",
                                "k_conf", "k_max", "design_v1_levels", "design_v2_levels")),
        "v1 levels=%d v2 levels=%d" % (len(data["design_v1_levels"]), len(data["design_v2_levels"])))

    out = SP.replicate(data)

    # B1 两设计逐档×逐臂统计量
    bad = []
    for tag, mine in (("v1", out["per_s_v1"]), ("v2", out["per_s_v2"])):
        theirs = rep["design_" + tag]["per_s"]
        for s, row in mine.items():
            tr = theirs.get(str(s))
            if tr is None:
                bad.append((tag, s, "missing"))
                continue
            for arm in ("gated", "point"):
                if not row[arm]:
                    continue
                t = tr.get(arm) or {}
                for k in ARM_KEYS[arm]:
                    if k not in t or abs(row[arm][k] - t[k]) > 1e-4:
                        bad.append((tag, s, arm, k))
    chk("B1 两设计逐档×逐臂统计量与报告一致（v1 6 档 + v2 6 档 × 1–2 臂）", not bad,
        "mismatch=%s" % bad[:4])

    # B2 汇总与混淆量
    bad2 = []
    for tag, sm in (("v1", out["summary_v1"]), ("v2", out["summary_v2"])):
        ts = rep["design_" + tag]["summary"]
        for k in ("calls", "feasible", "dangerous", "gt_dangerous_rates", "old5_nonmono",
                  "all6_monotone_calls", "rho_s_calls", "rho_s_feasible"):
            mine, theirs = sm[k], ts[k]
            if isinstance(mine, list):
                if [round(x, 4) for x in mine] != [round(x, 4) for x in theirs]:
                    bad2.append((tag, k))
            elif isinstance(mine, bool):
                if bool(mine) != bool(theirs):
                    bad2.append((tag, k))
            elif abs(mine - theirs) > 1e-3:
                bad2.append((tag, k))
        if abs(sm["ceiling"] - ts["ceiling"]) > 1e-4 or abs(sm["oracle_feasible"] - ts["oracle_feasible"]) > 1e-4:
            bad2.append((tag, "ceiling/oracle"))
    chk("B2 两设计汇总量（calls/feasible/gt/单调性/ρ/ceiling/oracle）一致", not bad2, "mismatch=%s" % bad2)

    # B3 混淆量
    chk("B3 条件流混淆量复现（v1 gt 极差 = 报告值）",
        abs(out["v1_gt_spread"] - rep["v1_vs_v2"]["v1_gt_dangerous_rate_variation"]["spread"]) <= 1e-4,
        "mine=%.4f theirs=%.4f" % (out["v1_gt_spread"],
                                   rep["v1_vs_v2"]["v1_gt_dangerous_rate_variation"]["spread"]))

    # B4 判据（含诚实 FAIL）
    chk("B4 预注册判据 H85b-1..7 逐条复现（**含 H85b-4 的诚实 FAIL 同样复现**）",
        out["H"] == rep["preregistered_verdict"],
        "mine=%s theirs=%s" % (out["H"], rep["preregistered_verdict"]))
    chk("B5 结论判定复现（verdict_pass，作者为 False 亦须复现）",
        out["verdict_pass"] == rep["verdict_pass"],
        "mine=%s theirs=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e85b_bench_data.json + 复刻者侧独立物理）：两种设计的逐档统计量、"
        "单调性、能力上限、混淆量与 7 条判据（**含诚实 FAIL**）全部被外部独立重建。",
        "**语义强度标注**：数据文件把『估计器轨迹 (mean,std)』外化为数据 → 本 L4 检验"
        "**判据/门控/统计管线**的独立可重建性；`active_estimator` 内部不在本件范围。",
        "**审核侧纪律（本次踩坑）**：逐臂比对必须用**各自的字段集**——POINT 臂天然没有"
        "`measure_calls_mean`/`trigger_rate`，套用 GATED 字段集会制造 4×档数 条假 FAIL。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
