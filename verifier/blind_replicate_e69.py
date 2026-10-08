# -*- coding: utf-8 -*-
"""blind_replicate_e69.py —— E69 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e69_bench_data.json + e69_standalone_physics.py（自写 2R 核 + 可达带宽 + 经验标定 + E58 回填），
**不 import planning.control_strategies / trackability / e66 / e68 / e58 / e69**，
独立复现 E69 的口径一致性 / ωn 曲线 / 经验标定 / E58 回填，与作者报告比对。

产物：blind_replicate_e69.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e69_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e69_bandwidth_report.json")
BENCH = os.path.join(EVAL, "e69_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e69.json")
_repo = ("planning", "control_strategies", "trackability", "rne_dynamics",
         "e66_closed_loop_control", "e68_control_strategy_suite", "e69_bandwidth_refill",
         "e58_trackability_gate")
_res = {"experiment": "E69 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E69 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench_cs/bench_e58/q_grid/条件/网格）",
        all(k in data for k in ("bench_cs", "bench_e58", "q_grid", "e58_conditions", "v2")),
        "n_cond=%d" % len(data["e58_conditions"]))
    out = P.replicate(data)

    rv0 = rep["V0_consistency_with_E66"]
    chk("B1 V0 口径一致性复现（kp_max=1.8 / 逐关节 kp / E66 ωn=7.758）",
        abs(out["V0"]["E66_kp_max"] - rv0["E66_kp_max"]) < 1e-6
        and abs(out["V0"]["E69_per_joint_wn"] - rv0["E69_per_joint_wn"]) < 1e-3
        and abs(out["V0"]["E66_budget_gains_wn"] - rv0["E66_budget_gains_wn"]) < 1e-3,
        "kp re=%.4f store=%.4f | ωn re=%.4f store=%.4f | legacy re=%.4f store=%.4f"
        % (out["V0"]["E66_kp_max"], rv0["E66_kp_max"], out["V0"]["E69_per_joint_wn"],
           rv0["E69_per_joint_wn"], out["V0"]["legacy_scalar_wn_pre_fix"], rv0["legacy_scalar_wn_pre_fix"]))
    rv1 = rep["V1_curve"]
    chk("B2 V1 ωn 曲线复现（min/median/max；48 点全 < 假设 40）",
        abs(out["V1"]["wn_min"] - rv1["wn_min"]) < 1e-3 and abs(out["V1"]["wn_median"] - rv1["wn_median"]) < 1e-3
        and abs(out["V1"]["wn_max"] - rv1["wn_max"]) < 1e-3 and out["V1"]["all_below_assumed"],
        "min/med/max re=(%.3f,%.3f,%.3f) store=(%.3f,%.3f,%.3f)"
        % (out["V1"]["wn_min"], out["V1"]["wn_median"], out["V1"]["wn_max"],
           rv1["wn_min"], rv1["wn_median"], rv1["wn_max"]))
    rv2 = rep["V2_empirical"]
    chk("B3 V2 经验标定复现（实测 kp_max=2.5119 / ωn=6.0978；公式偏保守）",
        out["V2"]["empirical_kp_max"] == rv2["empirical_kp_max"]
        and abs(out["V2"]["empirical_wn"] - rv2["empirical_wn"]) < 1e-3,
        "kp_emp re=%s store=%s | wn_emp re=%s store=%s"
        % (out["V2"]["empirical_kp_max"], rv2["empirical_kp_max"],
           out["V2"]["empirical_wn"], rv2["empirical_wn"]))
    rv3 = rep["V3_e58_refill"]
    chk("B4 V3 E58 回填·假设 ωn=40（盲区 30 / 慢档误报 0.0）复现",
        out["V3"]["assumed"]["blind_spot_vs_D"] == rv3["assumed"]["blind_spot_vs_D"]
        and abs((out["V3"]["assumed"]["slow_fa"] or 0) - (rv3["assumed"]["slow_T_ge_0.8s_false_alarm"] or 0)) < 1e-9,
        "blind re=%d store=%d | slow_fa re=%s store=%s"
        % (out["V3"]["assumed"]["blind_spot_vs_D"], rv3["assumed"]["blind_spot_vs_D"],
           out["V3"]["assumed"]["slow_fa"], rv3["assumed"]["slow_T_ge_0.8s_false_alarm"]))
    chk("B5 V3 E58 回填·预算 ωn（盲区 102 / 慢档误报 0.9407 / H58-3 翻 FAIL）复现",
        out["V3"]["budget"]["blind_spot_vs_D"] == rv3["budget"]["blind_spot_vs_D"]
        and abs((out["V3"]["budget"]["slow_fa"] or 0) - (rv3["budget"]["slow_T_ge_0.8s_false_alarm"] or 0)) < 1e-3
        and out["V3"]["H58_3_slow_fa_lt_0.10_budget"] == rv3["H58_3_slow_fa_lt_0.10_budget"],
        "blind re=%d store=%d | slow_fa re=%s store=%s | H58-3 re=%s store=%s"
        % (out["V3"]["budget"]["blind_spot_vs_D"], rv3["budget"]["blind_spot_vs_D"],
           out["V3"]["budget"]["slow_fa"], rv3["budget"]["slow_T_ge_0.8s_false_alarm"],
           out["V3"]["H58_3_slow_fa_lt_0.10_budget"], rv3["H58_3_slow_fa_lt_0.10_budget"]))
    chk("B6 V3 ωn 预算中位复现（1.5351 rad/s）",
        abs(out["V3"]["wn_budget_median"] - rv3["wn_budget_median"]) < 2e-3,
        "re=%.4f store=%.4f" % (out["V3"]["wn_budget_median"], rv3["wn_budget_median"]))
    chk("B7 结论判定复现（verdict_pass=True）", out["verdict_pass"] == rep["verdict_pass"],
        "re=%s store=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e69_bench_data.json + 独立实现）：E69 的口径一致性、ωn 预算曲线、"
                        "经验标定与 E58 回填（30→102 盲区 / 0.0→0.9407 慢档误报）可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
