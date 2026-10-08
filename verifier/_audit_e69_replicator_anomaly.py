# -*- coding: utf-8 -*-
"""_audit_e69_replicator_anomaly.py —— E69 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e69_bandwidth_report.json，验证**报告内部算术自洽**、
关键派生量可从存储字段重算、篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。V3 盲区计数依赖 E58/trackability（已审），本件只验其逻辑自洽。

产物：_audit_e69_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e69_bandwidth_report.json")
OUT = os.path.join(EVAL, "_audit_e69_replicator_anomaly.json")

_res = {"experiment": "E69 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    # D1：V0 ok 自洽
    v0 = rep["V0_consistency_with_E66"]
    s1 = (v0["kp_match"] is True and abs(v0["E66_budget_gains_wn"] - v0["E69_per_joint_wn"]) < 1e-6)
    ch.append(chk("D1_V0_ok_selfconsistent", v0["ok"] is bool(s1),
                  "E66_wn=%.4f E69_wn=%.4f ok=%s" % (v0["E66_budget_gains_wn"], v0["E69_per_joint_wn"], v0["ok"])))
    # D2：V1 曲线自洽（median 在 min/max 间，all_below_assumed，倍数 ≈ 40/median）
    v1 = rep["V1_curve"]
    s2 = (v1["wn_min"] <= v1["wn_median"] <= v1["wn_max"]
          and v1["all_below_assumed"] is True
          and abs(v1["assumed_over_budget_median_x"] - 40.0 / v1["wn_median"]) < 0.1)
    ch.append(chk("D2_V1_curve_selfconsistent", s2,
                  "min=%.3f med=%.3f max=%.3f x=%.2f" % (v1["wn_min"], v1["wn_median"], v1["wn_max"], v1["assumed_over_budget_median_x"])))
    # D3：V2 公式偏保守（wn 比 >1，kp 比 <1）
    v2 = rep["V2_empirical"]
    s3 = (v2["formula_over_empirical_wn"] > 1.0 and v2["formula_over_empirical_kp"] < 1.0
          and abs(v2["empirical_wn"] - math.sqrt(v2["empirical_kp_max"] / v2["M_eff"])) < 1e-2)
    ch.append(chk("D3_V2_formula_conservative", s3,
                  "wn_ratio=%.3f kp_ratio=%.3f" % (v2["formula_over_empirical_wn"], v2["formula_over_empirical_kp"])))
    # D4：V3 盲区逻辑
    v3 = rep["V3_e58_refill"]
    s4 = (v3["assumed"]["blind_spot_vs_D"] > 0 and v3["budget"]["blind_spot_vs_D"] > 0
          and v3["assumed"]["blind_spot_vs_D"] < v3["budget"]["blind_spot_vs_D"]
          and v3["H58_3_slow_fa_lt_0.10_assumed"] is True and v3["H58_3_slow_fa_lt_0.10_budget"] is False)
    ch.append(chk("D4_V3_blindspot_logic", s4,
                  "assumed=%d budget=%d" % (v3["assumed"]["blind_spot_vs_D"], v3["budget"]["blind_spot_vs_D"])))
    # D5：V3b 建议 k_omega
    v3b = rep["V3b_komega_recalibration"]
    all_fa_ge_01 = all((v["slow_false_alarm"] or 0) >= 0.10 for v in v3b["scan"].values())
    ch.append(chk("D5_V3b_recommendation", (v3b["recommended_k_omega"] is None) == all_fa_ge_01,
                  "rec=%s" % v3b["recommended_k_omega"]))
    # D6：V4 结论引用 V2 数字
    v4 = rep["V4_conclusion"]
    ch.append(chk("D6_V4_cites_V2", ("%.4f" % v2["empirical_kp_max"] in v4["finding_4"])
                  and ("%.3f" % v2["empirical_wn"] in v4["finding_4"]),
                  "V4 finding_4 references kp=%.4f wn=%.3f" % (v2["empirical_kp_max"], v2["empirical_wn"])))
    # D7：verdict 自洽
    s7 = rep["verdict_pass"] is (v0["ok"] and v1["all_below_assumed"]
                                 and v3["H58_1_blind_exists_assumed"] and v3["budget"]["blind_spot_vs_D"] > 0)
    ch.append(chk("D7_verdict_self_consistent", s7, "verdict_pass=%s" % rep["verdict_pass"]))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


import math  # noqa: E402


def main():
    print("=" * 88)
    print("E69 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V0 ok=false → D1 必报", lambda r: r["V0_consistency_with_E66"].__setitem__("ok", False), ["D1"])
    _tamp("R2 篡改 V1 all_below_assumed=false → D2 必报",
          lambda r: r["V1_curve"].__setitem__("all_below_assumed", False), ["D2"])
    _tamp("R3 篡改 V2 empirical_kp=0.5 → D3 必报", lambda r: r["V2_empirical"].__setitem__("empirical_kp_max", 0.5), ["D3"])
    _tamp("R4 篡改 V3 budget blind=0 → D4 必报", lambda r: r["V3_e58_refill"]["budget"].__setitem__("blind_spot_vs_D", 0), ["D4"])
    _tamp("R5 篡改 verdict_pass=false → D7 必报", lambda r: r.__setitem__("verdict_pass", False), ["D7"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["replicator_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n第三方审核：算术 %d/%d ｜ 篡改 %d/%d ｜ pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
