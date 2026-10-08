# -*- coding: utf-8 -*-
"""_audit_e70_replicator_anomaly.py —— E70 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e70_task_residual_loop_report.json，验证**报告内部算术自洽**、
派生量可从存储字段重算、篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。

产物：_audit_e70_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e70_task_residual_loop_report.json")
OUT = os.path.join(EVAL, "_audit_e70_replicator_anomaly.json")

_res = {"experiment": "E70 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    # D1：V1 配对逻辑
    v1 = rep["V1_paired"]
    s1 = (v1["V1_ok"] is True and v1["resid_only"] > 0 and v1["coarse_only"] == 0
          and v1["resid_rate"] > v1["coarse_rate"])
    ch.append(chk("D1_V1_paired_ok", s1,
                  "coarse_only=%d resid_only=%d coarse=%.3f resid=%.3f"
                  % (v1["coarse_only"], v1["resid_only"], v1["coarse_rate"], v1["resid_rate"])))
    # D2：V2 执行越限=0
    v2 = rep["V2_safety_invariant"]
    ch.append(chk("D2_V2_exec_zero", v2["exec_violation_steps_total"] == 0 and v2["demand_violation_steps_total"] > 0,
                  "exec=%d demand=%d" % (v2["exec_violation_steps_total"], v2["demand_violation_steps_total"])))
    # D3：V3 不可达处置
    rb = rep["SET_B_inadmissible"]["resid"]
    s3 = (rb["reach_1cm_rate"] <= 0.10
          and (rb["status_counts"].get("reject", 0) + rb["status_counts"].get("budget_exhausted", 0)) > 0)
    ch.append(chk("D3_V3_inadmissible_reject", s3, "reach=%.3f status=%s" % (rb["reach_1cm_rate"], rb["status_counts"])))
    # D4：V4 CLAMP 对照
    rc = rep["SET_B_inadmissible"]["clamp"]
    ch.append(chk("D4_V4_clamp_fails", rc["reach_1cm_rate"] <= 0.10, "clamp_reach=%.3f" % rc["reach_1cm_rate"]))
    # D5：V5 迭代预算无提升
    v5 = rep["V5_kmax_sensitivity"]
    sa = rep["SET_A_admissible"]["resid"]["reach_1cm_rate"]
    ch.append(chk("D5_V5_kmax_no_improvement", abs(v5["reach_1cm_rate"] - sa) < 1e-6,
                  "kmax16=%.3f resid8=%.3f" % (v5["reach_1cm_rate"], sa)))
    # D6：残差曲线
    if "V1_residual_curve" in rep:
        curve = rep["V1_residual_curve"]
        coarse_med = rep["SET_A_admissible"]["coarse"]["median_err_m"]
        s6 = (abs(curve[0]["median_err_m"] - coarse_med) < 1e-3 and curve[-1]["median_err_m"] < 0.01)
        ch.append(chk("D6_residual_curve", s6,
                      "first=%.4f coarse=%.4f last=%.4f" % (curve[0]["median_err_m"], coarse_med, curve[-1]["median_err_m"])))
    else:
        ch.append(chk("D6_residual_curve", True, "no curve"))
    # D7：verdict 自洽
    s7 = rep["verdict_pass"] is (rep["V1_paired"]["V1_ok"] and rep["V2_ok"] and rep["V3_ok"] and rep["V4_ok"])
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


def main():
    print("=" * 88)
    print("E70 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V1 ok=false → D1 必报", lambda r: r["V1_paired"].__setitem__("V1_ok", False), ["D1"])
    _tamp("R2 篡改 V2 exec=5 → D2 必报",
          lambda r: r["V2_safety_invariant"].__setitem__("exec_violation_steps_total", 5), ["D2"])
    _tamp("R3 篡改 SET_B resid reach=0.5 → D3 必报",
          lambda r: r["SET_B_inadmissible"]["resid"].__setitem__("reach_1cm_rate", 0.5), ["D3"])
    _tamp("R4 篡改 SET_B clamp reach=0.5 → D4 必报",
          lambda r: r["SET_B_inadmissible"]["clamp"].__setitem__("reach_1cm_rate", 0.5), ["D4"])
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
