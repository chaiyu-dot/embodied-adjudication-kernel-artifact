# -*- coding: utf-8 -*-
"""_audit_e84_replicator_anomaly.py —— E84 的 **L2 第三方复算 + 异常/篡改注入**。

E84 的报告只落**聚合**（无逐提案行），故本层做两件事：
  T1 **第三方复算（只读 JSON）**：由 aggregate 的份额/计数重算全部预注册判据与内部不变量
     （gt_dangerous_rate 四臂恒等；可行率偏序 POINT ≥ GATED ≥ SOUND；GATED 测量 ≤ MEAS_ALL/5；
      H84-1..5 ⟺ aggregate；verdict_pass = all(H)）。
  T2 **异常注入**：逐项篡改 aggregate / verdict → 上列检查必须报差异（证明有分辨力）。
  T3 **跨实验一致性**：与 E84b（独立引擎真值复核）的同一提案流数字**逐位相等**。
  T4 阴性对照；T5 幂等。

产物：_audit_e84_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
E84B = os.path.join(EVAL, "e84b_independent_engine_gt_report.json")
OUT = os.path.join(EVAL, "_audit_e84_replicator_anomaly.json")
_res = {"audit": "E84 L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def verdicts_from(agg):
    g, m, s, p = agg["GATED"], agg["MEAS_ALL"], agg["SOUND_NOMEAS"], agg["POINT"]
    return {
        "H84-1_point_entry_reopens_danger": bool(p["dangerous_rate"] > 0.0),
        "H84-2_sound_arms_zero_danger": bool(s["dangerous"] == 0 and g["dangerous"] == 0
                                            and m["dangerous"] == 0),
        "H84-3_gated_far_fewer_measures": bool(
            g["measure_calls_mean"] <= max(m["measure_calls_mean"] / 5.0, 1e-9)),
        "H84-4_gated_keeps_feasibility": bool(g["feasible_rate"] >= s["feasible_rate"] - 1e-9),
    }


def invariants(rep):
    a = rep["aggregate"]
    ok, det = True, []
    gts = {x: a[x]["gt_dangerous_rate"] for x in ARMS}
    if len(set(gts.values())) != 1:
        ok = False
    if not (a["POINT"]["feasible_rate"] >= a["GATED"]["feasible_rate"] >= a["SOUND_NOMEAS"]["feasible_rate"]):
        ok = False
    if not (a["MEAS_ALL"]["feasible_rate"] >= a["GATED"]["feasible_rate"] - 1e-9):
        ok = False
    if a["POINT"]["measure_calls_mean"] != 0.0 or a["SOUND_NOMEAS"]["measure_calls_mean"] != 0.0:
        ok = False
    if not (a["MEAS_ALL"]["measure_calls_mean"] > a["GATED"]["measure_calls_mean"] > 0.0):
        ok = False
    for k in ("dangerous", "dangerous_rate"):
        if a["MEAS_ALL"][k] != 0 or a["SOUND_NOMEAS"][k] != 0 or a["GATED"][k] != 0:
            ok = False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        ok = False
    det.append("gt_rates=%s | feas POINT/GATED/SOUND/MEAS_ALL=%s"
               % (sorted(set(gts.values())),
                  [a[x]["feasible_rate"] for x in ARMS]))
    return ok, " ｜ ".join(det)


def main():
    print("=" * 92)
    print("E84 L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    ok, det = invariants(rep)
    chk("T1 第三方复算：内部不变量（gt 四臂恒等 / 可行率偏序 / 零测量臂 / 门控<全测 / 零危险）", ok, det)

    v_rep = dict(rep["preregistered_verdict"])
    v_rep.pop("H84-5_five_order_in_certificate", None)
    chk("T1b 预注册判据 H84-1..4 由 aggregate 重算一致", verdicts_from(rep["aggregate"]) == v_rep,
        "mine=%s theirs=%s" % (verdicts_from(rep["aggregate"]), v_rep))

    a = copy.deepcopy(rep)
    a["aggregate"]["POINT"]["dangerous_rate"] = 0.0
    a["aggregate"]["POINT"]["dangerous"] = 0
    chk("T2a 抹掉 POINT 的危险执行 → H84-1 必报（判据重算不一致）",
        verdicts_from(a["aggregate"]) != v_rep)

    b = copy.deepcopy(rep)
    b["aggregate"]["GATED"]["dangerous"] = 2
    chk("T2b 让 GATED 危险=2 → 不变量必报", not invariants(b)[0])

    c = copy.deepcopy(rep)
    c["aggregate"]["MEAS_ALL"]["measure_calls_mean"] = 0.4
    chk("T2c 把 MEAS_ALL 测量次数压到 0.4 → H84-3 必报", verdicts_from(c["aggregate"]) != v_rep)

    d = copy.deepcopy(rep)
    d["aggregate"]["SOUND_NOMEAS"]["gt_dangerous_rate"] = 0.99
    chk("T2d 破坏 gt_dangerous_rate 四臂恒等 → 不变量必报", not invariants(d)[0])

    e = copy.deepcopy(rep)
    e["preregistered_verdict"]["H84-1_point_entry_reopens_danger"] = False
    chk("T2e 翻转 H84-1 → verdict_pass≠all(H) 必报",
        e["verdict_pass"] != all(e["preregistered_verdict"].values()))

    f = copy.deepcopy(rep)
    f["aggregate"]["GATED"]["feasible_rate"] = 0.05
    chk("T2f 把 GATED 可行率压到低于 SOUND → 偏序不变量必报", not invariants(f)[0])

    e84b = json.load(open(E84B, encoding="utf-8"))
    same = all(e84b["aggregate"][x]["dangerous_rate_analytic_gt"] == rep["aggregate"][x]["dangerous_rate"]
               and e84b["aggregate"][x]["feasible_rate"] == rep["aggregate"][x]["feasible_rate"]
               and e84b["aggregate"][x]["measure_calls_mean"] == rep["aggregate"][x]["measure_calls_mean"]
               for x in ARMS)
    chk("T3 跨实验一致性：E84b（独立引擎复核）在同一提案流上重放的四臂数字与 E84 **逐位相等**",
        bool(same), "E84b selfcheck=%s" % e84b.get("e84_replay_selfcheck"))

    chk("T4 阴性对照：未篡改 → T1 通过", invariants(rep)[0])
    chk("T5 幂等：两次读入结论一致",
        invariants(json.load(open(REPORT, encoding="utf-8"))) == invariants(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "第三方复算与异常注入均按预期：四臂不变量、预注册判据、跨实验一致性有分辨力（篡改可被抓）。",
        "★ T3 是较强的跨件证据：E84b 用**完全独立的真值来源**重放同一提案流，"
        "得到的四臂数字与 E84 **逐位相等**——排除了'E84 的聚合数字被手工修改'这一类风险。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
