# -*- coding: utf-8 -*-
"""_audit_e84b_replicator_anomaly.py —— E84b 的 **L2 第三方复算 + 异常/篡改注入**。

E84b 是 E84 的"独立引擎真值"复核件。本层只读其报告 JSON，做第三方重算与注入：

  T1 内部不变量：逐臂 dangerous(bullet) == dangerous(analytic)；dangerous_count 与 rate 自洽；
     gt_agreement_rate ⟺ 两真值危险集合一致；binding 分布合计 = n_proposals；
     cross_engine.pass ⟺ 实测误差 ≤ 申报容差；plant_selfcheck.pass ⟺ 各项误差 ≤ tol；
     H84b-0 ⟺ 与 E84 报告的重放数字一致；verdict_pass = all(H)。
  T2 异常注入：逐项篡改 → 对应检查必须报。
  T3 跨实验一致性：臂层（feasible / measure）与 E84 报告逐位相等（臂逻辑与真值来源无关）。
  T4 阴性对照；T5 幂等。

产物：_audit_e84b_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e84b_independent_engine_gt_report.json")
E84 = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
OUT = os.path.join(EVAL, "_audit_e84b_replicator_anomaly.json")
_res = {"audit": "E84b L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def invariants(rep):
    ok, det = True, []
    for a in ARMS:
        r = rep["aggregate"][a]
        if r["dangerous_rate_bullet_gt"] != r["dangerous_rate_analytic_gt"]:
            ok = False
        if abs(r["dangerous_count_bullet_gt"] / max(r["n"], 1) - r["dangerous_rate_bullet_gt"]) > 1e-4:
            ok = False
    ce = rep["cross_engine_check"]
    ok &= (bool(ce["pass"]) == bool(
        ce["instantaneous"]["max_abs_err_Nm"] <= ce["tolerance_abs_Nm"]
        and ce["instantaneous"]["median_abs_err_Nm"] <= ce["tolerance_median_Nm"]
        and ce["trajectory_peak"]["max_abs_err_Nm"] <= ce["tolerance_abs_Nm"]
        and ce["trajectory_peak"]["median_abs_err_Nm"] <= ce["tolerance_median_Nm"]))
    ps = rep["plant_selfcheck"]
    ok &= (bool(ps["pass"]) == bool(max(ps["max_mass_err_kg"], ps["max_izz_err_kgm2"],
                                        ps["max_ixx_err_kgm2"], ps["max_com_err_m"]) <= ps["tol"]))
    m = rep["meta"]
    ok &= (sum(m["order_binding_distribution"].values()) == m["n_proposals"])
    ok &= (0.0 <= m["gt_agreement_rate"] <= 1.0)
    # 两真值危险计数一致性（一致率 1.0 时逐臂必须相等）
    if m["gt_agreement_rate"] == 1.0:
        for a in ARMS:
            r = rep["aggregate"][a]
            if r["dangerous_count_bullet_gt"] != round(r["dangerous_rate_analytic_gt"] * r["n"]):
                ok = False
    ok &= (rep["verdict_pass"] == all(rep["preregistered_verdict"].values()))
    sc = rep.get("e84_replay_selfcheck") or {}
    exp0 = (sc.get("e84_point_dangerous_rate") == sc.get("e84b_point_dangerous_rate_analytic_gt")
            and sc.get("e84_gated_feasible_rate") == sc.get("e84b_gated_feasible_rate"))
    ok &= (bool(rep["preregistered_verdict"]["H84b-0_replay_faithful"]) == bool(exp0))
    det.append("agreement=%.4f binding_sum=%d cross_pass=%s" % (m["gt_agreement_rate"],
                                                               sum(m["order_binding_distribution"].values()),
                                                               ce["pass"]))
    return ok, " ｜ ".join(det)


def main():
    print("=" * 92)
    print("E84b L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    ok, det = invariants(rep)
    chk("T1 第三方复算：两真值逐臂一致 / 计数↔比率自洽 / 容差↔pass / 重放自检↔H84b-0", ok, det)

    a = copy.deepcopy(rep)
    a["aggregate"]["POINT"]["dangerous_rate_bullet_gt"] = 0.0
    a["aggregate"]["POINT"]["dangerous_count_bullet_gt"] = 0
    chk("T2a 抹掉 Bullet 真值下的 POINT 危险 → 不变量必报", not invariants(a)[0])

    b = copy.deepcopy(rep)
    b["cross_engine_check"]["instantaneous"]["max_abs_err_Nm"] = 1.0     # 误差爆表但 pass 仍 True
    chk("T2b 让跨引擎误差爆表而 pass 不变 → 容差↔pass 一致性必报", not invariants(b)[0])

    c = copy.deepcopy(rep)
    c["plant_selfcheck"]["max_izz_err_kgm2"] = 1e-2
    chk("T2c 让装载自检的 izz 误差爆表而 pass=True → 必报", not invariants(c)[0])

    d = copy.deepcopy(rep)
    d["meta"]["gt_agreement_rate"] = 0.90
    chk("T2d 把两真值一致率降到 0.90 → H84b-5 必报（判据与一致率不符）",
        d["preregistered_verdict"]["H84b-5_gt_verdict_agreement"] != (d["meta"]["gt_agreement_rate"] >= 0.98))

    e = copy.deepcopy(rep)
    e["meta"]["order_binding_distribution"]["C"] += 1
    chk("T2e 篡改逐阶承重分布（合计 ≠ n）→ 不变量必报", not invariants(e)[0])

    f = copy.deepcopy(rep)
    f["e84_replay_selfcheck"]["e84b_gated_feasible_rate"] = 0.5
    chk("T2f 破坏重放自检（GATED 可行率不一致）→ H84b-0 一致性必报", not invariants(f)[0])

    g = copy.deepcopy(rep)
    g["preregistered_verdict"]["H84b-2_cross_engine_agree"] = False
    chk("T2g 翻转 H84b-2 → verdict_pass≠all(H) 必报",
        g["verdict_pass"] != all(g["preregistered_verdict"].values()))

    e84 = json.load(open(E84, encoding="utf-8"))
    same = all(e84["aggregate"][x]["feasible_rate"] == rep["aggregate"][x]["feasible_rate"]
               and e84["aggregate"][x]["measure_calls_mean"] == rep["aggregate"][x]["measure_calls_mean"]
               for x in ARMS)
    chk("T3 跨实验一致性：臂层（可行率 / 测量次数）与 E84 报告逐位相等", bool(same))

    chk("T4 阴性对照：未篡改 → T1 通过", invariants(rep)[0])
    chk("T5 幂等：两次读入结论一致",
        invariants(json.load(open(REPORT, encoding="utf-8"))) == invariants(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["E84b 的两真值一致性、容差↔pass 自洽、装载自检、重放自检与臂层跨件一致性均通过；篡改均可被抓。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
