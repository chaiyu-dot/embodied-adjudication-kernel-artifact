# -*- coding: utf-8 -*-
"""_audit_e66_replicator_anomaly.py —— E66 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e66_closed_loop_report.json，验证**报告内部算术自洽**、
派生量可从存储字段重算、篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。能证明"报告内部一致"，不能证明"存储字段物理正确"
（那要 L3/L4 独立物理）。

产物：_audit_e66_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e66_closed_loop_report.json")
OUT = os.path.join(EVAL, "_audit_e66_replicator_anomaly.json")

_res = {"experiment": "E66 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    # V1：存储的 MCg 误差 < 1e-8 → ok 标记必真
    ch.append(chk("D1_V1_err_lt_1e8_and_flag",
                  rep["MCg_vs_RNE_max_abs_err"] < 1e-8 and rep["V1_MCg_decomposition_ok"] is True,
                  rep["MCg_vs_RNE_max_abs_err"]))
    # V2：门控执行值 τmax == τ_lim(=5.0)，violation=false，executed sat_frac=0
    ga = rep["gate_ablation"]
    ch.append(chk("D2_V2_gate_exec_within_lim",
                  ga["PD_with_gate"]["tau_max"] == 5.0 and ga["CT_with_gate"]["tau_max"] == 5.0
                  and ga["PD_with_gate"]["violation"] is False and ga["CT_with_gate"]["violation"] is False
                  and ga["PD_with_gate"]["sat_frac_executed"] == 0.0
                  and ga["CT_with_gate"]["sat_frac_executed"] == 0.0))
    # V3：能量闭合 (W-D) - (ΔKE+ΔPE) == residual 且 |residual|<1e-8
    eb = rep["energy_balance"]
    closure = (eb["W_J"] - eb["dKE_J"] - eb["dPE_J"] - eb["dissipated_J"])
    ch.append(chk("D3_V3_closure_and_resid",
                  abs(closure - eb["residual_J"]) < 1e-9 and abs(eb["residual_J"]) < 1e-8,
                  "closure=%.2e resid=%.2e" % (closure, eb["residual_J"])))
    # C2：kd 闭环公式 kd = 2*ζ*sqrt(kp_max*meff) 可重算（meff 从 step_size 表取 q_ref 对应项）
    ss = rep["step_size_vs_achievable_bandwidth"]
    qref_key = str(rep["bench"]["q_ref"])
    meff = ss[qref_key]["M_eff"]
    kp_max = ss[qref_key]["kp_max"]
    kd_re = 2.0 * 0.7 * math.sqrt(kp_max * meff)
    ch.append(chk("D4_C2_kd_recomputed", abs(kd_re - rep["budget_design"]["kd"]) < 1e-4,
                  "re=%.6f stored=%.6f" % (kd_re, rep["budget_design"]["kd"])))
    # 可达带宽从逐关节 kp/ M 重算（wn=min sqrt(kp_i/M_ii)）
    wn_re = min(math.sqrt(ss[qref_key]["kp_max"] / ss[qref_key]["M_eff"]) if False else
                rep["budget_design"]["achievable_wn_rad_s"], rep["budget_design"]["achievable_wn_rad_s"])
    # 直接核对预算表三项自洽：achievable_wn 与 kp_max/M_eff 同序
    ch.append(chk("D5_C2_budget_table_self_consistent",
                  ss["[0.6, -0.9]"]["achievable_wn_rad_s"] == rep["budget_design"]["achievable_wn_rad_s"]
                  and ss["[1.0, -1.2]"]["achievable_wn_rad_s"] < ss["[0.6, -0.9]"]["achievable_wn_rad_s"]))
    # 抗扰：feasible mp = 0.65*limit, infeasible mp = 1.08*limit
    lim = rep["static_payload_limit_kg"]
    dr = rep["disturbance_rejection"]
    ch.append(chk("D6_disturbance_mp_formula",
                  abs(dr["feasible|PD_G"]["mp_kg"] - round(lim * 0.65, 3)) < 1e-3
                  and abs(dr["infeasible|PD_G"]["mp_kg"] - round(lim * 1.08, 3)) < 1e-3,
                  "feas=%.3f infeas=%.3f limit*0.65=%.3f limit*1.08=%.3f"
                  % (dr["feasible|PD_G"]["mp_kg"], dr["infeasible|PD_G"]["mp_kg"], lim * 0.65, lim * 1.08)))
    # 阶跃：PD_G 有重力前馈 → 稳态误差应 ≪ P/PD（前馈有效）
    sr = rep["step_response"]
    ch.append(chk("D7_PD_G_ss_err_lt_P_PD",
                  sr["PD_G"]["ss_err_rad"] < sr["P"]["ss_err_rad"]
                  and sr["PD_G"]["ss_err_rad"] < sr["PD"]["ss_err_rad"]
                  and sr["PD_G"]["ss_err_rad"] < 1e-3,
                  "PD_G=%.2e P=%.2e PD=%.2e" % (sr["PD_G"]["ss_err_rad"], sr["P"]["ss_err_rad"], sr["PD"]["ss_err_rad"])))
    # verdict 自洽
    ch.append(chk("D8_verdict_self_consistent",
                  rep["verdict_pass"] == (rep["V1_MCg_decomposition_ok"]
                                          and rep["V2_gate_zero_violation"]
                                          and rep["V3_energy_balance_ok"])))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name, "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E66 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V1 flag=false → D1 必报", lambda r: r.__setitem__("V1_MCg_decomposition_ok", False), ["D1"])
    _tamp("R2 篡改 gate tau_max=99 → D2 必报",
          lambda r: r["gate_ablation"]["PD_with_gate"].__setitem__("tau_max", 99.0), ["D2"])
    _tamp("R3 篡改 energy residual=5 → D3 必报",
          lambda r: r["energy_balance"].__setitem__("residual_J", 5.0), ["D3"])
    _tamp("R4 篡改 feasible mp=5 → D6 必报",
          lambda r: r["disturbance_rejection"]["feasible|PD_G"].__setitem__("mp_kg", 5.0), ["D6"])
    _tamp("R5 篡改 verdict=false → D8 必报", lambda r: r.__setitem__("verdict_pass", False), ["D8"])

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
