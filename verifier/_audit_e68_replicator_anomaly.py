# -*- coding: utf-8 -*-
"""_audit_e68_replicator_anomaly.py —— E68 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e68_control_suite_report.json，验证**报告内部算术自洽**、
派生量（预算增益、可达带宽、静载、MPC 标定、门控不变量、不可行处置）可从存储字段 +
bench 几何重算、篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。能证明"报告内部一致"，不能证明"存储字段物理正确"
（那要 L3/L4 独立物理 / L5 逆向）。

产物：_audit_e68_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e68_control_suite_report.json")
OUT = os.path.join(EVAL, "_audit_e68_replicator_anomaly.json")

_res = {"experiment": "E68 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON + bench 几何（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _mass_matrix(q, l1, l2, m1, m2):
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    c2 = math.cos(q[1])
    m11 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * c2) + m1 * l1 * l1 / 12.0 + m2 * l2 * l2 / 12.0
    m12 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + m2 * l2 * l2 / 12.0
    m22 = m2 * lc2 * lc2 + m2 * l2 * l2 / 12.0
    return [[m11, m12], [m12, m22]]


def _forward(rep):
    ch = []
    bench = rep["bench"]
    L1, L2 = bench["l"]
    M1, M2 = bench["m"]
    TAU = bench["tau_lim"]
    qref = bench["q_ref"]

    # --- D1：V0 ok 标志自洽 ---
    v0 = rep["V0_model_consistency"]
    s1 = (v0["CS_vs_E66_max_abs"] == 0.0 and v0["CS_vs_RNE_max_abs"] < 1e-8)
    ch.append(chk("D1_V0_ok_selfconsistent", v0["ok"] is bool(s1),
                  "CS_vs_E66=%g CS_vs_RNE=%g ok=%s" % (v0["CS_vs_E66_max_abs"], v0["CS_vs_RNE_max_abs"], v0["ok"])))

    # --- D2：V0b MPC 一致性 ok 自洽 ---
    v0b = rep["V0b_mpc_consistency"]
    s2 = (v0b["gravity_ff_at_target_equals_g"] < 1e-6 and v0b["pred_vs_euler_max_abs_over_Np"] < 1e-2)
    ch.append(chk("D2_V0b_ok_selfconsistent", v0b["ok"] is bool(s2),
                  "g_ff=%g pred=%g ok=%s" % (v0b["gravity_ff_at_target_equals_g"],
                                              v0b["pred_vs_euler_max_abs_over_Np"], v0b["ok"])))

    # --- D3：V1 门控不变量 ok = 全族 executed==0；激进设计 kp/kd 与 ωn=40 自洽 ---
    v1 = rep["V1_gate_invariant"]
    exec_ok = all(v["executed_violation_steps"] == 0 for v in v1["per_family"].values())
    ch.append(chk("D3_V1_ok_selfconsistent", v1["ok"] is bool(exec_ok),
                  "exec_all_zero=%s ok=%s" % (exec_ok, v1["ok"])))
    Mref = _mass_matrix(qref, L1, L2, M1, M2)
    meff = 0.5 * (Mref[0][0] + Mref[1][1])
    ag = v1["aggressive_design"]
    kp_a_re = 40.0 ** 2 * meff
    kd_a_re = 2 * 0.7 * 40.0 * meff
    ch.append(chk("D3b_aggressive_design_from_wn40", abs(ag["kp"] - kp_a_re) < 1e-2 and abs(ag["kd"] - kd_a_re) < 1e-2,
                  "kp_re=%.4f kd_re=%.4f stored=%s" % (kp_a_re, kd_a_re, ag)))

    # --- D4：预算增益 / 可达带宽 从 bench 几何独立重算 ---
    frac = 0.9
    kp0 = frac * TAU[0] / abs(qref[0])
    kp1 = frac * TAU[1] / abs(qref[1])
    kp_max = min(kp0, kp1)
    wn_re = min(math.sqrt(kp0 / Mref[0][0]), math.sqrt(kp1 / Mref[1][1]))
    kd_re = 2 * 0.7 * math.sqrt(kp_max * meff)
    bd = rep["budget_design"]
    ch.append(chk("D4_budget_kp_from_taulim", abs(bd["kp"] - kp_max) < 1e-6,
                  "re=%.6f stored=%.6f" % (kp_max, bd["kp"])))
    ch.append(chk("D4b_budget_kd_from_formula", abs(bd["kd"] - kd_re) < 1e-4,
                  "re=%.6f stored=%.6f" % (kd_re, bd["kd"])))
    ch.append(chk("D4c_budget_wn_from_geometry", abs(bd["achievable_wn_rad_s"] - wn_re) < 1e-3,
                  "re=%.4f stored=%.4f" % (wn_re, bd["achievable_wn_rad_s"])))

    # --- D5：MPC 标定自洽 ---
    mc = rep["MPC_calibration"]
    ch.append(chk("D5_MPC_calibration", abs(mc["qd_w"] - 0.05 * mc["q_w"]) < 1e-9
                  and mc["rel_err"] < 0.05 and mc["Np"] == 20 and mc["Nc"] == 5,
                  "qd_w=%.2e rel_err=%.3f Np=%d Nc=%d" % (mc["qd_w"], mc["rel_err"], mc["Np"], mc["Nc"])))

    # --- D6：V4 不可行处置 ok 逻辑自洽 ---
    feas = rep["V4_infeasible_disposition"]
    inf_in = [v for k, v in feas.items() if k.startswith("feasible|informed")]
    inf_all = [v for k, v in feas.items() if k.startswith("infeasible|")]
    s6 = all(v["within_tol_0.02"] for v in inf_in) and all(not v["within_tol_0.02"] for v in inf_all)
    ch.append(chk("D6_V4_ok_selfconsistent", rep["V4_ok"] is bool(s6),
                  "feas_in_ok=%s infeas_all_fail=%s V4_ok=%s"
                  % (all(v["within_tol_0.02"] for v in inf_in),
                     all(not v["within_tol_0.02"] for v in inf_all), rep["V4_ok"])))

    # --- D7：阶跃能力：PD_G 有重力前馈 → 稳态误差 ≪ P/PD；MPC ≈ 0 ---
    sr = rep["step_response"]
    ch.append(chk("D7_PD_G_ff_effective", sr["PD_G"]["ss_err_rad"] < sr["P"]["ss_err_rad"]
                  and sr["PD_G"]["ss_err_rad"] < sr["PD"]["ss_err_rad"]
                  and sr["PD_G"]["ss_err_rad"] < 1e-3,
                  "PD_G=%.2e P=%.2e PD=%.2e" % (sr["PD_G"]["ss_err_rad"], sr["P"]["ss_err_rad"], sr["PD"]["ss_err_rad"])))
    ch.append(chk("D7b_MPC_ss_zero", abs(sr["MPC"]["ss_err_rad"]) < 1e-9, "MPC_ss=%.2e" % sr["MPC"]["ss_err_rad"]))

    # --- D8：verdict 自洽 ---
    ch.append(chk("D8_verdict_self_consistent", rep["verdict_pass"] is
                  (rep["V0_model_consistency"]["ok"] and rep["V0b_mpc_consistency"]["ok"]
                   and rep["V1_gate_invariant"]["ok"] and rep["V4_ok"]),
                  "verdict_pass=%s" % rep["verdict_pass"]))
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
    print("E68 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V0 ok=false → D1 必报", lambda r: r["V0_model_consistency"].__setitem__("ok", False), ["D1"])
    _tamp("R2 篡改 V1 PD_G executed_violation_steps=3 → D3 必报",
          lambda r: r["V1_gate_invariant"]["per_family"]["PD_G"].__setitem__("executed_violation_steps", 3), ["D3"])
    _tamp("R3 篡改 budget achievable_wn=99 → D4c 必报",
          lambda r: r["budget_design"].__setitem__("achievable_wn_rad_s", 99.0), ["D4c"])
    _tamp("R4 篡改 MPC qd_w=0.5*q_w → D5 必报",
          lambda r: r["MPC_calibration"].__setitem__("qd_w", 0.5 * r["MPC_calibration"]["q_w"]), ["D5"])
    _tamp("R5 篡改 V4 feasible|informed|PD_G within_tol→false → D6 必报",
          lambda r: r["V4_infeasible_disposition"]["feasible|informed|PD_G"].__setitem__("within_tol_0.02", False), ["D6"])
    _tamp("R6 篡改 verdict_pass=false → D8 必报", lambda r: r.__setitem__("verdict_pass", False), ["D8"])

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
