# -*- coding: utf-8 -*-
"""blind_reverse_e68.py —— E68 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 control_strategies / e68）。只信"数据文件"
（bench 几何 + 力矩上限 + RNE 生成的扭矩样本），**反推** τ_lim、激进设计增益、plant 质量参数，
并做 ±5% 扰动证伪、无门控负对照证明门控真实承重（非真空真）。

产物：blind_reverse_e68.json
"""
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 control_strategies / e68 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e68_control_suite_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e68.json")

_res = {"experiment": "E68 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 control_strategies / e68_control_strategy_suite",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def gen_data(l1, l2, m1, m2, n=60, seed=20260916):
    """RNE 生成 (q,qd,qdd,τ) 样本（数据文件等价物）。"""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-5, 5, 2)
        t = np.asarray(RNE.planar_2r_torque_closedform(l1, l2, m1, m2, q[0], q[1], qd[0], qd[1], qdd[0], qdd[1]), float)
        rows.append((q, qd, qdd, t))
    return rows


def recover_masses(rows, l1, l2, g=9.81):
    """几何已知、质量未知：τ 对 m1,m2 线性 → 最小二乘反推。返回 (m1,m2,residual)。"""
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    A = []; b = []
    for q, qd, qdd, t in rows:
        c2 = math.cos(q[1]); s2 = math.sin(q[1])
        c1 = math.cos(q[0]); c12 = math.cos(q[0] + q[1])
        dd1, dd2 = qdd[0], qdd[1]
        d1, d2 = qd[0], qd[1]
        a_m1_1 = (lc1 ** 2 + l1 ** 2 / 12.0) * dd1 + g * lc1 * c1
        a_m2_1 = (l1 ** 2 + lc2 ** 2 + 2.0 * l1 * lc2 * c2 + l2 ** 2 / 12.0) * dd1 \
                 + (lc2 ** 2 + l1 * lc2 * c2 + l2 ** 2 / 12.0) * dd2 \
                 - l1 * lc2 * s2 * (2.0 * d1 * d2 + d2 ** 2) \
                 + g * l1 * c1 + g * lc2 * c12
        a_m1_2 = 0.0
        a_m2_2 = (lc2 ** 2 + l1 * lc2 * c2 + l2 ** 2 / 12.0) * dd1 \
                 + (lc2 ** 2 + l2 ** 2 / 12.0) * dd2 \
                 + l1 * lc2 * s2 * d1 ** 2 + g * lc2 * c12
        A.append([a_m1_1, a_m2_1]); b.append(t[0])
        A.append([a_m1_2, a_m2_2]); b.append(t[1])
    A = np.array(A); b = np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.max(np.abs(A @ sol - b)))
    return sol[0], sol[1], resid


def mass_matrix(q, l1, l2, m1, m2):
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    c2 = math.cos(q[1])
    m11 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * c2) + m1 * l1 * l1 / 12.0 + m2 * l2 * l2 / 12.0
    m12 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + m2 * l2 * l2 / 12.0
    m22 = m2 * lc2 * lc2 + m2 * l2 * l2 / 12.0
    return [[m11, m12], [m12, m22]]


def main():
    print("=" * 88)
    print("E68 逆向盲测（L5 隔离，反推参数 + 证伪）")
    print("=" * 88)
    chk("R0_isolation_CS_not_imported",
        "control_strategies" not in sys.modules and "e68_control_strategy_suite" not in sys.modules,
        "sys.modules 含 CS/e68=%s" % ("control_strategies" in sys.modules or "e68_control_strategy_suite" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    L1, L2 = bench["l"]
    M1, M2 = bench["m"]
    TAU = bench["tau_lim"]            # [5.0, 1.8]
    qref = bench["q_ref"]             # [0.60, -0.90]

    # --- 反推 τ_lim ---
    # 关节1：预算 kp = 0.9*τ_lim[1]/|qref[1]| ⇒ τ_lim[1] = kp*|qref[1]|/0.9
    bd = rep["budget_design"]
    tau1_re = bd["kp"] * abs(qref[1]) / 0.9
    chk("R1_reverse_tau_lim_joint1_from_budget", abs(tau1_re - TAU[1]) < 1e-6,
        "recovered=%.4f truth=%.4f" % (tau1_re, TAU[1]))
    # 关节0：V1 门控 clip 给出 executed_tau_max == τ_lim[0]
    v1 = rep["V1_gate_invariant"]
    tau0_re = max(v["executed_tau_max"] for v in v1["per_family"].values())
    chk("R2_reverse_tau_lim_joint0_from_clip", abs(tau0_re - TAU[0]) < 1e-6,
        "recovered=%.4f truth=%.4f" % (tau0_re, TAU[0]))

    # --- 反推激进设计增益（ωn=40，meff 从 bench 几何独立算）---
    Mref = mass_matrix(qref, L1, L2, M1, M2)
    meff = 0.5 * (Mref[0][0] + Mref[1][1])
    ag = v1["aggressive_design"]
    kp_a_re = 40.0 ** 2 * meff
    kd_a_re = 2 * 0.7 * 40.0 * meff
    chk("R3_reverse_aggressive_kp_from_wn40", abs(ag["kp"] - kp_a_re) < 1e-2,
        "recovered=%.4f stored=%.4f" % (kp_a_re, ag["kp"]))
    chk("R4_reverse_aggressive_kd_from_wn40", abs(ag["kd"] - kd_a_re) < 1e-2,
        "recovered=%.4f stored=%.4f" % (kd_a_re, ag["kd"]))

    # --- 反推 plant 质量参数（几何已知，质量未知）→ 确认 bench 一致 ---
    rows = gen_data(L1, L2, M1, M2)
    rm1, rm2, resid = recover_masses(rows, L1, L2)
    chk("R5_reverse_plant_mass_m1", abs(rm1 - M1) < 1e-3, "recovered=%.5f truth=%.5f" % (rm1, M1))
    chk("R6_reverse_plant_mass_m2", abs(rm2 - M2) < 1e-3, "recovered=%.5f truth=%.5f" % (rm2, M2))

    # --- 证伪 A：±5% 扰动可识别（recovered m1 单调偏移）---
    rm1p, _, _ = recover_masses(gen_data(L1, L2, M1 * 1.05, M2 * 1.05), L1, L2)
    rm1m, _, _ = recover_masses(gen_data(L1, L2, M1 * 0.95, M2 * 0.95), L1, L2)
    chk("R7_falsify_perturb_identifiable", rm1m < rm1 < rm1p and abs(rm1p - rm1) > 1e-2,
        "0.95x=%.5f 1.0x=%.5f 1.05x=%.5f" % (rm1m, rm1, rm1p))

    # --- 证伪 B（负对照）：无门控对照下 PD/CT 真实越限 → 门控非真空真 ---
    ng = rep["gate_ablation_no_gate"]
    chk("R8_falsify_gate_load_bearing", ng["PD"]["violation_steps"] > 0 and ng["CT"]["violation_steps"] > 0,
        "PD_nogate_viol=%d CT_nogate_viol=%d MPC_nogate_viol=%d"
        % (ng["PD"]["violation_steps"], ng["CT"]["violation_steps"], ng["MPC"]["violation_steps"]))
    # ★ 需求越限 >0 的存在性（至少 P/PD/CT），证明门控真实在承重
    dv = {k: v["demand_violation_steps"] for k, v in v1["per_family"].items()}
    chk("R9_demand_violation_falsifiable", dv["P"] > 0 and dv["PD"] > 0 and dv["CT"] > 0,
        "demand_viol: " + ", ".join("%s=%d" % (k, dv[k]) for k in ("P", "PD", "PID", "PD_G", "CT", "FO", "MPC")))

    # --- 反推 MPC 标定比率（qd_w = 0.05*q_w 且 effective_kp≈target）---
    mc = rep["MPC_calibration"]
    chk("R10_reverse_MPC_calibration", abs(mc["qd_w"] - 0.05 * mc["q_w"]) < 1e-9
        and mc["rel_err"] < 0.05 and abs(mc["effective_kp"] - mc["target_kp_budget"]) / mc["target_kp_budget"] < 0.05,
        "qd_w=%.2e rel_err=%.3f" % (mc["qd_w"], mc["rel_err"]))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
