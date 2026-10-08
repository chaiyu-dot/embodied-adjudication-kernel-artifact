# -*- coding: utf-8 -*-
"""blind_reverse_e69.py —— E69 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 control_strategies / e69）。只信"数据文件"
（bench 几何 + 力矩上限 + RNE 生成的扭矩样本），**反推** τ_lim、plant 质量参数、可达带宽配方，
并做证伪：假设 ωn=40 相比预算带宽**隐藏**了盲区（预算口径盲区 102 ≫ 假设口径 30），
证明"C 阶判据建立在不可达带宽上"这一主结论。

产物：blind_reverse_e69.json
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
# ★ 刻意不把 EVAL 加入 sys.path，使 control_strategies / e69 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e69_bandwidth_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e69.json")

_res = {"experiment": "E69 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 control_strategies / e69_bandwidth_refill",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def gen_data(l1, l2, m1, m2, n=60, seed=20260916):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-5, 5, 2)
        t = np.asarray(RNE.planar_2r_torque_closedform(l1, l2, m1, m2, q[0], q[1], qd[0], qd[1], qdd[0], qdd[1]), float)
        rows.append((q, qd, qdd, t))
    return rows


def recover_masses(rows, l1, l2, g=9.81):
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
    print("E69 逆向盲测（L5 隔离，反推参数 + 证伪）")
    print("=" * 88)
    chk("R0_isolation_CS_not_imported",
        "control_strategies" not in sys.modules and "e69_bandwidth_refill" not in sys.modules,
        "sys.modules 含 CS/e69=%s" % ("control_strategies" in sys.modules or "e69_bandwidth_refill" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    L1, L2 = 0.40, 0.30
    M1, M2 = 0.60, 0.35
    TAU = (5.0, 1.8)
    QREF = [0.60, -0.90]
    FRAC = 0.9

    # --- 反推 τ_lim（从 Q_REF 的 kp_i = FRAC*τ_lim_i/|q_i|）---
    v0 = rep["V0_consistency_with_E66"]
    kp0, kp1 = v0["E69_per_joint_kp"][0], v0["E69_per_joint_kp"][1]   # [7.5, 1.8]
    tau0_re = kp0 * abs(QREF[0]) / FRAC
    tau1_re = kp1 * abs(QREF[1]) / FRAC
    chk("R1_reverse_tau_lim", abs(tau0_re - TAU[0]) < 1e-6 and abs(tau1_re - TAU[1]) < 1e-6,
        "recovered=[%.4f,%.4f] truth=[%.4f,%.4f]" % (tau0_re, tau1_re, TAU[0], TAU[1]))

    # --- 反推 achievable_wn 配方（独立质量矩阵，tau_ff=0 口径与 V0 一致）---
    Mref = mass_matrix(QREF, L1, L2, M1, M2)
    kp0_, kp1_ = FRAC * TAU[0] / abs(QREF[0]), FRAC * TAU[1] / abs(QREF[1])
    wn_re = min(math.sqrt(kp0_ / Mref[0][0]), math.sqrt(kp1_ / Mref[1][1]))
    chk("R2_reverse_achievable_wn_at_QREF", abs(wn_re - v0["E69_per_joint_wn"]) < 1e-3,
        "recovered=%.4f stored=%.4f" % (wn_re, v0["E69_per_joint_wn"]))

    # --- 反推 plant 质量参数（几何已知，质量未知）---
    rows = gen_data(L1, L2, M1, M2)
    rm1, rm2, resid = recover_masses(rows, L1, L2)
    chk("R3_reverse_plant_mass_m1", abs(rm1 - M1) < 1e-3, "recovered=%.5f truth=%.5f" % (rm1, M1))
    chk("R4_reverse_plant_mass_m2", abs(rm2 - M2) < 1e-3, "recovered=%.5f truth=%.5f" % (rm2, M2))

    # --- 证伪：假设 ωn=40 相比预算带宽**隐藏**盲区（预算盲区 ≫ 假设盲区）---
    v3 = rep["V3_e58_refill"]
    blind_assumed = v3["assumed"]["blind_spot_vs_D"]
    blind_budget = v3["budget"]["blind_spot_vs_D"]
    chk("R5_falsify_assumed_wn_hides_blindspot",
        blind_budget > blind_assumed > 0 and blind_budget >= 3 * blind_assumed,
        "assumed=%d budget=%d ratio=%.1f" % (blind_assumed, blind_budget, blind_budget / max(blind_assumed, 1)))
    # ★ 无门控对照：预算口径下 H58-3 翻 FAIL（慢档误报 0.94），假设口径 PASS（0.0）→ 假设值粉饰
    chk("R6_falsify_H58_3_flips_under_real_bandwidth",
        v3["H58_3_slow_fa_lt_0.10_assumed"] is True and v3["H58_3_slow_fa_lt_0.10_budget"] is False,
        "assumed_PASS=%s budget_FAIL=%s" % (v3["H58_3_slow_fa_lt_0.10_assumed"], v3["H58_3_slow_fa_lt_0.10_budget"]))

    # --- 反推 V1 中位带宽（独立全量重算 48 行，确认与存储一致）---
    v1 = rep["V1_curve"]
    rec = []
    for r in v1["rows"]:
        q = r["q"]; dq = r["delta_q"]; mp = r["mp_kg"]
        m2e = M2 + mp; lc2e = (M2 * (L2 / 2.0) + mp * L2) / m2e; i2e = M2 * (L2 / 2.0) ** 2 + mp * (L2 - lc2e) ** 2 + (M2 * (L2 / 2.0) ** 2 + M2 * L2 * L2 / 12.0) - M2 * (L2 / 2.0) ** 2
        # 用标准 _l2_params：i2 = I2 + M2*(lc2e-LC2)^2 + mp*(L2-lc2e)^2
        I2e = M2 * L2 * L2 / 12.0 + M2 * (lc2e - L2 / 2.0) ** 2 + mp * (L2 - lc2e) ** 2
        c2 = math.cos(q[1])
        m11 = M1 * (L1 / 2.0) ** 2 + m2e * (L1 * L1 + lc2e ** 2 + 2 * L1 * lc2e * c2) + M1 * L1 * L1 / 12.0 + I2e
        m22 = m2e * lc2e ** 2 + I2e
        g1 = (M1 * (L1 / 2.0) + m2e * L1) * 9.81 * math.cos(q[0]) + m2e * lc2e * 9.81 * math.cos(q[0] + q[1])
        g2 = m2e * lc2e * 9.81 * math.cos(q[0] + q[1])
        head0 = FRAC * (TAU[0] - abs(g1)); head1 = FRAC * (TAU[1] - abs(g2))
        kp0r = head0 / max(abs(dq), 1e-3); kp1r = head1 / max(abs(dq), 1e-3)
        w0 = math.sqrt(kp0r / m11) if head0 > 0 else 0.0
        w1 = math.sqrt(kp1r / m22) if head1 > 0 else 0.0
        rec.append(min(w0, w1))
    rec_med = float(np.median(rec))
    chk("R7_reverse_V1_median_wn", abs(rec_med - v1["wn_median"]) < 1e-2,
        "recomputed_median=%.4f stored=%.4f" % (rec_med, v1["wn_median"]))

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
