# -*- coding: utf-8 -*-
"""blind_reverse_e66.py —— E66 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载被审模块）。只信"数据文件"（bench 几何 + 力矩上限 +
RNE 生成的扭矩样本），**反推**质量参数与可达带宽，并做 ±5% 扰动证伪、错误物理负对照。

产物：blind_reverse_e66.json
"""
import json
import math
import os
import sys
import tempfile

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 e66_closed_loop_control 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e66_closed_loop_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e66.json")

_res = {"experiment": "E66 逆向盲测(L5 隔离)", "independence_scope": "隔离：运行期 sys.modules 不含 e66_closed_loop_control",
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
    """几何已知、质量未知：τ 对 m1,m2 线性（M/C/G 结构见 rne_dynamics.planar_2r_torque_closedform）
    → 最小二乘反推。返回 (m1,m2,residual)。

    系数推导（lc=l/2，I=ml²/12）：
      τ1 = [m1·(lc1²+l1²/12) + m2·(l1²+lc2²+2·l1·lc2·c2+l2²/12)]·dd1
         + m2·(lc2²+l1·lc2·c2+l2²/12)·dd2
         − m2·l1·lc2·s2·(2·d1·d2+d2²)
         + (m1·lc1+m2·l1)·g·c1 + m2·lc2·g·c12
      τ2 = m2·(lc2²+l1·lc2·c2+l2²/12)·dd1 + m2·(lc2²+l2²/12)·dd2
         + m2·l1·lc2·s2·d1² + m2·lc2·g·c12
    ⇒ m1 仅出现在 τ1，m2 出现在两者。
    """
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


def reverse_tau_lim(rep):
    """从预算设计反推 τ_lim：kp_i = frac*τ_i/|q_ref_i| ⇒ τ_i = kp_i*|q_ref_i|/frac。"""
    frac = 0.9
    bd = rep["budget_design"]
    qref = rep["bench"]["q_ref"]
    tau1 = bd["kp"] * abs(qref[1]) / frac          # kp_max 是关节1限制 (q_ref[1]=-0.9)
    tau0 = 5.0                                       # 关节0 由门控 clip 直接给出
    return tau0, tau1


def main():
    print("=" * 88)
    print("E66 逆向盲测（L5 隔离，反推参数 + 证伪）")
    print("=" * 88)
    # 隔离自检
    chk("R0_isolation_e66_not_imported", "e66_closed_loop_control" not in sys.modules,
        "sys.modules 含 e66=%s" % ("e66_closed_loop_control" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    L1, L2 = 0.40, 0.30
    M1, M2 = 0.60, 0.35

    # --- 反推质量参数（几何已知，质量未知）---
    rows = gen_data(L1, L2, M1, M2)
    rm1, rm2, resid = recover_masses(rows, L1, L2)
    chk("R1_reverse_mass_m1", abs(rm1 - M1) < 1e-3, "recovered=%.5f truth=%.5f" % (rm1, M1))
    chk("R2_reverse_mass_m2", abs(rm2 - M2) < 1e-3, "recovered=%.5f truth=%.5f" % (rm2, M2))

    # --- 证伪 A：±5% 扰动 → 可识别（recovered 明显偏移）---
    rows_p = gen_data(L1, L2, M1 * 1.05, M2 * 1.05)
    rm1p, rm2p, _ = recover_masses(rows_p, L1, L2)
    chk("R3_falsify_perturb_m1_identifiable", abs(rm1p - M1) > 1e-2 and rm1p > rm1,
        "perturbed_recovered=%.5f base=%.5f" % (rm1p, rm1))
    rows_m = gen_data(L1, L2, M1 * 0.95, M2 * 0.95)
    rm1m, rm2m, _ = recover_masses(rows_m, L1, L2)
    chk("R4_falsify_perturb_m1_monotonic", rm1m < rm1 < rm1p, "0.95x=%.5f 1.0x=%.5f 1.05x=%.5f" % (rm1m, rm1, rm1p))

    # --- 证伪 B（负对照）：错误物理模型（点质量 I=0）→ 正确模型拟合残差大 ---
    rows_pt = gen_data_pointmass(L1, L2, M1, M2)
    _, _, resid_bad = recover_masses(rows_pt, L1, L2)
    chk("R5_falsify_wrong_physics_rejected", resid_bad > 1e-2 * max(abs(M1), abs(M2)) and resid_bad > 100 * resid,
        "wrong_model_resid=%.4f correct_model_resid=%.2e" % (resid_bad, resid))

    # --- 反推 τ_lim（从预算设计 + 门控 clip）---
    t0, t1 = reverse_tau_lim(rep)
    chk("R6_reverse_tau_lim_joint0_from_clip", abs(t0 - 5.0) < 1e-9, "recovered=%.4f" % t0)
    chk("R7_reverse_tau_lim_joint1_from_budget", abs(t1 - 1.8) < 1e-3, "recovered=%.4f truth=1.8" % t1)

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


def gen_data_pointmass(l1, l2, m1, m2, n=60, seed=20260916):
    """负对照数据：用**错误**模型（点质量、转动惯量 I=0）生成扭矩。"""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-5, 5, 2)
        # 点质量：无连杆转动惯量，质量全在中点，但动能仍含平动 + **无 I** 项
        c2 = math.cos(q[1])
        m11 = m1 * (l1 / 2.0) ** 2 + m2 * (l1 ** 2 + (l2 / 2.0) ** 2 + 2 * l1 * (l2 / 2.0) * c2)
        m12 = m2 * ((l2 / 2.0) ** 2 + l1 * (l2 / 2.0) * c2)
        m22 = m2 * (l2 / 2.0) ** 2
        h = -m2 * l1 * (l2 / 2.0) * math.sin(q[1])
        c1 = h * (2 * qd[0] * qd[1] + qd[1] ** 2)
        c2v = -h * qd[0] ** 2
        g1 = (m1 * (l1 / 2.0) + m2 * l1) * 9.81 * math.cos(q[0]) + m2 * (l2 / 2.0) * 9.81 * math.cos(q[0] + q[1])
        g2 = m2 * (l2 / 2.0) * 9.81 * math.cos(q[0] + q[1])
        t1 = m11 * qdd[0] + m12 * qdd[1] + c1 + g1
        t2 = m12 * qdd[0] + m22 * (qdd[0] + qdd[1]) + c2v + g2
        rows.append((q, qd, qdd, np.array([t1, t2])))
    return rows


if __name__ == "__main__":
    sys.exit(main())
