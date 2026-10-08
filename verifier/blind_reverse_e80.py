# -*- coding: utf-8 -*-
"""blind_reverse_e80.py —— E80 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 e80_three_paradigm_dynamics）。只信"数据文件"
（bench 几何 + 力矩样本），**反推** plant 质量参数并做 ±5% 扰动证伪；并以**负对照**证明
C2（证伪见证）与 C3（KE 独立一致）确有分辨力——即把范式/KE 故意弄坏，必被检出，
否则这两个对拍就是空对拍。

产物：blind_reverse_e80.json
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
# ★ 刻意不把 EVAL 加入 sys.path，使 e80_three_paradigm_dynamics 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402
from planning import lagrange_dynamics as LAG                               # noqa: E402

REPORT = os.path.join(EVAL, "e80_three_paradigm_dynamics_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e80.json")

_res = {"experiment": "E80 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e80_three_paradigm_dynamics",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def recover_masses(rows, l1, l2, g=9.81):
    """几何已知、质量未知：τ 对 m1,m2 线性 → 最小二乘反推。返回 (m1,m2,residual)。"""
    A = []
    b = []
    for q, qd, qdd, t in rows:
        c2 = math.cos(q[1]); s2 = math.sin(q[1])
        c1 = math.cos(q[0]); c12 = math.cos(q[0] + q[1])
        dd1, dd2 = qdd[0], qdd[1]
        d1, d2 = qd[0], qd[1]
        a_m1_1 = (l1 ** 2 / 3.0) * dd1 + g * (l1 / 2.0) * c1
        a_m2_1 = (l1 ** 2 + (l2 / 2.0) ** 2 + 2.0 * l1 * (l2 / 2.0) * c2 + l2 ** 2 / 12.0) * dd1 \
            + ((l2 / 2.0) ** 2 + l1 * (l2 / 2.0) * c2 + l2 ** 2 / 12.0) * dd2 \
            - l1 * (l2 / 2.0) * s2 * (2.0 * d1 * d2 + d2 ** 2) \
            + g * l1 * c1 + g * (l2 / 2.0) * c12
        a_m1_2 = 0.0
        a_m2_2 = ((l2 / 2.0) ** 2 + l1 * (l2 / 2.0) * c2 + l2 ** 2 / 12.0) * dd1 \
            + ((l2 / 2.0) ** 2 + l2 ** 2 / 12.0) * dd2 \
            + l1 * (l2 / 2.0) * s2 * d1 ** 2 + g * (l2 / 2.0) * c12
        A.append([a_m1_1, a_m2_1]); b.append(t[0])
        A.append([a_m1_2, a_m2_2]); b.append(t[1])
    A = np.array(A); b = np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.max(np.abs(A @ sol - b)))
    return sol[0], sol[1], resid


def gen_data(l1, l2, m1, m2, n=40, seed=20260916):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-8, 8, 2)
        t = np.asarray(RNE.planar_2r_torque_closedform(l1, l2, m1, m2, q[0], q[1], qd[0], qd[1], qdd[0], qdd[1]), float)
        rows.append((q, qd, qdd, t))
    return rows


def main():
    print("=" * 88)
    print("E80 逆向盲测（L5 隔离，反推参数 + 负对照证伪分辨力）")
    print("=" * 88)
    chk("R0_isolation_e80_not_imported",
        "e80_three_paradigm_dynamics" not in sys.modules,
        "sys.modules 含 e80=%s" % ("e80_three_paradigm_dynamics" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    g = bench["g"]
    tol_C1, tol_C2, tol_C3 = bench["tol_C1"], bench["tol_C2"], bench["tol_C3"]

    # --- 反推 model[0] 的 plant 质量（从存储的 τ 样本，几何已知）---
    l1, l2, m1t, m2t = bench["models"][0]
    rows0 = []
    for r in rep["A_three_paradigm"]["rows"]:
        if r["model"] == bench["models"][0]:
            q = np.array(r["q"], float); qd = np.array(r["qd"], float); qdd = np.array(r["qdd"], float)
            t = np.array(r["tau_rne"], float)
            rows0.append((q, qd, qdd, t))
    rm1, rm2, resid = recover_masses(rows0, l1, l2, g)
    chk("R1_reverse_plant_mass_from_stored_tau", abs(rm1 - m1t) < 1e-3 and abs(rm2 - m2t) < 1e-3
        and resid < 1e-6, "recovered=(%.5f,%.5f) truth=(%.5f,%.5f) resid=%.2e" % (rm1, rm2, m1t, m2t, resid))

    # --- ±5% 扰动可识别（生成器扰动，reverse 仍锁定）---
    rm1p, _, _ = recover_masses(gen_data(l1, l2, m1t * 1.05, m2t * 1.05), l1, l2, g)
    rm1m, _, _ = recover_masses(gen_data(l1, l2, m1t * 0.95, m2t * 0.95), l1, l2, g)
    chk("R2_falsify_mass_perturbation_identifiable", rm1m < rm1 < rm1p and abs(rm1p - rm1) > 1e-2,
        "0.95x=%.5f 1.0x=%.5f 1.05x=%.5f" % (rm1m, rm1, rm1p))

    # --- R3：C2 证伪见证具备分辨力（负对照）—— 故意坏的范式必被检出 ---
    P = LAG.default_params(l1, l2, m1t, m2t, g)
    rng = np.random.RandomState(bench["seed"] + 1)
    worstB_good_rne = 0.0
    worstB_bad = 0.0
    for _ in range(10):
        q = rng.uniform(*bench["q_range"], 2); qd = rng.uniform(*bench["qd_range"], 2)
        qdd = rng.uniform(*bench["qdd_range"], 2)
        good = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        bad = np.asarray(LAG.coriolis_dropped(q, qd, qdd, P), float)
        lk = RNE.planar_2r_model(l1, l2, m1t, m2t, plane="xy", q1=float(q[0]), q2=float(q[1]))
        rne = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0.0, -g, 0.0)), float)
        worstB_good_rne = max(worstB_good_rne, float(np.max(np.abs(good - rne))))
        worstB_bad = max(worstB_bad, float(np.max(np.abs(good - bad))))
    chk("R3_falsify_C2_witness_resolves_broken_paradigm", worstB_bad > tol_C2 and worstB_good_rne < tol_C1,
        "bad_vs_good=%.3e good_vs_rne=%.3e tol_C2=%.0e" % (worstB_bad, worstB_good_rne, tol_C2))
    # 与存储的 B 一致（存储 witnesses 真实非手填）
    chk("R3b_C2_stored_matches_independent", abs(worstB_bad - rep["B_falsification_witness"]["max_bad_vs_good"]) < 1e-12,
        "recomp=%.3e stored=%.3e" % (worstB_bad, rep["B_falsification_witness"]["max_bad_vs_good"]))

    # --- R4：C3 KE 独立一致具备分辨力（负对照）—— 故意坏的 KE 必被检出 ---
    pk = {"l1": l1, "l2": l2, "m1": m1t, "m2": m2t, "lc1": l1 / 2.0, "lc2": l2 / 2.0,
          "i1": m1t * l1 ** 2 / 12.0, "i2": m2t * l2 ** 2 / 12.0, "g": g, "mp": 0.0}
    rng = np.random.RandomState(bench["seed"] + 2)
    worstC_real = 0.0
    worstC_bad = 0.0
    for _ in range(60):
        q = rng.uniform(*bench["q_range"], 2); qd = rng.uniform(*bench["qd_range"], 2)
        t_lag = float(LAG.kinetic(q, qd, P))
        t_ek = float(RNE.kinetic_energy(
            RNE.planar_2r_model(l1, l2, m1t, m2t, plane="xy", q1=float(q[0]), q2=float(q[1])), q, qd))
        # 故意坏的 KE：只算平移、丢掉旋转 ½Iω²（常见实现错误）
        # t_lag 的平移部分 ≈ ½m1|v1|²+½m2|v2|²；旋转 = ½I1 qd1²+½I2(qd1+qd2)²
        bad_ke = t_lag - 0.5 * (m1t * l1 ** 2 / 12.0) * qd[0] ** 2 \
            - 0.5 * (m2t * l2 ** 2 / 12.0) * (qd[0] + qd[1]) ** 2
        worstC_real = max(worstC_real, abs(t_lag - t_ek) / max(abs(t_ek), 1e-9))
        worstC_bad = max(worstC_bad, abs(t_lag - bad_ke) / max(abs(bad_ke), 1e-9))
    chk("R4_falsify_C3_ke_crosscheck_resolves_broken_ke", worstC_bad > tol_C3 and worstC_real < tol_C3,
        "bad_ke_rel=%.2e real_rel=%.2e tol_C3=%.0e" % (worstC_bad, worstC_real, tol_C3))
    chk("R4b_C3_stored_matches_independent", abs(worstC_real - rep["C_ke_crosscheck"]["max_rel_dev"]) < 1e-15,
        "recomp=%.3e stored=%.3e" % (worstC_real, rep["C_ke_crosscheck"]["max_rel_dev"]))

    # --- R5：三范式算法独立签名（自创 RNG，不依赖实验 seed）---
    rng = np.random.RandomState(777)
    err_cf, err_lag = 0.0, 0.0
    for _ in range(60):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-8, 8, 2)
        lk = RNE.planar_2r_model(l1, l2, m1t, m2t, plane="xy", q1=float(q[0]), q2=float(q[1]))
        rne = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0.0, -g, 0.0)), float)
        cf = np.array(RNE.planar_2r_torque_closedform(l1, l2, m1t, m2t, q[0], q[1], qd[0], qd[1], qdd[0], qdd[1], g=g), float)
        lag = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        err_cf = max(err_cf, float(np.max(np.abs(rne - cf))))
        err_lag = max(err_lag, float(np.max(np.abs(rne - lag))))
    chk("R5_three_paradigm_algorithmic_signature", err_cf < 1e-10 and 1e-9 <= err_lag <= 1e-7,
        "rne_vs_cf=%.2e rne_vs_lag=%.2e" % (err_cf, err_lag))

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
