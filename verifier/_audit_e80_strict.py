# -*- coding: utf-8 -*-
"""_audit_e80_strict.py —— E80 的**严格审核（L3，独立物理）**。

不 import 任何 e80 模块；独立推导平面 2R 的 M/C/g（审计者自写公式），与**参考实现**
`rne_dynamics`（信任源）+ **第四范式** `lagrange_dynamics`（算法不同）对拍，验证：

  · C1 三范式一致：存储的 tau_rne/tau_cf/tau_lag 每行都可被独立重算复现，且聚合
    max|Δτ| < tol_C1（1e-7）；并核验"RNE vs 闭式≈机器精度、RNE/Lagrange≈1e-8"的
    截断特征签名（说明 Lagrange 的 ~1e-8 来自中心差分，而非三方同错）。
  · C2 证伪见证：独立复算 part_B（lagrange_torque vs coriolis_dropped），确认坏范式
    必然被检出（max|Δτ| > tol_C2=1e-3）。
  · C3 KE 独立一致：独立复算 part_C（lagrange.kinetic vs energy_kernel.kinetic_energy_2r），
    确认相对偏差 < tol_C3=1e-12。
  · verdict 自洽。

并做篡改用例（改报告字段 → 必有检查报出）。
产物：_audit_e80_strict.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                       # noqa: E402
from planning import lagrange_dynamics as LAG                  # noqa: E402
from planning.energy_kernel import kinetic_energy_2r           # noqa: E402

REPORT = os.path.join(EVAL, "e80_three_paradigm_dynamics_report.json")
OUT = os.path.join(EVAL, "_audit_e80_strict.json")

_res = {"experiment": "E80 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写 M/C/g + rne_dynamics(参考) + lagrange_dynamics(第四范式)，不 import e80",
        "checks": [], "tamper": []}


# ---- 审计者自写的平面 2R 动力学（不经过 e80 / 也不经过 e80 之外的任何实验函数）----
def _aud_mass(l1, l2, m1, m2, q):
    c2 = math.cos(q[1])
    m11 = m1 * (l1 / 2.0) ** 2 + m2 * (l1 * l1 + (l2 / 2.0) ** 2 + 2 * l1 * (l2 / 2.0) * c2) + m1 * l1 ** 2 / 12.0 + m2 * l2 ** 2 / 12.0
    m12 = m2 * ((l2 / 2.0) ** 2 + l1 * (l2 / 2.0) * c2) + m2 * l2 ** 2 / 12.0
    m22 = m2 * (l2 / 2.0) ** 2 + m2 * l2 ** 2 / 12.0
    return np.array([[m11, m12], [m12, m22]], float)


def _aud_coriolis(l1, l2, m1, m2, q, qd):
    h = -m2 * l1 * (l2 / 2.0) * math.sin(q[1])
    return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2], float)


def _aud_gravity(l1, l2, m1, m2, q, g):
    g1 = (m1 * (l1 / 2.0) + m2 * l1) * g * math.cos(q[0]) + m2 * (l2 / 2.0) * g * math.cos(q[0] + q[1])
    g2 = m2 * (l2 / 2.0) * g * math.cos(q[0] + q[1])
    return np.array([g1, g2], float)


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    A = rep["A_three_paradigm"]
    B = rep["B_falsification_witness"]
    C = rep["C_ke_crosscheck"]
    bench = rep["bench"]
    g = bench["g"]
    tol_C1, tol_C2, tol_C3 = bench["tol_C1"], bench["tol_C2"], bench["tol_C3"]

    # --- S1：审计者 M/C/g vs 参考 RNE（信任锚）+ vs Lagrange（第四范式）---
    err_aud_rne = 0.0
    err_aud_lag = 0.0
    for row in A["rows"]:
        l1, l2, m1, m2 = row["model"]
        q = np.array(row["q"], float); qd = np.array(row["qd"], float); qdd = np.array(row["qdd"], float)
        lhs = (_aud_mass(l1, l2, m1, m2, q) @ qdd
               + _aud_coriolis(l1, l2, m1, m2, q, qd)
               + _aud_gravity(l1, l2, m1, m2, q, g))
        lk = RNE.planar_2r_model(l1, l2, m1, m2, plane="xy", q1=float(q[0]), q2=float(q[1]))
        rne = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0.0, -g, 0.0)), float)
        P = LAG.default_params(l1, l2, m1, m2, g)
        lag = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        err_aud_rne = max(err_aud_rne, float(np.max(np.abs(lhs - rne))))
        err_aud_lag = max(err_aud_lag, float(np.max(np.abs(lhs - lag))))
    ch.append(_chk("S1_auditor_MCg_vs_RNE_lt_1e8_and_vs_Lagrange_lt_1e7",
                   err_aud_rne < 1e-8 and err_aud_lag < 1e-7,
                   "aud-vs-rne=%.2e aud-vs-lag=%.2e" % (err_aud_rne, err_aud_lag)))

    # --- S2：独立重算存储的 tau_rne/cf/lag（每行）并比对（容忍 10 位四舍五入）---
    worst_tau = 0.0
    for row in A["rows"]:
        l1, l2, m1, m2 = row["model"]
        q = np.array(row["q"], float); qd = np.array(row["qd"], float); qdd = np.array(row["qdd"], float)
        lk = RNE.planar_2r_model(l1, l2, m1, m2, plane="xy", q1=float(q[0]), q2=float(q[1]))
        t_rne = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0.0, -g, 0.0)), float)
        t_cf = np.array(RNE.planar_2r_torque_closedform(l1, l2, m1, m2, q[0], q[1], qd[0], qd[1], qdd[0], qdd[1], g=g), float)
        P = LAG.default_params(l1, l2, m1, m2, g)
        t_lag = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        worst_tau = max(worst_tau,
                        float(np.max(np.abs(t_rne - np.array(row["tau_rne"], float)))),
                        float(np.max(np.abs(t_cf - np.array(row["tau_cf"], float)))),
                        float(np.max(np.abs(t_lag - np.array(row["tau_lag"], float)))))
    ch.append(_chk("S2_stored_tau_reproducible_lt_1e6", worst_tau < 1e-6,
                   "max|recomputed-stored|=%.2e" % worst_tau))

    # --- S3：聚合 max_abs 复算 == 存储，且 < tol_C1 ---
    mx_recomp = max(max(r["d_rne_cf"], r["d_rne_lag"], r["d_cf_lag"]) for r in A["rows"])
    ch.append(_chk("S3_max_abs_recomputed_eq_stored_and_lt_tolC1",
                   abs(mx_recomp - A["max_abs"]) < 1e-10 and A["max_abs"] < tol_C1,
                   "recomp=%.3e stored=%.3e tol=%.0e" % (mx_recomp, A["max_abs"], tol_C1)))

    # --- S4：特征签名 —— RNE vs 闭式≈机器精度；RNE/Lagrange≈1e-8（中心差分截断，非三方同错）---
    mp = A["max_pairwise"]
    sig = (mp["rne_vs_cf"] < 1e-10) and (mp["rne_vs_lag"] >= 1e-9) and (mp["rne_vs_lag"] < tol_C1)
    ch.append(_chk("S4_truncation_signature_rne_cf_tiny_lag_1e8",
                   sig, "rne_vs_cf=%.2e rne_vs_lag=%.2e" % (mp["rne_vs_cf"], mp["rne_vs_lag"])))

    # --- S5：C2 证伪见证独立复算（seed+1，model[0]）---
    l1, l2, m1, m2 = bench["models"][0]
    P = LAG.default_params(l1, l2, m1, m2, g)
    rng = np.random.RandomState(bench["seed"] + 1)
    worstB = 0.0
    for _ in range(10):
        q = rng.uniform(*bench["q_range"], 2); qd = rng.uniform(*bench["qd_range"], 2)
        qdd = rng.uniform(*bench["qdd_range"], 2)
        good = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        bad = np.asarray(LAG.coriolis_dropped(q, qd, qdd, P), float)
        worstB = max(worstB, float(np.max(np.abs(good - bad))))
    recomputed_detected = bool(worstB > tol_C2)
    ch.append(_chk("S5_C2_falsification_recomputed_and_stored_consistent",
                   recomputed_detected and abs(worstB - B["max_bad_vs_good"]) < 1e-12
                   and B["detected"] == recomputed_detected and B["detected"] is True,
                   "recomp=%.3e stored=%.3e" % (worstB, B["max_bad_vs_good"])))

    # --- S6：C3 KE 独立一致复算（seed+2，60 状态）---
    pk = {"l1": l1, "l2": l2, "m1": m1, "m2": m2, "lc1": l1 / 2.0, "lc2": l2 / 2.0,
          "i1": m1 * l1 ** 2 / 12.0, "i2": m2 * l2 ** 2 / 12.0, "g": g, "mp": 0.0}
    rng = np.random.RandomState(bench["seed"] + 2)
    worstC = 0.0
    for _ in range(60):
        q = rng.uniform(*bench["q_range"], 2); qd = rng.uniform(*bench["qd_range"], 2)
        t_lag = float(LAG.kinetic(q, qd, P))
        t_ek = float(kinetic_energy_2r(q, qd, pk))
        worstC = max(worstC, abs(t_lag - t_ek) / max(abs(t_ek), 1e-9))
    recomputed_ok = bool(worstC < tol_C3)
    ch.append(_chk("S6_C3_ke_crosscheck_recomputed_and_stored_consistent",
                   recomputed_ok and abs(worstC - C["max_rel_dev"]) < 1e-15
                   and C["all_ok"] == recomputed_ok and C["all_ok"] is True,
                   "recomp=%.3e stored=%.3e tol=%.0e" % (worstC, C["max_rel_dev"], tol_C3)))

    # --- S7：verdict 自洽 ---
    ch.append(_chk("S7_verdict_self_consistent",
                   rep["verdict_pass"] is (A["all_ok"] and B["all_ok"] and C["all_ok"])
                   and rep["verdict_pass"] is True))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught),
                           "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E80 严格审核（L3 独立物理，不 import e80）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例（改报告字段 → 必有检查报出）")
    _tamp("R1 篡改 A.max_abs=9.9 → S3 必报",
          lambda r: r["A_three_paradigm"].__setitem__("max_abs", 9.9), ["S3"])
    _tamp("R2 篡改 B.detected=False → S5 必报",
          lambda r: r["B_falsification_witness"].__setitem__("detected", False), ["S5"])
    _tamp("R3 篡改 C.all_ok=False → S6 必报",
          lambda r: r["C_ke_crosscheck"].__setitem__("all_ok", False), ["S6"])
    _tamp("R4 篡改 verdict_pass=False → S7 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S7"])
    _tamp("R5 篡改第一行 tau_lag 数据 → S2 必报",
          lambda r: r["A_three_paradigm"]["rows"][0]["tau_lag"].__setitem__(0,
                  r["A_three_paradigm"]["rows"][0]["tau_lag"][0] + 100.0), ["S2"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s"
          % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
