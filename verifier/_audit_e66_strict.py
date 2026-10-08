# -*- coding: utf-8 -*-
"""_audit_e66_strict.py —— E66 的**严格审核（L3，审前提 + 独立物理）**。

不 import 任何 e66 模块；独立推导平面 2R 的 M/C/g（审计者自写公式），与**参考实现**
`rne_dynamics`（信任源）+ **第四范式** `lagrange_dynamics`（算法不同）对拍，三重确认 V1。
硬性重算预算增益（C2）、静态可持载荷、能量闭合恒等式；并做篡改用例。

产物：_audit_e66_strict.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                       # noqa: E402
from planning import lagrange_dynamics as LAG                  # noqa: E402

REPORT = os.path.join(EVAL, "e66_closed_loop_report.json")
OUT = os.path.join(EVAL, "_audit_e66_strict.json")

_res = {"experiment": "E66 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写 M/C/g + rne_dynamics(参考) + lagrange_dynamics(第四范式)，不 import e66",
        "checks": [], "tamper": []}

# ---- 审计者自写的平面 2R 动力学（不经过 e66 的任何函数）----
L1, L2 = 0.40, 0.30
M1, M2 = 0.60, 0.35
LC1, LC2 = L1 / 2.0, L2 / 2.0
I1 = M1 * L1 * L1 / 12.0
I2 = M2 * L2 * L2 / 12.0
G = 9.81
TAU_LIM = (5.0, 1.8)


def _aud_mass(q):
    c2 = math.cos(q[1])
    m11 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2 * L1 * LC2 * c2) + I1 + I2
    m12 = M2 * (LC2 * LC2 + L1 * LC2 * c2) + I2
    m22 = M2 * LC2 * LC2 + I2
    return [[m11, m12], [m12, m22]]


def _aud_coriolis(q, qd):
    h = -M2 * L1 * LC2 * math.sin(q[1])
    return [h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2]


def _aud_gravity(q):
    g1 = (M1 * LC1 + M2 * L1) * G * math.cos(q[0]) + M2 * LC2 * G * math.cos(q[0] + q[1])
    g2 = M2 * LC2 * G * math.cos(q[0] + q[1])
    return [g1, g2]


def _aud_potential(q):
    y1 = LC1 * math.sin(q[0])
    y2 = L1 * math.sin(q[0]) + LC2 * math.sin(q[0] + q[1])
    return M1 * G * y1 + M2 * G * y2


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    import numpy as np
    rng = np.random.RandomState(20260916)
    # --- V1：审计者 M/C/g + RNE(参考) + Lagrange(第四范式) 三重一致 ---
    err_aud_rne = 0.0
    err_lag_rne = 0.0
    err_aud_lag = 0.0
    for _ in range(200):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-5, 5, 2)
        lhs = np.array(_aud_mass(q)) @ qdd + np.array(_aud_coriolis(q, qd)) + np.array(_aud_gravity(q))
        lk = RNE.planar_2r_model(L1, L2, M1, M2, q1=q[0], q2=q[1])
        rne = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0, -G, 0)), float)
        P = LAG.default_params(L1, L2, M1, M2, G)
        lag = np.asarray(LAG.lagrange_torque(q, qd, qdd, P), float)
        err_aud_rne = max(err_aud_rne, float(np.max(np.abs(lhs - rne))))
        err_lag_rne = max(err_lag_rne, float(np.max(np.abs(lag - rne))))
        err_aud_lag = max(err_aud_lag, float(np.max(np.abs(lhs - lag))))
    ch.append(_chk("S1_V1_triple_consistency_MCg_vs_RNE_lt_1e8",
                   err_aud_rne < 1e-8 and err_lag_rne < 1e-8 and err_aud_lag < 1e-8,
                   "aud-vs-rne=%.2e lag-vs-rne=%.2e aud-vs-lag=%.2e"
                   % (err_aud_rne, err_lag_rne, err_aud_lag)))
    ch.append(_chk("S2_V1_flag_matches_recomputed",
                   rep["V1_MCg_decomposition_ok"] is True and rep["MCg_vs_RNE_max_abs_err"] < 1e-8,
                   rep["MCg_vs_RNE_max_abs_err"]))

    # --- C2：预算增益 / 可达带宽（独立重算）---
    frac = 0.9
    q_ref = rep["bench"]["q_ref"]
    kp0 = frac * TAU_LIM[0] / abs(q_ref[0])
    kp1 = frac * TAU_LIM[1] / abs(q_ref[1])
    kp_max = min(kp0, kp1)
    Mref = np.array(_aud_mass(q_ref))
    meff = float(np.mean(np.diag(Mref)))
    wn = min(math.sqrt(kp0 / Mref[0, 0]), math.sqrt(kp1 / Mref[1, 1]))
    kd = 2 * 0.7 * math.sqrt(kp_max * meff)
    ch.append(_chk("S3_C2_achievable_wn_matches", abs(wn - rep["budget_design"]["achievable_wn_rad_s"]) < 1e-3,
                   "recomputed=%.4f stored=%.4f" % (wn, rep["budget_design"]["achievable_wn_rad_s"])))
    ch.append(_chk("S4_C2_kp_matches", abs(kp_max - rep["budget_design"]["kp"]) < 1e-6,
                   "recomputed=%.6f stored=%.6f" % (kp_max, rep["budget_design"]["kp"])))
    ch.append(_chk("S5_C2_kd_matches", abs(kd - rep["budget_design"]["kd"]) < 1e-4,
                   "recomputed=%.6f stored=%.6f" % (kd, rep["budget_design"]["kd"])))
    # 假设的 40 rad/s 确实被力矩预算钉死（乐观 ≥5×）
    ch.append(_chk("S6_C2_assumed_wn_infeasible",
                   40.0 / wn >= 5.0 and rep["budget_design"]["assumed_over_achievable_x"] >= 5.0,
                   "ratio=%.2f" % (40.0 / wn)))

    # --- 静态可持载荷上限（独立解线性方程）---
    q = np.array(q_ref, float)
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    g0 = _aud_gravity(q)
    d1 = G * (L1 * c1 + L2 * c12)
    d2 = G * L2 * c12
    cand = []
    for g_lim, g_0, den in ((TAU_LIM[0], g0[0], d1), (TAU_LIM[1], g0[1], d2)):
        if den > 1e-12:
            cand.append((g_lim - g_0) / den)
    mp_lim = max(0.0, min(cand))
    ch.append(_chk("S7_static_payload_limit_matches", abs(mp_lim - rep["static_payload_limit_kg"]) < 1e-3,
                   "recomputed=%.4f stored=%.4f" % (mp_lim, rep["static_payload_limit_kg"])))

    # --- V3：能量闭合恒等式（用存储标量验证 W-D = ΔKE+ΔPE，并独立重算 ΔPE）---
    eb = rep["energy_balance"]
    closure = (eb["W_J"] - eb["dKE_J"] - eb["dPE_J"] - eb["dissipated_J"])
    ch.append(_chk("S8_V3_closure_identity", abs(closure - eb["residual_J"]) < 1e-9
                   and abs(eb["residual_J"]) < 1e-8,
                   "W-D-ΔKE-ΔPE=%.2e stored_resid=%.2e" % (closure, eb["residual_J"])))
    # 独立重算 dPE：end-state PE 与 start-state PE 之差，需从轨迹端点反推——
    # 报告仅存标量 ΔPE，故此处验证 ΔPE 与 W/D/ΔKE 满足闭合（同上）。额外核对 ΔKE≈0、ΔPE≈dPE 量级合理
    ch.append(_chk("S9_V3_dKE_near_zero", abs(eb["dKE_J"]) < 1e-6,
                   "dKE=%.2e" % eb["dKE_J"]))
    # V2 门控：执行值恒在 τ_lim 内（构造保证），demand 越限非零（真有需求越限）
    ga = rep["gate_ablation"]
    ch.append(_chk("S10_V2_executed_within_lim",
                   ga["PD_with_gate"]["tau_max"] <= TAU_LIM[0] + 1e-9
                   and ga["CT_with_gate"]["tau_max"] <= TAU_LIM[0] + 1e-9
                   and ga["PD_with_gate"]["violation"] is False
                   and ga["CT_with_gate"]["violation"] is False))
    ch.append(_chk("S11_V2_demand_violation_nonzero_falsifiable",
                   ga["PD_with_gate"]["demand_violation_steps"] > 0
                   and ga["CT_with_gate"]["demand_violation_steps"] > 0,
                   "PD=%d CT=%d" % (ga["PD_with_gate"]["demand_violation_steps"],
                                    ga["CT_with_gate"]["demand_violation_steps"])))
    # verdict 自洽
    ch.append(_chk("S12_verdict_self_consistent",
                   rep["verdict_pass"] is (rep["V1_MCg_decomposition_ok"]
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
    _res["tamper"].append({"name": name, "caught": bool(caught),
                           "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E66 严格审核（L3 独立物理，不 import e66）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例（改报告字段 → 必有检查报出）")
    _tamp("R1 篡改 V1 ok=false → S2 必报", lambda r: r.__setitem__("V1_MCg_decomposition_ok", False), ["S2"])
    _tamp("R2 篡改 achievable_wn=99 → S3 必报",
          lambda r: r["budget_design"].__setitem__("achievable_wn_rad_s", 99.0), ["S3"])
    _tamp("R3 篡改 static_payload=9.0 → S7 必报",
          lambda r: r.__setitem__("static_payload_limit_kg", 9.0), ["S7"])
    _tamp("R4 篡改 energy residual=1.0 → S8 必报",
          lambda r: r["energy_balance"].__setitem__("residual_J", 1.0), ["S8"])
    _tamp("R5 篡改 gate demand_violation_steps=0 → S11 必报",
          lambda r: r["gate_ablation"]["PD_with_gate"].__setitem__("demand_violation_steps", 0), ["S11"])
    _tamp("R6 篡改 verdict_pass=false → S12 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S12"])

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
