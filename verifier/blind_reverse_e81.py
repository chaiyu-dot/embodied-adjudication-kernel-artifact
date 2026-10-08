# -*- coding: utf-8 -*-
"""blind_reverse_e81.py —— E81 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 e81_dynamic_speed_ceiling）。独立重算三者，
并以敏感性与负对照证明 min/绑定逻辑有分辨力（非同义反复）：
  · 冲量随 m_payload 变化 → 绑定会切换（说明 impulse 在 m=2.0 口径确为最紧，非冻结）
  · 科氏随 τ_lim[1] 单调 → v_coriolis 实变（说明科氏扫掠是真实物理，非写死数）
  · 最坏构型在 v_coriolis 处 joint2 饱和（τ2≈τ_lim[1]）= 饱和签名
  · 丢掉动态天花板（直接用运动学）会给出危险值 2.957 > 冲量上限 → 证明动态天花板确有必要

产物：blind_reverse_e81.json
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
# ★ 刻意不把 EVAL 加入 sys.path，使 e81_dynamic_speed_ceiling 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e81_dynamic_speed_ceiling_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e81.json")

_res = {"experiment": "E81 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e81_dynamic_speed_ceiling", "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _jac(q, L1, L2):
    q1, q2 = q[0], q[1]
    return np.array([[-L1 * math.sin(q1) - L2 * math.sin(q1 + q2), -L2 * math.sin(q1 + q2)],
                     [L1 * math.cos(q1) + L2 * math.cos(q1 + q2), L2 * math.cos(q1 + q2)]], float)


def _tau_at(L1, L2, M1, M2, g, q, v_ee, d, sig_min):
    J = _jac(q, L1, L2)
    sm = np.linalg.svd(J, compute_uv=False)[-1]
    if sm < sig_min:
        return None
    qd = np.linalg.pinv(J) @ (v_ee * np.asarray(d, float))
    t1, t2 = RNE.planar_2r_torque_closedform(L1, L2, M1, M2, q[0], q[1], qd[0], qd[1], 0.0, 0.0, g=g)
    return np.array([t1, t2], float)


def _bin(L1, L2, M1, M2, g, q, d, tl, sig_min):
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        tau = _tau_at(L1, L2, M1, M2, g, q, mid, d, sig_min)
        if tau is None:
            return None
        if abs(tau[0]) <= tl[0] and abs(tau[1]) <= tl[1]:
            lo = mid
        else:
            hi = mid
    return lo


def _sweep_coriolis(L1, L2, M1, M2, g, tl, sig_min, n_q1, n_q2, n_dir):
    q1s = np.linspace(-math.pi, math.pi, n_q1, endpoint=False)
    q2s = np.linspace(-math.pi, math.pi, n_q2, endpoint=False)
    dirs = [np.array([math.cos(t), math.sin(t)], float)
            for t in np.linspace(0, 2 * math.pi, n_dir, endpoint=False)]
    vmin = float("inf")
    worst = None
    nvalid = 0
    for q1 in q1s:
        for q2 in q2s:
            q = np.array([q1, q2], float)
            for d in dirs:
                vc = _bin(L1, L2, M1, M2, g, q, d, tl, sig_min)
                if vc is None:
                    continue
                nvalid += 1
                if vc < vmin:
                    vmin = vc
                    worst = (np.array([q1, q2], float), np.asarray(d, float))
    return vmin, nvalid, worst


def main():
    print("=" * 88)
    print("E81 逆向盲测（L5 隔离，独立重算 + 敏感性/负对照证伪分辨力）")
    print("=" * 88)
    chk("R0_isolation_e81_not_imported",
        "e81_dynamic_speed_ceiling" not in sys.modules,
        "sys.modules 含 e81=%s" % ("e81_dynamic_speed_ceiling" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    L1, L2, M1, M2, g = bench["L1"], bench["L2"], bench["M1"], bench["M2"], bench["g"]
    tl = bench["TAU_LIM"]
    sig_min = bench["sigma_min_min"]

    A, B, C = rep["A_kinematic"], rep["B_impulse"], rep["C_coriolis"]

    # --- R1：v_kinematic 确为 E73 工作空间包络上限（独立确认）---
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin_e73 = float(r73["B_envelope_curve"]["v_max_on_path"])
    chk("R1_reverse_kinematic_cap_from_E73", abs(A["v_kinematic_mps"] - v_kin_e73) < 1e-9,
        "stored=%.4f e73=%.4f" % (A["v_kinematic_mps"], v_kin_e73))

    # --- R2：v_impulse 从冲量-动量第一性原理重算 + 敏感性（m→绑定切换）---
    v_imp_re = bench["F_allow_N"] * bench["dt_contact_s"] / ((1.0 + bench["restitution"]) * bench["m_payload_kg"])
    v_imp_m1 = bench["F_allow_N"] * bench["dt_contact_s"] / ((1.0 + bench["restitution"]) * 1.0)
    # m=1.0 → v_impulse=3.75 > v_coriolis=2.133 → 绑定应切到 coriolis（证明 impulse 在 m=2.0 确为最紧）
    bind_m1 = "coriolis" if min(v_kin_e73, v_imp_m1, C["v_coriolis_mps"]) == C["v_coriolis_mps"] else "impulse"
    chk("R2_reverse_impulse_and_mass_sensitivity", abs(v_imp_re - B["v_impulse_mps"]) < 1e-9
        and v_imp_m1 > C["v_coriolis_mps"] and bind_m1 == "coriolis",
        "v_imp=%.4f v_imp(m=1)=%.4f v_cor=%.4f bind@m1=%s" % (v_imp_re, v_imp_m1, C["v_coriolis_mps"], bind_m1))

    # --- R3：v_coriolis 独立扫掠重算 + 饱和签名（worst_config 处 joint2 触 τ_lim[1]）---
    v_cor_re, nvalid_re, worst_re = _sweep_coriolis(L1, L2, M1, M2, g, tl, sig_min,
                                                    bench["n_q1"], bench["n_q2"], bench["n_dir"])
    wc = C["worst_config"]
    tau_cap = _tau_at(L1, L2, M1, M2, g, np.array(wc["q"], float), C["v_coriolis_mps"],
                      np.array(wc["dir"], float), sig_min)
    sat_ok = (tau_cap is not None) and (abs(tau_cap[1]) >= tl[1] - 1e-2)
    chk("R3_reverse_coriolis_sweep_and_saturation_signature",
        abs(v_cor_re - C["v_coriolis_mps"]) < 1e-2 and nvalid_re == C["n_valid_configs_dirs"] and sat_ok,
        "v_cor_re=%.4f stored=%.4f nvalid=%d tau2=%.3f tl1=%.2f"
        % (v_cor_re, C["v_coriolis_mps"], nvalid_re, tau_cap[1] if tau_cap is not None else -1, tl[1]))

    # --- R4：科氏扫掠随 τ_lim[1] 单调（降限更紧、升限更松）→ 真实物理非冻结 ---
    v_lo = _sweep_coriolis(L1, L2, M1, M2, g, [tl[0], max(0.5, tl[1] - 0.8)], sig_min,
                           bench["n_q1"], bench["n_q2"], bench["n_dir"])[0]
    v_hi = _sweep_coriolis(L1, L2, M1, M2, g, [tl[0], tl[1] + 1.2], sig_min,
                           bench["n_q1"], bench["n_q2"], bench["n_dir"])[0]
    chk("R4_coriolis_monotone_in_tau_lim", v_lo < C["v_coriolis_mps"] - 1e-3 and v_hi > C["v_coriolis_mps"] + 1e-3,
        "v@tl2-=%.4f v@base=%.4f v@tl2+=%.4f" % (v_lo, C["v_coriolis_mps"], v_hi))

    # --- R5：绑定逻辑分辨力 + 负对照（丢掉动态天花板→危险值）---
    v_kin, v_imp, v_cor = A["v_kinematic_mps"], B["v_impulse_mps"], C["v_coriolis_mps"]
    # 扰动 v_impulse +1.0（2.875，仍 < kinematic 2.957 但 > coriolis 2.133）→ 绑定应切到 coriolis
    v_imp_pert = v_imp + 1.0
    v_dyn_pert = min(v_kin, v_imp_pert, v_cor)
    bind_pert = "coriolis" if v_dyn_pert == v_cor else ("impulse" if v_dyn_pert == v_imp_pert else "kinematic")
    # 负对照：丢掉 min，直接用运动学 → 2.957 越过冲量上限（危险）
    negctrl_danger = v_kin > v_imp + 1e-9
    chk("R5_binding_logic_discriminating_and_negative_control",
        bind_pert == "coriolis" and abs(v_dyn_pert - v_cor) < 1e-9 and negctrl_danger,
        "bind@pert=%s v_dyn_pert=%.4f kinematic_only=%.4f impulse=%.4f danger=%s"
        % (bind_pert, v_dyn_pert, v_kin, v_imp, negctrl_danger))

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
