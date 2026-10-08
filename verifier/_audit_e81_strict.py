# -*- coding: utf-8 -*-
"""_audit_e81_strict.py —— E81 的**严格审核（L3，独立物理）**。

不 import 任何 e81 模块；独立推导平面 2R 雅可比 + 用**参考实现** `rne_dynamics`（闭式）重算
科氏/力矩饱和扫掠，并交叉验证运动学上限（E73 工作空间包络）、冲量上限（E79 同式重算）、
动态天花板 min(三者) 与绑定约束；并做篡改用例。

产物：_audit_e81_strict.json
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

REPORT = os.path.join(EVAL, "e81_dynamic_speed_ceiling_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
E79_REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
OUT = os.path.join(EVAL, "_audit_e81_strict.json")

_res = {"experiment": "E81 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写雅可比 + rne_dynamics(参考)，不 import e81；E73/E79 作外部数据交叉",
        "checks": [], "tamper": []}


def _jac(q, L1, L2):
    q1, q2 = q[0], q[1]
    return np.array([[-L1 * math.sin(q1) - L2 * math.sin(q1 + q2), -L2 * math.sin(q1 + q2)],
                     [L1 * math.cos(q1) + L2 * math.cos(q1 + q2), L2 * math.cos(q1 + q2)]], float)


def _tau_at(L1, L2, M1, M2, g, q, v_ee, d, sigma_min_min):
    J = _jac(q, L1, L2)
    sm = np.linalg.svd(J, compute_uv=False)[-1]
    if sm < sigma_min_min:
        return None
    qd = np.linalg.pinv(J) @ (v_ee * np.asarray(d, float))
    t1, t2 = RNE.planar_2r_torque_closedform(L1, L2, M1, M2, q[0], q[1], qd[0], qd[1], 0.0, 0.0, g=g)
    return np.array([t1, t2], float)


def _bin(L1, L2, M1, M2, g, q, d, tl, sigma_min_min):
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        tau = _tau_at(L1, L2, M1, M2, g, q, mid, d, sigma_min_min)
        if tau is None:
            return None
        if abs(tau[0]) <= tl[0] and abs(tau[1]) <= tl[1]:
            lo = mid
        else:
            hi = mid
    return lo


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    b = rep["bench"]
    A, B, C = rep["A_kinematic"], rep["B_impulse"], rep["C_coriolis"]
    L1, L2, M1, M2, g = b["L1"], b["L2"], b["M1"], b["M2"], b["g"]
    tl = b["TAU_LIM"]

    # --- S1：运动学上限 == E73 工作空间包络（外部数据交叉）---
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin_e73 = float(r73["B_envelope_curve"]["v_max_on_path"])
    ch.append(_chk("S1_kinematic_matches_E73", abs(A["v_kinematic_mps"] - v_kin_e73) < 1e-9
                   and A["source"].startswith("E73"), "stored=%.4f e73=%.4f" % (A["v_kinematic_mps"], v_kin_e73)))

    # --- S2：冲量上限独立重算 + 与 E79 存储一致 ---
    v_imp_re = b["F_allow_N"] * b["dt_contact_s"] / ((1.0 + b["restitution"]) * b["m_payload_kg"])
    r79 = json.load(open(E79_REPORT, encoding="utf-8"))
    v_imp_e79 = float(r79["C_verdict_coupling"]["rows"]["_v_star_impulse_mps"])
    ch.append(_chk("S2_impulse_recomputed_and_matches_E79", abs(v_imp_re - B["v_impulse_mps"]) < 1e-9
                   and abs(v_imp_re - v_imp_e79) < 1e-9 and abs(B["abs_err_vs_e79"]) < 1e-9,
                   "recomp=%.6f stored=%.6f e79=%.6f" % (v_imp_re, B["v_impulse_mps"], v_imp_e79)))

    # --- S3：科氏/力矩饱和上限 —— 独立重算扫掠 + 饱和签名验证 ---
    q1s = np.linspace(-math.pi, math.pi, b["n_q1"], endpoint=False)
    q2s = np.linspace(-math.pi, math.pi, b["n_q2"], endpoint=False)
    dirs = [np.array([math.cos(t), math.sin(t)], float)
            for t in np.linspace(0, 2 * math.pi, b["n_dir"], endpoint=False)]
    vmin = float("inf"); nvalid = 0
    for q1 in q1s:
        for q2 in q2s:
            q = np.array([q1, q2], float)
            for d in dirs:
                vc = _bin(L1, L2, M1, M2, g, q, d, tl, b["sigma_min_min"])
                if vc is None:
                    continue
                nvalid += 1
                if vc < vmin:
                    vmin = vc
    # 饱和签名：在存储 v_coriolis 处，worst_config 的 τ 应贴合 τ_lim（joint2 绑）
    wc = C["worst_config"]
    tau_at_cap = _tau_at(L1, L2, M1, M2, g, np.array(wc["q"], float), C["v_coriolis_mps"],
                         np.array(wc["dir"], float), b["sigma_min_min"])
    sat_ok = (tau_at_cap is not None) and (abs(tau_at_cap[1]) >= tl[1] - 1e-2)
    ch.append(_chk("S3_coriolis_recomputed_and_saturation_signature",
                   abs(vmin - C["v_coriolis_mps"]) < 1e-2 and nvalid > 0
                   and nvalid == C["n_valid_configs_dirs"] and sat_ok,
                   "recomp=%.4f stored=%.4f nvalid=%d sat_tau2=%.3f tl1=%.2f"
                   % (vmin, C["v_coriolis_mps"], nvalid, tau_at_cap[1] if tau_at_cap is not None else -1, tl[1])))

    # --- S4：动态天花板 = min(三者) + 绑定约束 ---
    v_dyn_re = min(A["v_kinematic_mps"], B["v_impulse_mps"], C["v_coriolis_mps"])
    bind_re = ("kinematic" if v_dyn_re == A["v_kinematic_mps"]
               else "impulse" if v_dyn_re == B["v_impulse_mps"] else "coriolis")
    ch.append(_chk("S4_dynamic_cap_and_binding", abs(v_dyn_re - rep["v_dynamic_cap_mps"]) < 1e-9
                   and bind_re == rep["binding_constraint"]
                   and rep["v_dynamic_cap_mps"] < A["v_kinematic_mps"] - 1e-9,
                   "recomp=%.4f stored=%.4f bind=%s" % (v_dyn_re, rep["v_dynamic_cap_mps"], bind_re)))

    # --- S5：verdict 自洽 ---
    cr = rep["criteria"]
    ch.append(_chk("S5_verdict_self_consistent",
                   rep["verdict_pass"] is (cr["C1_dynamic_tighter_than_kinematic"]
                                          and cr["C2_impulse_matches_e79"]
                                          and cr["C3_coriolis_sweep_converged"])
                   and rep["verdict_pass"] is True))
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
    print("E81 严格审核（L3 独立物理，不 import e81）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 A v_kinematic=9.9 → S1 必报", lambda r: r["A_kinematic"].__setitem__("v_kinematic_mps", 9.9), ["S1"])
    _tamp("R2 篡改 B v_impulse=9.9 → S2 必报", lambda r: r["B_impulse"].__setitem__("v_impulse_mps", 9.9), ["S2"])
    _tamp("R3 篡改 C v_coriolis=9.9 → S3 必报", lambda r: r["C_coriolis"].__setitem__("v_coriolis_mps", 9.9), ["S3"])
    _tamp("R4 篡改 v_dynamic_cap=9.9 → S4 必报", lambda r: r.__setitem__("v_dynamic_cap_mps", 9.9), ["S4"])
    _tamp("R5 篡改 verdict_pass=False → S5 必报", lambda r: r.__setitem__("verdict_pass", False), ["S5"])

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
