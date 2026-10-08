# -*- coding: utf-8 -*-
"""blind_replicate_e74.py —— **盲复刻（含重跑仿真）**：只给 e74 数据文件，从零重建实验。

只给一份 ``e74_tracking_bandwidth_report.json``，在**隔离目录**里（PYTHONPATH=""，不 import
任何仓库模块）：
  ① 从 bench 规格重建台架与场景（模型约定已在数据文件内）；
  ② **重跑仿真**抽样点（用报告里给出的增益），验证报告的扫频表是真跑出来的、不是编的；
  ③ 从报告的扫频表**独立重算**两条边界（无饱和 / 任意）；
  ④ 独立重算二阶闭式 −3 dB 与"口径因子"、E71 α 去额关系；
  ⑤ 判定数据文件是否**自足**。

自包含：只用 stdlib + numpy。产物：blind_replicate_e74.json
用法：python blind_replicate_e74.py [report.json]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "e74_tracking_bandwidth_report.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(REPORT)), "blind_replicate_e74.json")

_BANNED = ("e66", "e67", "e74", "planning", "control_strategies", "rne_dynamics")


def _guard():
    return sorted({m.split(".")[0] for m in list(sys.modules) if m.split(".")[0] in _BANNED})


def _num(x, nd=5):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ---------------- 独立重建：运动学 / 动力学 / 积分 ----------------
def ik(p, L1, L2, elbow=-1.0):
    x, y = float(p[0]), float(p[1])
    r = max(1e-9, math.hypot(x, y))
    c2 = max(-1.0, min(1.0, (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    q2 = elbow * math.acos(c2)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2])


def fk(q, L1, L2):
    return np.array([L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
                     L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1])])


def jac(q, L1, L2):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def l2p(P):
    """末端载荷参数化（数据文件 bench.payload.model 声明）。"""
    mp = float(P.get("mp", 0.0))
    m2 = P["M2"] + mp
    lc2 = (P["M2"] * P["LC2"] + mp * P["L2"]) / m2
    i2 = P["I2"] + P["M2"] * (lc2 - P["LC2"]) ** 2 + mp * (P["L2"] - lc2) ** 2
    return m2, lc2, i2


def mass_matrix(q, P):
    m2, lc2, i2 = l2p(P)
    c2 = math.cos(q[1])
    m11 = P["M1"] * P["LC1"] ** 2 + m2 * (P["L1"] ** 2 + lc2 ** 2 + 2 * P["L1"] * lc2 * c2) + P["I1"] + i2
    m12 = m2 * (lc2 ** 2 + P["L1"] * lc2 * c2) + i2
    m22 = m2 * lc2 ** 2 + i2
    return np.array([[m11, m12], [m12, m22]])


def coriolis(q, qd, P):
    m2, lc2, _ = l2p(P)
    h = -m2 * P["L1"] * lc2 * math.sin(q[1])
    return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2])


def gravity(q, P):
    m2, lc2, _ = l2p(P)
    g1 = (P["M1"] * P["LC1"] + m2 * P["L1"]) * P["G"] * math.cos(q[0]) + m2 * lc2 * P["G"] * math.cos(q[0] + q[1])
    g2 = m2 * lc2 * P["G"] * math.cos(q[0] + q[1])
    return np.array([g1, g2])


def deriv(st, tau, P):
    q, qd = st[:2], st[2:4]
    rhs = np.asarray(tau, float) - coriolis(q, qd, P) - gravity(q, P) - P["b_visc"] * qd
    return np.concatenate([qd, np.linalg.solve(mass_matrix(q, P), rhs)])


def rk4(st, tau, dt, P):
    k1 = deriv(st, tau, P)
    k2 = deriv(st + 0.5 * dt * k1, tau, P)
    k3 = deriv(st + 0.5 * dt * k2, tau, P)
    k4 = deriv(st + dt * k3, tau, P)
    return st + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def track(f, gains, sc, bench, amp_override=None):
    """独立重跑 E67-C 同款跟踪仿真（PD_G + 速度前馈 + 门控）；amp_override 供 F 段口径表使用。"""
    P = {"L1": bench["L1"], "L2": bench["L2"], "G": bench["G"],
         "M1": bench["link_params"]["M1"], "M2": bench["link_params"]["M2"],
         "LC1": bench["link_params"]["LC1"], "LC2": bench["link_params"]["LC2"],
         "I1": bench["link_params"]["I1"], "I2": bench["link_params"]["I2"],
         "b_visc": bench["b_visc"], "mp": 0.0}
    TAU = np.asarray(bench["TAU_LIM"], float)
    dt = bench["dt"]
    kp = np.asarray(gains[0], float)
    kd = np.asarray(gains[1], float)
    ctr = np.asarray(sc["track_center"], float)
    amp, T, settle, ff = (sc["amp_m"] if amp_override is None else amp_override), sc["T_s"], sc["settle_s"], sc["ff"]
    n = int(round(T / dt))
    ns = int(round(settle / dt))
    w = 2 * math.pi * f
    st = np.zeros(4)
    errs, viol = [], 0
    for i in range(n):
        t = i * dt
        p = ctr + amp * np.array([math.cos(w * t), 0.5 * math.sin(w * t)])
        p_dot = amp * np.array([-w * math.sin(w * t), 0.5 * w * math.cos(w * t)])
        q_ref = ik(p, bench["L1"], bench["L2"])
        if ff:
            qd_ref = np.linalg.solve(jac(q_ref, bench["L1"], bench["L2"]), p_dot)
            raw = kp * (q_ref - st[:2]) + kd * (qd_ref - st[2:]) + gravity(st[:2], P)
        else:
            raw = kp * (q_ref - st[:2]) + kd * (0.0 - st[2:]) + gravity(st[:2], P)
        if np.any(np.abs(raw) > TAU + 1e-9):
            viol += 1
        tau = np.clip(raw, -TAU, TAU)
        if i >= ns:
            errs.append(float(np.linalg.norm(fk(st[:2], bench["L1"], bench["L2"]) - p)))
        st = rk4(st, tau, dt, P)
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return {"rms_err_m": _num(rms), "sat_frac_demand": _num(viol / n), "n_steps": n}


def track_joint(pr, bench):
    """独立重跑 **E 段关节空间带载荷探针**（q_ref=q0+amp·[sin,sin]，mp 载荷）。"""
    lp = bench["link_params"]
    P = {"L1": bench["L1"], "L2": bench["L2"], "G": bench["G"], "b_visc": bench["b_visc"], "mp": pr["mp_kg"],
         "M1": lp["M1"], "M2": lp["M2"], "LC1": lp["LC1"], "LC2": lp["LC2"], "I1": lp["I1"], "I2": lp["I2"]}
    TAU = np.asarray(bench["TAU_LIM"], float)
    dt = bench["dt"]
    kp = np.asarray(pr["kp"], float)
    kd = np.asarray(pr["kd"], float)
    q0 = np.asarray(pr["q0"], float)
    amp, f, T = pr["amp_rad"], pr["freq_hz"], pr["T_s"]
    settle = bench["scenario"]["settle_s"]
    n, ns, w = int(round(T / dt)), int(round(settle / dt)), 2 * math.pi * f
    st = np.zeros(4)
    errs, traw = [], []
    for i in range(n):
        t = i * dt
        s, cs = math.sin(w * t), math.cos(w * t)
        q_ref = q0 + amp * np.array([s, s])
        qd_ref = amp * w * np.array([cs, cs])
        raw = kp * (q_ref - st[:2]) + kd * (qd_ref - st[2:]) + gravity(st[:2], P)
        traw.append(np.array(raw, float))
        tau = np.clip(raw, -TAU, TAU)
        if i >= ns:
            errs.append(float(np.linalg.norm(fk(st[:2], bench["L1"], bench["L2"])
                                             - fk(q_ref, bench["L1"], bench["L2"]))))
        st = rk4(st, tau, dt, P)
    ta = np.asarray(traw, float).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return {"rms_err_m": _num(rms), "n_steps": n,
            # ★ 逐关节量也独立重算：否则篡改这些新字段将无人可察
            "demand_peak_per_joint": [_num(np.max(np.abs(ta[:, 0]))), _num(np.max(np.abs(ta[:, 1])))],
            "violation_steps_per_joint": [int(np.sum(np.abs(ta[:, 0]) > TAU[0] + 1e-9)),
                                          int(np.sum(np.abs(ta[:, 1]) > TAU[1] + 1e-9))],
            "sat_frac_per_joint": [_num(np.mean(np.abs(ta[:, 0]) > TAU[0] + 1e-9), 5),
                                   _num(np.mean(np.abs(ta[:, 1]) > TAU[1] + 1e-9), 5)]}


def f3db_formula(wn, zeta):
    a = 1.0 - 2 * zeta * zeta
    return wn * math.sqrt(a + math.sqrt(a * a + 1.0)) / (2 * math.pi)


def boundary(rows, tol, unsat, sat_tol):
    ok = []
    for k, v in rows.items():
        if v["rms_err_m"] is None or v["diverged"]:
            continue
        if v["rms_err_m"] <= tol and (not unsat or v["sat_frac_demand"] <= sat_tol):
            ok.append(float(k))
    return round(max(ok), 1) if ok else None


def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    A, B, C, D = rep["A_freq_domain_bw"], rep["B_task_tolerance_bw"], rep["C_e67c_anchor"], rep["D_calibration"]
    sc = bench["scenario"]
    checks = []

    def chk(n, ok, d=""):
        checks.append({"name": n, "pass": bool(ok), "detail": str(d)})

    chk("B1_isolation_no_repo_modules", not _guard(), "loaded=%s" % _guard())

    spec = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params", "zeta",
            "wn_design", "b_visc", "scenario", "payload"]
    chk("B2_full_spec_present", all(k in bench for k in spec),
        "missing=%s" % [k for k in spec if k not in bench])

    # B3 **重跑仿真**抽样点：报告的扫频表须是真跑出来的，且**落盘精度足以逐位复现**
    #    判据：全部抽样点 |Δrms| ≤ 6e-6（= 报告 rms 5 位小数的舍入上限）。
    #    （此前的 4 位小数增益会在低增益行造成 ~2e-3 偏差 —— 已修为全精度落盘。）
    n_sim, n_exact, max_abs = 0, 0, 0.0
    for name, row in B["by_wn"].items():
        gains = (row["kp"], row["kd"])
        for f in (0.3, 0.5, 1.0):
            k = "%.1f" % f
            if k not in row["rows"] or row["rows"][k]["rms_err_m"] is None:
                continue
            r = track(f, gains, sc, bench)
            n_sim += 1
            dab = abs(r["rms_err_m"] - row["rows"][k]["rms_err_m"])
            max_abs = max(max_abs, dab)
            if dab <= 6e-6:
                n_exact += 1
    chk("B3_resimulate_sampled_points", n_sim >= 12 and n_exact == n_sim,
        "重跑 %d 点：逐位一致 %d，最大绝对偏差 %.2e（限 6e-6）" % (n_sim, n_exact, max_abs))

    # B4 从扫频表独立重算两条边界
    b_bad = []
    for name, row in B["by_wn"].items():
        for tk, pt in row["per_tol"].items():
            tol = float(tk.split("=")[1][:-1])
            for kind, unsat in (("any", False), ("unsaturated", True)):
                rb = boundary(row["rows"], tol, unsat, bench["sat_tol"])
                if rb != pt[kind]["hz"]:
                    b_bad.append((name, tk, kind, pt[kind]["hz"], rb))
    chk("B4_boundaries_recompute", not b_bad, "反例=%d %s" % (len(b_bad), b_bad[:2]))

    # B5 二阶闭式 + 口径因子
    a_bad = []
    ratios = []
    for name, v in A["by_wn"].items():
        fa = f3db_formula(v["wn_rad_s"], bench["zeta"])
        if abs(fa - v["analytic_hz"]) > 2e-3:
            a_bad.append((name, v["analytic_hz"], round(fa, 3)))
        if v["measured_hz"] is not None:
            ratios.append(v["measured_hz"] / fa)
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios)) if len(ratios) >= 2 else None
    chk("B5_analytic_bw_and_ratio", not a_bad and (spread is None or spread <= 0.30),
        "bad=%s ratio_range=[%.3f,%.3f] spread=%s" % (a_bad[:2], min(ratios) if ratios else 0,
                                                      max(ratios) if ratios else 0, _num(spread, 4) if spread else None))

    # B6 E71 α 去额关系
    exp = math.sqrt(bench["alpha_e71"])
    chk("B6_alpha_derate_consistent",
        abs(exp - D["derate_ratio_expected_sqrt_alpha"]) < 1e-3 and D["derate_rel_dev"] <= 0.02,
        "sqrt(alpha)=%.4f vs reported %.4f" % (exp, D["derate_ratio_expected_sqrt_alpha"]))

    # B7 锚点复核（E67-C 原增益）
    rowsC = C["rows"]

    def passes(f, tol):
        v = rowsC.get("%.1f" % f)
        return bool(v and v["rms_err_m"] is not None and v["rms_err_m"] <= tol)
    chk("B7_e67c_anchor_consistent",
        passes(0.2, 0.01) and (not passes(0.5, 0.01)) and passes(0.5, 0.02),
        "pass0.2(1cm)=%s fail0.5(1cm)=%s pass0.5(2cm)=%s" % (passes(0.2, 0.01), not passes(0.5, 0.01), passes(0.5, 0.02)))

    # B9 **E 段带载荷关节空间探针**重跑：rms **与逐关节量**都要对得上
    #    （只比 rms 会漏掉对 violation_steps_per_joint / demand_peak_per_joint 的篡改）
    e_bad, e_n, e_max = [], 0, 0.0
    for nm, pr in (rep.get("E_identification_probes", {}).get("probes") or {}).items():
        r = track_joint(pr, bench)
        e_n += 1
        dab = abs(r["rms_err_m"] - pr["rms_err_m"])
        e_max = max(e_max, dab)
        if dab > 6e-6:
            e_bad.append((nm, "rms", pr["rms_err_m"], r["rms_err_m"]))
        for j in (0, 1):
            if abs(r["demand_peak_per_joint"][j] - pr["demand_peak_per_joint"][j]) > 1e-4:
                e_bad.append((nm, "peak%d" % j, pr["demand_peak_per_joint"][j], r["demand_peak_per_joint"][j]))
            if r["violation_steps_per_joint"][j] != pr["violation_steps_per_joint"][j]:
                e_bad.append((nm, "viol%d" % j, pr["violation_steps_per_joint"][j], r["violation_steps_per_joint"][j]))
            if abs(r["sat_frac_per_joint"][j] - pr["sat_frac_per_joint"][j]) > 1e-5:
                e_bad.append((nm, "sat%d" % j, pr["sat_frac_per_joint"][j], r["sat_frac_per_joint"][j]))
    chk("B9_E_probes_resimulate", e_n >= 3 and not e_bad,
        "重跑 %d 个载荷探针 × {rms,逐关节峰值,逐关节越限步,逐关节饱和率}：最大|Δrms|=%.2e，反例=%d %s"
        % (e_n, e_max, len(e_bad), e_bad[:3]))

    # B10 **F 段（振幅口径）**重跑抽样点（振幅 ≠ 基准，验证该表亦真跑出来）
    f_n, f_bad, f_max = 0, [], 0.0
    for nm, v in (rep.get("F_amplitude_calibration", {}).get("by_wn") or {}).items():
        gains = (B["by_wn"][nm]["kp"], B["by_wn"][nm]["kd"])
        for amp in (bench["amp_sweep"][0], bench["amp_sweep"][-1]):
            pt = v["per_amp"]["amp=%.2fm" % amp]
            for fk in ("0.3", "0.6"):
                if fk not in pt["rows"] or pt["rows"][fk]["rms_err_m"] is None:
                    continue
                r = track(float(fk), gains, sc, bench, amp_override=amp)
                f_n += 1
                dab = abs(r["rms_err_m"] - pt["rows"][fk]["rms_err_m"])
                f_max = max(f_max, dab)
                if dab > 6e-6:
                    f_bad.append((nm, amp, fk, pt["rows"][fk]["rms_err_m"], r["rms_err_m"]))
    chk("B10_F_amplitude_resimulate", f_n >= 4 and not f_bad,
        "重跑 %d 个振幅口径点，最大绝对偏差 %.2e，反例=%d" % (f_n, f_max, len(f_bad)))

    self_contained = all(c["pass"] for c in checks)
    chk("B8_blind_self_contained", self_contained, "全项通过" if self_contained else "存在失败项")

    npass = sum(c["pass"] for c in checks)
    out = {"audit": "blind_replicate_e74", "input_file": os.path.basename(REPORT), "blind": True,
           "isolation": {"repo_modules_loaded": _guard(), "imports": ["stdlib", "numpy"]},
           "resimulated_points": n_sim,
           "n_pass": npass, "n_total": len(checks),
           "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("blind replicate e74: pass=%d/%d" % (npass, len(checks)))
    for c in checks:
        print("  [%s] %-36s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
    return out


if __name__ == "__main__":
    run()
