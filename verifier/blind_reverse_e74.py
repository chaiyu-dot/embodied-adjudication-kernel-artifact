# -*- coding: utf-8 -*-
"""blind_reverse_e74.py —— **外部逆向盲测**：连数据文件里声明的参数也不信，从数据反推并证伪。

只给 ``e74_tracking_bandwidth_report.json``，隔离运行（PYTHONPATH=""，不 import 任何仓库模块）：

  R2 **由报告增益反推设计 ωn**：ωn_i = sqrt(kp_i / M_ii(q_ref))，与声明的 wn_design 比；
  R3 **由 A 段扫频点反推 −3 dB 交越**，与 measured_hz 比；
  R4 **证伪式可识别性**：对每个声明参数施加 ±5% 扰动并**重跑仿真**（覆盖两种工况：
     EE 椭圆跟踪 + **带载荷的关节空间探针**），看数据能否分辨；
  R5 汇总"数据识别得出"vs"仅声明"；R5b 核验 G 的区制依赖；R6 规格自足；R7 基线复现。

自包含：只用 stdlib + numpy。产物：blind_reverse_e74.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "e74_tracking_bandwidth_report.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(REPORT)), "blind_reverse_e74.json")

_BANNED = ("e66", "e67", "e74", "planning")
REL_TOL = 1e-3


def _guard():
    return sorted({m.split(".")[0] for m in list(sys.modules) if m.split(".")[0] in _BANNED})


def _num(x, nd=6):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ---------------- 独立运动学 / 动力学（含末端载荷 mp） ----------------
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
    """末端载荷参数化（数据文件里由 bench.payload.model 声明）。"""
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
    rhs = np.asarray(tau, float) - coriolis(st[:2], st[2:4], P) - gravity(st[:2], P) - P["b_visc"] * st[2:4]
    return np.concatenate([st[2:4], np.linalg.solve(mass_matrix(st[:2], P), rhs)])


def rk4(st, tau, dt, P):
    k1 = deriv(st, tau, P)
    k2 = deriv(st + 0.5 * dt * k1, tau, P)
    k3 = deriv(st + 0.5 * dt * k2, tau, P)
    k4 = deriv(st + dt * k3, tau, P)
    return st + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def build_P(bench, ov=None, mp=None):
    lp = bench["link_params"]
    P = {"L1": bench["L1"], "L2": bench["L2"], "G": bench["G"], "b_visc": bench["b_visc"],
         "M1": lp["M1"], "M2": lp["M2"], "LC1": lp["LC1"], "LC2": lp["LC2"], "I1": lp["I1"], "I2": lp["I2"],
         "mp": 0.0}
    if mp is not None:
        P["mp"] = float(mp)
    if ov:
        P.update(ov)
    return P


def sim_rms(probe, bench, ov=None, ov_sc=None, ov_tau=None, ov_mp=None):
    """probe: {"mode","gains","f","q0","amp","T"}；mode ∈ {ee, joint}。"""
    P = build_P(bench, ov, ov_mp)
    sc = dict(bench["scenario"])
    if ov_sc:
        sc.update(ov_sc)
    dt = bench["dt"]
    tau_lim = np.asarray(bench["TAU_LIM"] if ov_tau is None else ov_tau, float)
    kp = np.asarray(probe["gains"][0], float)
    kd = np.asarray(probe["gains"][1], float)
    w = 2 * math.pi * probe["f"]
    T = probe.get("T", sc["T_s"])
    settle = sc["settle_s"]
    n, ns = int(round(T / dt)), int(round(settle / dt))
    st = np.zeros(4)
    errs = []
    ctr = np.asarray(sc["track_center"], float)
    q0 = np.asarray(probe.get("q0", sc["track_center"]), float)
    for i in range(n):
        t = i * dt
        s, cs = math.sin(w * t), math.cos(w * t)
        if probe["mode"] == "ee":
            p = ctr + sc["amp_m"] * np.array([math.cos(w * t), 0.5 * math.sin(w * t)])
            p_dot = sc["amp_m"] * np.array([-w * math.sin(w * t), 0.5 * w * math.cos(w * t)])
            q_ref = ik(p, P["L1"], P["L2"])
            qd_ref = np.linalg.solve(jac(q_ref, P["L1"], P["L2"]), p_dot)
            target = p
        else:
            q_ref = q0 + probe["amp"] * np.array([s, s])
            qd_ref = probe["amp"] * w * np.array([cs, cs])
            target = fk(q_ref, P["L1"], P["L2"])
        raw = kp * (q_ref - st[:2]) + kd * (qd_ref - st[2:]) + gravity(st[:2], P)
        if i >= ns:
            errs.append(float(np.linalg.norm(fk(st[:2], P["L1"], P["L2"]) - target)))
        st = rk4(st, np.clip(raw, -tau_lim, tau_lim), dt, P)
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def f3db_formula(wn, zeta):
    a = 1.0 - 2 * zeta * zeta
    return wn * math.sqrt(a + math.sqrt(a * a + 1.0)) / (2 * math.pi)


def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    A, B, E = rep["A_freq_domain_bw"], rep["B_task_tolerance_bw"], rep["E_identification_probes"]
    checks = []

    def chk(n, ok, d=""):
        checks.append({"name": n, "pass": bool(ok), "detail": str(d)})

    chk("R1_isolation_no_repo_modules", not _guard(), "loaded=%s" % _guard())

    # R2 由报告增益**反推设计 ωn**
    q_ref = np.asarray(bench["Q_REF"], float)
    Mq = mass_matrix(q_ref, build_P(bench))
    rec, bad2 = {}, []
    for name, row in B["by_wn"].items():
        wns = [math.sqrt(row["kp"][i] / float(Mq[i, i])) for i in range(2)]
        rec[name] = {"wn_from_joint0": _num(wns[0], 4), "wn_from_joint1": _num(wns[1], 4),
                     "declared": row["wn_rad_s"]}
        if abs(wns[0] - row["wn_rad_s"]) > 1e-3 or abs(wns[1] - row["wn_rad_s"]) > 1e-3:
            bad2.append(name)
    chk("R2_reverse_wn_from_gains", not bad2, "反推不一致=%s" % bad2)

    # R3 由 A 段扫频点**反推 −3 dB 交越**
    bad3 = []
    for name, v in A["by_wn"].items():
        pts = {float(k): val["gain_db"] for k, val in v["points"].items()}
        prev, est = None, None
        for f in sorted(pts):
            if pts[f] <= -3.0:
                if prev is None:
                    est = float(f)
                else:
                    f0, g0 = prev
                    tt = (g0 + 3.0) / (g0 - pts[f]) if abs(g0 - pts[f]) > 1e-12 else 0.0
                    est = float(f0 * (f / f0) ** max(0.0, min(1.0, tt)))
                break
            prev = (f, pts[f])
        if v["measured_hz"] is not None and (est is None or abs(est - v["measured_hz"]) > 2e-3):
            bad3.append((name, v["measured_hz"], None if est is None else round(est, 3)))
    chk("R3_reverse_f3db_from_points", not bad3, "反例=%s" % bad3[:3])

    # ---- 探针集：EE 椭圆（原场景）+ 带载荷关节空间（E 段，压到关节 1 上限）----
    probes, ref = [], {}
    for nm, f in (("e69_min", 0.3), ("e69_median", 0.3), ("e69_max", 0.3),
                  ("e69_max", 1.0), ("e69_max", 2.0)):
        p = {"mode": "ee", "gains": (B["by_wn"][nm]["kp"], B["by_wn"][nm]["kd"]), "f": f,
             "T": bench["scenario"]["T_s"], "tag": "ee:%s@%.1f" % (nm, f)}
        probes.append(p)
        ref[p["tag"]] = B["by_wn"][nm]["rows"]["%.1f" % f]["rms_err_m"]
    for nm, pr in E["probes"].items():
        p = {"mode": "joint", "gains": (pr["kp"], pr["kd"]), "f": pr["freq_hz"],
             "q0": pr["q0"], "amp": pr["amp_rad"], "T": pr["T_s"], "mp": pr["mp_kg"],
             "tag": "E:%s" % nm}
        probes.append(p)
        ref[p["tag"]] = pr["rms_err_m"]

    def dev(ov=None, ov_sc=None, ov_tau=None, ov_mp=None, subset=None):
        """返回 (最大绝对偏差, 最大"落地相对"偏差)；相对分母设 1 cm 地板（免小 rms 行被舍入放大）。"""
        ab, rl = 0.0, 0.0
        for p in (subset if subset is not None else probes):
            mp = p.get("mp") if ov_mp is None else ov_mp
            rr = sim_rms(p, bench, ov, ov_sc, ov_tau, mp)
            r0 = ref[p["tag"]]
            ab = max(ab, abs(rr - r0))
            rl = max(rl, abs(rr - r0) / max(abs(r0), 0.01))
        return ab, rl

    # R4 证伪式可识别性
    ident = {}
    for k in ("L1", "L2", "G", "M1", "M2", "LC1", "LC2", "I1", "I2", "b_visc"):
        base = bench[k] if k in bench else bench["link_params"][k]
        dv = 0.0
        for eps in (0.05, -0.05):
            dv = max(dv, dev(ov={k: base * (1 + eps)})[1])
        ident[k] = dv
    for i in (0, 1):
        dv = 0.0
        for eps in (0.05, -0.05):
            tl = list(bench["TAU_LIM"])
            tl[i] = tl[i] * (1 + eps)
            dv = max(dv, dev(ov_tau=tl)[1])
        ident["TAU_LIM[%d]" % i] = dv
    mp0 = bench["payload"]["mp_probe_kg"]
    dv = 0.0
    for eps in (0.05, -0.05):
        dv = max(dv, dev(ov_mp=mp0 * (1 + eps))[1])
    ident["mp_kg"] = dv
    for k in ("amp_m", "track_center"):
        dv = 0.0
        for eps in (0.05, -0.05):
            if k == "amp_m":
                dv = max(dv, dev(ov_sc={"amp_m": bench["scenario"]["amp_m"] * (1 + eps)})[1])
            else:
                c = list(bench["scenario"]["track_center"])
                dv = max(dv, dev(ov_sc={"track_center": [c[0] * (1 + eps), c[1]]})[1])
        ident[k] = dv

    identified = sorted(k for k, v in ident.items() if v > REL_TOL)
    not_identified = sorted(k for k, v in ident.items() if v <= REL_TOL)
    chk("R4_identifiability_computed", len(ident) >= 14,
        "n=%d identified=%d not_identified=%s" % (len(ident), len(identified), not_identified))

    ab0, rl0 = dev()
    chk("R5_identifiability_characterized", len(identified) >= 14,
        "识别得出 %d / 仅声明 %s" % (len(identified), not_identified if not_identified else "无"))

    # R5b 机制：G 在**未饱和/线性**区制被精确重力前馈相消 → 不可观测；**饱和**后暴露
    base_G = bench["G"]
    unsat = [p for p in probes if p["tag"] == "ee:e69_min@0.3"]
    g_unsat = max(dev(ov={"G": base_G * (1 + e)}, subset=unsat)[1] for e in (0.05, -0.05))
    chk("R5b_gravity_revealed_by_saturation",
        (g_unsat <= max(2.0 * rl0, 1e-4)) and ident["G"] > REL_TOL,
        "G 敏感度：未饱和探针=%.2e（≈基线 %.2e）｜全探针=%.2e" % (g_unsat, rl0, ident["G"]))

    # R6 规格自足
    spec = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params", "zeta", "wn_design",
            "b_visc", "scenario", "Q_REF", "payload"]
    miss = [k for k in spec if k not in bench]
    chk("R6_full_spec_present", not miss, "missing=%s" % miss)

    # R7 基线复现
    chk("R7_baseline_reproduces", ab0 <= 6e-6, "基线最大绝对偏差=%.2e（限 6e-6=报告 5 位舍入）" % ab0)

    npass = sum(c["pass"] for c in checks)
    out = {"audit": "blind_reverse_e74", "input_file": os.path.basename(REPORT), "blind": True, "reverse": True,
           "rel_tol": REL_TOL, "isolation": {"repo_modules_loaded": _guard(), "imports": ["stdlib", "numpy"]},
           "probe_count": len(probes), "probe_tags": [p["tag"] for p in probes],
           "reverse_inferred_wn": rec,
           "identifiability": {k: {"sensitivity": _num(v), "identified": bool(v > REL_TOL)}
                               for k, v in sorted(ident.items())},
           "identified_params": identified, "not_identified_params": not_identified,
           "n_pass": npass, "n_total": len(checks), "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("blind-reverse e74: pass=%d/%d | identified=%d/%d not_identified=%s"
          % (npass, len(checks), len(identified), len(ident), not_identified))
    for c in checks:
        print("  [%s] %-36s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
    return out


if __name__ == "__main__":
    run()
