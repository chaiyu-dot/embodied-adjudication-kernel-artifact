# -*- coding: utf-8 -*-
"""blind_replicate_e73.py —— **盲复刻**：只给一个数据文件，从零反推 E73 实验并复跑。

用途（第三方"盲测"口径）：把本文件 + ``e73_velocity_spectrum_report.json`` 拷到一个**干净目录**，
``PYTHONPATH=""`` 下运行——**不 import 任何仓库模块、不读落地记忆、不碰作者源码**。
复刻者只有"数据文件"一份输入，必须：

  ① 从 data 里读出实验台架参数；
  ② **提出假设并检验**（如 v_max 是否 = 关节速度上限 × σ_max(J)），而非照抄作者结论；
  ③ 用第一性原理从零重算每一个可从数据文件重建的量；
  ④ **判定数据文件是否自足**——哪些量单凭该文件无法重建（缺什么参数），如实报告。

不 import 作者模块；只用 stdlib + numpy。产物：blind_replicate_e73.json

用法：python blind_replicate_e73.py [report.json]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "e73_velocity_spectrum_report.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(REPORT)), "blind_replicate_e73.json")

# 盲区守卫：本次运行**不得**加载任何仓库模块
_BANNED = ("e66", "e67", "e73", "planning", "control_strategies", "rne_dynamics")


def _guard():
    return sorted({m.split(".")[0] for m in list(sys.modules)
                   if m.split(".")[0] in _BANNED})


def _num(x, nd=4):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ---------------- 纯第一性原理重建（仅依赖台架参数） ----------------
def jac(q, L1, L2):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def fk(q, L1, L2):
    return (L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
            L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1]))


def svd2(J):
    s = np.linalg.svd(J, compute_uv=False)
    return float(s[0]), float(s[1])


def mass_matrix(q, P):
    c2 = math.cos(q[1])
    m11 = P["M1"] * P["LC1"] ** 2 + P["M2"] * (P["L1"] ** 2 + P["LC2"] ** 2 + 2 * P["L1"] * P["LC2"] * c2) + P["I1"] + P["I2"]
    m12 = P["M2"] * (P["LC2"] ** 2 + P["L1"] * P["LC2"] * c2) + P["I2"]
    m22 = P["M2"] * P["LC2"] ** 2 + P["I2"]
    return np.array([[m11, m12], [m12, m22]])


def coriolis(q, qd, P):
    h = -P["M2"] * P["L1"] * P["LC2"] * math.sin(q[1])
    return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2])


def gravity(q, P):
    g1 = (P["M1"] * P["LC1"] + P["M2"] * P["L1"]) * P["G"] * math.cos(q[0]) + P["M2"] * P["LC2"] * P["G"] * math.cos(q[0] + q[1])
    g2 = P["M2"] * P["LC2"] * P["G"] * math.cos(q[0] + q[1])
    return np.array([g1, g2])


def tau_at(q, qd, qdd, P):
    return mass_matrix(q, P) @ np.asarray(qdd, float) + coriolis(q, np.asarray(qd, float), P) + gravity(q, P)


def max_accel(q, u, P):
    u = np.asarray(u, float)
    TAU = np.array(P["TAU_LIM"], float)

    def ok(s):
        return bool(np.all(np.abs(tau_at(q, [0.0, 0.0], s * u, P)) <= TAU + 1e-12))
    lo, hi = 1e-9, 1e4
    if not ok(lo):
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def qdd_cap(q, P):
    M = mass_matrix(q, P)
    return [P["TAU_LIM"][i] / float(M[i, i]) for i in range(2)]


def max_accel_capped(q, u, cap, P):
    u = np.asarray(u, float)
    TAU = np.array(P["TAU_LIM"], float)

    def ok(s):
        if not np.all(np.abs(s * u) <= np.asarray(cap, float) + 1e-12):
            return False
        return bool(np.all(np.abs(tau_at(q, [0.0, 0.0], s * u, P)) <= TAU + 1e-12))
    lo, hi = 1e-9, 1e4
    if not ok(lo):
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def budget_wn(P):
    kp = [0.9 * P["TAU_LIM"][i] / max(abs(P["Q_REF"][i]), 1e-9) for i in range(2)]
    M = mass_matrix(P["Q_REF"], P)
    return min(math.sqrt(kp[i] / float(M[i, i])) for i in range(2))


# ---------------- 主流程 ----------------
def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep.get("bench", {})
    L1, L2 = bench.get("L1"), bench.get("L2")
    cap = bench.get("qdot_cap")
    TAU = bench.get("TAU_LIM")
    A, B, C, D = (rep["A_polar_spectrum"], rep["B_envelope_curve"],
                  rep["C_workspace_map"], rep["D_calibration"])
    checks = []

    def chk(n, ok, d=""):
        checks.append({"name": n, "pass": bool(ok), "detail": str(d)})

    # B1 隔离：未加载任何仓库模块
    chk("B1_isolation_no_repo_modules", not _guard(), "loaded=%s" % _guard())

    # B2 台架参数解析：运动学层所需
    kin_need = ["L1", "L2", "G", "TAU_LIM", "qdot_cap", "Q_REF"]
    kin_have = [k for k in kin_need if k in bench]
    chk("B2_bench_kinematic_present", all(k in bench for k in ("L1", "L2", "qdot_cap")),
        "have=%s missing=%s" % (kin_have, [k for k in kin_need if k not in bench]))

    # B2b 完整规格：运动学 + 惯性 + **模型约定** + 积分步长（缺任一项即"不自足"）
    spec_need = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params"]
    spec_missing = [k for k in spec_need if k not in bench]
    chk("B2b_full_spec_present", not spec_missing, "missing=%s" % spec_missing)

    # B3 假设检验 H1：v_max == qdot_cap · σ_max(J(q))（速度包络=速度上限×最大奇异值）
    nd = C["dirs"]
    tol_of = lambda sg: cap * sg * (1.0 - math.cos(math.pi / nd)) + 2e-4
    h1_bad = []
    for cell in C["rows"]:
        q = cell["q"]
        smax, _ = svd2(jac(q, L1, L2))
        v_hi = cap * smax
        if not (v_hi + 1e-9 >= cell["v_max"] >= v_hi - tol_of(smax)):
            h1_bad.append((cell["q"], cell["v_max"], _num(v_hi)))
    chk("B3_hypothesis_vmax_eq_cap_sigma_max", not h1_bad,
        "反例数=%d %s" % (len(h1_bad), h1_bad[:3]))

    # B4 A 段 σ 与奇异判据独立重算
    sig_bad = []
    for cfg in A["configs"]:
        smax, smin = svd2(jac(cfg["q"], L1, L2))
        if abs(smax - cfg["sigma_max"]) > 1e-4 or abs(smin - cfg["sigma_min"]) > 1e-6:
            sig_bad.append((cfg["q"], cfg["sigma_max"], _num(smax), cfg["sigma_min"], _num(smin, 6)))
        sing = smin <= 1e-3 * smax
        if sing != cfg["is_singular"]:
            sig_bad.append((cfg["q"], "singular flag", cfg["is_singular"], sing))
    chk("B4_sigma_and_singularity", not sig_bad, "反例=%s" % sig_bad[:3])

    # B5 A 段极坐标速度谱逐方向独立重算（v_ee = cap·|J u|）
    pol_bad = []
    cfg0 = A["configs"][0]
    J0 = jac(cfg0["q"], L1, L2)
    for row in cfg0["rows"][:12]:
        th = math.radians(row["deg"])
        u = np.array([math.cos(th), math.sin(th)])
        v = cap * float(np.linalg.norm(J0 @ u))
        if abs(v - row["v_ee"]) > tol_of(float(np.linalg.norm(J0 @ u))) + 1e-4:
            pol_bad.append((row["deg"], row["v_ee"], _num(v)))
    chk("B5_polar_v_recompute", not pol_bad, "反例=%s" % pol_bad[:3])

    # B6 路径几何（x,y = fk(q)）独立重算
    geo_bad = []
    for pt in B["curve"]:
        x, y = fk(pt["q"], L1, L2)
        if abs(x - pt["x"]) > 1e-3 or abs(y - pt["y"]) > 1e-3:
            geo_bad.append((pt["q"], (pt["x"], pt["y"]), (_num(x), _num(y))))
    chk("B6_path_geometry_recompute", not geo_bad, "反例=%d %s" % (len(geo_bad), geo_bad[:2]))

    # B7 路径 v_max 独立重算（= cap·σ_max）
    pv_bad = []
    for pt in B["curve"]:
        smax, _ = svd2(jac(pt["q"], L1, L2))
        v_hi = cap * smax
        if not (v_hi + 1e-9 >= pt["v_max"] >= v_hi - tol_of(smax)):
            pv_bad.append((pt["q"], pt["v_max"], _num(v_hi)))
    chk("B7_path_v_recompute", not pv_bad, "反例=%d %s" % (len(pv_bad), pv_bad[:2]))

    # B8 统计（从 rows 重算）
    v = [c["v_max"] for c in C["rows"]]
    st_ok = (round(min(v), 4) == C["v_min"] and round(max(v), 4) == C["v_max"]
             and round(float(np.median(v)), 4) == C["v_median"])
    chk("B8_stats_recompute", st_ok, "v=%s/%s/%s" % (C["v_min"], C["v_median"], C["v_max"]))

    # ---- B9/B10 动力学层自足性判定 ----
    P_keys = ["M1", "M2", "LC1", "LC2", "I1", "I2"]
    link = bench.get("link_params", {}) if isinstance(bench.get("link_params"), dict) else {}
    have_p = [k for k in P_keys if k in link]
    missing_p = [k for k in P_keys if k not in link]
    dyn_status = "self_contained" if not missing_p else "UNDER_DETERMINED"
    chk("B9_dynamic_layer_self_contained", not missing_p,
        "status=%s missing_inertial_params=%s" % (dyn_status, missing_p))

    dyn_ok = False
    dyn_detail = "数据文件未含惯性参数 → a_max / a_max_diag / jerk / torque_only 无法仅凭该文件重建"
    if not missing_p:
        P = {"L1": L1, "L2": L2, "G": bench["G"], "TAU_LIM": TAU, "Q_REF": bench["Q_REF"],
             "M1": link["M1"], "M2": link["M2"], "LC1": link["LC1"], "LC2": link["LC2"],
             "I1": link["I1"], "I2": link["I2"]}
        ang = np.linspace(0.0, 2 * math.pi, nd, endpoint=False)
        uvs = [np.array([math.cos(t), math.sin(t)]) for t in ang]
        a_mis = ad_mis = 0
        for cell in C["rows"]:
            q = cell["q"]
            J = jac(q, L1, L2)
            ba = max(float(np.linalg.norm(J @ (max_accel(q, u, P) * u))) for u in uvs)
            capq = qdd_cap(q, P)
            bad = max(float(np.linalg.norm(J @ (max_accel_capped(q, u, capq, P) * u))) for u in uvs)
            if abs(ba - cell["a_max"]) > 5e-2:
                a_mis += 1
            if abs(bad - cell["a_max_diag"]) > 5e-2:
                ad_mis += 1
        wn = budget_wn(P)
        jerk_ok = abs(D["wn_nominal"] - round(wn, 3)) < 1e-2
        dyn_ok = (a_mis == 0 and ad_mis == 0 and jerk_ok)
        dyn_detail = "a_mis=%d ad_mis=%d wn=%.3f(report %s)" % (a_mis, ad_mis, wn, D["wn_nominal"])
    chk("B10_dynamic_layer_recompute", dyn_ok, dyn_detail)

    # B11 综合："仅凭数据文件"自足性判定（运动学层 + 动力学层）
    # 注意：用**精确名**集合，禁用前缀匹配（"B1" 会误吞 "B10" —— 前缀碰撞，与 whitelist 漏更同类）
    KIN_NAMES = {"B1_isolation_no_repo_modules", "B2_bench_kinematic_present",
                 "B2b_full_spec_present",
                 "B3_hypothesis_vmax_eq_cap_sigma_max", "B4_sigma_and_singularity",
                 "B5_polar_v_recompute", "B6_path_geometry_recompute",
                 "B7_path_v_recompute", "B8_stats_recompute"}
    kin_ok = all(c["pass"] for c in checks if c["name"] in KIN_NAMES)
    self_contained = bool(kin_ok and not missing_p and dyn_ok)
    chk("B11_blind_self_contained", self_contained,
        "kinematic_layer=OK dynamic_layer=%s" % dyn_status)

    npass = sum(c["pass"] for c in checks)
    out = {"audit": "blind_replicate_e73", "input_file": os.path.basename(REPORT),
           "blind": True, "isolation": {"repo_modules_loaded": _guard(),
                                        "imports": ["stdlib", "numpy"]},
           "kinematic_layer": "REPRODUCED" if kin_ok else "FAILED",
           "dynamic_layer": dyn_status,
           "missing_params": missing_p,
           "n_pass": npass, "n_total": len(checks),
           "audit_pass": bool(npass == len(checks)),
           "checks": checks}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("blind replicate: pass=%d/%d | kinematic=%s dynamic=%s missing=%s"
          % (npass, len(checks), out["kinematic_layer"], dyn_status, missing_p))
    for c in checks:
        print("  [%s] %-38s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
    return out


if __name__ == "__main__":
    run()
