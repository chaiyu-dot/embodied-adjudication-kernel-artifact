# -*- coding: utf-8 -*-
"""_audit_e73_strict.py —— 对 E73 速度谱曲线化的**严格审核**（零 API，比常规 18/18 更进一层）。

常规 _audit_e73（18/18）重算报告数值；本审核**审问前提与模型**，专挑"数值对但道理错"的隐患：

  F1  报告结构 + **全报告递归扫描**，无 inf/NaN 泄漏（JSON 诚实）
  F2  **闭式恒等**：密方向(720)下 v_max(q) == QD_CAP·σ_max(J(q))（速度包络=关节速度上限×最大奇异值）
  F3  **torque-never-binds 显式核验**：抽样 (q,u) 处 max_speed 恒 == QD_CAP，且 qd=QD_CAP·u 的力矩在限内
  F4  速度原语**可行性**：报告所用 s 处 |τ| ≤ τ_lim
  F5  速度原语**最优性（不保守）**：s(1+1e-6) 必越限（除非 s==cap）
  F6  二分**前提单调性**：可行集是 [0,s*] 前缀（否则二分无效）
  F7  加速度原语可行 + 最优
  F8  可实现性界自洽：qdd_cap==τ_i/M_ii；a_max_diag ≤ a_max；界有效
  F9  路径**几何一致**：fk(q) ≈ (x,y)，r==|p|，s 单调
  F10 统计自洽（min/median/max 从 rows 重算）
  F11 **独立闭式复现 E67**：4·max σ_max(7×7) == 3.0461；独立二分 == 66.1436
  F12 jerk 一致（wn·a_max）
  F13 口径一致（纯力矩速度>cap 口径、a_diag<a_max、奇异≥1、caveats≥5）
  F14 确定性（两跑一致）
  R1 篡改 v_max → F2 必 FAIL
  R2 篡改曲线 q（破坏 fk 一致）→ F9 必 FAIL
  R3 向报告注入 inf → F1 必 FAIL
  R4 篡改统计 → F10 必 FAIL
  R5 干净报告全正向 PASS

产物：_audit_e73_strict.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e67_testbed_completeness as E67     # noqa: E402
import e66_closed_loop_control as E66      # noqa: E402
import e73_velocity_spectrum_envelope as E73  # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
TAU = E66.TAU_LIM
QD_CAP = 4.0
E67_V, E67_A = 3.0461, 66.1436
RNG = np.random.RandomState(20260917)


def _chk(name, cond, detail=""):
    return {"name": name, "pass": bool(cond), "detail": str(detail)}


def _finite_scan(obj, path="$"):
    """递归扫描 inf/NaN；返回坏路径列表。"""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _finite_scan(v, "%s.%s" % (path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _finite_scan(v, "%s[%d]" % (path, i))
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    return bad


def _speed_ok(q, u, s):
    return bool(np.all(np.abs(E67.tau_at(q, s * np.asarray(u, float), 0.0)) <= TAU + 1e-12))


def _accel_ok(q, u, s):
    return bool(np.all(np.abs(E67.tau_at(q, [0.0, 0.0], s * np.asarray(u, float))) <= TAU + 1e-12))


def _sgm(q):
    return float(np.linalg.svd(E67.jac(q), compute_uv=False)[0])


def load():
    return json.load(open(REPORT, encoding="utf-8"))


# ===================== 正向 =====================
def forward(rep):
    ch = []
    need = ["A_polar_spectrum", "B_envelope_curve", "C_workspace_map", "D_calibration",
            "criteria", "verdict_pass"]
    missing = [k for k in need if k not in rep]
    ch.append(_chk("F1a_structure", not missing, "missing=%s" % missing))
    bad = _finite_scan(rep)
    ch.append(_chk("F1b_no_inf_nan", not bad, "非有限=%s" % bad[:5]))
    if missing:
        return ch
    A, B, C, D = (rep["A_polar_spectrum"], rep["B_envelope_curve"],
                  rep["C_workspace_map"], rep["D_calibration"])

    # ---- F2a 闭式恒等（模型层）：密方向下 v_max == QD_CAP·σ_max ----
    cells = C["rows"]
    idx = RNG.choice(len(cells), size=min(60, len(cells)), replace=False)
    worst = 0.0
    dense = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
    for i in idx:
        q = cells[i]["q"]
        J = E67.jac(q)
        v_dense = max(float(np.linalg.norm(J @ (E67.max_speed_along(q, [math.cos(t), math.sin(t)])
                                               * np.array([math.cos(t), math.sin(t)]))))
                      for t in dense)
        worst = max(worst, abs(v_dense - QD_CAP * _sgm(q)))
    ch.append(_chk("F2a_closed_form_v", worst <= 2e-3,
                   "max|v_dense - cap*sigma_max| = %.2e" % worst))

    # ---- F2b 报告存储值 == 物理重算（逐单元格，抓"自洽的谎"） ----
    n_dir = C["dirs"]
    ang = np.linspace(0.0, 2 * math.pi, n_dir, endpoint=False)
    uvs = [np.array([math.cos(t), math.sin(t)]) for t in ang]
    v_mis, a_mis, ad_mis = [], [], []
    for cell in cells:
        q = cell["q"]
        J = E67.jac(q)
        bv = max(float(np.linalg.norm(J @ (E67.max_speed_along(q, u) * u))) for u in uvs)
        ba = max(float(np.linalg.norm(J @ (E67.max_accel_along(q, u) * u))) for u in uvs)
        cap_ = E73.qdd_cap_of(q)
        bad = max(float(np.linalg.norm(J @ (E73.max_accel_capped(q, u, cap_) * u))) for u in uvs)
        if abs(bv - cell["v_max"]) > 2e-3:
            v_mis.append((cell["q"], cell["v_max"], round(bv, 4)))
        if abs(ba - cell["a_max"]) > 5e-2:
            a_mis.append((cell["q"], cell["a_max"], round(ba, 4)))
        if abs(bad - cell["a_max_diag"]) > 5e-2:
            ad_mis.append((cell["q"], cell["a_max_diag"], round(bad, 4)))
    ch.append(_chk("F2b_stored_values_match_physics",
                   not (v_mis or a_mis or ad_mis),
                   "v_mis=%d a_mis=%d ad_mis=%d %s" % (len(v_mis), len(a_mis), len(ad_mis),
                                                       (v_mis + a_mis + ad_mis)[:3])))

    # ---- F3 torque-never-binds 显式核验（抽样） ----
    cap_ok = True
    for i in idx:
        q = cells[i]["q"]
        for t in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
            u = np.array([math.cos(t), math.sin(t)])
            sv = E67.max_speed_along(q, u)
            if abs(sv - QD_CAP) > 1e-9:
                cap_ok = False
            if not _speed_ok(q, u, QD_CAP):
                cap_ok = False
    ch.append(_chk("F3_speed_cap_driven", cap_ok,
                   "全部抽样方向 max_speed==QD_CAP 且 qd=cap 时刻力矩在限内"))

    # ---- F4/F5/F6 速度原语 可行 / 最优 / 单调 ----
    pts = [(cells[i]["q"], np.array([math.cos(t), math.sin(t)]))
           for i in RNG.choice(len(cells), 40, replace=False)
           for t in [RNG.uniform(0, 2 * math.pi)]]
    feas_ok = tight_ok = mono_ok = True
    s_hist = []
    for q, u in pts:
        s = E67.max_speed_along(q, u)
        s_hist.append(s)
        if not _speed_ok(q, u, s):
            feas_ok = False
        if abs(s - QD_CAP) > 1e-9 and _speed_ok(q, u, s * (1 + 1e-6)):
            tight_ok = False
        # 单调：可行集是前缀
        if not (_speed_ok(q, u, 0.5 * s) and _speed_ok(q, u, 0.999 * s)):
            mono_ok = False
        if abs(s - QD_CAP) > 1e-6 and _speed_ok(q, u, s * 1.001):
            mono_ok = False
    ch.append(_chk("F4_speed_primitive_feasible", feas_ok,
                   "40 抽样 (q,u)：返回 s 处 |τ|≤τ_lim"))
    ch.append(_chk("F5_speed_primitive_tight", tight_ok,
                   "s_range=[%.3f,%.3f]（越限不可再放大）" % (min(s_hist), max(s_hist))))
    ch.append(_chk("F6_feasibility_prefix_monotone", mono_ok, "可行集=[0,s*] 前缀（二分前提成立）"))

    # ---- F7 加速度原语 可行 + 最优 ----
    a_feas = a_tight = True
    for q, u in pts:
        sa = E67.max_accel_along(q, u)
        if not _accel_ok(q, u, sa):
            a_feas = False
        if _accel_ok(q, u, sa * (1 + 1e-6)):
            a_tight = False
    ch.append(_chk("F7_accel_primitive_ok", a_feas and a_tight,
                   "feasible=%s tight=%s" % (a_feas, a_tight)))

    # ---- F8 可实现性界自洽 ----
    q0 = C["at_a"]
    cap = E73.qdd_cap_of(q0)
    M = E66.mass_matrix(q0)
    qdd_ok = all(abs(cap[i] - TAU[i] / float(M[i, i])) < 1e-9 for i in range(2)) and all(c > 0 for c in cap)
    diag_le = C["a_max_diag"] < C["a_max"] and C["a_max_diag"] > 0
    capped_feas, active = True, False
    for t in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        u = np.array([math.cos(t), math.sin(t)])
        sc = E73.max_accel_capped(q0, u, cap)
        if not (_accel_ok(q0, u, sc) and np.all(np.abs(sc * u) <= np.asarray(cap) + 1e-9)):
            capped_feas = False
        if sc < E67.max_accel_along(q0, u) - 1e-6:
            active = True
    ch.append(_chk("F8_realizability_bound", qdd_ok and diag_le and capped_feas and active,
                   "qdd_ok=%s diag<%s feas=%s active=%s" % (qdd_ok, diag_le, capped_feas, active)))

    # ---- F9 路径几何一致 ----
    geo_ok, smono = True, True
    prev_s = -1.0
    for pt in B["curve"]:
        p = E67.fk(pt["q"])
        if abs(p[0] - pt["x"]) > 1e-3 or abs(p[1] - pt["y"]) > 1e-3:
            geo_ok = False
        if pt["s"] < prev_s - 1e-12:
            smono = False
        prev_s = pt["s"]
    ch.append(_chk("F9_path_geometry", geo_ok and smono, "geo=%s s单调=%s" % (geo_ok, smono)))

    # ---- F10 统计自洽 ----
    v = [r["v_max"] for r in cells]
    a = [r["a_max"] for r in cells]
    ad = [r["a_max_diag"] for r in cells]
    st = (round(min(v), 4) == C["v_min"] and round(max(v), 4) == C["v_max"]
          and round(float(np.median(v)), 4) == C["v_median"]
          and round(min(a), 4) == C["a_min"] and round(max(a), 4) == C["a_max"]
          and round(float(np.median(a)), 4) == C["a_median"]
          and round(max(ad), 4) == C["a_max_diag"])
    ch.append(_chk("F10_stats_consistent", st, "v=%s/%s/%s a=%s/%s/%s" % (
        C["v_min"], C["v_median"], C["v_max"], C["a_min"], C["a_median"], C["a_max"])))

    # ---- F11 独立闭式复现 E67 ----
    qs = np.linspace(-1.8, 1.8, 7)
    v_closed = max(QD_CAP * _sgm([float(x), float(y)]) for x in qs for y in qs)
    a_indep = max(E67.max_accel_along([float(x), float(y)], np.array([math.cos(t), math.sin(t)]))
                  * float(np.linalg.norm(E67.jac([float(x), float(y)])
                                         @ np.array([math.cos(t), math.sin(t)])))
                  for x in qs for y in qs
                  for t in np.linspace(0, 2 * math.pi, 16, endpoint=False))
    rp = C["reproduce_e67"]
    ch.append(_chk("F11_reproduce_closedform",
                   abs(v_closed - E67_V) <= 1e-3 and abs(a_indep - E67_A) <= 1e-3
                   and abs(rp["v_max"] - v_closed) <= 1e-3 and abs(rp["a_max"] - a_indep) <= 1e-3,
                   "closed v=%.4f indep a=%.4f | report %s/%s" % (v_closed, a_indep, rp["v_max"], rp["a_max"])))

    # ---- F12 jerk ----
    _, _, wn = E66.budget_gains(E66.Q_REF)
    jk = (abs(D["wn_nominal"] - round(wn, 3)) < 1e-2
          and abs(D["jerk_envelope"]["at_e67_settings"] - round(wn * E67_A, 3)) < 5e-1
          and abs(D["jerk_envelope"]["max"] - round(wn * C["a_max"], 3)) < 5e-1)
    ch.append(_chk("F12_jerk_consistent", jk, "wn=%.3f" % wn))

    # ---- F13 口径一致 ----
    hn = (D["torque_only_v_nominal"] > C["v_max"] and C["a_max_diag"] < C["a_max"]
          and C["n_singular"] >= 1 and len(D["caveats"]) >= 5
          and abs(C["reproduce_e67"]["v_max"] - C["v_max"]) <= 1e-9)
    ch.append(_chk("F13_honesty_consistent", hn,
                   "tq_only=%.2f vmax=%.2f" % (D["torque_only_v_nominal"], C["v_max"])))

    # ---- F14 确定性 ----
    det = (E67.max_speed_along([0.7, -1.0], [1.0, 0.0]) == E67.max_speed_along([0.7, -1.0], [1.0, 0.0])
           and E73.envelope_at([0.7, -1.0]) == E73.envelope_at([0.7, -1.0])
           and _sgm([-1.8, 0.0]) == _sgm([-1.8, 0.0]))
    ch.append(_chk("F14_determinism", bool(det), "non-deterministic"))
    return ch


# ===================== 反向 =====================
def reverse(rep):
    ch = []
    bad = copy.deepcopy(rep)
    bad["C_workspace_map"]["rows"][0]["v_max"] = 99.0     # 只改单格存储值（不改聚合）→ F2b 必抓
    ch.append(_chk("R1_tamper_cell_v_detected",
                   any(c["name"] == "F2b_stored_values_match_physics" and not c["pass"] for c in forward(bad)),
                   "改单格 v_max→F2b 必报"))

    bad2 = copy.deepcopy(rep)
    bad2["B_envelope_curve"]["curve"][5]["q"] = [9.0, 9.0]
    ch.append(_chk("R2_tamper_geometry_detected",
                   any(c["name"] == "F9_path_geometry" and not c["pass"] for c in forward(bad2)),
                   "破坏 fk 一致→F9 必报"))

    bad3 = copy.deepcopy(rep)
    bad3["D_calibration"]["wn_nominal"] = float("inf")
    ch.append(_chk("R3_inject_inf_detected",
                   any(c["name"] == "F1b_no_inf_nan" and not c["pass"] for c in forward(bad3)),
                   "注入 inf→F1b 必报"))

    bad4 = copy.deepcopy(rep)
    bad4["C_workspace_map"]["v_median"] = -5.0
    ch.append(_chk("R4_tamper_stats_detected",
                   any(c["name"] == "F10_stats_consistent" and not c["pass"] for c in forward(bad4)),
                   "改中位数→F10 必报"))

    fwd = forward(rep)
    ch.append(_chk("R5_clean_passes", all(c["pass"] for c in fwd),
                   "clean=%d/%d" % (sum(c["pass"] for c in fwd), len(fwd))))
    return ch


def main():
    rep = load()
    fwd, rev = forward(rep), reverse(rep)
    allc = fwd + rev
    npass = sum(c["pass"] for c in allc)
    out = {"audit": "_audit_e73_strict", "experiment": "e73_velocity_spectrum_envelope",
           "pass": npass, "total": len(allc), "verdict": bool(npass == len(allc)),
           "audit_pass": bool(npass == len(allc)), "n_pass": npass, "n_total": len(allc),
           "forward": fwd, "reverse": rev, "checks": allc}
    with open(os.path.join(EVAL, "_audit_e73_strict.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("strict audit pass = %d/%d  verdict=%s" % (npass, len(allc), out["verdict"]))
    for c in allc:
        flag = "PASS" if c["pass"] else "FAIL"
        print("  [%s] %-34s %s" % (flag, c["name"], c["detail"]))
    return out


if __name__ == "__main__":
    main()
