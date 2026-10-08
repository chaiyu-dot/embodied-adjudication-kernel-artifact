# -*- coding: utf-8 -*-
"""blind_reverse_e73.py —— **外部逆向盲测**：连数据文件里声明的参数也不信，从数据反推并证伪。

比 blind_replicate_e73.py 更进一步：
  · 上一轮：不 import 作者模块，用数据文件里**声明的**台架参数重建；
  · 这一轮：**不信任声明值** ——
      (R2) 只用路径点的 (q,x,y) **最小二乘反解** L1,L2；
      (R3) 只用极坐标谱的 (q,θ,v_ee) **反解**关节速度上限 cap；
      (R4) 对每个声明参数施加 ±5% 扰动，重算其依赖的报告量——
           **数据能否分辨**（相对偏差 > 阈值）→ 该参数**被数据识别**；否则**仅声明、未被数据约束**；
      (R5) 汇总"数据识别得出"vs"仅声明"参数集，判定数据文件是否**完全由数据锁定**。

自包含：只用 stdlib + numpy，不 import 任何仓库模块（运行期自检 sys.modules）。
产物：blind_reverse_e73.json　用法：python blind_reverse_e73.py [report.json]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "e73_velocity_spectrum_report.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(REPORT)), "blind_reverse_e73.json")

_BANNED = ("e66", "e67", "e73", "planning", "control_strategies", "rne_dynamics")
REL_TOL = 1e-3          # 数据"能否分辨"的相对偏差阈值


def _guard():
    return sorted({m.split(".")[0] for m in list(sys.modules) if m.split(".")[0] in _BANNED})


def _num(x, nd=6):
    return round(float(x), nd) if math.isfinite(float(x)) else None


# ---------------- 独立重建（仅用台架参数，无任何作者代码） ----------------
def jac(q, L1, L2):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def smax_of(q, L1, L2):
    return float(np.linalg.svd(jac(q, L1, L2), compute_uv=False)[0])


def mass_matrix(q, P):
    c2 = math.cos(q[1])
    m11 = P["M1"] * P["LC1"] ** 2 + P["M2"] * (P["L1"] ** 2 + P["LC2"] ** 2 + 2 * P["L1"] * P["LC2"] * c2) + P["I1"] + P["I2"]
    m12 = P["M2"] * (P["LC2"] ** 2 + P["L1"] * P["LC2"] * c2) + P["I2"]
    m22 = P["M2"] * P["LC2"] ** 2 + P["I2"]
    return np.array([[m11, m12], [m12, m22]])


def tau_at(q, qdd, P):
    """加速度口径：qd=0 ⇒ Coriolis=0；τ = M(q)·qdd + g(q)。"""
    g1 = (P["M1"] * P["LC1"] + P["M2"] * P["L1"]) * P["G"] * math.cos(q[0]) + P["M2"] * P["LC2"] * P["G"] * math.cos(q[0] + q[1])
    g2 = P["M2"] * P["LC2"] * P["G"] * math.cos(q[0] + q[1])
    return mass_matrix(q, P) @ np.asarray(qdd, float) + np.array([g1, g2])


def max_accel(q, u, P):
    u = np.asarray(u, float)
    TAU = np.array(P["TAU_LIM"], float)

    def ok(s):
        return bool(np.all(np.abs(tau_at(q, s * u, P)) <= TAU + 1e-12))
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


def a_map(cells, P, nd):
    """给定参数下的 a_max 预测（与报告同口径：nd 个方向取最大）。"""
    uvs = [np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0.0, 2 * math.pi, nd, endpoint=False)]
    out = []
    for cell in cells:
        q = cell["q"]
        J = jac(q, P["L1"], P["L2"])
        out.append(max(float(np.linalg.norm(J @ (max_accel(q, u, P) * u))) for u in uvs))
    return out


def v_map(cells, L1, L2, cap, nd):
    uvs = [np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0.0, 2 * math.pi, nd, endpoint=False)]
    out = []
    for cell in cells:
        J = jac(cell["q"], L1, L2)
        out.append(max(float(np.linalg.norm(J @ (cap * u))) for u in uvs))
    return out


def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    A, B, C = rep["A_polar_spectrum"], rep["B_envelope_curve"], rep["C_workspace_map"]
    nd = C["dirs"]
    checks, notes = [], []

    def chk(n, ok, d=""):
        checks.append({"name": n, "pass": bool(ok), "detail": str(d)})

    # R1 隔离
    chk("R1_isolation_no_repo_modules", not _guard(), "loaded=%s" % _guard())

    # R2 只用路径点 (q,x,y) 最小二乘反解 L1,L2
    Amat, bvec = [], []
    for pt in B["curve"]:
        q1, q2 = pt["q"]
        Amat.append([math.cos(q1), math.cos(q1 + q2)])
        bvec.append(pt["x"])
        Amat.append([math.sin(q1), math.sin(q1 + q2)])
        bvec.append(pt["y"])
    sol, *_ = np.linalg.lstsq(np.array(Amat), np.array(bvec), rcond=None)
    L1_r, L2_r = float(sol[0]), float(sol[1])
    e1, e2 = abs(L1_r - bench["L1"]), abs(L2_r - bench["L2"])
    chk("R2_reverse_L1_L2_from_geometry", e1 <= 1e-3 and e2 <= 1e-3,
        "反推 L1=%.5f L2=%.5f | 声明 %.3f/%.3f | 误差 %.2e/%.2e" % (L1_r, L2_r, bench["L1"], bench["L2"], e1, e2))

    # R3 只用极坐标谱 (q,θ,v_ee) 反解 qdot_cap（v_ee = cap·|J u| 精确）
    cfg = A["configs"][0]
    J0 = jac(cfg["q"], L1_r, L2_r)
    ratios = []
    for row in cfg["rows"]:
        th = math.radians(row["deg"])
        u = np.array([math.cos(th), math.sin(th)])
        n = float(np.linalg.norm(J0 @ u))
        if n > 1e-6:
            ratios.append(row["v_ee"] / n)
    cap_r = float(np.median(ratios))
    e_cap = abs(cap_r - bench["qdot_cap"])
    chk("R3_reverse_qdot_cap_from_polar", e_cap <= 1e-3,
        "反推 cap=%.6f | 声明 %.3f | 误差 %.2e | n=%d" % (cap_r, bench["qdot_cap"], e_cap, len(ratios)))

    # R4 证伪式可识别性
    rng = np.random.RandomState(20260918)
    cells = [C["rows"][i] for i in rng.choice(len(C["rows"]), size=min(30, len(C["rows"])), replace=False)]
    P0 = {"L1": bench["L1"], "L2": bench["L2"], "G": bench["G"],
          "TAU_LIM": list(bench["TAU_LIM"]), "Q_REF": bench["Q_REF"],
          **{k: bench["link_params"][k] for k in ("M1", "M2", "LC1", "LC2", "I1", "I2")}}

    def rel_dev(pred, target):
        return max(abs(p - t) / max(abs(t), 1e-9) for p, t in zip(pred, target))

    a_ref = [c["a_max"] for c in cells]
    a_base = a_map(cells, P0, nd)
    base_tol = rel_dev(a_base, a_ref)
    v_ref = [c["v_max"] for c in cells]
    v_base = v_map(cells, P0["L1"], P0["L2"], bench["qdot_cap"], nd)
    base_tol_v = rel_dev(v_base, v_ref)
    chk("R4a_declared_reproduces_report", base_tol <= 5e-3 and base_tol_v <= 5e-3,
        "a_base_rel=%.2e v_base_rel=%.2e" % (base_tol, base_tol_v))

    ident = {}
    for k in ["L1", "L2"]:
        dev = 0.0
        for eps in (0.05, -0.05):
            Pq = dict(P0)
            Pq[k] = P0[k] * (1 + eps)
            dev = max(dev, rel_dev(a_map(cells, Pq, nd), a_ref))
            dev = max(dev, rel_dev(v_map(cells, Pq["L1"], Pq["L2"], bench["qdot_cap"], nd), v_ref))
        ident[k] = dev
    ident["qdot_cap"] = 0.0                    # 速度层：仅由 v 图分辨
    for eps in (0.05, -0.05):
        dev = rel_dev(v_map(cells, P0["L1"], P0["L2"], bench["qdot_cap"] * (1 + eps), nd), v_ref)
        ident["qdot_cap"] = max(ident["qdot_cap"], dev)
    for k in ["M1", "M2", "LC1", "LC2", "I1", "I2", "G"]:
        dev = 0.0
        for eps in (0.05, -0.05):
            Pq = dict(P0)
            Pq[k] = P0[k] * (1 + eps)
            dev = max(dev, rel_dev(a_map(cells, Pq, nd), a_ref))
        ident[k] = dev
    for i in (0, 1):
        dev = 0.0
        for eps in (0.05, -0.05):
            Pq = dict(P0)
            tl = list(P0["TAU_LIM"])
            tl[i] = tl[i] * (1 + eps)
            Pq["TAU_LIM"] = tl
            dev = max(dev, rel_dev(a_map(cells, Pq, nd), a_ref))
        ident["TAU_LIM[%d]" % i] = dev

    identified = sorted(k for k, v in ident.items() if v > REL_TOL)
    not_identified = sorted(k for k, v in ident.items() if v <= REL_TOL)
    EXPECT = 12   # L1,L2,qdot_cap + M1,M2,LC1,LC2,I1,I2,G + TAU_LIM[0],TAU_LIM[1]
    chk("R4b_identifiability_computed", len(ident) == EXPECT,
        "n=%d(期望 %d) identified=%d not_identified=%d | sens_min=%.2e" % (
            len(ident), EXPECT, len(identified), len(not_identified),
            min(ident.values()) if ident else 0.0))

    # R5 数据文件是否被数据完全锁定（全部参数都被数据识别）
    fully_pinned = (not not_identified)
    chk("R5_data_fully_pinned_by_data", fully_pinned,
        "仅声明(未被数据约束)=%s" % not_identified if not_identified else "全部声明参数均被数据识别")

    # R6 规格完整（模型约定 + 惯性 + 步长）
    spec_need = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params"]
    miss = [k for k in spec_need if k not in bench]
    chk("R6_full_spec_present", not miss, "missing=%s" % miss)

    if not_identified:
        notes.append("以下参数**未被数据约束**（数据改 ±5% 仍与报告一致）→ 复刻者只能接受声明值，"
                     "不能从数据独立验证：" + ", ".join(not_identified))

    npass = sum(c["pass"] for c in checks)
    out = {"audit": "blind_reverse_e73", "input_file": os.path.basename(REPORT),
           "blind": True, "reverse": True, "rel_tol": REL_TOL,
           "isolation": {"repo_modules_loaded": _guard(), "imports": ["stdlib", "numpy"]},
           "reverse_inferred": {"L1": _num(L1_r), "L2": _num(L2_r), "qdot_cap": _num(cap_r),
                                "declared": {"L1": bench["L1"], "L2": bench["L2"], "qdot_cap": bench["qdot_cap"]},
                                "err": {"L1": _num(e1), "L2": _num(e2), "qdot_cap": _num(e_cap)}},
           "identifiability": {k: {"sensitivity": _num(v), "identified": bool(v > REL_TOL)}
                               for k, v in sorted(ident.items())},
           "identified_params": identified, "not_identified_params": not_identified,
           "notes": notes,
           "n_pass": npass, "n_total": len(checks), "audit_pass": bool(npass == len(checks)),
           "checks": checks}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("blind-reverse: pass=%d/%d | L1=%.5f L2=%.5f cap=%.5f | identified=%d not_identified=%s"
          % (npass, len(checks), L1_r, L2_r, cap_r, len(identified), not_identified))
    for c in checks:
        print("  [%s] %-40s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
    return out


if __name__ == "__main__":
    run()
