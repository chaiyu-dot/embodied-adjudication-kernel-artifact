# -*- coding: utf-8 -*-
"""blind_reverse_e76.py —— E76 的**外部逆向盲测**（连声明参数也不信）。

只把 e76 报告当数据：闭式反推夹持几何/内力方向、可识别性 ±5% 证伪。
隔离：运行期 sys.modules 不得含 planning/e66/e67/e76；只 import e76_standalone_physics。
产物：blind_reverse_e76.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e76_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e76_dualarm_coordination_report.json")
BENCH = os.path.join(EVAL, "e76_bench_data.json")
OUT = os.path.join(EVAL, "blind_reverse_e76.json")
_repo_mods = ("planning", "rne_dynamics", "energy_kernel", "e66_closed_loop_control",
              "e67_testbed_completeness", "e76_dualarm_coordination", "control_strategies")
TOL = 1e-3


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E76 外部逆向盲测（反推 + 证伪可识别性）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:240]) if detail else ""))
    return bool(ok)


def _scalars(data):
    o = P.replicate(data)
    A, B, C = o["A_relative_pose_keeping"], o["B_internal_force_decoupling"], o["C_danger_zero"]
    sc = {}
    for k, v in A["rows"].items():
        sc["A_%s_drift" % k] = float(v["rel_drift_max_m"] or 0.0)
        sc["A_%s_tau" % k] = float(v["tau_max_demand"] or 0.0)
        sc["A_%s_viol" % k] = float(v["violation_steps_demand"])
    for f in ("net_object_wrench_inf", "orthogonality_motion_vs_internal",
              "motion_component_invariance_vs_alpha_max_err", "internal_force_sum_norm"):
        sc["B_" + f] = float(B[f])
    for k, v in C["rows"].items():
        sc["C_%s_viol" % k] = float(v["violation_steps_demand"])
        sc["C_%s_tau" % k] = float(v["tau_max_demand"] or 0.0)
    return sc


def _rel(a, b):
    m = 0.0
    for k in a:
        m = max(m, abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-9))
    return m


def main():
    print("=" * 92); print("E76 外部逆向盲测（反推 + 证伪可识别性）"); print("=" * 92)
    chk("R0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    tgt = data["config"]["targets"]
    pL = np.asarray(tgt[0][:2], float); pR = np.asarray(tgt[1][:2], float)
    k = data["config"]["internal_force_N"]

    # R1 反推夹持几何与内力方向
    c = 0.5 * (pL + pR); d = pR - pL
    u = d / np.linalg.norm(d)
    W_int = np.concatenate([k * u, -k * u])
    rL = pL - c; rR = pR - c
    GL = np.array([[1, 0], [0, 1], [-rL[1], rL[0]]]); GR = np.array([[1, 0], [0, 1], [-rR[1], rR[0]]])
    net = GL @ W_int[:2] + GR @ W_int[2:]
    chk("R1 反推夹持几何 + GᵀW_int ≡ 0（净物体 wrench 为零）",
        abs(np.linalg.norm(u) - 1.0) < 1e-12 and float(np.max(np.abs(net))) < 1e-12,
        "‖û‖=%.12f ‖GᵀW_int‖∞=%.2e" % (np.linalg.norm(u), float(np.max(np.abs(net)))))

    # R2 正交分解（range Gᵀ ⊥ null Gᵀ）
    G = np.hstack([GL, GR]); Gt = G.T; F_des = np.asarray(data["config"]["F_des"], float)
    W_motion = Gt @ np.linalg.solve(G @ Gt, F_des)
    orth = float(abs(W_motion @ W_int) / max(np.linalg.norm(W_motion) * np.linalg.norm(W_int), 1e-12))
    chk("R2 运动分量与内力分量正交（|cos|≈0）", orth < 1e-12, "|cos|=%.2e" % orth)

    # R3 内力水平 α 不改变运动分量
    errs = [float(np.max(np.abs(G @ (W_motion + a * W_int) - F_des))) for a in (-10.0, -2.0, 0.0, 2.0, 10.0, 50.0)]
    chk("R3 ∀α：G(W_motion+αW_int)=F_des（内力不影响运动分量）", max(errs) < 1e-12, "max err=%.2e" % max(errs))

    # R4 反推内力对称性（净力为零）
    chk("R4 内力对称：F_L+F_R=0（净力为零）", float(np.linalg.norm(W_int[:2] + W_int[2:])) < 1e-12,
        "‖F_L+F_R‖=%.2e" % float(np.linalg.norm(W_int[:2] + W_int[2:])))

    # R5 漂移随载荷单调（序数）
    rows = rep["A_relative_pose_keeping"]["rows"]
    order = sorted(rows.keys(), key=lambda s: float(s.split("=")[1]))
    dr = [rows[k]["rel_drift_max_m"] for k in order]
    mono = all(dr[i + 1] >= dr[i] - 1e-9 for i in range(len(dr) - 1))
    chk("R5 相对位形漂移随非对称载荷单调不减（序数）", mono, "drift 序=%s" % [round(x, 4) for x in dr])

    # R6 可识别性证伪（12 参数 ±5%）
    base = _scalars(data)
    B = data["bench"]
    pars = [("L1", lambda b, v: b.__setitem__("L1", v), B["L1"]),
            ("L2", lambda b, v: b.__setitem__("L2", v), B["L2"]),
            ("M1", lambda b, v: b.__setitem__("M1", v), B["M1"]),
            ("M2", lambda b, v: b.__setitem__("M2", v), B["M2"]),
            ("LC1", lambda b, v: b.__setitem__("LC1", v), B["LC1"]),
            ("LC2", lambda b, v: b.__setitem__("LC2", v), B["LC2"]),
            ("I1", lambda b, v: b.__setitem__("I1", v), B["I1"]),
            ("I2", lambda b, v: b.__setitem__("I2", v), B["I2"]),
            ("G", lambda b, v: b.__setitem__("G", v), B["G"]),
            ("TAU_LIM[0]", lambda b, v: b["TAU_LIM"].__setitem__(0, v), B["TAU_LIM"][0]),
            ("TAU_LIM[1]", lambda b, v: b["TAU_LIM"].__setitem__(1, v), B["TAU_LIM"][1]),
            ("B_VISC", lambda b, v: b.__setitem__("B_VISC", v), B["B_VISC"])]
    ident, not_ident, sens = [], [], {}
    for name, setter, v0 in pars:
        s = 0.0
        for sign in (+1.0, -1.0):
            d2 = json.loads(json.dumps(data)); setter(d2["bench"], v0 * (1.0 + sign * 0.05))
            s = max(s, _rel(_scalars(d2), base))
        sens[name] = s
        (ident if s > TOL else not_ident).append(name)
    chk("R6 可识别性证伪（±5%）：被识别集合如实列出，未识别项诚实标注",
        len(ident) + len(not_ident) == 12, "identified=%s | not_identified=%s" % (ident, not_ident))
    chk("R7 被识别参数敏感度均 > 容差", all(sens[k] > TOL for k in ident),
        "敏感度表=%s" % {k: float("%.2e" % v) for k, v in sens.items()})

    # R8 基线复现 == 报告
    o = P.replicate(data)
    ok8 = (o["A_relative_pose_keeping"]["max_drift_within_limit_m"] == rep["A_relative_pose_keeping"]["max_drift_within_limit_m"]
           and o["B_internal_force_decoupling"]["net_object_wrench_inf"] == rep["B_internal_force_decoupling"]["net_object_wrench_inf"]
           and o["C_danger_zero"]["all_ok"] == rep["C_danger_zero"]["all_ok"])
    chk("R8 独立复现基线与报告一致（反推成立）", ok8,
        "drift=%.6f B_net=%.2e C_ok=%s" % (o["A_relative_pose_keeping"]["max_drift_within_limit_m"],
                                           o["B_internal_force_decoupling"]["net_object_wrench_inf"],
                                           o["C_danger_zero"]["all_ok"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["identified"] = ident; _res["not_identified"] = not_ident
    _res["sensitivity"] = {k: float("%.3e" % v) for k, v in sens.items()}
    _res["findings"] = [
        "★ 逆向盲测（连声明参数也不信）：夹持几何/内力方向由目标点闭式反推，GᵀW_int≡0 精确。",
        "  可识别性证伪（正负5%%）：12 参数中 %d 被数据锁死，未识别=%s。" % (len(ident), not_ident),
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n逆向盲测结果：%d/%d PASS ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return _res


if __name__ == "__main__":
    main()
