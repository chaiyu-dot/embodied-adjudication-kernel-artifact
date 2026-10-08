# -*- coding: utf-8 -*-
"""_audit_e76_replicator_anomaly.py —— **第三方复刻者视角**的异常排查审核（零 API）。

不信任作者代码：只凭 e76 报告数字，自己独立实现夹持代数（夹持矩阵/内力/正交分解）重算 B 段，
并核验 A/C 段内部自洽（漂移单调、超限场景越限>0、上限内越限=0）与 verdict 自洽；再构造篡改。
**不 import planning.* / e66 / e67 / e76 / standalone**。
产物：_audit_e76_replicator_anomaly.json
"""
import copy
import json
import math
import os

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e76_dualarm_coordination_report.json")


def _chk(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": str(detail)[:400]}


def _algebra(cfg):
    pL = np.asarray(cfg["targets"][0][:2], float); pR = np.asarray(cfg["targets"][1][:2], float)
    c = 0.5 * (pL + pR); u = (pR - pL) / np.linalg.norm(pR - pL)
    k = cfg["internal_force_N"]
    W = np.concatenate([k * u, -k * u])
    rL = pL - c; rR = pR - c
    GL = np.array([[1, 0], [0, 1], [-rL[1], rL[0]]]); GR = np.array([[1, 0], [0, 1], [-rR[1], rR[0]]])
    G = np.hstack([GL, GR])
    net = GL @ W[:2] + GR @ W[2:]
    F_des = np.asarray(cfg["F_des"], float)
    Wm = G.T @ np.linalg.solve(G @ G.T, F_des)
    orth = float(abs(Wm @ W) / max(np.linalg.norm(Wm) * np.linalg.norm(W), 1e-12))
    errs = max(float(np.max(np.abs(G @ (Wm + a * W) - F_des))) for a in (-5.0, -1.0, 0.0, 1.0, 5.0, 20.0))
    return {"net": float(np.max(np.abs(net))), "orth": orth, "alpha": errs,
            "sum": float(np.linalg.norm(W[:2] + W[2:]))}


def main():
    rep = json.load(open(REPORT, encoding="utf-8"))
    cfg = rep["config"]; A, B, C = rep["A_relative_pose_keeping"], rep["B_internal_force_decoupling"], rep["C_danger_zero"]
    al = _algebra(cfg)
    checks = []

    # T1 B 段代数独立重算
    t1 = (abs(al["net"] - B["net_object_wrench_inf"]) < 1e-15
          and abs(al["orth"] - B["orthogonality_motion_vs_internal"]) < 1e-18
          and abs(al["alpha"] - B["motion_component_invariance_vs_alpha_max_err"]) < 1e-15
          and abs(al["sum"] - B["internal_force_sum_norm"]) < 1e-15)
    checks.append(_chk("T1_internal_force_algebra_independent", t1,
                       "‖GᵀW_int‖=%.2e orth=%.2e α=%.2e" % (al["net"], al["orth"], al["alpha"])))

    # T2 A 段：上限内漂移 ≤1cm，超限场景越限>0（先可行后可控）
    safe = [v for k, v in A["rows"].items() if v["within_static_limit"]]
    over = [v for k, v in A["rows"].items() if not v["within_static_limit"]]
    t2 = (all(v["rel_drift_max_m"] <= cfg["drift_tol_m"] for v in safe)
          and all(v["violation_steps_demand"] > 0 for v in over))
    checks.append(_chk("T2_static_limit_boundary_consistent", t2,
                       "上限内 %d 行漂移≤1cm；超限 %d 行越限>0" % (len(safe), len(over))))

    # T3 A 段漂移随载荷单调
    order = sorted(A["rows"].keys(), key=lambda s: float(s.split("=")[1]))
    dr = [A["rows"][k]["rel_drift_max_m"] for k in order]
    t3 = all(dr[i + 1] >= dr[i] - 1e-9 for i in range(len(dr) - 1)) and A["drift_monotone_in_mp"]
    checks.append(_chk("T3_drift_monotone_in_load", t3, "drift=%s" % [round(x, 4) for x in dr]))

    # T4 C 段：上限内越限==0
    t4 = all(v["violation_steps_demand"] == 0 and not v["diverged"] for v in C["rows"].values())
    checks.append(_chk("T4_danger_zero_within_limit", t4, "C 段 %d 行越限==0" % len(C["rows"])))

    # T5 内部自洽 + verdict
    c = rep["criteria"]
    inv = bool(c["C1_drift_le_1cm_within_static_limit_and_monotone"] and A["all_ok"]
               and c["C2_internal_decoupled_exact"] and B["all_ok"]
               and c["C3_danger_zero_within_limit"] and C["all_ok"])
    checks.append(_chk("T5_verdict_self_consistent", inv == bool(rep["verdict_pass"]),
                       "inv=%s verdict=%s" % (inv, rep["verdict_pass"])))

    # T6 静态上限声明一致（0.3572 kg）
    t6 = abs(A["static_hold_limit_kg"] - 0.3572) < 1e-3
    checks.append(_chk("T6_static_hold_limit_declared", t6, "limit=%.4f kg" % A["static_hold_limit_kg"]))

    # T7-T9 篡改
    bad = copy.deepcopy(rep); bad["B_internal_force_decoupling"]["net_object_wrench_inf"] = 0.5
    checks.append(_chk("T7_tamper_internal_detected",
                       abs(_algebra(bad["config"])["net"] - bad["B_internal_force_decoupling"]["net_object_wrench_inf"]) > 1e-15,
                       "改 ‖GᵀW_int‖ → 独立重算必报"))
    bad2 = copy.deepcopy(rep); bad2["A_relative_pose_keeping"]["rows"]["mp=0.20"]["violation_steps_demand"] = 0
    # 上限内该行若本为 0 则无信息；改为改超限行
    bad2 = copy.deepcopy(rep)
    for k, v in bad2["A_relative_pose_keeping"]["rows"].items():
        if not v["within_static_limit"]:
            v["violation_steps_demand"] = 0
    checks.append(_chk("T8_tamper_over_limit_detected",
                       not all(v["violation_steps_demand"] > 0 for v in bad2["A_relative_pose_keeping"]["rows"].values()
                               if not v["within_static_limit"]),
                       "抹超限行越限 → T2 类必报"))
    bad3 = copy.deepcopy(rep); bad3["verdict_pass"] = False
    c3 = bad3["criteria"]
    inv3 = bool(c3["C1_drift_le_1cm_within_static_limit_and_monotone"] and bad3["A_relative_pose_keeping"]["all_ok"]
                and c3["C2_internal_decoupled_exact"] and bad3["B_internal_force_decoupling"]["all_ok"]
                and c3["C3_danger_zero_within_limit"] and bad3["C_danger_zero"]["all_ok"])
    checks.append(_chk("T9_tamper_verdict_detected", inv3 != bool(bad3["verdict_pass"]),
                       "翻转 verdict → T5 类必报"))

    npass = sum(x["pass"] for x in checks)
    out = {"audit": "_audit_e76_replicator_anomaly", "experiment": "e76_dualarm_coordination",
           "pass": npass, "total": len(checks), "verdict": bool(npass == len(checks)),
           "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(os.path.join(EVAL, "_audit_e76_replicator_anomaly.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly audit pass = %d/%d  verdict=%s" % (npass, len(checks), out["verdict"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
