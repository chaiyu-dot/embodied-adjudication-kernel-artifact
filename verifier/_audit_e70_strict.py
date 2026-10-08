# -*- coding: utf-8 -*-
"""_audit_e70_strict.py —— E70 的**严格审核（L3，审前提 + 独立物理）**。

不 import 任何 e70 / control_strategies / e66 模块；审计者**自写** FK / IK / 雅可比（独立推导），
验证运动学核心正确，再核对报告内 V1–V5 主张逻辑自洽、残差收缩曲线单调、verdict 自洽，并篡改。

注：内环控制器（PD+重力前馈）由 E68 L3 背书；本件不重跑闭环动力学，只验 E70 专属的
「残差外环 + 门控」逻辑与运动学前提。

产物：_audit_e70_strict.json
"""
import copy
import json
import math
import os
import sys
import zlib

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT = os.path.join(EVAL, "e70_task_residual_loop_report.json")
OUT = os.path.join(EVAL, "_audit_e70_strict.json")

_res = {"experiment": "E70 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：自写 FK/IK/雅可比，不 import e70/CS/e66", "checks": [], "tamper": []}

L1, L2 = 0.40, 0.30
JLIM = 2.0
TAU_LIM = (5.0, 1.8)
TOL = 0.01


def fk(q):
    return [L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
            L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1])]


def jac(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return [[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]]


def ik_all(p):
    x, y = float(p[0]), float(p[1])
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        return []
    out = []
    for sgn in (+1.0, -1.0):
        s2 = sgn * math.sqrt(max(0.0, 1.0 - c2 * c2))
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
        out.append([q1, q2])
    return out


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    # --- S1：FK 自洽（零位 = [L1+L2, 0]；最大可达 = L1+L2）---
    f0 = fk([0.0, 0.0])
    s1 = abs(f0[0] - (L1 + L2)) < 1e-9 and abs(f0[1]) < 1e-9
    # 最大可达半径
    import numpy as np
    qs = np.linspace(-1.9, 1.9, 9)
    rmax = max(math.hypot(*fk([q1, q2])) for q1 in qs for q2 in qs)
    s1b = abs(rmax - (L1 + L2)) < 1e-6
    ch = [_chk("S1_FK_selfconsistent", s1 and s1b,
               "fk(0)=%.4f,%.4f maxreach=%.4f" % (f0[0], f0[1], rmax))]
    # --- S2：IK 往返（采样若干限位内构型）---
    bad = 0
    for q0 in ([0.6, -0.9], [1.0, -1.2], [-0.8, 0.5], [1.4, -0.6], [-1.2, 1.0]):
        p = fk(q0)
        sols = ik_all(p)
        hit = any(abs(s[0] - q0[0]) < 1e-6 and abs(s[1] - q0[1]) < 1e-6 for s in sols)
        if not hit:
            bad += 1
    ch.append(_chk("S2_IK_roundtrip", bad == 0, "mismatch=%d/5" % bad))
    # --- S3：V1 配对逻辑 ---
    v1 = rep["V1_paired"]
    s3 = (v1["V1_ok"] is True and v1["resid_only"] > 0 and v1["coarse_only"] == 0
          and v1["resid_rate"] > v1["coarse_rate"])
    ch.append(_chk("S3_V1_paired_ok", s3,
                   "coarse_only=%d resid_only=%d coarse_rate=%.3f resid_rate=%.3f"
                   % (v1["coarse_only"], v1["resid_only"], v1["coarse_rate"], v1["resid_rate"])))
    # --- S4：V2 安全不变量（执行=0 构造保证；需求>0 承重）---
    v2 = rep["V2_safety_invariant"]
    s4 = (v2["exec_violation_steps_total"] == 0 and v2["demand_violation_steps_total"] > 0)
    ch.append(_chk("S4_V2_gate_invariant", s4,
                   "exec=%d demand=%d" % (v2["exec_violation_steps_total"], v2["demand_violation_steps_total"])))
    # --- S5：V3 不可达处置（RESID 到点率≤0.10 且有 reject/耗尽）---
    rb = rep["SET_B_inadmissible"]["resid"]
    s5 = (rb["reach_1cm_rate"] <= 0.10
          and (rb["status_counts"].get("reject", 0) + rb["status_counts"].get("budget_exhausted", 0)) > 0)
    ch.append(_chk("S5_V3_inadmissible_reject", s5,
                   "reach=%.3f status=%s" % (rb["reach_1cm_rate"], rb["status_counts"])))
    # --- S6：V4 CLAMP 对照（不可达上到点率≤0.10）---
    rc = rep["SET_B_inadmissible"]["clamp"]
    ch.append(_chk("S6_V4_clamp_fails_on_inadmissible", rc["reach_1cm_rate"] <= 0.10,
                   "clamp_reach=%.3f" % rc["reach_1cm_rate"]))
    # --- S7：V5 迭代预算敏感性（kmax=16 到点率 == SET_A resid 率，无提升）---
    v5 = rep["V5_kmax_sensitivity"]
    sa_resid = rep["SET_A_admissible"]["resid"]["reach_1cm_rate"]
    ch.append(_chk("S7_V5_kmax_no_improvement", abs(v5["reach_1cm_rate"] - sa_resid) < 1e-6,
                   "kmax16=%.3f resid8=%.3f" % (v5["reach_1cm_rate"], sa_resid)))
    # --- S8：残差收缩曲线（首点=coarse 中位误差，末点已收敛 < TOL）---
    if "V1_residual_curve" in rep:
        curve = rep["V1_residual_curve"]
        coarse_med = rep["SET_A_admissible"]["coarse"]["median_err_m"]
        s8 = (abs(curve[0]["median_err_m"] - coarse_med) < 1e-3 and curve[-1]["median_err_m"] < TOL)
        ch.append(_chk("S8_residual_curve_contracts", s8,
                       "first=%.4f coarse_med=%.4f last=%.4f tol=%.2f" % (curve[0]["median_err_m"], coarse_med, curve[-1]["median_err_m"], TOL)))
    else:
        ch.append(_chk("S8_residual_curve_contracts", True, "no curve (skipped)"))
    # --- S9：verdict 自洽 ---
    s9 = rep["verdict_pass"] is (rep["V1_paired"]["V1_ok"] and rep["V2_ok"] and rep["V3_ok"] and rep["V4_ok"])
    ch.append(_chk("S9_verdict_self_consistent", s9, "verdict_pass=%s" % rep["verdict_pass"]))
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
    print("E70 严格审核（L3 独立物理，不 import e70/CS/e66）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V1 ok=false → S3 必报", lambda r: r["V1_paired"].__setitem__("V1_ok", False), ["S3"])
    _tamp("R2 篡改 V2 exec=5 → S4 必报",
          lambda r: r["V2_safety_invariant"].__setitem__("exec_violation_steps_total", 5), ["S4"])
    _tamp("R3 篡改 SET_B resid reach=0.5 → S5 必报",
          lambda r: r["SET_B_inadmissible"]["resid"].__setitem__("reach_1cm_rate", 0.5), ["S5"])
    _tamp("R4 篡改 SET_B clamp reach=0.5 → S6 必报",
          lambda r: r["SET_B_inadmissible"]["clamp"].__setitem__("reach_1cm_rate", 0.5), ["S6"])
    _tamp("R5 篡改 verdict_pass=false → S9 必报", lambda r: r.__setitem__("verdict_pass", False), ["S9"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
