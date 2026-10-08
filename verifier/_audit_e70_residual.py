# -*- coding: utf-8 -*-
"""_audit_e70_residual.py —— E70（端到端残差环 / C3）的**正反双向审核**（零 API、可复跑）。

正向
  F1 报告完整 + verdict_pass
  F2 独立重算：抽 6 个单元重跑，与报告逐位对照（不信任打印）
  F3 判据 ↔ 实现：TOL / K_MAX / 限位内线搜索 / 牛顿步 逐条对应
  F4 口径登记：容差定义、迭代预算、内环控制器、粗提案规则必须显式
  F5 **bench 缺陷已修**：SET_A 全部目标在限位内有 IK 解（对照 E57/Tier 的 46% 无解）

反向（构造反例/攻击）
  R1 目标集真可达：逐条断言存在限位内解且 |p| < L1+L2（排除"其实到不了"）
  R2 夹取对照有效：SET_B 中必须有目标**确实被夹**（否则 CLAMP 对照是空的）
  R3 残差环没作弊：内环只接收 q_ref（不接收 goal）；误差由**实际 q** 算（不是 setpoint）
  R4 门控同义反复检查：exec 越限 = 0 且 demand 越限 > 0
  R5 配对统计方向正确：仅 RESID 到点 > 0 且 仅 COARSE 到点 = 0
  R6 残差整体收缩：曲线前 3 步严格下降（否则外环无效）
  R7 不可行集处置正确：RESID 到点率 ≤ 0.10 且状态 ∈ {reject, budget_exhausted}

用法：python _audit_e70_residual.py     产物：_audit_e70_residual.json
"""
import json
import math
import os
import re
import sys

import numpy as np

def _find_src(start):
    """向上查找含 ``planning/control_strategies.py`` 的目录（**不假设固定两层布局**）。"""
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "planning", "control_strategies.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.dirname(os.path.abspath(start))


_SRC = _find_src(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e70_task_residual_loop as E70   # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EVAL, "_audit_e70_residual.json")
res = {"checks": []}


def chk(name, cond, detail=""):
    res["checks"].append({"check": name, "pass": bool(cond), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + detail) if detail else ""))


def main():
    rep = json.load(open(E70.REPORT, encoding="utf-8"))
    src = open(E70.__file__, encoding="utf-8").read()

    print("=== F 正向审核 ===")
    need = ["SET_A_admissible", "SET_B_inadmissible", "V1_paired", "V2_safety_invariant",
            "V5_kmax_sensitivity", "verdict_pass"]
    chk("F1 报告含全部必需键", all(k in rep for k in need),
        "缺失=%s" % [k for k in need if k not in rep])
    chk("F1b verdict_pass=True", rep["verdict_pass"] is True)

    # F2 独立重算（抽 6 个 SET_A 目标）
    A, B = E70.build_goal_sets(rep["design"]["n_goals_admissible"],
                               rep["design"]["n_goals_inadmissible"])
    diffs = []
    for gi in range(min(6, len(A))):
        r1 = E70.arm_coarse(np.array(A[gi]), "A|%d" % gi)
        r2 = E70.arm_resid(np.array(A[gi]), "A|%d" % gi)
        diffs.append((r1["err_m"], r2["err_m"], r2["iters"]))
    chk("F2 独立重算 6 单元（COARSE 误差 / RESID 误差与迭代数均有限）",
        all(d[0] is not None and d[1] is not None and d[2] >= 1 for d in diffs),
        "样例 %s" % [tuple(round(x, 4) if isinstance(x, float) else x for x in d) for d in diffs[:3]])
    chk("F2b RESID 误差显著小于 COARSE（逐单元）",
        all(d[1] < d[0] for d in diffs),
        "ratio 中位=%.4f" % float(np.median([d[1] / max(d[0], 1e-12) for d in diffs])))

    # F2c ★ 全量重算 SET_A 两臂，与报告汇总逐项对照（不只查"有限"）
    rows_c = [E70.arm_coarse(np.array(g), "A|%d" % i) for i, g in enumerate(A)]
    rows_r = [E70.arm_resid(np.array(g), "A|%d" % i) for i, g in enumerate(A)]

    def _agg(rows):
        errs = [r["err_m"] for r in rows if r["err_m"] is not None]
        return (round(sum(1 for r in rows if r.get("reach_1cm")) / len(rows), 4),
                round(float(np.median(errs)), 4) if errs else None,
                round(float(np.median([r["iters"] for r in rows])), 2))

    ac, ar = _agg(rows_c), _agg(rows_r)
    rc, rr = rep["SET_A_admissible"]["coarse"], rep["SET_A_admissible"]["resid"]
    chk("F2c 全量重算 SET_A（48×2 臂）与报告汇总一致",
        ac[0] == rc["reach_1cm_rate"] and ar[0] == rr["reach_1cm_rate"]
        and abs((ac[1] or 0) - (rc["median_err_m"] or 0)) < 1e-4
        and abs((ar[1] or 0) - (rr["median_err_m"] or 0)) < 1e-4
        and ar[2] == rr["median_iters"],
        "coarse 重算=%s 报告=%s ｜ resid 重算=%s 报告=%s"
        % (ac, (rc["reach_1cm_rate"], rc["median_err_m"], rc["median_iters"]),
           ar, (rr["reach_1cm_rate"], rr["median_err_m"], rr["median_iters"])))

    # F3 判据↔实现
    ok3 = all(s in src for s in ("TOL = 0.01", "K_MAX = 8", "alpha *= 0.5", "np.linalg.solve(Ji, e)"))
    chk("F3 判据式与实现逐条对应（TOL / K_MAX / 线搜索 / 牛顿步）", ok3)

    # F4 口径登记
    caveats = [
        "容差 TOL=%.3f m（E57/Tier 从未定义过，本件显式定义）" % rep["design"]["tol_m"],
        "迭代预算 K_MAX=%d（另有 V5 敏感性到 16）" % rep["design"]["K_MAX"],
        "内环 = %s" % rep["design"]["inner_controller"],
        "粗提案 = %s" % rep["design"]["coarse_proposer"],
    ]
    res["F4_caveats"] = caveats
    print("     口径登记：" + " ｜ ".join(caveats))

    # F5 bench 缺陷已修
    n_sol = sum(1 for p in A if any(np.all(np.abs(s) <= E70.JLIM) for s in E70.ik_all(p)))
    chk("F5 SET_A 全部目标在限位内有 IK 解（修 46%% 无解缺陷）", n_sol == len(A),
        "%d/%d" % (n_sol, len(A)))

    print("\n=== R 反向审核（构造反例/攻击） ===")

    # R1 目标集真可达
    inrange = all(0.0 < float(np.linalg.norm(p)) < (E70.L1 + E70.L2) * 0.99 for p in A)
    chk("R1 SET_A 目标均在可达环内且有限位内解", inrange and n_sol == len(A),
        "n=%d" % len(A))

    # R2 夹取对照有效
    B_pts = [b["p"] for b in B]
    clamped = 0
    for p in B_pts:
        sols = E70.ik_all(p)
        if sols and all(float(np.max(np.abs(s))) > E70.JLIM for s in sols):
            clamped += 1
    chk("R2 SET_B 中存在**确实被夹取**的目标（CLAMP 对照非空）", clamped > 0,
        "被夹=%d/%d ｜ kinds=%s" % (clamped, len(B_pts), rep["design"].get("SET_B_kinds")))

    # R3 残差环没作弊
    sig = re.search(r"def inner_run\(([^)]*)\)", src)
    chk("R3a 内环只接收 q_ref（不接收 goal）",
        sig is not None and "goal" not in sig.group(1), sig.group(1) if sig else "?")
    chk("R3b 误差由**实际 q** 计算（FK(q_end)/FK(q_cur)）",
        "goal - fk(q_end)" in src and "goal - fk(q_cur)" in src)

    # R4 门控同义反复检查
    v2 = rep["V2_safety_invariant"]
    chk("R4 exec 越限 = 0（构造保证）且 demand 越限 > 0（门控在承重）",
        v2["exec_violation_steps_total"] == 0 and v2["demand_violation_steps_total"] > 0,
        "exec=%d demand=%d" % (v2["exec_violation_steps_total"], v2["demand_violation_steps_total"]))

    # R5 配对方向
    p = rep["V1_paired"]
    chk("R5 配对：仅 RESID 到点 >0 且 仅 COARSE 到点 =0",
        p["resid_only"] > 0 and p["coarse_only"] == 0, str(p))

    # R6 残差整体收缩
    cur = rep.get("V1_residual_curve", [])
    chk("R6 残差曲线前 3 步严格下降（外环有效）",
        len(cur) >= 3 and cur[0]["median_err_m"] > cur[1]["median_err_m"] > cur[2]["median_err_m"],
        " → ".join("%.4f" % c["median_err_m"] for c in cur[:4]))

    # R7 不可行集处置
    rb = rep["SET_B_inadmissible"]["resid"]
    st = rb["status_counts"]
    chk("R7 SET_B：到点 ≤0.10 且状态 ∈ {reject, budget_exhausted}（不夹取、不假装成功）",
        rb["reach_1cm_rate"] <= 0.10 and set(st) <= {"reject", "budget_exhausted"},
        "到点=%.4f 状态=%s" % (rb["reach_1cm_rate"], st))

    nfail = sum(1 for c in res["checks"] if not c["pass"])
    res["n_pass"] = len(res["checks"]) - nfail
    res["n_total"] = len(res["checks"])
    res["audit_pass"] = (nfail == 0)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n审核结果：%d/%d PASS%s → %s"
          % (res["n_pass"], res["n_total"], "" if res["audit_pass"] else "（存在 FAIL）", OUT))
    if nfail:
        print("FAIL 项：" + " ｜ ".join(c["check"] for c in res["checks"] if not c["pass"]))


if __name__ == "__main__":
    main()
