# -*- coding: utf-8 -*-
"""_audit_e67_testbed.py —— E67 理想台架补全的**正反双向审核**（零 API、纯 CPU）。

为什么需要
----------
E67 此前**没有审核脚本**，是控制侧唯一"无人复核"的产物（2026-09-17 补齐）。
本审核分两路：
  · 正向：报告↔实现↔数字三者自洽；关键量**独立重算**（换算法、换路径）；
  · 反向：构造反例 / 边界 / 退化，检验结论不是空断言，且**篡改能被抓住**。

重点盯住本轮新补的口径问题
--------------------------
``gate=True`` 时控制器返回的是**已 clip** 的力矩 → 用执行值统计"越限"恒为 0（同义反复）。
故审核强制检查：① 所有 gate=True 场景 ``violation_steps_executed == 0``；
② ``violation_steps_demand`` 必须存在；③ ``demand ≥ executed`` 逐场景成立；
④ 把 gate 关掉后 executed 必须**真的** > 0（证明该字段有分辨力，不是永真）。

产物：_audit_e67_testbed.json
"""
import json
import math
import os
import sys
import zlib

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
import e66_closed_loop_control as E66      # noqa: E402
import e67_testbed_completeness as E67     # noqa: E402
from planning import control_strategies as CS   # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e67_testbed_report.json")
OUT = os.path.join(EVAL, "_audit_e67_testbed.json")
TAU_LIM = E67.TAU_LIM

_res = {"experiment": "E67 台架补全 正反双向审核", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------- 独立实现（不复用 E67 的算法）
def fk_alt(q, base=(0.0, 0.0)):
    x = base[0] + E67.L1 * math.cos(q[0]) + E67.L2 * math.cos(q[0] + q[1])
    y = base[1] + E67.L1 * math.sin(q[0]) + E67.L2 * math.sin(q[0] + q[1])
    return np.array([x, y])


def jac_alt(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-E67.L1 * s1 - E67.L2 * s12, -E67.L2 * s12],
                     [E67.L1 * c1 + E67.L2 * c12, E67.L2 * c12]])


def tau_alt(q, qd, qdd):
    """独立写法：直接展开 2R 的 M/C/g（不经 E66 的函数），用于模型身份交叉验证。"""
    c2, s2 = math.cos(q[1]), math.sin(q[1])
    m1, m2 = E66.M1, E66.M2
    l1, l2, lc1, lc2 = E66.L1, E66.L2, E66.LC1, E66.LC2
    i1 = m1 * l1 * l1 / 12.0
    i2 = m2 * l2 * l2 / 12.0
    M11 = m1 * lc1 ** 2 + m2 * (l1 * l1 + lc2 ** 2 + 2 * l1 * lc2 * c2) + i1 + i2
    M12 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + i2
    M22 = m2 * lc2 ** 2 + i2
    h = -m2 * l1 * lc2 * s2
    C1 = h * qd[1] * qd[1] + 2 * h * qd[0] * qd[1]
    C2 = -h * qd[0] * qd[0]
    g1 = (m1 * lc1 + m2 * l1) * E66.G * math.cos(q[0]) + m2 * lc2 * E66.G * math.cos(q[0] + q[1])
    g2 = m2 * lc2 * E66.G * math.cos(q[0] + q[1])
    return np.array([M11 * qdd[0] + M12 * qdd[1] + C1 + g1,
                     M12 * qdd[0] + M22 * qdd[1] + C2 + g2])


def vmax_alt(q, qdot_lim=4.0, n_dir=180, n_bis=70):
    """**独立实现**的最大末端速度：细方向网格 + 纯二分（无早退短路，与 E67 算法不同）。"""
    ang = np.linspace(0, 2 * math.pi, n_dir, endpoint=False)
    best = 0.0
    for th in ang:
        u = np.array([math.cos(th), math.sin(th)])
        if np.any(np.abs(tau_alt(q, np.zeros(2), np.zeros(2))) > TAU_LIM + 1e-12):
            continue
        lo, hi = 0.0, qdot_lim
        if np.any(np.abs(tau_alt(q, qdot_lim * u, np.zeros(2))) <= TAU_LIM + 1e-12):
            s = qdot_lim
        else:
            for _ in range(n_bis):
                mid = 0.5 * (lo + hi)
                if np.any(np.abs(tau_alt(q, mid * u, np.zeros(2))) > TAU_LIM + 1e-12):
                    hi = mid
                else:
                    lo = mid
            s = lo
        best = max(best, float(np.linalg.norm(jac_alt(q) @ (s * u))))
    return best


def e_invariants(E):
    """E 段口径不变量 → 返回 (是否全部成立, 违反清单)。

    规则：① gate=True 行 executed 越限必须 = 0；
          ② demand 字段必须存在；
          ③ ``tau_max_demand ≥ tau_max_executed``（clip 只会减小绝对值）；
          ④ 结论三布尔与明细一致。
    """
    bad = []
    for tag, row in (E.get("scenarios") or {}).items():
        gate_on = tag.endswith("gate_on")
        for sc, d in row.items():
            if "error" in d:
                bad.append("%s/%s 记录为 error" % (tag, sc))
                continue
            if "tau_max_demand" not in d or "violation_steps_demand" not in d:
                bad.append("%s/%s 缺 demand 字段" % (tag, sc))
                continue
            if gate_on and d.get("violation_steps_executed") != 0:
                bad.append("%s/%s gate_on 却 executed 越限=%s" % (tag, sc, d.get("violation_steps_executed")))
            de, dm = d.get("tau_max_executed"), d.get("tau_max_demand")
            if de is not None and dm is not None and dm < de - 1e-9:
                bad.append("%s/%s demand(%.3f) < executed(%.3f)" % (tag, sc, dm, de))
    c = E.get("conclusion") or {}
    g1a = (E.get("scenarios") or {}).get("aggressive_wn40|gate_on", {}).get("G1_dual_arm_independent", {})
    g1b = (E.get("scenarios") or {}).get("aggressive_wn40|gate_off", {}).get("G1_dual_arm_independent", {})
    g2a = (E.get("scenarios") or {}).get("aggressive_wn40|gate_on", {}).get("G2_track_1hz_ff", {})
    g2b = (E.get("scenarios") or {}).get("aggressive_wn40|gate_off", {}).get("G2_track_1hz_ff", {})
    exp_load = bool((g1a.get("violation_steps_demand") or 0) > 0 or (g2a.get("violation_steps_demand") or 0) > 0)
    exp_zero = bool(g1a.get("violation_steps_executed") == 0 and g2a.get("violation_steps_executed") == 0)
    exp_off = bool((g1b.get("violation_steps_demand") or 0) > 0 or (g2b.get("violation_steps_demand") or 0) > 0)
    if c.get("gate_load_bearing") != exp_load:
        bad.append("结论 gate_load_bearing 与明细不符")
    if c.get("gate_on_executed_zero") != exp_zero:
        bad.append("结论 gate_on_executed_zero 与明细不符")
    if c.get("gate_off_really_exceeds") != exp_off:
        bad.append("结论 gate_off_really_exceeds 与明细不符")
    return (len(bad) == 0), bad


def main():
    print("=" * 92)
    print("E67 台架补全 正反双向审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    A, B, C, D, E = (rep["A_velocity_spectrum"], rep["B_dual_arm"], rep["C_moving_target"],
                     rep["D_contact_friction"], rep["E_gate_ablation"])

    # ================= 正向 =================
    print("\n[正向] 报告↔实现↔数字自洽 + 关键量独立重算")

    chk("F1 报告五段齐全（A 速度谱 / B 双臂 / C 跟踪 / D 接触摩擦 / E 门控消融）+ 口径注",
        all(k in rep for k in ("A_velocity_spectrum", "B_dual_arm", "C_moving_target",
                               "D_contact_friction", "E_gate_ablation", "demand_vs_executed_note")))

    # F2 FK/IK 往返
    #   ★ 审核自查（2026-09-17）：`E67.ik` 是**单分支**（elbow=-1，肘下），对 q2>0 的构型
    #     关节角必然不往返（解到另一支），但**末端位置**必须一致。故拆成两条检查：
    #       F2a 任务空间往返：FK(ik(p)) == p（p 由 FK 采样生成 → 保证可达）
    #       F2b 关节空间往返：仅对 **ik 所选分支（q2<0）** 断言，并对 q2>0 断言"位置一致、角度跳支"
    rng = np.random.RandomState(zlib.crc32(b"e67_fkik") % (2 ** 31))
    worst_p = 0.0
    worst_q_same = 0.0
    n_same = n_other = 0
    worst_p_other = 0.0
    for _ in range(400):
        q = [float(rng.uniform(-1.9, 1.9)), float(rng.uniform(-1.9, 1.9))]
        p = E67.fk(q)
        q2 = E67.ik(p)
        worst_p = max(worst_p, float(np.linalg.norm(E67.fk(q2) - p)))
        if q[1] < 0:                      # ik 所选分支
            n_same += 1
            worst_q_same = max(worst_q_same, float(np.max(np.abs(np.asarray(q2) - np.asarray(q)))))
        else:                             # 另一分支：位置须一致，角度必然不同
            n_other += 1
            worst_p_other = max(worst_p_other, float(np.linalg.norm(E67.fk(q2) - p)))
    chk("F2a 任务空间 IK 往返一致（400 随机构型，含两支；末端 ≤1e-12）",
        worst_p <= 1e-12 and worst_p_other <= 1e-12,
        "max|Δp|=%.2e（其中另一分支 %d 例 max|Δp|=%.2e）" % (worst_p, n_other, worst_p_other))
    chk("F2b 关节空间往返仅对 ik 所选分支（q2<0）成立 → 已记录单分支局限",
        worst_q_same <= 1e-9, "同支 %d 例 max|Δq|=%.2e ｜ 另一支 %d 例（预期不往返）"
        % (n_same, worst_q_same, n_other))
    _res.setdefault("notes", []).append(
        "E67.ik 为单分支（elbow=-1）：对 q2>0 构型，FK/IK 在任务空间一致但关节角会跳到另一支。"
        "本台架所有参考均由该单支生成，`_target_feasible` 的可行性也只针对该支 —— 若另一支更优，"
        "当前实现不会选到（已知局限，未影响本轮结论）。")

    # F3 A 段独立重算 v_max（换算法：细网格 + 纯二分 + 独立 M/C/g 展开）
    q_at = A["at"]["q"] if A.get("at") else None
    v_alt = vmax_alt(q_at) if q_at else float("nan")
    chk("F3 A 段 v_max 独立重算一致（换算法/换实现；相对误差 <1e-3）",
        q_at is not None and abs(v_alt - A["v_max_ee"]) / max(A["v_max_ee"], 1e-12) < 1e-3,
        "报告=%.4f  独立=%.4f  q=%s" % (A["v_max_ee"], v_alt, q_at))

    # F4 A 段 jerk = ωn · a_max，ωn 取 budget_gains（复算）
    wn = E66.budget_gains(E66.Q_REF)[2]
    jerk_exp = round(wn * A["a_max_ee"], 3)
    chk("F4 A 段 jerk 上界 = ωn·a_max 且 ωn 与 budget_gains 复算一致（且标注为推得值）",
        abs(jerk_exp - A["jerk_bound"]) < 1e-6 and "推得" in A.get("jerk_bound_note", ""),
        "报告=%.3f  复算=%.3f（ωn=%.3f）" % (A["jerk_bound"], jerk_exp, wn))

    # F5 模型身份交叉验证：E67/E66 用的 M/C/g 与独立展开式一致
    deg = 0.0
    for _ in range(60):
        q = [float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))]
        qd = [float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))]
        qdd = [float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))]
        deg = max(deg, float(np.max(np.abs(E67.tau_at(q, qd, qdd) - tau_alt(q, qd, qdd)))))
    deg_cs = float(np.max(np.abs(E66.mass_matrix([0.6, -0.9]) - CS.mass_matrix([0.6, -0.9]))))
    chk("F5 模型身份：E66/E67 的 τ=M·q̈+C+g 与**独立展开式**一致；且与 control_strategies 同源",
        deg < 1e-12 and deg_cs < 1e-12, "独立展开 max|Δτ|=%.2e ｜ vs CS max|ΔM|=%.2e" % (deg, deg_cs))

    # F6 C 段独立重算（1.0 Hz，无前馈）
    r_rep = C["no_ff"]["rows"]["1.00Hz"]
    r_re = E67._track_arm(1.0, False)
    chk("F6 C 段 1.0 Hz 无前馈独立重算一致（rms/τmax executed/demand）",
        abs((r_re["rms_err_m"] or 0) - (r_rep["rms_err_m"] or 0)) < 1e-9
        and r_re["violation_steps_demand"] == r_rep["violation_steps_demand"]
        and r_re["tau_max_demand"] == r_rep["tau_max_demand"],
        "重算 rms=%.5f demand越限=%d ｜ 报告 rms=%.5f demand越限=%d"
        % (r_re["rms_err_m"], r_re["violation_steps_demand"],
           r_rep["rms_err_m"], r_rep["violation_steps_demand"]))

    # F7 D 段独立重算（mu=0.4）并与解析式一致性检查
    d04 = D["friction_cone_max_push"]["mu=0.4"]
    re04 = E67.max_push([0.7, -1.0], 0.4)
    chk("F7 D 段 μ=0.4 最大切向力独立重算一致，且锥利用 ≤1、Ft ≤ μ·Fn",
        abs(re04["max_tangential_N"] - d04["max_tangential_N"]) < 1e-9
        and d04["cone_utilization"] <= 1.0 + 1e-9
        and d04["max_tangential_N"] <= 0.4 * d04["at_normal_N"] + 1e-9,
        "报告=%.4f N 重算=%.4f N ｜ 锥利用=%.4f" % (d04["max_tangential_N"], re04["max_tangential_N"],
                                                   d04["cone_utilization"]))

    # F8 E 段口径不变量（含 gate_on executed 恒 0 / demand 字段 / demand≥executed / 结论一致）
    ok8, bad8 = e_invariants(E)
    chk("F8 E 段口径不变量全部成立（executed 恒 0、demand 存在、demand≥executed、结论一致）",
        ok8, "违反=%s" % (bad8 or "无"))

    # F9 ★ 关键结论对门控的稳健性：双臂独立到点在 gate 关掉时同样到点
    g1_off = E["scenarios"]["budget|gate_off"]["G1_dual_arm_independent"]
    chk("F9 B 段「独立双末端到点 0.0000 m」不依赖门控（gate_off 也到点）",
        g1_off["errL_m"] == 0.0 and g1_off["errR_m"] == 0.0,
        "gate_off errL=%.5f errR=%.5f（此时 executed 越限 %d 步、τmax=%.3f）"
        % (g1_off["errL_m"], g1_off["errR_m"], g1_off["violation_steps_executed"],
           g1_off["tau_max_executed"]))

    # F10 台架四项点名能力齐备 + 数值可追溯
    caps = {"v_max_ee": A.get("v_max_ee"), "a_max_ee": A.get("a_max_ee"),
            "dual_arm_errL": B["independent"]["errL_m"],
            "track_1cm_ff": C["ff"]["max_trackable_speed_mps_tol1cm"],
            "friction_mu0.8": D["friction_cone_max_push"]["mu=0.8"]["max_tangential_N"]}
    chk("F10 用户点名四项能力齐备且非空（速度谱 / 双臂 / 跟踪 / 接触摩擦）",
        all(v is not None for v in caps.values()), str(caps))

    # F11 双臂目标可行性标签对**所选分支**正确（防"标签是单支产物"的误读）
    branch = {}
    for tag, p, base in (("L", [0.30, 0.25], E67.BASE_L), ("R", [0.55, 0.35], E67.BASE_R)):
        qs = [E67.ik(p, base, elbow=-1.0), E67.ik(p, base, elbow=+1.0)]
        branch[tag] = {"chosen_in_limits": bool(max(abs(qs[0][0]), abs(qs[0][1])) <= E67.JLIM),
                       "other_branch_in_limits": bool(max(abs(qs[1][0]), abs(qs[1][1])) <= E67.JLIM),
                       "q_chosen": [round(v, 4) for v in qs[0]]}
    rep_lab = {"L": B["target_feasibility_L"]["in_joint_limits"], "R": B["target_feasibility_R"]["in_joint_limits"]}
    chk("F11 双臂目标可行性标签与**所选分支**的限位判定一致",
        all(rep_lab[k] == branch[k]["chosen_in_limits"] for k in ("L", "R")), str(branch))

    # ================= 反向 =================
    print("\n[反向] 构造反例 / 边界 / 退化，检验结论非空且篡改可被抓")

    # R1 越工作空间：|p| > L1+L2 → ik 必然落不到该点（IK 不应假装成功）
    p_out = np.array([1.2, 0.0])
    q_out = E67.ik(p_out)
    err_out = float(np.linalg.norm(E67.fk(q_out) - p_out))
    chk("R1 越工作空间目标：ik 不假装成功（FK(ik(p)) ≠ p 明显偏离）",
        err_out > 0.3, "|p|=%.2f (L1+L2=%.2f) → 残差 %.4f m" % (np.linalg.norm(p_out),
                                                              E67.L1 + E67.L2, err_out))

    # R2 门控消融对照非空：gate_off + 激进增益必须真的越限
    g_off_agg = E["scenarios"]["aggressive_wn40|gate_off"]
    tot = sum((d.get("violation_steps_demand") or 0) for d in g_off_agg.values())
    chk("R2 门控消融对照非空（gate_off+激进增益确实越限）", tot > 0,
        "无门控 demand 越限合计 %d 步" % tot)

    # R3 「executed 恒 0」不是永真断言：关掉门控后该字段必须 > 0（有分辨力）
    exec_on = sum((d.get("violation_steps_executed") or 0) for d in E["scenarios"]["budget|gate_on"].values())
    exec_off = sum((d.get("violation_steps_executed") or 0) for d in E["scenarios"]["budget|gate_off"].values())
    chk("R3 executed 字段有分辨力（gate_on=0 但 gate_off>0，非永真）",
        exec_on == 0 and exec_off > 0, "gate_on=%d ｜ gate_off=%d" % (exec_on, exec_off))

    # R4 摩擦单调性：μ 增大 → 最大切向力不减，且不超防滑上限
    mus = [0.2, 0.4, 0.6, 0.8]
    fts = [D["friction_cone_max_push"]["mu=%.1f" % m]["max_tangential_N"] for m in mus]
    mono = all(fts[i] <= fts[i + 1] + 1e-9 for i in range(len(fts) - 1))
    within = all(D["friction_cone_max_push"]["mu=%.1f" % m]["max_tangential_N"]
                 <= m * D["friction_cone_max_push"]["mu=%.1f" % m]["at_normal_N"] + 1e-9 for m in mus)
    chk("R4 摩擦物理单调性（μ↑ ⇒ Ft↑，且 Ft ≤ μ·Fn）", mono and within, str(dict(zip(mus, fts))))

    # R5 跟踪误差随频率不降（0.2→1.0 Hz 段；2 Hz 后的回落系饱和效应，不纳入单调性断言）
    seg = [C["no_ff"]["rows"]["%.2fHz" % f]["rms_err_m"] for f in (0.2, 0.5, 1.0)]
    chk("R5 跟踪误差随频率单调增（0.2/0.5/1.0 Hz，无前馈）",
        all(seg[i] <= seg[i + 1] + 1e-9 for i in range(len(seg) - 1)),
        str(["%.5f" % v for v in seg]))

    # R6 确定性：1 Hz 跟踪连算两次必须完全一致
    a1 = E67._track_arm(1.0, False)
    a2 = E67._track_arm(1.0, False)
    chk("R6 确定性（同参数两次运行结果逐字段一致）",
        json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True))

    # R7 阳性对照：篡改 demand（×0.5）必须被 F8 的不变量抓住
    bad = json.loads(json.dumps(E))
    for tag, row in bad["scenarios"].items():
        for sc, d in row.items():
            if "tau_max_demand" in d and d["tau_max_demand"] is not None:
                d["tau_max_demand"] = round(d["tau_max_demand"] * 0.5, 4)
    ok7, why7 = e_invariants(bad)
    chk("R7 阳性对照：篡改 tau_max_demand ×0.5 必被抓住", (not ok7) and len(why7) > 0,
        "抓到 %d 项，例：%s" % (len(why7), why7[:2]))

    # R8 阴性对照：未篡改 → 不报任何违反
    ok8b, why8b = e_invariants(E)
    chk("R8 阴性对照：未篡改报告 → 违反项为 0", ok8b and not why8b)

    # R9 篡改结论布尔也必须被抓
    bad2 = json.loads(json.dumps(E))
    bad2["conclusion"]["gate_off_really_exceeds"] = not bad2["conclusion"]["gate_off_really_exceeds"]
    ok9, why9 = e_invariants(bad2)
    chk("R9 篡改结论布尔（翻转 gate_off_really_exceeds）必被抓住", not ok9, str(why9))

    # R10 _num 对非有限值返回 None（报告必须是可移植的标准 JSON）
    nan_ok = (E67._num(float("nan")) is None) and (E67._num(float("inf")) is None) \
        and (E67._num(1.23456, 3) == 1.235)
    txt = json.dumps(rep)
    chk("R10 非有限值映射为 None 且报告文本无裸 NaN/Infinity（可移植 JSON）",
        nan_ok and ("NaN" not in txt) and ("Infinity" not in txt))

    # R11 需求口径的"承重"事实必须留在报告里（预算增益下也已越限）
    bg = E["scenarios"]["budget|gate_on"]["G1_dual_arm_independent"]
    chk("R11 报告保留了「预算增益下 demand 亦已越限」这一事实（不被 executed=0 掩盖）",
        (bg.get("violation_steps_demand") or 0) > 0 and bg.get("violation_steps_executed") == 0,
        "budget|gate_on 双臂：demand 越限 %s 步、τmax 需求 %.3f ／ 执行 %.3f"
        % (bg.get("violation_steps_demand"), bg.get("tau_max_demand") or -1, bg.get("tau_max_executed") or -1))

    # R12 双臂对称协同漂移≈0 是平凡的（已标注），非对称负载才是有效读数
    sym = B["coordinated_carry"]["rel_drift_max_m"]
    asym = B["coordinated_carry_asym_load"]["rel_drift_max_m"]
    chk("R12 对称协同漂移≈0（平凡，已在 note 标注）｜非对称负载漂移非零（有效读数）",
        sym <= 1e-9 and asym > 1e-4,
        "对称=%.6f m ｜ 非对称(+0.25kg)=%.6f m" % (sym, asym))

    _res["n_pass"] = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_total"] = len(_res["checks"])
    _res["audit_pass"] = bool(_res["n_pass"] == _res["n_total"])
    _res["findings"] = [
        "★ 口径发现：E67 原报告只记 τmax（执行值，恒=5.0），把「预算增益下需求即已越限」这一事实藏住了；"
        "补齐 demand 口径后实测：双臂独立到点 demand 越限 82 步、需求 τmax=6.677（上限 5.0）；"
        "1.0 Hz 跟踪 demand 越限 12 步、τmax=5.639。",
        "★ 结论稳健性：双臂独立到点 0.0000 m 在门控关闭时同样成立（gate_off 也 0.0），"
        "故该结论不依赖 clip 干预；被 clip 的 82/6000 步（1.4%）未改变末端结果。",
        "★ 新发现：激进增益（ωn=40）下 1.0 Hz 跟踪 rms 由 41.6 mm 降到 1.66 mm（within_tol=True），"
        "代价是 405/6000 步处于饱和 → C 段报告的「最大可跟踪速度」是**保守增益下的值**，"
        "不是硬件物理上限；引用时必须写明口径。",
        "A 段 jerk 上界随 ωn 修正由 341 → 513.15 m/s³（ωn 由 5.162 → 7.758，B2 修复的连带影响）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, ensure_ascii=False, indent=2)
    print("\n审核结果：%d/%d PASS ｜ audit_pass = %s" % (_res["n_pass"], _res["n_total"], _res["audit_pass"]))
    print("wrote", OUT)
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
