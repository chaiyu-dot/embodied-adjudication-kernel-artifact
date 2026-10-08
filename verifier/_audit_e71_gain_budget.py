# -*- coding: utf-8 -*-
"""_audit_e71_gain_budget.py —— E71 增益预算化优化的**正反双向审核**（零 API、纯 CPU）。

为什么需要
----------
E71 是 P0-3（C4 抗饱和）的核心交付：把 E67 E 段暴露的「目标点预算增益不约束过渡段
（双臂独立到点 82/6000 步需求越限、需求 τmax 6.677 > 5.0）」用**沿轨迹预算化的 PD 增益**修掉。
本审核分两路：
  · 正向：报告↔实现↔数字三者自洽；关键量**独立重算**（A 段 before/after 闭环仿真、B 段跟踪采样）；
  · 反向：构造反例 / 边界 / 退化，检验结论不是空断言，且**篡改能被抓住**。

重点盯住本轮新增口径
--------------------
① 「越限」必须用**门控前需求值**（E67 口径铁律）；② C2「到点保持」须用 **1cm 任务容差**
（苛求 ==0 是错的：刚度↓→整定变慢，固定 6s 末残差 0.24/0.12 mm 仍远优于容差）；
③ B 段「无饱和可达」边界 = None 是**真实发现**（1cm 跟踪容差与零饱和在该台架不可同时零裕度满足）。

产物：_audit_e71_gain_budget.json
"""
import json
import math
import os
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
import e66_closed_loop_control as E66      # noqa: E402
import e67_testbed_completeness as E67     # noqa: E402
from planning import control_strategies as CS   # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e71_gain_budget_report.json")
OUT = os.path.join(EVAL, "_audit_e71_gain_budget.json")
TAU_LIM = np.asarray(CS.TAU_LIM, float)
FRAC = 0.9
ZETA = 0.7
T_B = 6.0
TGT_B = [[0.30, 0.25], [0.55, 0.35]]
TOL_CM = 0.01

_res = {"experiment": "E71 增益预算化优化 正反双向审核", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def peak_fn(kp, kd):
    """与 e71 同源的闭环仿真实测峰值（权威口径）。"""
    r = E67.dual_sim(TGT_B, T=T_B, gains=(kp, kd), gate=True)
    return np.asarray(r["tau_max_demand_per_joint"], float)


# ---------------------------------------------------------------- 报告不变量（反向篡改检测用）
def e71_invariants(rep):
    """报告内部自洽 → 返回 (是否全部成立, 违反清单)。

    规则：① 报告的 C1/C2/C3 布尔须与 A/B 段数据重算一致；
          ② B 段两条边界须与 rows 重算一致；
          ③ verdict_pass 须 = C1∧C2∧C3。
    """
    bad = []
    A = rep["A_trajectory_budget"]
    B = rep["B_capability_curve"]
    C = rep["C_consistency"]
    d = A["delta"]
    tb = A["trajectory_budget"]
    c1 = (tb["alpha"] <= 1.0 + 1e-9) and (tb["ratio_after"] <= 1.0 + 1e-9)
    c2 = (d["errL_m"][1] <= TOL_CM + 1e-9) and (d["errR_m"][1] <= TOL_CM + 1e-9)
    c3 = (d["demand_violation_steps"][1] == 0)
    if C["C1_alpha_le_1_and_within_budget"] != c1:
        bad.append("C1 布尔与数据不符")
    if C["C2_endpoint_precision_kept"] != c2:
        bad.append("C2 布尔与数据不符")
    if C["C3_demand_violation_removed"] != c3:
        bad.append("C3 布尔与数据不符")
    rows = B["rows"]
    any_b = max([v["max_trackable_hz_any"] for v in rows.values() if v["max_trackable_hz_any"]], default=None)
    uns_b = max([v["max_trackable_hz_unsaturated"] for v in rows.values()
                 if v["max_trackable_hz_unsaturated"]], default=None)
    if B["boundary_any_hz"] != any_b:
        bad.append("boundary_any_hz 与 rows 不符")
    if B["boundary_unsaturated_hz"] != uns_b:
        bad.append("boundary_unsaturated_hz 与 rows 不符")
    if rep.get("verdict_pass") != (c1 and c2 and c3):
        bad.append("verdict_pass 与 C1∧C2∧C3 不符")
    return (len(bad) == 0), bad


def main():
    print("=" * 92)
    print("E71 增益预算化优化 正反双向审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    A = rep["A_trajectory_budget"]
    B = rep["B_capability_curve"]
    C = rep["C_consistency"]

    # ================= 正向 =================
    print("\n[正向] 报告↔实现↔数字自洽 + 关键量独立重算")

    chk("F1 报告三段齐全（A 轨迹预算 / B 能力曲线 / C 一致性）+ verdict 字段",
        all(k in rep for k in ("A_trajectory_budget", "B_capability_curve",
                               "C_consistency", "verdict_pass", "verdict_note")))

    # F2 独立重算 A 段「before」（目标点预算增益，闭环仿真）
    kp_hi = float(A["target_pose_budget"]["kp"][0])
    kd_hi = float(A["target_pose_budget"]["kd"][0])
    before = E67.dual_sim(TGT_B, T=T_B, gains=(kp_hi, kd_hi), gate=True)
    chk("F2 A 段 before 独立重算一致（需求越限步数 / 需求 τmax 逐关节）",
        before["violation_steps_demand"] == A["delta"]["demand_violation_steps"][0]
        and abs(before["tau_max_demand"] - A["delta"]["tau_max_demand"][0]) < 1e-6
        and np.allclose(before["tau_max_demand_per_joint"], A["delta"]["tau_max_demand_per_joint"][0], atol=1e-4),
        "重算 越限=%d τmax=%.4f 逐关节%s ｜ 报告 越限=%d τmax=%.4f 逐关节%s"
        % (before["violation_steps_demand"], before["tau_max_demand"], before["tau_max_demand_per_joint"],
           A["delta"]["demand_violation_steps"][0], A["delta"]["tau_max_demand"][0],
           A["delta"]["tau_max_demand_per_joint"][0]))

    # F3 独立重算 budget_gains_trajectory（与 e71 同参）
    kp_hi_a = np.array([kp_hi, kp_hi])
    kd_hi_a = np.array([kd_hi, kd_hi])
    kp_tb, kd_tb, det = CS.budget_gains_trajectory(
        [0.0, 0.0], [0.5, -0.7], T=T_B, tau_lim=TAU_LIM, zeta=ZETA, frac=FRAC,
        peak_fn=peak_fn, kp_hi=kp_hi_a, kd_hi=kd_hi_a)
    chk("F3 轨迹预算化增益独立重算一致（α / ratio_after / peak_after）",
        abs(det["alpha"] - A["trajectory_budget"]["alpha"]) < 1e-4
        and abs(det["ratio_after"] - A["trajectory_budget"]["ratio_after"]) < 1e-4
        and np.allclose(det["peak_after_Nm"], A["trajectory_budget"]["peak_after_Nm"], atol=1e-3),
        "重算 α=%.4f ratio_after=%.4f peak_after=%s ｜ 报告 α=%.4f ratio_after=%.4f peak_after=%s"
        % (det["alpha"], det["ratio_after"], det["peak_after_Nm"],
           A["trajectory_budget"]["alpha"], A["trajectory_budget"]["ratio_after"],
           A["trajectory_budget"]["peak_after_Nm"]))

    # F4 独立重算 A 段「after」（预算化增益，闭环仿真）
    after = E67.dual_sim(TGT_B, T=T_B, gains=(kp_tb, kd_tb), gate=True)
    chk("F4 A 段 after 独立重算一致（需求越限=0 / 需求 τmax / 端点误差≤1cm）",
        after["violation_steps_demand"] == 0
        and abs(after["tau_max_demand"] - A["delta"]["tau_max_demand"][1]) < 1e-6
        and after["errL_m"] <= TOL_CM and after["errR_m"] <= TOL_CM,
        "重算 越限=%d τmax=%.4f errL=%.5f errR=%.5f ｜ 报告 越限=%d τmax=%.4f errL=%.5f errR=%.5f"
        % (after["violation_steps_demand"], after["tau_max_demand"], after["errL_m"], after["errR_m"],
           A["delta"]["demand_violation_steps"][1], A["delta"]["tau_max_demand"][1],
           A["delta"]["errL_m"][1], A["delta"]["errR_m"][1]))

    # F5 B 段采样独立重算（wn=40 @ 1.0Hz，含速度前馈）
    wn40 = B["rows"]["wn=40"]
    kp40 = np.asarray(wn40["kp"][:2], float)
    kd40 = np.asarray(wn40["kd"][:2], float)
    r40 = E67._track_arm(1.0, True, gains=(kp40, kd40))
    rep40 = wn40["per_freq"]["1.00Hz"]
    n40 = max(int(r40["n_steps"]), 1)
    sf40 = r40["violation_steps_demand"] / n40
    chk("F5 B 段 wn=40@1.0Hz 独立重算一致（rms / sat_frac=越限步÷n / within_tol）",
        abs((r40["rms_err_m"] or 0) - (rep40["rms_err_m"] or 0)) < 1e-9
        and abs(sf40 - rep40["sat_frac_demand"]) < 1e-4
        and bool(r40["within_tol"]) == bool(rep40["within_tol_1cm"]),
        "重算 rms=%.6f sat=%.4f tol=%s ｜ 报告 rms=%.6f sat=%.4f tol=%s"
        % (r40["rms_err_m"], sf40, r40["within_tol"],
           rep40["rms_err_m"], rep40["sat_frac_demand"], rep40["within_tol_1cm"]))

    # F6 C1/C2/C3 从数据重算与报告一致（不变量自检）
    ok6, bad6 = e71_invariants(rep)
    chk("F6 C1/C2/C3 与 A/B 数据重算一致 + verdict 自洽", ok6, "违反=%s" % (bad6 or "无"))

    # F7 B 段两条边界与 rows 自洽 + 能力数值标注口径（无饱和=可作主张；饱和=不作主张）
    any_b = max([v["max_trackable_hz_any"] for v in B["rows"].values() if v["max_trackable_hz_any"]], default=None)
    uns_b = max([v["max_trackable_hz_unsaturated"] for v in B["rows"].values()
                 if v["max_trackable_hz_unsaturated"]], default=None)
    chk("F7 B 段边界自洽；且报告注明「无饱和可作主张 / 饱和不作主张」",
        B["boundary_any_hz"] == any_b and B["boundary_unsaturated_hz"] == uns_b
        and "unsaturated" in B["note"].lower() and "saturated" in B["note"].lower(),
        "boundary_any=%.2f boundary_unsat=%s（报告 any=%.2f unsat=%s）"
        % (any_b, uns_b, B["boundary_any_hz"], B["boundary_unsaturated_hz"]))

    # F8 能力数值标注增益口径（防当硬件上限）：每行有 wn_rad_s，且 design 引用带宽
    design = str(B.get("design", ""))
    cap_fields = ("wn_rad_s" in next(iter(B["rows"].values())))
    caps_labeled = ("ωn" in design) or ("wn" in design.lower()) or ("增益" in design)
    chk("F8 B 段每行标增益口径（wn_rad_s 存在 + design 引带宽，非裸硬件上限）",
        cap_fields and caps_labeled, "wn_rad_s 存在=%s ｜ design 含带宽标记=%s" % (cap_fields, caps_labeled))

    # ================= 反向 =================
    print("\n[反向] 构造反例 / 边界 / 退化，检验结论非空且篡改可被抓")

    # R1 阴性对照：未篡改报告 → 不变量全过
    ok1, why1 = e71_invariants(rep)
    chk("R1 阴性对照：未篡改报告 → 违反项为 0", ok1 and not why1)

    # R2 阳性对照：篡改 A.delta 的 after 越限步数（0→100）→ 必被抓
    bad = json.loads(json.dumps(rep))
    bad["A_trajectory_budget"]["delta"]["demand_violation_steps"][1] = 100
    ok2, why2 = e71_invariants(bad)
    chk("R2 阳性对照：篡改 after 越限步数 0→100 必被抓住", (not ok2) and len(why2) > 0,
        "抓到 %d 项，例：%s" % (len(why2), why2[:2]))

    # R3 阳性对照：翻转 verdict_pass → 必被抓
    bad3 = json.loads(json.dumps(rep))
    bad3["verdict_pass"] = not bad3["verdict_pass"]
    ok3, why3 = e71_invariants(bad3)
    chk("R3 阳性对照：翻转 verdict_pass 必被抓住", not ok3, str(why3))

    # R4 泄漏检查：before 的 82 步越限是**门控前需求**（预 clip）。关掉门控后动力学轨迹变，
    # 需求越限步数会变（69），但违规不消失（仍 >0）→ 证明违规是真实需求、非门控产物；
    # 且 gate_on 执行=0（被 clip）vs gate_off 执行>0（无 clip）→ 门控确实承重。
    before_off = E67.dual_sim(TGT_B, T=T_B, gains=(kp_hi, kd_hi), gate=False)
    chk("R4 越限是真实需求（非门控产物）：gate_off 后 demand 仍越限；gate_on 执行=0 证明门控承重",
        before_off["violation_steps_demand"] > 0 and before["violation_steps_demand"] > 0
        and before_off["violation_steps_executed"] > 0 and before["violation_steps_executed"] == 0,
        "gate_on demand 越限=%d（执行=0，承重）｜ gate_off demand 越限=%d（执行=%d，无 clip）"
        % (before["violation_steps_demand"], before_off["violation_steps_demand"],
           before_off["violation_steps_executed"]))

    # R5 确定性：budget_gains_trajectory 连算两次 α 完全一致
    _, _, det_a = CS.budget_gains_trajectory([0.0, 0.0], [0.5, -0.7], T=T_B, tau_lim=TAU_LIM,
                                             zeta=ZETA, frac=FRAC, peak_fn=peak_fn,
                                             kp_hi=kp_hi_a, kd_hi=kd_hi_a)
    _, _, det_b = CS.budget_gains_trajectory([0.0, 0.0], [0.5, -0.7], T=T_B, tau_lim=TAU_LIM,
                                             zeta=ZETA, frac=FRAC, peak_fn=peak_fn,
                                             kp_hi=kp_hi_a, kd_hi=kd_hi_a)
    chk("R5 确定性（同参数两次预算化 α 一致）", abs(det_a["alpha"] - det_b["alpha"]) < 1e-9,
        "α_a=%.6f α_b=%.6f" % (det_a["alpha"], det_b["alpha"]))

    # R6 边界健全性：peak_fn 返回极小（不承压）→ α=1（无需缩放）；返回极大 → α 远小于 1
    kp1, kd1, det1 = CS.budget_gains_trajectory(
        [0.0, 0.0], [0.5, -0.7], T=T_B, tau_lim=TAU_LIM, zeta=ZETA, frac=FRAC,
        peak_fn=lambda kp, kd: np.array([0.0, 0.0]), kp_hi=kp_hi_a, kd_hi=kd_hi_a)
    kp2, kd2, det2 = CS.budget_gains_trajectory(
        [0.0, 0.0], [0.5, -0.7], T=T_B, tau_lim=TAU_LIM, zeta=ZETA, frac=FRAC,
        peak_fn=lambda kp, kd: np.array([1000.0, 1000.0]), kp_hi=kp_hi_a, kd_hi=kd_hi_a)
    chk("R6 边界健全性：零承压→α=1；千倍承压→α<<1",
        abs(det1["alpha"] - 1.0) < 1e-9 and det2["alpha"] < 0.5,
        "零承压 α=%.4f ｜ 千倍承压 α=%.4f" % (det1["alpha"], det2["alpha"]))

    # R7 C4/C5 弱单调：审计独立重算 e71 的单调陈述
    keys = list(B["rows"].keys())
    freqs = list(B["rows"][keys[0]]["per_freq"].keys())
    mono_ok = True
    for fk in freqs:
        seq_rms = [B["rows"][k]["per_freq"][fk]["rms_err_m"] for k in keys]
        seq_sat = [B["rows"][k]["per_freq"][fk]["sat_frac_demand"] for k in keys]
        if not all(seq_rms[i] >= seq_rms[i + 1] - 5e-3 for i in range(len(seq_rms) - 1)):
            mono_ok = False
        if not all(seq_sat[i] <= seq_sat[i + 1] + 1e-9 for i in range(len(seq_sat) - 1)):
            mono_ok = False
    chk("R7 C4/C5 弱单调独立重算成立（rms 随增益↓、sat 随增益↑）", mono_ok,
        "频率=%s" % freqs)

    # R8 非有限值映射为 None（报告可移植标准 JSON）
    nan_ok = (E67._num(float("nan")) is None) and (E67._num(float("inf")) is None) \
        and (E67._num(1.23456, 3) == 1.235)
    txt = json.dumps(rep)
    chk("R8 非有限值→None 且报告文本无裸 NaN/Infinity（可移植 JSON）",
        nan_ok and ("NaN" not in txt) and ("Infinity" not in txt))

    # R9 刚度代价的诚实性：α<1 ⇒ 预算化付出了带宽代价（wn_after < wn_before），但端点仍在容差内
    chk("R9 刚度代价诚实报告（α<1 ⇒ ωn_after<ωn_before；且端点误差≤1cm 不因缩放退化出容差）",
        A["trajectory_budget"]["alpha"] < 1.0 - 1e-9
        and A["trajectory_budget"]["wn_after_rad_s"] < A["trajectory_budget"]["wn_before_rad_s"]
        and A["delta"]["errL_m"][1] <= TOL_CM and A["delta"]["errR_m"][1] <= TOL_CM,
        "α=%.4f ωn %.3f→%.3f ｜ errL=%.5f errR=%.5f（容差 %.2f m）"
        % (A["trajectory_budget"]["alpha"], A["trajectory_budget"]["wn_before_rad_s"],
           A["trajectory_budget"]["wn_after_rad_s"], A["delta"]["errL_m"][1],
           A["delta"]["errR_m"][1], TOL_CM))

    _res["n_pass"] = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_total"] = len(_res["checks"])
    _res["audit_pass"] = bool(_res["n_pass"] == _res["n_total"])
    _res["findings"] = [
        "★ 核心修复成立：目标点预算增益下双臂独立到点 **需求越限 82 步、τmax 6.677 > 5.0**；"
        "轨迹预算化增益（α=%.4f，二分为闭环仿真实测峰值）将需求越限 **82→0**、τmax→4.5（=0.9·τ_lim，绑定关节），"
        "代价是带宽 ↓（ωn %.3f→%.3f）：固定 6s 末残差 0.24/0.12 mm，仍远优于 1cm 任务容差。"
        % (A["trajectory_budget"]["alpha"], A["trajectory_budget"]["wn_before_rad_s"],
           A["trajectory_budget"]["wn_after_rad_s"]),
        "★ 口径发现：before 的 82 步越限是**门控前需求**（预 clip），gate_off 后 demand 仍越限（R4 证明），"
        "故不能用「门控吸收了」来掩盖——这正是 E67 E 段暴露的根因：budget_gains 只约束目标位形静态力矩。",
        "★ 新发现：B 段「无饱和可达」边界 = None——在 1cm 跟踪容差下，达到容差所需的最低增益"
        "（wn=6@0.2Hz）sat_frac≈1.07% 略超 1% 阈值。即 **1cm 跟踪容差与零饱和不可同时零裕度满足**，"
        "须接受边际饱和或放宽容差；该读数**不作能力主张**，只作设计参考。",
        "★ 能力数值纪律落地：B 段每行标注 wn 增益口径（非硬件上限）；饱和运行（无裕度、有热/磨损代价）单列。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, ensure_ascii=False, indent=2)
    print("\n审核结果：%d/%d PASS ｜ audit_pass = %s" % (_res["n_pass"], _res["n_total"], _res["audit_pass"]))
    print("wrote", OUT)
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
