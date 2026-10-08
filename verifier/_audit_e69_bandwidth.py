# -*- coding: utf-8 -*-
"""_audit_e69_bandwidth.py —— E69（C2 回填）的**正反双向审核**（零 API、纯 CPU、可复跑）。

正向
  F1 报告完整 + verdict_pass
  F2 独立重算：逐关节 kp/ωn、E66 配方的复现（不信任打印）
  F3 判据 ↔ 实现：head=frac·(τ_lim−τ_ff) → kp=head/|Δq| → ωn=sqrt(kp/M_ii) → min
  F4 口径登记：两套台架的 τ_lim / 质量 / Δq / T_s 必须显式分列
  F5 **E58 未被改动**（其报告仍记 ωn=40，脚本仍用 OMEGA_N）

反向（构造反例/攻击）
  R1 单调性：ωn 对 |Δq| / mp / τ_ff 单调不增
  R2 边界：Δq→0 不发散（min_step 兜底）；head≤0 → ωn=0
  R3 回填是否偷偷保留假设值：逐条 budget ωn 必须全部 ≠ 40，且 max < 40
  R4 经验标定有效：网格上必须存在**从不合格→合格**的翻转；且判据必须用 gate=False（否则同义反复）
  R5 两类 ωn 逐条不同（回填真的生效，不是换了标签）
  R6 推荐的 k_omega 可复现：用该值重算 → 慢档误报 <0.10 且盲区 >0
  R7 分母口径：V3（全体 slow）与 V3b（D_ok 子集）的统计口径不同 → 必须登记

用法：python _audit_e69_bandwidth.py     产物：_audit_e69_bandwidth.json
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

from planning import control_strategies as CS          # noqa: E402
from planning.trackability import trackability_margin  # noqa: E402
import e58_trackability_gate as E58                    # noqa: E402
import e66_closed_loop_control as E66                  # noqa: E402
import e69_bandwidth_refill as E69                     # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EVAL, "_audit_e69_bandwidth.json")
res = {"checks": []}


def chk(name, cond, detail=""):
    res["checks"].append({"check": name, "pass": bool(cond), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + detail) if detail else ""))


def main():
    rep = json.load(open(E69.REPORT, encoding="utf-8"))
    Q = [0.60, -0.90]

    # ---------------- 正向 ----------------
    print("=== F 正向审核 ===")
    need = ["V0_consistency_with_E66", "V1_curve", "V2_empirical", "V3_e58_refill",
            "V3b_komega_recalibration", "V4_conclusion", "verdict_pass"]
    chk("F1 报告含全部必需键", all(k in rep for k in need),
        "缺失=%s" % [k for k in need if k not in rep])
    chk("F1b verdict_pass=True", rep["verdict_pass"] is True)

    # F2 独立重算
    kp_b, kd_b, wn_b = E66.budget_gains(Q)
    wn, det = CS.achievable_omega_n(np.asarray(Q), np.asarray(Q), tau_lim=CS.TAU_LIM,
                                    frac=0.9, tau_ff=np.zeros(2))
    hand = []
    M = CS.mass_matrix(Q)
    for i, dq in enumerate(Q):
        head = 0.9 * CS.TAU_LIM[i]
        kp = head / abs(dq)
        hand.append(math.sqrt(kp / float(M[i, i])))
    chk("F2 逐关节 ωn 独立重算一致", abs(min(hand) - wn) < 1e-9,
        "hand=%.6f impl=%.6f" % (min(hand), wn))
    chk("F2b E66 kp_max 复现", abs(kp_b - 1.8) < 1e-9, "kp=%.4f" % kp_b)
    wn_per_joint = min(math.sqrt((0.9 * E66.TAU_LIM[i] / abs(Q[i])) / float(M[i, i])) for i in range(2))
    chk("F2c E66.budget_gains 的 ωn 已修为**逐关节口径**（与手算一致）",
        abs(wn_per_joint - wn_b) < 1e-6, "手算=%.6f budget_gains=%.6f" % (wn_per_joint, wn_b))
    _legacy_scalar = math.sqrt(kp_b / float(np.mean(np.diag(M))))
    chk("F2d 历史 bug 值得以保留（min kp ÷ mean M = %.4f ≠ 逐关节 %.4f）" % (_legacy_scalar, wn_b),
        abs(_legacy_scalar - wn_b) > 1e-3, "legacy=%.4f correct=%.4f" % (_legacy_scalar, wn_b))

    # F3 判据↔实现
    src = open(os.path.join(_SRC, "planning", "control_strategies.py"), encoding="utf-8").read()
    ok_impl = all(s in src for s in ("head = frac * (tau_lim - tau_ff)",
                                     "k = float(head[i]) / max(abs(float(dq[i])), min_step)",
                                     "w = math.sqrt(k / mii)"))
    chk("F3 判据式与实现逐条对应（head → kp → ωn）", ok_impl)

    # F4 口径登记
    caveats = [
        "E69 V1/V2 台架：τ_lim=[5,1.8]，质量 0.6/0.35（=E66/E68 同一台架）",
        "E69 V3 台架：τ_lim=[8,4]，质量随 load∈{0.5,1,2} 缩放（=E58 原始 bench，未改）",
        "E69 frac=0.9；E58 k_omega=2.0 / k_margin=2.0 / T_s=0.001",
    ]
    res["F4_caveats"] = caveats
    print("     口径登记：" + " ｜ ".join(caveats))

    # F5 E58 未被改动
    e58_path = os.path.join(EVAL, "e58_trackability_gate.py")
    e58_src = open(e58_path, encoding="utf-8").read()
    e58_rep = json.load(open(os.path.join(EVAL, "e58_trackability_report.json"), encoding="utf-8"))
    chk("F5 E58 脚本仍用假设 ωn=40（未被回填篡改）",
        "OMEGA_N, ZETA, T_S = 40.0, 0.7, 0.001" in e58_src)
    chk("F5b E58 原报告仍记 ωn=40（历史记录保持原样）",
        abs(float(e58_rep["design"]["actuator"]["omega_n"]) - 40.0) < 1e-9,
        "E58 report ωn=%.1f" % e58_rep["design"]["actuator"]["omega_n"])

    # ---------------- 反向 ----------------
    print("\n=== R 反向审核（构造反例/攻击） ===")

    # R1 单调性
    base = CS.achievable_omega_n(np.asarray(Q), np.array([0.6, 0.9]), tau_lim=CS.TAU_LIM, frac=0.9)[0]
    dq_inc = [CS.achievable_omega_n(np.asarray(Q), np.array([d, d]), tau_lim=CS.TAU_LIM, frac=0.9)[0]
              for d in (0.1, 0.3, 0.6, 1.2, 2.4)]
    chk("R1a ωn 对 |Δq| 单调不增", all(dq_inc[i] >= dq_inc[i + 1] - 1e-12 for i in range(len(dq_inc) - 1)),
        " ".join("%.2f" % v for v in dq_inc))
    mp_inc = [CS.achievable_omega_n(np.asarray(Q), np.array([0.6, 0.9]), tau_lim=CS.TAU_LIM,
                                    frac=0.9, mp=m)[0] for m in (0.0, 0.1, 0.3, 0.6)]
    chk("R1b ωn 对负载 mp 单调不增", all(mp_inc[i] >= mp_inc[i + 1] - 1e-12 for i in range(len(mp_inc) - 1)),
        " ".join("%.3f" % v for v in mp_inc))
    ff_inc = [CS.achievable_omega_n(np.asarray(Q), np.array([0.6, 0.9]), tau_lim=CS.TAU_LIM,
                                    frac=0.9, tau_ff=np.array([f, f]))[0] for f in (0.0, 1.0, 2.0, 4.0)]
    chk("R1c ωn 对前馈 τ_ff 单调不增", all(ff_inc[i] >= ff_inc[i + 1] - 1e-12 for i in range(len(ff_inc) - 1)),
        " ".join("%.3f" % v for v in ff_inc))

    # R2 边界
    tiny = CS.achievable_omega_n(np.asarray(Q), np.array([1e-9, 1e-9]), tau_lim=CS.TAU_LIM, frac=0.9)[0]
    chk("R2a Δq→0 时 ωn 有限（min_step 兜底，不发散）", np.isfinite(tiny) and tiny > 0,
        "ωn(Δq=1e-9)=%.3f" % tiny)
    zero = CS.achievable_omega_n(np.asarray(Q), np.array([0.6, 0.9]), tau_lim=np.array([1.0, 1.0]),
                                 frac=0.9, tau_ff=np.array([5.0, 5.0]))[0]
    chk("R2b 前馈吃掉预算 → ωn=0（判不可达而非给负值）", zero == 0.0, "ωn=%.3f" % zero)

    # R3 回填没偷偷保留假设值
    rows = json.load(open(os.path.join(EVAL, "_e69_e58_rows.json"), encoding="utf-8"))
    wns = [r["wn_budget_rad_s"] for r in rows]
    chk("R3a 逐条 budget ωn 全部 ≠ 40", all(abs(w - 40.0) > 1e-9 for w in wns))
    chk("R3b max(budget ωn) < 40（假设值确属不可达）", max(wns) < 40.0, "max=%.3f" % max(wns))
    e69_src = open(E69.__file__, encoding="utf-8").read()
    hard40 = re.findall(r"=\s*40\.0\s*(?:#|$|\n)", e69_src)
    chk("R3c E69 本体不硬编码 40.0 作为 ωn（只从 E58.OMEGA_N 读）",
        len(hard40) == 0 and "E58.OMEGA_N" in e69_src, "hardcoded40=%d" % len(hard40))

    # R4 经验标定有效性
    grid = rep["V2_empirical"]["grid"]
    has_flip = any(g["violation_steps"] > 0 or g["diverged"] for g in grid) and \
        any(g["violation_steps"] == 0 and not g["diverged"] for g in grid)
    chk("R4a 标定网格存在 合格↔不合格 翻转（标定有信息量）", has_flip,
        "n_ok=%d n_bad=%d" % (sum(1 for g in grid if g["violation_steps"] == 0 and not g["diverged"]),
                              sum(1 for g in grid if g["violation_steps"] > 0 or g["diverged"])))
    chk("R4b 标定判据用 gate=False（否则越限统计同义反复）",
        "gate=False" in e69_src, "grid 中 demand 越限非零 = %s"
        % any(g["violation_steps"] > 0 for g in grid))

    # R5 两类 ωn 逐条不同
    diff = [abs(r["C_margin_assumed"] - r["C_margin_budget"]) for r in rows]
    chk("R5 假设口径与预算口径逐条 margin 不同（回填真的生效）", all(d > 1e-9 for d in diff),
        "min|Δmargin|=%.3e" % min(diff))

    # R6 推荐 k_omega / 或"无解"必须给出结构性解释（不得静默返回 None）
    v3b = rep["V3b_komega_recalibration"]
    rec = v3b["recommended_k_omega"]
    scan = v3b["scan"]
    if rec is not None:
        nD = blind = slow_tot = slow_rej = 0
        for r in rows:
            if not r["D_ok"]:
                continue
            nD += 1
            tr = E58.traj([r["_qt"][0], r["_qt"][1]], r["T_s"], E58.N_JUDGE)
            m, _ = trackability_margin(max(r["wn_budget_rad_s"], 1e-6), E58.ZETA, tr,
                                       T_s=E58.T_S, k_omega=rec, k_margin=E58.K_MARGIN)
            if m < 0.0:
                blind += 1
                if r["T_s"] >= 0.8:
                    slow_rej += 1
            if r["T_s"] >= 0.8:
                slow_tot += 1
        fa = slow_rej / slow_tot if slow_tot else 1.0
        chk("R6a 推荐 k_omega 复现：慢档误报 <0.10 且盲区 >0",
            fa < 0.10 and blind > 0, "k_omega=%.3f → 盲区 %d／慢档误报 %.3f" % (rec, blind, fa))
    else:
        # ★ 无解本身就是发现：必须 (a) 有显式结构性解释 (b) 误报在极宽松 k_omega 下已收敛到地板
        f5 = rep["V4_conclusion"].get("finding_5", "")
        floor_ok = (scan["0.02"]["slow_false_alarm"] == scan["0.05"]["slow_false_alarm"]
                    and scan["0.02"]["slow_false_alarm"] > 0.0)
        chk("R6a 无可行 k_omega 时必须给出结构性解释（finding_5 含反应时间地板）",
            ("k_margin" in f5) and ("地板" in f5), f5[:70])
        chk("R6b 误报在 k_omega→0.02 后不再下降（存在结构性地板，非调参问题）", floor_ok,
            "FA(0.05)=%s FA(0.02)=%s" % (scan["0.05"]["slow_false_alarm"], scan["0.02"]["slow_false_alarm"]))

    # R7 分母口径登记
    v3fa = rep["V3_e58_refill"]["budget"]["slow_T_ge_0.8s_false_alarm"]
    v3bfa = rep["V3b_komega_recalibration"]["scan"]["2.0"]["slow_false_alarm"]
    chk("R7 两表分母口径不同已登记（V3=全体 slow；V3b=D_ok 子集）",
        abs((v3fa or 0) - (v3bfa or 0)) > 1e-9
        and rep["V3b_komega_recalibration"].get("denominator") == "D_ok only",
        "V3=%.3f V3b=%.3f" % (v3fa or 0, v3bfa or 0))

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
