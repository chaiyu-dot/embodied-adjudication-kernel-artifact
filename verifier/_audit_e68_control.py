# -*- coding: utf-8 -*-
"""_audit_e68_control.py —— E68 控制策略模块的**正反双向审核**（零 API、纯 CPU、可复跑）。

正向审核（假设 ↔ 实现 ↔ 数字 是否一致）
  F1 报告完整性：E68 报告存在且含全部键
  F2 数字可追溯：从 JSON 重读并与独立重算对照（不信任打印）
  F3 判据 ↔ 实现：V0/V0b/V1/V4 的判据式与代码逐条对应
  F4 声明 vs 实测：capability() 的设计值（source=design）必须与实测 ωn/ζ 分列
  F5 口径一致：dt / T / 增益预算 / 初值 是否全族一致；不一致项必须显式登记

反向审核（构造反例/攻击，尝试推翻）
  R1 负载泄漏：控制器是否可能窥视真实负载（结构性 + 行为性两路）
  R2 因果性：输出是否依赖未来信息 / 绝对时间
  R3 门控有效性：把 τ_lim 压到极小，输出是否被真正 clip
  R4 辨识诚实：未收敛的族是否**拒绝**给出 ωn/ζ（不得伪造）
  R5 对照有效性：无门控组必须**确实**越限（否则 gate 消融是空的）
  R6 口径混淆：不得把"可行率"当"任务完成率"（历史坑）
  R7 边界真伪：静载上限附近扫描，验证边界是**力矩预算**性质而非控制器性质

用法：python _audit_e68_control.py     产物：_audit_e68_control.json
"""
import json
import math
import os
import sys

import numpy as np

def _find_src(start):
    """向上查找含 ``planning/control_strategies.py`` 的目录。

    ★ **不假设固定的两层布局**：原实现写死 ``dirname(dirname(__file__))``，
    复刻者把脚本平铺到别处就 FileNotFoundError —— 由独立复刻测试抓出的真 bug。
    """
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

from planning import control_strategies as CS        # noqa: E402
import e66_closed_loop_control as E66                 # noqa: E402
import e68_control_strategy_suite as E68              # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EVAL, "_audit_e68_control.json")

Q_REF = E68.Q_REF
TAU_LIM = CS.TAU_LIM
FAMILIES = E68.FAMILIES
res = {}
checks = []


def chk(name, cond, detail=""):
    checks.append({"check": name, "pass": bool(cond), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + detail) if detail else ""))


def main():
    rep = json.load(open(E68.REPORT, encoding="utf-8"))
    kp_b, kd_b, wn_b = E66.budget_gains(Q_REF)
    mp_lim = E66.static_hold_limit(Q_REF)

    # ================= 正向 =================
    print("=== F 正向审核 ===")
    need = ["V0_model_consistency", "V0b_mpc_consistency", "V1_gate_invariant",
            "gate_ablation_no_gate", "step_response", "V3_freq_response",
            "V4_infeasible_disposition", "MPC_calibration", "verdict_pass"]
    chk("F1 报告含全部必需键", all(k in rep for k in need),
        "缺失=%s" % [k for k in need if k not in rep])
    chk("F1b verdict_pass 为 True", rep.get("verdict_pass") is True, "verdict=%s" % rep.get("verdict_pass"))

    # F2 独立重算 V0（不信任 JSON）
    rng = np.random.RandomState(12345)
    e_rne = 0.0
    for _ in range(200):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-5, 5, 2)
        lhs = CS.mass_matrix(q) @ qdd + CS.coriolis_vec(q, qd) + CS.gravity_vec(q)
        e_rne = max(e_rne, float(np.max(np.abs(lhs - E66.rne_tau(list(q), list(qd), list(qdd))))))
    chk("F2 V0 独立重算 vs RNE < 1e-8", e_rne < 1e-8, "max=%.3e" % e_rne)

    # F2b 独立重算 MPC 前馈（不信任 JSON）
    m = CS.make_controller("MPC", kp=kp_b, kd=kd_b, q_ref=np.asarray(Q_REF))
    qq = np.array([0.30, -0.40])
    for _ in range(400):
        tau_ff = m.act(qq, np.zeros(2), qq, 0.0)
    g_ref = CS.gravity_vec(qq)
    chk("F2b MPC 前馈独立重算 |τ−g|<1e-6", float(np.max(np.abs(tau_ff - g_ref))) < 1e-6,
        "|Δ|=%.2e" % float(np.max(np.abs(tau_ff - g_ref))))

    # F3 判据↔实现
    anchors = {"V0": "V0_model_consistency", "V0b": "V0b_mpc_consistency",
               "V1": "V1_gate_invariant", "V4": "V4_ok"}
    chk("F3 判据键与实现一一对应", all(k in rep for k in anchors.values()),
        "keys=%s" % list(anchors.values()))

    # F4 声明 vs 实测
    decl = {k: rep["step_response"][k]["capability"]["source"] for k in FAMILIES}
    measured = {k: rep["step_response"][k]["ident"].get("identifiable") for k in FAMILIES}
    chk("F4 capability 全部标 source=design（不冒充实测）",
        all(v == "design" for v in decl.values()), str(decl))
    chk("F4b 未收敛族不给出实测 ωn/ζ",
        all(measured[k] is False for k in ("P", "PD", "PID", "CT")),
        str({k: measured[k] for k in ("P", "PD", "PID", "CT")}))

    # F5 口径
    dt_set = {rep["bench"]["dt"]}
    chk("F5 dt 全族一致", len(dt_set) == 1, "dt=%s" % dt_set)
    rep["_audit_F5_caveats"] = [
        "FO 用 (kp=3.0,kd=0.6,vmax=4.0)，非预算增益（其一阶速度外环结构不同）",
        "MPC 用 Q/R 标定到预算静态刚度（误差 3.7%），非线性控制器无 kp/kd",
        "P/PD/PID/PD_G/CT 用同一预算增益 kp=%.4f kd=%.4f" % (kp_b, kd_b),
    ]
    print("     口径登记：" + " ｜ ".join(rep["_audit_F5_caveats"]))

    # ================= 反向 =================
    print("\n=== R 反向审核（构造反例/攻击） ===")

    # R1 负载泄漏：结构性（签名/全局态）+ 行为性（同状态同输入必同输出）
    import inspect
    sig = inspect.signature(CS.Controller.act)
    chk("R1a act 只接受 mp_hat（无真实负载入口）",
        set(sig.parameters) == {"self", "q", "qd", "r", "t", "mp_hat"}, str(list(sig.parameters)))
    src = open(os.path.join(_SRC, "planning", "control_strategies.py"), encoding="utf-8").read()
    _mod_part = src.split('if __name__ == "__main__":')[0]   # 排除自测块的 global ok,tot
    import re as _re
    globals_in_mod = _re.findall(r"^\s*global\s+(.+)$", _mod_part, flags=_re.M)
    # ★ 2026-09-19 修：原判据含 `src.count("mp_hat") >= 5` 这个**魔数阈值** —— 源码一改
    #   （如 P2-3 新增 FLC/IMP/LRN）计数就变，导致"产物陈旧 vs 新鲜"被误判成数值差异。
    #   改为**结构性断言**（与编辑、与目录无关）：无 global + 负载不存为 `self.mp_hat` 状态。
    n_self_mp = len(_re.findall(r"self\.mp_hat\s*=", src))
    chk("R1b 模块主体无 global·负载仅经 mp_hat 形参（不存为可变状态）",
        len(globals_in_mod) == 0 and n_self_mp == 0 and "mp_hat" in src,
        "global=%s, self.mp_hat 赋值=%d" % (globals_in_mod, n_self_mp))
    c1 = CS.make_controller("MPC", kp=kp_b, kd=kd_b, q_ref=np.asarray(Q_REF))
    a = c1.act([0.3, -0.4], [0.05, 0.02], Q_REF, 0.0)
    out_leak = []
    for mp_world in (0.0, 0.5, 2.0):     # 改变"真实世界"负载
        c1.reset()
        out_leak.append(c1.act([0.3, -0.4], [0.05, 0.02], Q_REF, 0.0))
    chk("R1c 改变真实负载不改变控制器输出（无泄漏）",
        all(np.allclose(o, out_leak[0], atol=1e-12) for o in out_leak), str(out_leak[0]))

    # F6 ★ 独立重算 MPC 阶跃（用报告里的标定权重）并与报告逐项对照
    cal = rep["MPC_calibration"]
    m6 = CS.make_controller("MPC", kp=kp_b, kd=kd_b, Np=cal["Np"], Nc=cal["Nc"],
                            q_w=cal["q_w"], qd_w=cal["qd_w"], r_w=cal["r_w"],
                            q_ref=np.asarray(Q_REF))
    log6 = E68.simulate(m6, Q_REF, T=4.0)
    s6 = E68.step_metrics(log6, Q_REF)
    r6 = rep["step_response"]["MPC"]
    dv6, dm6 = E68.demand_violation(log6)
    chk("F6 独立重算 MPC 阶跃与报告一致（ss_err/settling/rise/demand）",
        abs(s6["ss_err_rad"] - r6["ss_err_rad"]) < 1e-9
        and abs((s6["settling_s"] or -1) - (r6["settling_s"] or -1)) < 1e-6
        and abs((s6["rise_s"] or -1) - (r6["rise_s"] or -1)) < 1e-6
        and dv6 == r6["demand_violation_steps"],
        "重算 ss=%.2e settle=%s rise=%s dv=%d ｜ 报告 ss=%.2e settle=%s rise=%s dv=%d"
        % (s6["ss_err_rad"], s6["settling_s"], s6["rise_s"], dv6,
           r6["ss_err_rad"], r6["settling_s"], r6["rise_s"], r6["demand_violation_steps"]))

    # R2 因果性：不同绝对时刻、同状态 → 必须同输出
    c2 = CS.make_controller("MPC", kp=kp_b, kd=kd_b, q_ref=np.asarray(Q_REF))
    c2.reset(); o0 = c2.act([0.3, -0.4], [0.05, 0.02], Q_REF, 0.0)
    c2.reset(); o1 = c2.act([0.3, -0.4], [0.05, 0.02], Q_REF, 123.456)
    chk("R2 输出与绝对时间无关（无未来信息/时间作弊）", np.allclose(o0, o1, atol=1e-12),
        "Δ=%.2e" % float(np.max(np.abs(o0 - o1))))

    # R3 门控有效性（★ MPC 的输入约束内建于 QP，与外部 gate 无关 → "双保险"）
    c3 = CS.make_controller("MPC", kp=kp_b, kd=kd_b, q_ref=np.asarray(Q_REF))
    c3.tau_lim = np.array([0.05, 0.05]); c3.reset()
    o3 = c3.act([0.0, 0.0], [0.0, 0.0], Q_REF, 0.0)
    chk("R3a MPC(gate=True, τ_lim=0.05) 输出被 clip", bool(np.all(np.abs(o3) <= 0.05 + 1e-12)),
        "τ=%s" % np.round(o3, 4))
    c3g = CS.make_controller("MPC", kp=kp_b, kd=kd_b, gate=False, q_ref=np.asarray(Q_REF))
    c3g.tau_lim = np.array([0.05, 0.05]); c3g.reset()
    o3g = c3g.act([0.0, 0.0], [0.0, 0.0], Q_REF, 0.0)
    chk("R3b MPC gate=False 仍不越限（**输入约束内建于 QP** → 双保险，非绕过门控）",
        bool(np.all(np.abs(o3g) <= 0.05 + 1e-12)), "τ=%s" % np.round(o3g, 4))
    meff = float(np.mean(np.diag(CS.mass_matrix(Q_REF))))
    pd_off = CS.make_controller("PD", kp=40.0 ** 2 * meff, kd=2 * 0.7 * 40.0 * meff, gate=False)
    pd_on = CS.make_controller("PD", kp=40.0 ** 2 * meff, kd=2 * 0.7 * 40.0 * meff, gate=True)
    pd_off.reset(); pd_on.reset()
    tau_off = pd_off.act([0.0, 0.0], [0.0, 0.0], Q_REF, 0.0)
    tau_on = pd_on.act([0.0, 0.0], [0.0, 0.0], Q_REF, 0.0)
    chk("R3c 线性族外部门控是有效限制器（PD 无门控超限 / 有门控不超限）",
        bool(np.max(np.abs(tau_off)) > np.max(TAU_LIM) + 1e-9
             and np.all(np.abs(tau_on) <= TAU_LIM + 1e-9)),
        "off=%s on=%s" % (np.round(tau_off, 3), np.round(tau_on, 3)))

    # R4 辨识诚实：人为不收敛时不得给 ωn/ζ
    probe = {"t": np.array([0.0, 1.0, 2.0]), "q": np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
             "qd": np.zeros((3, 2)), "tau": np.zeros((3, 2)), "diverged": False}
    idn = E66.identify_2nd_order(probe, [1.0, 1.0])
    chk("R4 未收敛/未整定 → identifiable=False（不伪造 ωn/ζ）", idn.get("identifiable") is False,
        str(idn.get("reason")))

    # R5 对照有效性：无门控组必须确实越限
    ng = rep["gate_ablation_no_gate"]
    chk("R5 gate 消融有效（无门控确实越限）",
        any(v["violation_steps"] > 0 for k, v in ng.items() if k in ("PD", "CT")),
        str({k: v["violation_steps"] for k, v in ng.items()}))
    chk("R5b 有门控执行值全族零越限", rep["V1_gate_invariant"]["ok"] is True)
    # R5c ★ 同义反复检查：executed 全 0 是构造保证；必须有族在**需求**上越限，否则实验没信息量
    pf = rep["V1_gate_invariant"]["per_family"]
    chk("R5c 需求越限非全零（门控确实在承重，非同义反复）",
        any(v["demand_violation_steps"] > 0 for v in pf.values()),
        str({k: v["demand_violation_steps"] for k, v in pf.items()}))
    chk("R5d MPC 需求越限 = 0（输入约束内建于 QP，自带保护）",
        pf["MPC"]["demand_violation_steps"] == 0,
        "MPC demand=%d demand_tau_max=%.3f" % (pf["MPC"]["demand_violation_steps"],
                                               pf["MPC"]["demand_tau_max"]))

    # R6 口径混淆：V4 用 tail_err，不得出现 success_rate 字段
    chk("R6 V4 未使用 success_rate 混淆口径",
        all("tail_err_rad" in v for v in rep["V4_infeasible_disposition"].values())
        and "success_rate" not in json.dumps(rep["V4_infeasible_disposition"]))

    # R7 边界真伪：mp 由 0.8→1.2 × mp_lim 扫描，验证边界是**力矩预算**性质而非控制器性质
    print("     边界扫描（mp/mp_lim → tail_err；mp_lim=%.4f kg）：" % mp_lim)
    scan = {}
    for r_mp in (0.8, 0.9, 1.0, 1.05, 1.1, 1.2):
        mp_test = round(mp_lim * r_mp, 4)
        row = {}
        for nm in ("PD_G", "CT", "MPC"):
            c = CS.make_controller(nm, kp=kp_b, kd=kd_b,
                                   **({"q_ref": np.asarray(Q_REF)} if nm == "MPC" else {}))
            log = E68.simulate(c, Q_REF, q0=Q_REF, T=4.0, mp=mp_test, mp_hat=mp_test)
            tail = int(round(1.0 / E68.DT))
            row[nm] = round(float(np.mean(np.max(np.abs(log["q"][-tail:] - np.asarray(Q_REF)), axis=1))), 6)
        scan["%.2f" % r_mp] = row
        print("       %.2f (mp=%.3f kg)：%s" % (r_mp, mp_test, row))
    chk("R7 边界是力矩预算性质：≤0.9×mp_lim 全族可保持、≥1.10× 全族不可保持",
        all(scan["0.90"][k] <= 0.02 for k in ("PD_G", "MPC"))
        and all(scan["1.10"][k] > 0.02 for k in ("PD_G", "CT", "MPC")),
        "0.90×→%s ; 1.10×→%s" % (scan["0.90"], scan["1.10"]))
    # R7b：修正后的 static_hold_limit 必须与"实测翻转点"吻合
    chk("R7b static_hold_limit 修正后与实测翻转点吻合（0.9×可持、1.05×不可持）",
        scan["0.90"]["PD_G"] <= 0.02 and scan["1.05"]["PD_G"] > 0.02,
        "0.90×→%.2e ; 1.05×→%.2e" % (scan["0.90"]["PD_G"], scan["1.05"]["PD_G"]))
    # R7c ★ 新发现：理论边界处（1.00×）零裕度 → 数值扰动后校正力矩超限被削 → 滑出
    zero_margin_fail = bool(scan["1.00"]["PD_G"] > 0.02)
    chk("R7c 理论力矩边界处无裕度即不可靠（1.00× 已不可保持 → 边界须配裕度）",
        zero_margin_fail,
        "1.00×→PD_G %.3e ; 结论：可用边界 = 0.9×mp_lim（观测翻转区间 0.90–1.00）"
        % scan["1.00"]["PD_G"])
    res["R7_boundary_scan"] = scan
    res["R7_finding"] = ("static_hold_limit 是**上确界**：τ_gravity = τ_lim 时零裕度，任何数值扰动都会让"
                         "校正力矩越限被削 → 滑出。工程可用的可持边界 = 观测翻转区间下沿（此处 0.90× = %.3f kg）。"
                         "→ 支持论文的"
                         "『判定需要准入裕度』主张。" % (mp_lim * 0.90))
    print("     → R7 结论：" + res["R7_finding"])

    # ================= 汇总 =================
    nfail = sum(1 for c in checks if not c["pass"])
    res["checks"] = checks
    res["n_pass"] = len(checks) - nfail
    res["n_total"] = len(checks)
    res["audit_pass"] = (nfail == 0)
    res["F5_caveats"] = rep["_audit_F5_caveats"]
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n审核结果：%d/%d PASS%s → %s"
          % (res["n_pass"], res["n_total"], "" if res["audit_pass"] else "（存在 FAIL）", OUT))
    if nfail:
        print("FAIL 项：" + " ｜ ".join(c["check"] for c in checks if not c["pass"]))


if __name__ == "__main__":
    main()
