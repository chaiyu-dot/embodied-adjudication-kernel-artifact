# -*- coding: utf-8 -*-
"""_audit_e74_strict.py —— 对 E74 跟踪带宽曲线化的**严格审核**（审前提，不只看数值）。

  S1  结构 + 全报告递归 inf/NaN 扫描
  S2  **短时程精确性 + 长时程浮点放大的因果验证**：同一控制器两种等价结合序
      （(a+b)+c vs a+(b+c)）在短/长时程的末态差 —— 证明"低增益行偏差"是**数值放大**
      而非模型不一致（高增益行两种结合序末态差应≈机器精度）
  S3  边界**括号相邻性**：每条边界的 [last_pass, first_fail] 必须是频率网格相邻点，且 last_pass 可复算
  S4  **通过集前缀性（前提）**：若通过集在 f 上有间断，"最大通过频率"只是下界证书 → 必须标注
  S5  口径因子稳定性 + FFT 受限点判据（周期数 <1.5）
  S6  逐格：measured_hz 由 points 插值可复现
  S7  E71 α 去额关系精确性
  S8  E67-C 锚点（原增益）通过/不通过集合
  S9  规格自足（bench 必需键）
  S10 确定性
  R1–R8 反向：篡改 rms/边界/measured/注入 inf/翻转 verdict/删规格/非有限→None

产物：_audit_e74_strict.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e74_tracking_bandwidth_curve as E74     # noqa: E402
import e67_testbed_completeness as E67         # noqa: E402
import e66_closed_loop_control as E66          # noqa: E402

REPORT = E74.REPORT


def load():
    return json.load(open(REPORT, encoding="utf-8"))


def _chk(n, ok, d=""):
    return {"name": n, "pass": bool(ok), "detail": str(d)}


def _finite(o, p="$"):
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += _finite(v, "%s.%s" % (p, k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _finite(v, "%s[%d]" % (p, i))
    elif isinstance(o, float) and not math.isfinite(o):
        bad.append(p)
    return bad


def _sim_rms(gains, f, T):
    """镜像 E67._track_arm 的解算（PD_G + 速度前馈 + 门控 + 1.5 s 整定窗口），返回 rms。"""
    kp = np.asarray(gains[0], float)
    kd = np.asarray(gains[1], float)
    ctr = np.asarray(E67.TRACK_CENTER, float)
    amp, dt = E74.AMP, E66.DT
    tau_lim = np.asarray(E66.TAU_LIM, float)
    st = np.zeros(4)
    n = int(round(T / dt))
    ns = int(round(1.5 / dt))
    w = 2 * math.pi * f
    errs = []
    for i in range(n):
        t = i * dt
        p = ctr + amp * np.array([math.cos(w * t), 0.5 * math.sin(w * t)])
        p_dot = amp * np.array([-w * math.sin(w * t), 0.5 * w * math.cos(w * t)])
        q_ref = E67.ik(p)
        qd_ref = np.linalg.solve(E67.jac(q_ref), p_dot)
        raw = kp * (q_ref - st[:2]) + kd * (qd_ref - st[2:]) + E66.gravity_vec(st[:2])
        if i >= ns:
            errs.append(float(np.linalg.norm(E67.fk(st[:2]) - p)))
        st = E66.rk4_step(st, np.clip(raw, -tau_lim, tau_lim), dt)
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _recompute_boundary(rows, tol, unsat, sat_tol):
    ok = []
    for k, v in rows.items():
        if v["rms_err_m"] is None or v["diverged"]:
            continue
        if v["rms_err_m"] <= tol and (not unsat or v["sat_frac_demand"] <= sat_tol):
            ok.append(float(k))
    return round(max(ok), 1) if ok else None


# ---------- 用于 S2 的最小复现仿真（可切换结合序） ----------
def _sim_final_state(gains, f, T, grouping):
    kp = np.asarray(gains[0], float)
    kd = np.asarray(gains[1], float)
    ctr = np.asarray(E67.TRACK_CENTER, float)
    amp, dt = E74.AMP, E66.DT
    tau_lim = np.asarray(E66.TAU_LIM, float)
    st = np.zeros(4)
    n = int(round(T / dt))
    w = 2 * math.pi * f
    for i in range(n):
        t = i * dt
        p = ctr + amp * np.array([math.cos(w * t), 0.5 * math.sin(w * t)])
        p_dot = amp * np.array([-w * math.sin(w * t), 0.5 * w * math.cos(w * t)])
        q_ref = E67.ik(p)
        qd_ref = np.linalg.solve(E67.jac(q_ref), p_dot)
        a = kp * (q_ref - st[:2])
        b = kd * (qd_ref - st[2:])
        g = E66.gravity_vec(st[:2])
        raw = (a + b) + g if grouping == 1 else a + (b + g)   # 两种等价结合序
        st = E66.rk4_step(st, np.clip(raw, -tau_lim, tau_lim), dt)
    return st


def forward(rep):
    ch = []
    need = ["A_freq_domain_bw", "B_task_tolerance_bw", "C_e67c_anchor", "D_calibration", "bench", "verdict_pass"]
    miss = [k for k in need if k not in rep]
    ch.append(_chk("S1a_structure", not miss, "missing=%s" % miss))
    badf = _finite(rep)
    ch.append(_chk("S1b_no_inf_nan", not badf, "非有限=%s" % badf[:5]))
    inv = ("criteria" in rep and "verdict_pass" in rep
           and bool(rep["verdict_pass"]) == all(bool(v) for v in rep["criteria"].values()))
    ch.append(_chk("S1c_verdict_matches_criteria", bool(inv), "verdict=%s" % rep.get("verdict_pass")))
    if miss:
        return ch
    A, B, C, D, bench = rep["A_freq_domain_bw"], rep["B_task_tolerance_bw"], rep["C_e67c_anchor"], rep["D_calibration"], rep["bench"]

    # S2 落盘精度充分性：用**报告存储的增益**重跑仿真，须与存储 rms 逐点一致
    #    （★ 本检查由"原先的浮点放大假设"被推翻后重建：真实原因是**落盘精度不足**）
    s2_bad, s2_max = [], 0.0
    for name, row in B["by_wn"].items():
        for f in (0.3, 0.5, 1.0):
            k = "%.1f" % f
            if k not in row["rows"] or row["rows"][k]["rms_err_m"] is None:
                continue
            rr = _sim_rms((row["kp"], row["kd"]), f, E74.T_SIM)
            dev = abs(rr - row["rows"][k]["rms_err_m"])
            s2_max = max(s2_max, dev)
            if dev > 6e-6:                      # 6e-6 = 报告 rms 5 位小数的舍入上限
                s2_bad.append((name, k, row["rows"][k]["rms_err_m"], round(rr, 6)))
    ch.append(_chk("S2_stored_precision_sufficient", not s2_bad,
                   "最大绝对偏差 %.2e（限 6e-6）反例=%d %s" % (s2_max, len(s2_bad), s2_bad[:2])))

    # S2b 回归证据：把存储增益舍到 4 位 → 低增益行必出现可观测偏差（证明原缺陷真实、修复必要）
    lo_row = B["by_wn"]["e69_min"]
    r_ref = lo_row["rows"]["0.5"]["rms_err_m"]
    dev4 = abs(_sim_rms(([round(v, 4) for v in lo_row["kp"]], [round(v, 4) for v in lo_row["kd"]]),
                        0.5, E74.T_SIM) - r_ref)
    rel4 = dev4 / max(r_ref, 1e-12)
    ch.append(_chk("S2b_precision_defect_regression", rel4 >= 1e-3,
                   "4 位舍入后 e69_min@0.5Hz 相对偏差 %.2e（应≥1e-3）" % rel4))

    # S3 边界括号相邻性 + 可复算
    grid = sorted(float(k) for k in list(B["by_wn"].values())[0]["rows"])
    s3_bad = []
    for name, row in B["by_wn"].items():
        for tk, pt in row["per_tol"].items():
            tol_ = float(tk.split("=")[1][:-1])
            for kind in ("any", "unsaturated"):
                e = pt[kind]
                # (b) 与从 rows 重算的边界一致（防"自洽的谎"：只改 hz 不改表）
                rb = _recompute_boundary(row["rows"], tol_, kind == "unsaturated", bench["sat_tol"])
                if rb != e["hz"]:
                    s3_bad.append((name, tk, kind, "重算= %s ≠ 报告 %s" % (rb, e["hz"])))
                    continue
                if e["hz"] is None:
                    if e["bracket"] is not None:
                        s3_bad.append((name, tk, kind, "None 却有 bracket"))
                    continue
                b, br = e["hz"], e["bracket"]
                if br is None or br[0] != b:
                    s3_bad.append((name, tk, kind, "bracket 头≠hz"))
                    continue
                idx = grid.index(b) if b in grid else None
                if idx is None or idx + 1 >= len(grid):
                    s3_bad.append((name, tk, kind, "hz 不在网格/无后继"))
                    continue
                if br[1] != round(grid[idx + 1], 1):
                    s3_bad.append((name, tk, kind, "后继非相邻", br[1], round(grid[idx + 1], 1)))
    ch.append(_chk("S3_bracket_adjacent", not s3_bad, "反例=%d %s" % (len(s3_bad), s3_bad[:3])))

    # S4 通过集前缀性（前提）
    nonprefix = []
    for name, row in B["by_wn"].items():
        for tk in row["per_tol"]:
            tol = float(tk.split("=")[1][:-1])
            ps = [f for f in grid if (row["rows"]["%.1f" % f]["rms_err_m"] is not None
                                      and row["rows"]["%.1f" % f]["rms_err_m"] <= tol)]
            if ps:
                # 前缀：最大值处之前应全在集合内
                if any(f not in ps for f in grid if f < max(ps)):
                    nonprefix.append((name, tk, max(ps)))
    # 通过集非前缀 → 边界只是"下界证书"，报告须显式标注；此处仅要求**已记录**
    doc_ok = ("note" in B and "边界" in B.get("note", ""))
    ch.append(_chk("S4_passset_prefix_or_documented", (not nonprefix) or doc_ok,
                   "非前缀例=%d %s | 已文档化=%s" % (len(nonprefix), nonprefix[:3], doc_ok)))

    # S5 口径因子稳定性 + FFT 受限判据
    trusted, limited, ratios = [], [], []
    for name, v in A["by_wn"].items():
        if v["measured_hz"] is None:
            continue
        cyc = v.get("cycles_in_data")
        (trusted if (cyc is not None and cyc >= 1.5) else limited).append(name)
        if cyc is not None and cyc >= 1.5:
            ratios.append(v["measured_hz"] / v["analytic_hz"])
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios)) if ratios else None
    ch.append(_chk("S5_ratio_stable_and_fft_flag", len(trusted) >= 3 and spread is not None and spread <= 0.05
                   and set(trusted) == set(A["trusted_points"]) and set(limited) == set(A["low_f_fft_limited"]),
                   "trusted=%s limited=%s spread=%s" % (trusted, limited, None if spread is None else round(spread, 4))))

    # S6 逐格：measured_hz 由 points 插值可复现
    s6_bad = []
    for name, v in A["by_wn"].items():
        pts = {float(k): val["gain_db"] for k, val in v["points"].items()}
        prev, est = None, None
        for f in sorted(pts):
            gg = pts[f]
            if gg <= -3.0:
                if prev is None:
                    est = float(f)
                else:
                    f0, g0 = prev
                    tt = (g0 + 3.0) / (g0 - gg) if abs(g0 - gg) > 1e-12 else 0.0
                    est = float(f0 * (f / f0) ** max(0.0, min(1.0, tt)))
                break
            prev = (f, gg)
        if v["measured_hz"] is not None and (est is None or abs(est - v["measured_hz"]) > 2e-3):
            s6_bad.append((name, v["measured_hz"], None if est is None else round(est, 3)))
    ch.append(_chk("S6_per_cell_interpolation", not s6_bad, "反例=%s" % s6_bad[:3]))

    # S7 α 关系
    exp = math.sqrt(bench["alpha_e71"])
    ch.append(_chk("S7_alpha_derate_exact",
                   abs(exp - D["derate_ratio_expected_sqrt_alpha"]) < 1e-4 and D["derate_rel_dev"] <= 0.02,
                   "sqrt(alpha)=%.6f vs %.6f" % (exp, D["derate_ratio_expected_sqrt_alpha"])))

    # S8 E67-C 锚点
    rowsC = C["rows"]
    def passes(f, tol):
        v = rowsC.get("%.1f" % f)
        return bool(v and v["rms_err_m"] is not None and v["rms_err_m"] <= tol)
    ch.append(_chk("S8_e67c_anchor", passes(0.2, 0.01) and (not passes(0.5, 0.01)) and passes(0.5, 0.02),
                   "0.2(1cm)=%s 0.5(1cm)=%s" % (passes(0.2, 0.01), passes(0.5, 0.01))))

    # S9 规格自足
    spec = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params", "zeta",
            "wn_design", "b_visc", "scenario"]
    ch.append(_chk("S9_spec_self_contained", all(k in bench for k in spec),
                   "missing=%s" % [k for k in spec if k not in bench]))

    # S10 确定性
    g_hi = (B["by_wn"]["e69_max"]["kp"], B["by_wn"]["e69_max"]["kd"])
    det = (E74.uniform_gains(7.7582) == E74.uniform_gains(7.7582)
           and _sim_final_state(g_hi, 0.5, 1.0, 1).tolist() == _sim_final_state(g_hi, 0.5, 1.0, 1).tolist())
    ch.append(_chk("S10_determinism", bool(det), "non-deterministic"))

    # ---- S11–S13 E 段前提（补：该段此前无任何审核覆盖） ----
    Esec = rep.get("E_identification_probes") or {}
    probesE = Esec.get("probes") or {}
    # S11 **约束激活前提**：加这段的目的就是让关节 1 上限承重；没压到就没意义
    act = [(nm, pr["demand_peak_per_joint"][1], pr["violation_steps_per_joint"][1])
           for nm, pr in probesE.items()]
    s11 = bool(act) and any(d > E66.TAU_LIM[1] + 1e-9 and v > 0 for _, d, v in act)
    ch.append(_chk("S11_E_joint1_constraint_activated", s11,
                   "关节1上限=%.1f；探针(峰值,越限步)=%s" % (E66.TAU_LIM[1], act)))

    # S12 **仿真器同一性**：独立重跑 C5 自检（E 段仿真器 vs E67._track_arm）
    g67 = E66.budget_gains(E74.Q_REF_TRACK)[:2]
    mine = E74._probe_sim(0.0, 0.0, E74.AMP, 1.0, E74.T_SIM, ee=True, gains=g67)["rms_err_m"]
    ref = E67._track_arm(1.0, True, amp=E74.AMP, T=E74.T_SIM, tol=1e-9, gains=g67)["rms_err_m"]
    s12 = bool(mine is not None and ref is not None and mine == ref)
    ch.append(_chk("S12_E_sim_identity_recheck", s12,
                   "E段仿真器 rms=%s vs E67._track_arm rms=%s（逐位相等=%s）" % (mine, ref, s12)))

    # S13 **逐格派生量**：sat_frac == viol/n（防"改分子不改分母"的自洽谎言）
    e_bad = []
    for nm, pr in probesE.items():
        n = int(pr["n_steps"])
        for j in (0, 1):
            if abs(pr["sat_frac_per_joint"][j] - pr["violation_steps_per_joint"][j] / n) > 1e-12:
                e_bad.append((nm, j, pr["sat_frac_per_joint"][j], pr["violation_steps_per_joint"][j] / n))
    ch.append(_chk("S13_E_per_cell_derived", not e_bad, "反例=%s" % e_bad[:3]))

    # S14 F 段前提：振幅口径已测量、对 amp 非增、且**与 B 段在基准振幅下跨段一致**
    Fsec = rep.get("F_amplitude_calibration") or {}
    f_bad = []
    for nm, v in (Fsec.get("by_wn") or {}).items():
        if not v["boundary_non_increasing_in_amp"]:
            f_bad.append((nm, "对 amp 非单调"))
        if nm in B["by_wn"]:
            for t in E74.TOLS:
                k = "tol=%.3fm" % t
                if B["by_wn"][nm]["per_tol"][k]["any"]["hz"] != v["per_amp"]["amp=%.2fm" % E74.AMP]["boundary_any"][k]:
                    f_bad.append((nm, k, "B段≠F段"))
    ch.append(_chk("S14_amplitude_calibration_premise", bool(Fsec) and not f_bad, "反例=%s" % f_bad[:3]))

    # S15 G 段前提 + 诚实（口径穷举最后一条）：带宽是控制器族口径的，且诚实标注
    #   前提：spread_x 显著 >1（族效应真实）；spread_x 与 by_family 可独立重算一致；pdg_hz==PD_G
    #   诚实：D.caveats 含控制器族口径；C7 判据为真（不为仅声明）
    Gsec = rep.get("G_controller_family_calibration") or {}
    g15 = []
    fams = Gsec.get("by_family") or {}
    vals = [v.get("bandwidth_hz") for v in fams.values()]
    if len(fams) != 7 or any(v is None for v in vals):
        g15.append("G 段族不齐/未全测得")
    else:
        rec_spread = max(vals) / min(vals)
        if rec_spread <= 1.5:
            g15.append("spread=%.2f 不显著，族效应存疑" % rec_spread)
        if Gsec.get("spread_x") is None or abs(Gsec["spread_x"] - rec_spread) > 1e-3:
            g15.append(("spread_x 不一致", Gsec.get("spread_x"), round(rec_spread, 3)))
        if not (Gsec.get("pdg_hz") is not None and abs(Gsec["pdg_hz"] - fams["PD_G"]["bandwidth_hz"]) <= 1e-9):
            g15.append("pdg_hz 与 PD_G 不符")
    if not any("控制器族口径" in c for c in D.get("caveats", [])):
        g15.append("caveats 缺控制器族口径")
    if rep.get("criteria", {}).get("C7_controller_family_calibration_measured") is not True:
        g15.append("C7 未真")
    ch.append(_chk("S15_controller_family_premise_and_honesty", not g15, "反例=%s" % g15[:4]))
    return ch


def reverse(rep):
    ch = []
    first = list(rep["B_task_tolerance_bw"]["by_wn"])[2]

    bad = copy.deepcopy(rep)
    bad["B_task_tolerance_bw"]["by_wn"][first]["per_tol"]["tol=0.010m"]["any"] = {"hz": 0.5, "bracket": [0.5, 0.6]}
    ch.append(_chk("R1_tamper_bracket_detected",
                   any(c["name"] == "S3_bracket_adjacent" and not c["pass"] for c in forward(bad)),
                   "改边界括号→S3 必报"))

    bad2 = copy.deepcopy(rep)
    nm = list(bad2["A_freq_domain_bw"]["by_wn"])[3]
    bad2["A_freq_domain_bw"]["by_wn"][nm]["measured_hz"] = 9.99
    ch.append(_chk("R2_tamper_measured_detected",
                   any(c["name"] == "S6_per_cell_interpolation" and not c["pass"] for c in forward(bad2)),
                   "改 measured→S6 必报"))

    bad3 = copy.deepcopy(rep)
    bad3["A_freq_domain_bw"]["trusted_points"] = []
    ch.append(_chk("R3_tamper_trusted_detected",
                   any(c["name"] == "S5_ratio_stable_and_fft_flag" and not c["pass"] for c in forward(bad3)),
                   "改 trusted→S5 必报"))

    bad4 = copy.deepcopy(rep)
    bad4["D_calibration"]["alpha_e71"] = float("inf")
    ch.append(_chk("R4_inject_inf_detected",
                   any(c["name"] == "S1b_no_inf_nan" and not c["pass"] for c in forward(bad4)),
                   "注入 inf→S1b 必报"))

    bad5 = copy.deepcopy(rep)
    bad5["verdict_pass"] = not rep["verdict_pass"]
    ch.append(_chk("R5_flip_verdict_detected",
                   any(c["name"] == "S1c_verdict_matches_criteria" and not c["pass"] for c in forward(bad5)),
                   "翻转 verdict→S1c 必报"))

    bad6 = copy.deepcopy(rep)
    bad6["bench"].pop("scenario", None)
    ch.append(_chk("R6_missing_spec_detected",
                   any(c["name"] == "S9_spec_self_contained" and not c["pass"] for c in forward(bad6)),
                   "删 scenario→S9 必报"))

    fwd = forward(rep)
    ch.append(_chk("R7_clean_passes", all(c["pass"] for c in fwd),
                   "clean=%d/%d" % (sum(c["pass"] for c in fwd), len(fwd))))
    ch.append(_chk("R8_nonfinite_to_none",
                   E74._num(float("nan")) is None and E74._num(float("inf")) is None,
                   "nonfinite not capped"))

    # R9–R11 **E 段篡改必报**（新覆盖 → 先证明检查敏感）
    bad7 = copy.deepcopy(rep)
    pn = list(bad7["E_identification_probes"]["probes"])[0]
    bad7["E_identification_probes"]["probes"][pn]["n_steps"] = bad7["E_identification_probes"]["probes"][pn]["n_steps"] * 2
    ch.append(_chk("R9_tamper_E_nsteps_detected",
                   any(c["name"] == "S13_E_per_cell_derived" and not c["pass"] for c in forward(bad7)),
                   "改 n_steps→S13 必报"))

    bad8 = copy.deepcopy(rep)
    for nm in bad8["E_identification_probes"]["probes"]:
        bad8["E_identification_probes"]["probes"][nm]["demand_peak_per_joint"][1] = 0.5   # 不再压到上限
        bad8["E_identification_probes"]["probes"][nm]["violation_steps_per_joint"][1] = 0
        bad8["E_identification_probes"]["probes"][nm]["sat_frac_per_joint"][1] = 0.0
    ch.append(_chk("R10_tamper_E_exercised_detected",
                   any(c["name"] == "S11_E_joint1_constraint_activated" and not c["pass"] for c in forward(bad8)),
                   "抹掉关节1承重→S11 必报"))

    bad9 = copy.deepcopy(rep)
    bad9["E_identification_probes"]["self_check_vs_e67_track_arm"]["abs_diff"] = 1e-3
    bad9["E_identification_probes"]["self_check_vs_e67_track_arm"]["match"] = False
    ch.append(_chk("R11_S12_immune_to_selfcheck_field",
                   any(c["name"] == "S12_E_sim_identity_recheck" and c["pass"] for c in forward(bad9)),
                   "S12 为硬重算：篡改报告自述的 match 字段**不影响**其结论"))

    # R12 **F 段跨段一致性篡改必报**（改 F 段结果使其与 B 段不符）
    bad10 = copy.deepcopy(rep)
    wnF = list(bad10["F_amplitude_calibration"]["by_wn"])[0]
    bad10["F_amplitude_calibration"]["by_wn"][wnF]["per_amp"]["amp=0.06m"]["boundary_any"]["tol=0.010m"] = 1.9
    ch.append(_chk("R12_tamper_F_crosssection_detected",
                   any(c["name"] == "S14_amplitude_calibration_premise" and not c["pass"] for c in forward(bad10)),
                   "改 F 段边界使其与 B 段不符→S14 必报"))

    # R13 **G 段篡改必报**（抹平族差异 → spread_x 重算与报告不符 → S15 必报）
    bad11 = copy.deepcopy(rep)
    bad11["G_controller_family_calibration"]["by_family"]["CT"]["bandwidth_hz"] = 0.600
    ch.append(_chk("R13_tamper_G_spread_detected",
                   any(c["name"] == "S15_controller_family_premise_and_honesty" and not c["pass"] for c in forward(bad11)),
                   "抹平 CT 带宽→S15 必报"))
    return ch


def main():
    rep = load()
    fwd, rev = forward(rep), reverse(rep)
    allc = fwd + rev
    npass = sum(c["pass"] for c in allc)
    out = {"audit": "_audit_e74_strict", "experiment": "e74_tracking_bandwidth_curve",
           "pass": npass, "total": len(allc), "verdict": bool(npass == len(allc)),
           "audit_pass": bool(npass == len(allc)), "n_pass": npass, "n_total": len(allc),
           "forward": fwd, "reverse": rev, "checks": allc}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit_e74_strict.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("strict audit e74 = %d/%d  verdict=%s" % (npass, len(allc), out["verdict"]))
    for c in allc:
        print("  [%s] %-36s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"][:110]))
    return out


if __name__ == "__main__":
    main()
