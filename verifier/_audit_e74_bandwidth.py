# -*- coding: utf-8 -*-
"""_audit_e74_bandwidth.py —— e74 的正向（F1–F10）+ 反向（R1–R7）双向审核。

纪律：正向**独立重算**（不复用报告自带的结论）；反向构造篡改证明审计**确实敏感**；
并额外做**全报告 inf/NaN 扫描**与**逐格一致性**（承 e73 严格审核教训）。

产物：_audit_e74_bandwidth.json
"""
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e74_tracking_bandwidth_curve as E74     # noqa: E402
import e66_closed_loop_control as E66          # noqa: E402

REPORT = E74.REPORT
ZETA = E74.ZETA


def load():
    return json.load(open(REPORT, encoding="utf-8"))


def _chk(n, ok, d=""):
    return {"name": n, "pass": bool(ok), "detail": str(d)}


def _finite_scan(o, p="$"):
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += _finite_scan(v, "%s.%s" % (p, k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _finite_scan(v, "%s[%d]" % (p, i))
    elif isinstance(o, float):
        if not math.isfinite(o):
            bad.append(p)
    return bad


def _recompute_boundary(rows, tol, require_unsat):
    ok = []
    for k, v in rows.items():
        f = float(k)
        if v["rms_err_m"] is None or v["diverged"]:
            continue
        if v["rms_err_m"] <= tol and (not require_unsat or v["sat_frac_demand"] <= E74.SAT_TOL):
            ok.append(f)
    return round(max(ok), 1) if ok else None


def forward(rep):
    ch = []
    need = ["A_freq_domain_bw", "B_task_tolerance_bw", "C_e67c_anchor", "D_calibration",
            "criteria", "verdict_pass", "bench"]
    miss = [k for k in need if k not in rep]
    ch.append(_chk("F1a_structure", not miss, "missing=%s" % miss))
    bad = _finite_scan(rep)
    ch.append(_chk("F1b_no_inf_nan", not bad, "非有限=%s" % bad[:5]))
    if miss:
        return ch
    A, B, C, D = rep["A_freq_domain_bw"], rep["B_task_tolerance_bw"], rep["C_e67c_anchor"], rep["D_calibration"]

    # F2 二阶闭式独立重算 + 口径因子一致性
    dev_bad = []
    for name, v in A["by_wn"].items():
        f_ana = E74.f3db_second_order(v["wn_rad_s"]) / (2 * math.pi)
        if abs(f_ana - v["analytic_hz"]) > 1e-3:
            dev_bad.append((name, v["analytic_hz"], round(f_ana, 3)))
        if "ratio" in v:
            r = v["measured_hz"] / v["analytic_hz"]
            if abs(r - v["ratio"]) > 5e-4:
                dev_bad.append((name, "ratio", v["ratio"], round(r, 4)))
    ch.append(_chk("F2_analytic_and_ratio_recompute", not dev_bad,
                   "bad=%s | ratio_mean=%s spread=%s n_trusted=%s"
                   % (dev_bad[:3], A["ratio_mean"], A["ratio_spread_rel"], A["n_trusted"])))

    # F3 两条边界：逐 (ωn,tol) 从 rows 重算并比对
    b_bad = []
    for name, row in B["by_wn"].items():
        for tk, pt in row["per_tol"].items():
            tol = float(tk.split("=")[1][:-1])
            for kind, req in (("any", False), ("unsaturated", True)):
                rb = _recompute_boundary(row["rows"], tol, req)
                if rb != pt[kind]["hz"]:
                    b_bad.append((name, tk, kind, pt[kind]["hz"], rb))
    ch.append(_chk("F3_boundaries_recompute_per_cell", not b_bad, "反例=%d %s" % (len(b_bad), b_bad[:3])))

    # F4 排序：any≥unsat；tol 越紧带宽非增；ωn 越大带宽非降
    c4 = True
    detail = []
    for name, row in B["by_wn"].items():
        prev = None
        for tol in E74.TOLS:
            pt = row["per_tol"]["tol=%.3fm" % tol]
            a, u = pt["any"]["hz"], pt["unsaturated"]["hz"]
            if a is not None and u is not None and u > a + 1e-9:
                c4 = False
                detail.append((name, tol, "unsat>any"))
            if a is not None and prev is not None and a > prev + 1e-9:
                c4 = False
                detail.append((name, tol, "tighter_tol_wider"))
            prev = a if a is not None else prev
    prevv = None
    for name, row in sorted(B["by_wn"].items(), key=lambda kv: kv[1]["wn_rad_s"]):
        a = row["per_tol"]["tol=0.010m"]["any"]["hz"]
        if a is not None and prevv is not None and a < prevv - 1e-9:
            c4 = False
            detail.append((name, "wn", "not_monotone"))
        prevv = a if a is not None else prevv
    ch.append(_chk("F4_boundary_ordering_monotone", c4, "%s" % detail[:4]))

    # F5 E67-C 锚点：从 rows 重算 1cm 的通过/不通过
    rowsC = C["rows"]

    def _passes(f, tol):
        v = rowsC.get("%.1f" % f)
        return bool(v and v["rms_err_m"] is not None and v["rms_err_m"] <= tol)
    ch.append(_chk("F5_e67c_anchor_recompute",
                   _passes(0.2, 0.01) == C["check_1cm"]["pass_0.2"]
                   and (not _passes(0.5, 0.01)) == C["check_1cm"]["fail_0.5"]
                   and _passes(0.2, 0.02) == C["check_2cm"]["pass_0.2"]
                   and _passes(0.5, 0.02) == C["check_2cm"]["pass_0.5"],
                   "pass0.2=%s fail0.5=%s" % (_passes(0.2, 0.01), not _passes(0.5, 0.01))))

    # ---- F11/F12 **E 段（可识别性探针）覆盖**（补：该段此前无任何审核覆盖）----
    Esec = rep.get("E_identification_probes")
    sc = (Esec or {}).get("self_check_vs_e67_track_arm", {})
    ch.append(_chk("F11_E_section_present_selfcheck",
                   bool(Esec) and bool(sc.get("match")) and bool(Esec.get("joint1_limit_exercised"))
                   and len(Esec.get("probes") or {}) >= 3
                   and sc.get("abs_diff") is not None and abs(sc["abs_diff"]) <= 1e-9,
                   "probes=%s match=%s exercised=%s diff=%s" % (
                       len((Esec or {}).get("probes") or {}), sc.get("match"),
                       (Esec or {}).get("joint1_limit_exercised"), sc.get("abs_diff"))))

    mp_spec = rep["bench"]["payload"]["mp_probe_kg"]
    e_bad = []
    for nm, pr in ((Esec or {}).get("probes") or {}).items():
        n = int(pr["n_steps"])
        if abs(pr["mp_kg"] - mp_spec) > 1e-12:
            e_bad.append((nm, "mp≠spec", pr["mp_kg"], mp_spec))
        for j in (0, 1):
            if abs(pr["sat_frac_per_joint"][j] - pr["violation_steps_per_joint"][j] / n) > 5e-5:
                e_bad.append((nm, j, "sat_frac≠viol/n"))
            if pr["violation_steps_per_joint"][j] > n or pr["violation_steps_per_joint"][j] < 0:
                e_bad.append((nm, j, "viol范围"))
            if pr["demand_peak_per_joint"][j] < 0 or pr["demand_peak_per_joint"][j] is None:
                e_bad.append((nm, j, "峰值非法"))
            # 越限即意味着峰值必须超过该关节上限（互为前提）
            over = pr["demand_peak_per_joint"][j] > E74.E66.TAU_LIM[j] + 1e-9
            if over != (pr["violation_steps_per_joint"][j] > 0):
                e_bad.append((nm, j, "峰值越限与越限步数不一致"))
    ch.append(_chk("F12_E_per_joint_selfconsistent", not e_bad, "反例=%s" % e_bad[:4]))

    # ---- F13 **F 段（振幅口径）覆盖** ----
    Fsec = rep.get("F_amplitude_calibration")
    f_bad = []
    if not Fsec:
        f_bad.append("F 段缺失")
    else:
        amps = sorted({pt["amp_m"] for v in Fsec["by_wn"].values() for pt in v["per_amp"].values()})
        if amps != sorted(rep["bench"]["amp_sweep"]):
            f_bad.append(("amp 与 bench.amp_sweep 不符", amps, rep["bench"]["amp_sweep"]))
        for nm, v in Fsec["by_wn"].items():
            if not v["boundary_non_increasing_in_amp"]:
                f_bad.append((nm, "B 对 amp 非单调"))
            for ak, pt in v["per_amp"].items():
                for t in E74.TOLS:
                    k = "tol=%.3fm" % t
                    rb = _recompute_boundary(pt["rows"], t, False)
                    if rb != pt["boundary_any"][k]:
                        f_bad.append((nm, ak, k, pt["boundary_any"][k], rb))
        # ★ 跨段一致：基准振幅下 F 段与 B 段必须给出同一边界
        for nm, wn in ((n, w) for n, w in E74.WN_AMP):
            brow = B["by_wn"].get(nm)
            frow = Fsec["by_wn"].get(nm)
            if brow and frow:
                for t in E74.TOLS:
                    k = "tol=%.3fm" % t
                    if brow["per_tol"][k]["any"]["hz"] != frow["per_amp"]["amp=%.2fm" % E74.AMP]["boundary_any"][k]:
                        f_bad.append((nm, k, "B 段与 F 段不一致",
                                      brow["per_tol"][k]["any"]["hz"],
                                      frow["per_amp"]["amp=%.2fm" % E74.AMP]["boundary_any"][k]))
    ch.append(_chk("F13_amplitude_section_consistent", not f_bad, "反例=%s" % f_bad[:3]))

    # ---- F14 **G 段（控制器族口径）覆盖** ----
    # 口径穷举最后一条：带宽依赖控制器族。G 由控制代码算出（非数据文件可独立复现），
    # 故覆盖=内部一致性（7 族齐全、全测得、spread_x 可独立重算、pdg_hz 与 by_family[PD_G] 一致）。
    Gsec = rep.get("G_controller_family_calibration")
    g_bad = []
    if not Gsec:
        g_bad.append("G 段缺失")
    else:
        fams = Gsec.get("by_family") or {}
        exp = ("P", "PD", "PID", "PD_G", "CT", "FO", "MPC")
        if tuple(fams.keys()) != exp:
            g_bad.append(("族集合不符", tuple(fams.keys())))
        vals = [v.get("bandwidth_hz") for v in fams.values()]
        if any(v is None for v in vals):
            g_bad.append("有族未测得")
        else:
            rec_spread = max(vals) / min(vals)
            if Gsec.get("spread_x") is None or abs(Gsec["spread_x"] - rec_spread) > 1e-3:
                g_bad.append(("spread_x 不一致", Gsec.get("spread_x"), round(rec_spread, 3)))
            if Gsec.get("pdg_hz") is None or abs(Gsec["pdg_hz"] - fams["PD_G"]["bandwidth_hz"]) > 1e-9:
                g_bad.append(("pdg_hz 不一致", Gsec.get("pdg_hz"), fams["PD_G"]["bandwidth_hz"]))
    ch.append(_chk("F14_controller_family_section_consistent", not g_bad, "反例=%s" % g_bad[:4]))

    # F6 α 去额关系
    exp = math.sqrt(E74.ALPHA_E71)
    ch.append(_chk("F6_alpha_derate_relation",
                   abs(exp - D["derate_ratio_expected_sqrt_alpha"]) < 1e-3
                   and D["derate_rel_dev"] <= 0.02,
                   "sqrt(alpha)=%.4f expected=%s rel_dev=%s" % (exp, D["derate_ratio_expected_sqrt_alpha"], D["derate_rel_dev"])))

    # F7 口径诚实
    ch.append(_chk("F7_calibration_honest",
                   len(D["caveats"]) >= 4 and B["sat_tol"] == E74.SAT_TOL
                   and all(v["gap"] is None or v["gap"] >= 0 for v in D["boundary_gap_tol1cm"].values()),
                   "caveats=%d" % len(D["caveats"])))

    # F8 不变量自洽
    inv = bool(rep["verdict_pass"] == all(bool(v) for v in rep["criteria"].values()))
    ch.append(_chk("F8_invariants_self_consistent", inv, "verdict=%s" % rep["verdict_pass"]))

    # F9 确定性（纯函数同输入同输出）
    det = (E74.uniform_gains(7.7582) == E74.uniform_gains(7.7582)
           and E74.f3db_second_order(5.269) == E74.f3db_second_order(5.269)
           and _recompute_boundary(list(B["by_wn"].values())[0]["rows"], 0.01, False)
           == _recompute_boundary(list(B["by_wn"].values())[0]["rows"], 0.01, False))
    ch.append(_chk("F9_determinism", bool(det), "non-deterministic"))

    # F10 规格自足（承 e73 盲测教训：数据文件须可重建）
    bench = rep["bench"]
    spec = ["model", "L1", "L2", "G", "TAU_LIM", "qdot_cap", "dt", "link_params", "zeta", "wn_design"]
    ch.append(_chk("F10_spec_self_contained", all(k in bench for k in spec),
                   "missing=%s" % [k for k in spec if k not in bench]))
    return ch


def reverse(rep):
    ch = []
    bad = copy.deepcopy(rep)
    first = list(bad["B_task_tolerance_bw"]["by_wn"])[2]
    bad["B_task_tolerance_bw"]["by_wn"][first]["rows"]["0.5"]["rms_err_m"] = 0.0001
    ch.append(_chk("R1_tamper_rms_detected",
                   any(c["name"] == "F3_boundaries_recompute_per_cell" and not c["pass"] for c in forward(bad)),
                   "改 rms 行→F3 必报"))

    bad2 = copy.deepcopy(rep)
    bad2["B_task_tolerance_bw"]["by_wn"][first]["per_tol"]["tol=0.010m"]["any"] = {"hz": 9.9, "bracket": [9.9, None]}
    ch.append(_chk("R2_tamper_boundary_detected",
                   any(c["name"] == "F3_boundaries_recompute_per_cell" and not c["pass"] for c in forward(bad2)),
                   "改边界→F3 必报"))

    bad3 = copy.deepcopy(rep)
    nm = list(bad3["A_freq_domain_bw"]["by_wn"])[3]
    bad3["A_freq_domain_bw"]["by_wn"][nm]["analytic_hz"] = 99.0
    ch.append(_chk("R3_tamper_analytic_detected",
                   any(c["name"] == "F2_analytic_and_ratio_recompute" and not c["pass"] for c in forward(bad3)),
                   "改解析值→F2 必报"))

    bad4 = copy.deepcopy(rep)
    bad4["D_calibration"]["alpha_e71"] = float("inf")
    ch.append(_chk("R4_inject_inf_detected",
                   any(c["name"] == "F1b_no_inf_nan" and not c["pass"] for c in forward(bad4)),
                   "注入 inf→F1b 必报"))

    bad5 = copy.deepcopy(rep)
    bad5["verdict_pass"] = not rep["verdict_pass"]
    ch.append(_chk("R5_flip_verdict_detected",
                   any(c["name"] == "F8_invariants_self_consistent" and not c["pass"] for c in forward(bad5)),
                   "翻转 verdict→F8 必报"))

    bad6 = copy.deepcopy(rep)
    bad6["bench"].pop("link_params", None)
    ch.append(_chk("R6_missing_spec_detected",
                   any(c["name"] == "F10_spec_self_contained" and not c["pass"] for c in forward(bad6)),
                   "删规格→F10 必报"))

    fwd = forward(rep)
    ch.append(_chk("R7_clean_passes", all(c["pass"] for c in fwd),
                   "clean=%d/%d" % (sum(c["pass"] for c in fwd), len(fwd))))
    ch.append(_chk("R8_nonfinite_to_none",
                   E74._num(float("nan")) is None and E74._num(float("inf")) is None
                   and E74._num(1.23456) == 1.2346, "nonfinite not capped"))

    # R9/R10 **E 段篡改必报**（该段此前无覆盖 → 先证明新检查确实敏感）
    bad7 = copy.deepcopy(rep)
    pn = list(bad7["E_identification_probes"]["probes"])[0]
    bad7["E_identification_probes"]["probes"][pn]["violation_steps_per_joint"][1] = 1  # 只改分子不改 sat_frac
    ch.append(_chk("R9_tamper_E_perjoint_detected",
                   any(c["name"] == "F12_E_per_joint_selfconsistent" and not c["pass"] for c in forward(bad7)),
                   "改逐关节越限步数→F12 必报"))

    bad8 = copy.deepcopy(rep)
    bad8["E_identification_probes"]["joint1_limit_exercised"] = False
    ch.append(_chk("R10_tamper_E_exercised_detected",
                   any(c["name"] == "F11_E_section_present_selfcheck" and not c["pass"] for c in forward(bad8)),
                   "翻 joint1_limit_exercised→F11 必报"))

    bad9 = copy.deepcopy(rep)
    bad9["bench"]["payload"]["mp_probe_kg"] = 2.0        # 与探针声明的 mp 不符
    ch.append(_chk("R11_tamper_E_mp_spec_detected",
                   any(c["name"] == "F12_E_per_joint_selfconsistent" and not c["pass"] for c in forward(bad9)),
                   "改 bench.payload.mp→F12 必报"))

    # R12/R13 **F 段篡改必报**
    bad10 = copy.deepcopy(rep)
    wnF = list(bad10["F_amplitude_calibration"]["by_wn"])[0]
    bad10["F_amplitude_calibration"]["by_wn"][wnF]["per_amp"]["amp=0.06m"]["rows"]["0.6"]["rms_err_m"] = 0.0001
    ch.append(_chk("R12_tamper_F_rows_detected",
                   any(c["name"] == "F13_amplitude_section_consistent" and not c["pass"] for c in forward(bad10)),
                   "改 F 段 rms 行→F13 必报"))

    bad11 = copy.deepcopy(rep)
    bad11["bench"]["amp_sweep"] = [0.03, 0.06, 0.09]
    ch.append(_chk("R13_tamper_F_ampspec_detected",
                   any(c["name"] == "F13_amplitude_section_consistent" and not c["pass"] for c in forward(bad11)),
                   "改 bench.amp_sweep→F13 必报"))

    # R14 **G 段篡改必报**（该段此前无覆盖 → 先证明新检查确实敏感）
    bad12 = copy.deepcopy(rep)
    bad12["G_controller_family_calibration"]["by_family"]["PD_G"]["bandwidth_hz"] = 9.9
    ch.append(_chk("R14_tamper_G_family_detected",
                   any(c["name"] == "F14_controller_family_section_consistent" and not c["pass"] for c in forward(bad12)),
                   "改 PD_G 带宽→F14 必报"))
    return ch


def main():
    rep = load()
    fwd, rev = forward(rep), reverse(rep)
    allc = fwd + rev
    npass = sum(c["pass"] for c in allc)
    out = {"audit": "_audit_e74_bandwidth", "experiment": "e74_tracking_bandwidth_curve",
           "pass": npass, "total": len(allc), "verdict": bool(npass == len(allc)),
           "audit_pass": bool(npass == len(allc)), "n_pass": npass, "n_total": len(allc),
           "forward": fwd, "reverse": rev, "checks": allc}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit_e74_bandwidth.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("audit pass = %d/%d  verdict=%s" % (npass, len(allc), out["verdict"]))
    for c in allc:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
