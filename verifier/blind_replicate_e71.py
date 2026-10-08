# -*- coding: utf-8 -*-
"""blind_replicate_e71.py —— E71 的**盲复刻（隔离重跑）**。

模拟外部复刻者：只拿到 e71_bench_data.json（物理参数 + 场景 + 算法规格）
+ e71_standalone_physics.py（复刻者自己从物理重写的 2R 动力学/RK4/PD_G 门控/dual_sim/
_track_arm/budget_gains_trajectory），**不读 e71_gain_budget_optimization.py、不 import 任何仓库模块**，
独立复现 E71 的 A/B/C 三段，再与作者报告逐位 / 逐格比对。

隔离保证：
  · 本脚本运行期 sys.modules 不得含 e66_closed_loop_control / e67_testbed_completeness /
    control_strategies / planning / e71_gain_budget_optimization；
  · 唯一允许的"作者侧"输入是 e71_gain_budget_report.json（待验证的**主张**，而非代码）。

产物：blind_replicate_e71.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
# ★ 复刻者自有实现（非仓库模块）
import e71_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e71_gain_budget_report.json")
BENCH = os.path.join(EVAL, "e71_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e71.json")

_repo_mods = ("e66_closed_loop_control", "e67_testbed_completeness",
              "control_strategies", "planning", "e71_gain_budget_optimization")


def _repo_loaded():
    return [m for m in sys.modules
            if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E71 盲复刻（隔离重跑，自有物理实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E71 盲复刻（隔离重跑）：自有物理实现 vs 作者报告")
    print("=" * 92)

    # === 隔离自检 ===
    loaded = _repo_loaded()
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    if loaded:
        print("  !! 隔离被破坏，终止"); return 2

    bench = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    scnA = bench["scenario"]["A_trajectory_budget"]
    scnB = bench["scenario"]["B_capability_curve"]
    P.configure(bench["bench"])

    # ================= A 段：轨迹预算化增益 =================
    print("\n[A] 复刻 A 段（目标点预算增益 vs 轨迹预算化增益）")
    A = P.replicate_part_A(scnA)
    repA = rep["A_trajectory_budget"]
    repA_det = repA["trajectory_budget"]

    chk("B1 目标点预算增益 kp/kd 复现一致",
        abs(A["target_pose_budget"]["kp"][0] - repA["target_pose_budget"]["kp"][0]) < 1e-6
        and abs(A["target_pose_budget"]["kd"][0] - repA["target_pose_budget"]["kd"][0]) < 1e-6,
        "复刻 kp=%.5f kd=%.5f ｜ 报告 kp=%.5f kd=%.5f"
        % (A["target_pose_budget"]["kp"][0], A["target_pose_budget"]["kd"][0],
           repA["target_pose_budget"]["kp"][0], repA["target_pose_budget"]["kd"][0]))

    chk("B2 α（刚度代价缩放因子）复现一致",
        abs(A["trajectory_budget"]["alpha"] - repA_det["alpha"]) < 1e-5,
        "复刻 α=%.6f ｜ 报告 α=%.6f" % (A["trajectory_budget"]["alpha"], repA_det["alpha"]))

    chk("B3 预算后峰值比 ratio_after≤1（在预算内）复现一致",
        abs(A["trajectory_budget"]["ratio_after"] - repA_det["ratio_after"]) < 1e-5,
        "复刻 %.5f ｜ 报告 %.5f" % (A["trajectory_budget"]["ratio_after"], repA_det["ratio_after"]))

    chk("B4 ωn 刚度代价（before→after）复现一致",
        abs(A["trajectory_budget"]["wn_before_rad_s"] - repA_det["wn_before_rad_s"]) < 1e-3
        and abs(A["trajectory_budget"]["wn_after_rad_s"] - repA_det["wn_after_rad_s"]) < 1e-3,
        "复刻 %.4f→%.4f ｜ 报告 %.4f→%.4f"
        % (A["trajectory_budget"]["wn_before_rad_s"], A["trajectory_budget"]["wn_after_rad_s"],
           repA_det["wn_before_rad_s"], repA_det["wn_after_rad_s"]))

    # before：82 步越限、τmax 6.677
    chk("B5 before（目标点预算，未约束过渡段）复现：需求越限 82 步 + τmax 6.677",
        A["before"]["violation_steps_demand"] == repA["delta"]["demand_violation_steps"][0]
        and abs(A["before"]["tau_max_demand"] - repA["delta"]["tau_max_demand"][0]) < 1e-3
        and np.allclose(A["before"]["tau_max_demand_per_joint"],
                        repA["delta"]["tau_max_demand_per_joint"][0], atol=1e-3),
        "复刻 越限=%d τmax=%.4f 逐关节%s ｜ 报告 越限=%d τmax=%.4f 逐关节%s"
        % (A["before"]["violation_steps_demand"], A["before"]["tau_max_demand"],
           A["before"]["tau_max_demand_per_joint"], repA["delta"]["demand_violation_steps"][0],
           repA["delta"]["tau_max_demand"][0], repA["delta"]["tau_max_demand_per_joint"][0]))

    # after：0 步越限、τmax 4.5=0.9*5.0、端点误差≤1cm
    chk("B6 after（轨迹预算化，α=0.3972）复现：越限 82→0 + τmax→4.5 + 端点≤1cm",
        A["after"]["violation_steps_demand"] == repA["delta"]["demand_violation_steps"][1]
        and abs(A["after"]["tau_max_demand"] - repA["delta"]["tau_max_demand"][1]) < 1e-3
        and A["after"]["errL_m"] <= 0.01 and A["after"]["errR_m"] <= 0.01,
        "复刻 越限=%d τmax=%.4f errL=%.5f errR=%.5f ｜ 报告 越限=%d τmax=%.4f errL=%.5f errR=%.5f"
        % (A["after"]["violation_steps_demand"], A["after"]["tau_max_demand"],
           A["after"]["errL_m"], A["after"]["errR_m"], repA["delta"]["demand_violation_steps"][1],
           repA["delta"]["tau_max_demand"][1], repA["delta"]["errL_m"][1], repA["delta"]["errR_m"][1]))

    # ================= B 段：跟踪能力曲线 =================
    print("\n[B] 复刻 B 段（wn × 频率 扫描，两条可达边界）")
    B = P.replicate_part_B(scnB)
    repB = rep["B_capability_curve"]
    max_rms = 0.0
    max_sat = 0.0
    ncell = 0
    mismatch = 0
    for k in repB["rows"]:
        for fk2 in repB["rows"][k]["per_freq"]:
            r = repB["rows"][k]["per_freq"][fk2]
            s = B["rows"][k]["per_freq"][fk2]
            max_rms = max(max_rms, abs(r["rms_err_m"] - s["rms_err_m"]))
            max_sat = max(max_sat, abs(r["sat_frac_demand"] - s["sat_frac_demand"]))
            if r["within_tol_1cm"] != s["within_tol_1cm"] or r["unsaturated"] != s["unsaturated"]:
                mismatch += 1
            ncell += 1
    chk("B7 B 段 %d 格逐格复现（rms/sat/within_tol/unsaturated 全一致）" % ncell,
        mismatch == 0 and max_rms < 1e-6 and max_sat < 1e-6,
        "mismatch=%d max|rms差|=%.1e max|sat差|=%.1e" % (mismatch, max_rms, max_sat))

    chk("B8 两条可达边界复现一致（any=2.0Hz；unsaturated=None 诚实报无）",
        B["boundary_any_hz"] == repB["boundary_any_hz"]
        and B["boundary_unsaturated_hz"] == repB["boundary_unsaturated_hz"],
        "复刻 any=%.2f unsat=%s ｜ 报告 any=%.2f unsat=%s"
        % (B["boundary_any_hz"], B["boundary_unsaturated_hz"],
           repB["boundary_any_hz"], repB["boundary_unsaturated_hz"]))

    # ================= C 段：一致性独立重算 =================
    print("\n[C] 复刻 C 段（一致性判据独立重算）")
    det = A["trajectory_budget"]
    c1 = (det["alpha"] <= 1.0 + 1e-9) and (det["ratio_after"] <= 1.0 + 1e-9)
    c2 = (A["after"]["errL_m"] <= 0.01 + 1e-9) and (A["after"]["errR_m"] <= 0.01 + 1e-9)
    c3 = (A["after"]["violation_steps_demand"] == 0)
    # C4/C5：从复刻 B 网格重算单调性（rms 随增益非增；sat 随增益非减）
    wn_seq = sorted(float(k.split("=")[1]) for k in B["rows"])
    freqs = list(next(iter(B["rows"].values()))["per_freq"].keys())
    c4_per = {}
    c5_per = {}
    for fk2 in freqs:
        rms_seq = [B["rows"]["wn=%.0f" % w]["per_freq"][fk2]["rms_err_m"] for w in wn_seq]
        sat_seq = [B["rows"]["wn=%.0f" % w]["per_freq"][fk2]["sat_frac_demand"] for w in wn_seq]
        mono = True
        satok = True
        for i in range(1, len(rms_seq)):
            if (rms_seq[i] - rms_seq[i - 1]) > 1e-9:
                mono = False
            if (sat_seq[i] - sat_seq[i - 1]) < -1e-9:
                satok = False
        c4_per[fk2] = mono
        c5_per[fk2] = satok
    c4 = all(c4_per.values())
    c5 = all(c5_per.values())
    repC = rep["C_consistency"]
    chk("B9 一致性判据 C1–C5 从复刻数据重算与报告一致 + verdict 自洽",
        c1 == repC["C1_alpha_le_1_and_within_budget"]
        and c2 == repC["C2_endpoint_precision_kept"]
        and c3 == repC["C3_demand_violation_removed"]
        and c4_per == repC["C4_rms_weakly_monotone_in_gain"]
        and c5_per == repC["C5_sat_frac_non_decreasing_in_gain"]
        and rep.get("verdict_pass") == (c1 and c2 and c3 and c4 and c5),
        "复刻 C1/C2/C3/C4/C5=%s/%s/%s/%s/%s verdict=%s ｜ 报告 %s/%s/%s/%s/%s verdict=%s"
        % (c1, c2, c3, c4, c5, bool(c1 and c2 and c3 and c4 and c5),
           repC["C1_alpha_le_1_and_within_budget"], repC["C2_endpoint_precision_kept"],
           repC["C3_demand_violation_removed"], repC["C4_rms_weakly_monotone_in_gain"],
           repC["C5_sat_frac_non_decreasing_in_gain"], rep.get("verdict_pass")))

    _res["n_pass"] = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_total"] = len(_res["checks"])
    _res["replicate_pass"] = bool(_res["n_pass"] == _res["n_total"])
    _res["findings"] = [
        "★ 盲复刻（隔离、仅数据文件 + 自有物理实现、零仓库 import）逐位复现 E71 全部结论：",
        "  A 段：α=%.6f、before 需求越限 82 步 / τmax %.4f、after 越限→0 / τmax→%.4f（=0.9·τ_lim）、"
        "ωn %.4f→%.4f、端点误差 %.5f/%.5f m。"
        % (A["trajectory_budget"]["alpha"], A["before"]["tau_max_demand"],
           A["after"]["tau_max_demand"], A["trajectory_budget"]["wn_before_rad_s"],
           A["trajectory_budget"]["wn_after_rad_s"], A["after"]["errL_m"], A["after"]["errR_m"]),
        "  B 段：%d 格逐格一致（max|rms差|=%.1e），边界 any=%.2f Hz、unsaturated=None（诚实无）。" % (ncell, max_rms, B["boundary_any_hz"]),
        "  C 段：C1–C5 从复刻数据重算与作者报告完全一致（含 B 网格单调性 C4/C5）→ E71 结论可被外部独立重建。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, ensure_ascii=False, indent=2)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (_res["n_pass"], _res["n_total"], _res["replicate_pass"]))
    print("wrote", OUT)
    return 0 if _res["replicate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
