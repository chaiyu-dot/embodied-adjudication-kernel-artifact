# -*- coding: utf-8 -*-
"""_audit_e71_strict.py —— E71 的**严格审核（审前提）**。

不 import 任何作者实验模块；只读 e71_gain_budget_report.json（作者主张），
独立核验每一条**前提**与**闭式恒等式**是否成立，并做篡改用例（改报告字段→必有检查报出）。

判据（审前提，非重跑仿真）：
  S1  α ≤ 1 且 ratio_after ≤ 1（预算可行，增益确被下调，换力矩余量）
  S2  闭环前提：after 逐关节峰值需求 ≤ FRAC·τ_lim（预算化二分的闭式目标）
  S3  after 需求越限步数 == 0（过渡段越限清零）
  S4  after 端点误差 ≤ 1cm 任务容差（精度未被牺牲）
  S5  before 确失败：越限 > 0 且峰值 > FRAC·τ_lim（证明预算化的必要性）
  S6  ωn 代价闭式：ωn_after ≈ √α · ωn_before
  S7  B 段物理直觉：rms 随增益非增、sat_frac 随增益非减
  S8  unsaturated 边界 = None 诚实（无任何 within_tol∧unsaturated 格）
  S9  verdict 自洽：verdict_pass == C1∧C2∧C3
  —— 篡改用例 R1–R5：改报告必被上述检查捕获 ——

产物：_audit_e71_strict.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e71_gain_budget_report.json")
OUT = os.path.join(EVAL, "_audit_e71_strict.json")

_res = {"experiment": "E71 严格审核（审前提）", "checks": [], "tamper": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:180]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    """从报告重算全部前提判据（纯数据，不信任任何自述字段）。"""
    ch = []
    frac = rep["frac"]
    tau_lim = rep["tau_lim"]
    A = rep["A_trajectory_budget"]
    det = A["trajectory_budget"]
    before = A["before"]
    after = A["after"]
    B = rep["B_capability_curve"]
    C = rep["C_consistency"]

    budget = [frac * t for t in tau_lim]
    # S1
    ch.append(_chk("S1_α≤1_and_ratio_after≤1",
                   det["alpha"] <= 1.0 + 1e-9 and det["ratio_after"] <= 1.0 + 1e-9,
                   "α=%.6f ratio_after=%.5f" % (det["alpha"], det["ratio_after"])))
    # S2
    pj = after["tau_max_demand_per_joint"]
    s2 = all(pj[i] <= budget[i] + 1e-6 for i in range(2))
    ch.append(_chk("S2_after_per_joint_within_FRAC_tau_lim",
                   s2, "after逐关节%s ≤ 预算%s" % (pj, [round(b, 4) for b in budget])))
    # S3
    ch.append(_chk("S3_after_demand_violation_zero",
                   after["violation_steps_demand"] == 0,
                   "越限步=%d" % after["violation_steps_demand"]))
    # S4
    ch.append(_chk("S4_after_endpoint_precision_kept",
                   after["errL_m"] <= 0.01 + 1e-9 and after["errR_m"] <= 0.01 + 1e-9,
                   "errL=%.5f errR=%.5f" % (after["errL_m"], after["errR_m"])))
    # S5
    s5 = (before["violation_steps_demand"] > 0
          and before["tau_max_demand"] > max(budget) + 1e-6)
    ch.append(_chk("S5_before_genuinely_fails_motivates_budget",
                   s5, "before 越限=%d τmax=%.4f > 预算max=%.4f"
                   % (before["violation_steps_demand"], before["tau_max_demand"], max(budget))))
    # S6
    pred_wn_a = math.sqrt(det["alpha"]) * det["wn_before_rad_s"]
    ch.append(_chk("S6_wn_cost_closed_form",
                   abs(pred_wn_a - det["wn_after_rad_s"]) < 2e-3,
                   "√α·ωn_b=%.5f vs ωn_a=%.5f" % (pred_wn_a, det["wn_after_rad_s"])))
    # S7
    wn_seq = sorted(float(k.split("=")[1]) for k in B["rows"])
    freqs = list(next(iter(B["rows"].values()))["per_freq"].keys())
    mono = True
    satok = True
    for f in freqs:
        rms = [B["rows"]["wn=%.0f" % w]["per_freq"][f]["rms_err_m"] for w in wn_seq]
        sat = [B["rows"]["wn=%.0f" % w]["per_freq"][f]["sat_frac_demand"] for w in wn_seq]
        for i in range(1, len(rms)):
            if (rms[i] - rms[i - 1]) > 1e-9:
                mono = False
            if (sat[i] - sat[i - 1]) < -1e-9:
                satok = False
    ch.append(_chk("S7_B_grid_monotonicity",
                   mono and satok, "rms非增=%s sat非减=%s" % (mono, satok)))
    # S8
    cnt = 0
    for row in B["rows"].values():
        for cell in row["per_freq"].values():
            if cell["within_tol_1cm"] and cell["unsaturated"]:
                cnt += 1
    s8 = (cnt == 0) and (B["boundary_unsaturated_hz"] is None)
    ch.append(_chk("S8_unsaturated_boundary_None_honest",
                   s8, "within_tol∧unsaturated 格数=%d 边界=%s" % (cnt, B["boundary_unsaturated_hz"])))
    # S9
    c1 = C["C1_alpha_le_1_and_within_budget"]
    c2 = C["C2_endpoint_precision_kept"]
    c3 = C["C3_demand_violation_removed"]
    ch.append(_chk("S9_verdict_self_consistent",
                   rep.get("verdict_pass") == (c1 and c2 and c3),
                   "verdict=%s C1∧C2∧C3=%s" % (rep.get("verdict_pass"), c1 and c2 and c3)))
    # S10 boundary_any_hz 由 per_freq within_tol_1cm 重算（不读存储的 max_trackable 字段，
    # 否则篡改单个 within_tol 格不会被任何检查捕获——这是 R5 暴露的覆盖缺口）
    any_recomp = []
    for row in B["rows"].values():
        ok = [float(f[:-2]) for f, cell in row["per_freq"].items() if cell["within_tol_1cm"]]
        any_recomp.append(max(ok) if ok else None)
    any_b = max([x for x in any_recomp if x is not None], default=None)
    s10 = (B["boundary_any_hz"] == any_b)
    ch.append(_chk("S10_boundary_any_hz_recomputed_from_within_tol",
                   s10, "重算=%.4f 存储=%.4f" % (any_b if any_b is not None else float("nan"),
                                                B["boundary_any_hz"] if B["boundary_any_hz"] is not None else float("nan"))))
    return ch


def _tamp(name, mut, expect_fail_names):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n_before = len(_res["checks"])   # 快照，避免篡改重算的 S 检查污染主 checks 列表
    ch = _forward(bad)
    caught = any((c[1] is False) and (c[0].startswith(tuple(expect_fail_names))) for c in ch)
    del _res["checks"][n_before:]    # 回滚篡改重算产生的重复 S 项
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name,
                           "caught": bool(caught),
                           "detail": "期望捕获 %s" % expect_fail_names})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  ("  —— 篡改被捕获" if caught else "  —— !! 篡改未被任何检查捕获 !!")))
    return bool(caught)


def main():
    print("=" * 92)
    print("E71 严格审核（审前提，纯数据、不信任自述）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例（改报告字段 → 必有检查报出）")
    _tamp("R1 篡改 after 逐关节峰值 > 预算 → S2 必报",
          lambda r: r["A_trajectory_budget"]["after"]["tau_max_demand_per_joint"].__setitem__(0, 5.5),
          ["S2"])
    _tamp("R2 篡改 before 越限步 = 0 → S5 必报",
          lambda r: r["A_trajectory_budget"]["before"].__setitem__("violation_steps_demand", 0),
          ["S5"])
    _tamp("R3 篡改 α = 1.1 > 1 → S1 必报",
          lambda r: r["A_trajectory_budget"]["trajectory_budget"].__setitem__("alpha", 1.1),
          ["S1"])
    _tamp("R4 篡改 unsaturated 边界 = 2.0 → S8 必报",
          lambda r: r["B_capability_curve"].__setitem__("boundary_unsaturated_hz", 2.0),
          ["S8"])
    # R5：翻转**绑定边界**的 B 格 within_tol_1cm（最高 within_tol 频率处的所有格），
    # 使重算 boundary_any_hz 必然改变，检验 S10 能否捕获边界与 per_freq 不一致。
    def _flip(r):
        rows = r["B_capability_curve"]["rows"]
        maxf = 0.0
        for row in rows.values():
            for f, cell in row["per_freq"].items():
                if cell["within_tol_1cm"]:
                    maxf = max(maxf, float(f[:-2]))
        for row in rows.values():
            for f, cell in row["per_freq"].items():
                if cell["within_tol_1cm"] and abs(float(f[:-2]) - maxf) < 1e-9:
                    cell["within_tol_1cm"] = False
    _tamp("R5 翻转绑定边界的 within_tol 格 → S10 必报", _flip, ["S10"])

    tpassed = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tpassed == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tpassed
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s"
          % (npass, len(fwd), tpassed, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
