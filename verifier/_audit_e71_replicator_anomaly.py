# -*- coding: utf-8 -*-
"""_audit_e71_replicator_anomaly.py —— E71 的**第三方复刻者排查**（纯数据、零作者模块 import）。

模拟外部复刻者的"拿到报告 JSON 后做的独立核验"：只读 e71_gain_budget_report.json，
不 import 任何作者实验模块（e71_* / e67_* / control_strategies）。它做三件事：

  (1) 内部自洽：C1–C5 布尔、B 段边界、verdict 是否都能从报告原始字段**独立重算**出来；
  (2) 口径诚实：无饱和边界=None 是否真实（没有任何 within_tol∧unsaturated 格）；
       能力数值是否带 wn 增益口径（非裸硬件上限）；刚度代价是否诚实（α<1⇒ωn↓）；
  (3) 篡改必报：构造几类改写 → 必有检查捕获（防止报告被静默篡改）。

注：本脚本不重跑闭环仿真（那是 blind_replicate_e71.py 的职责）；它只核验
"报告内部各字段之间"是否自洽——这是复刻者拿到一篇投稿时最先做的健全性检查。

产物：_audit_e71_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e71_gain_budget_report.json")
OUT = os.path.join(EVAL, "_audit_e71_replicator_anomaly.json")
TOL_CM = 0.01

_res = {"experiment": "E71 第三方复刻者排查（纯数据、零作者模块）", "checks": [], "tamper": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def _recompute_invariants(rep):
    """从报告原始字段独立重算全部不变量（不信任任何自述布尔）。返回 (ok, bad, 重算C1-C5)。"""
    A = rep["A_trajectory_budget"]
    B = rep["B_capability_curve"]
    C = rep["C_consistency"]
    tb = A["trajectory_budget"]
    before = A["before"]
    after = A["after"]
    frac = rep["frac"]
    tau_lim = rep["tau_lim"]
    budget = [frac * t for t in tau_lim]

    c1 = (tb["alpha"] <= 1.0 + 1e-9) and (tb["ratio_after"] <= 1.0 + 1e-9)
    c2 = (after["errL_m"] <= TOL_CM + 1e-9) and (after["errR_m"] <= TOL_CM + 1e-9)
    c3 = (after["violation_steps_demand"] == 0)
    # C4/C5 报告存为**逐频率**字典（{freq: bool}），重算须同结构比较
    c4 = {}
    c5 = {}
    wn_seq = sorted(float(k.split("=")[1]) for k in B["rows"])
    freqs = list(next(iter(B["rows"].values()))["per_freq"].keys())
    for f in freqs:
        rms = [B["rows"]["wn=%.0f" % w]["per_freq"][f]["rms_err_m"] for w in wn_seq]
        sat = [B["rows"]["wn=%.0f" % w]["per_freq"][f]["sat_frac_demand"] for w in wn_seq]
        c4[f] = all(rms[i] <= rms[i - 1] + 1e-9 for i in range(1, len(rms)))  # rms 随增益非增
        c5[f] = all(sat[i] >= sat[i - 1] - 1e-9 for i in range(1, len(sat)))  # sat 随增益非减

    bad = []
    if C["C1_alpha_le_1_and_within_budget"] != c1:
        bad.append("C1 不符")
    if C["C2_endpoint_precision_kept"] != c2:
        bad.append("C2 不符")
    if C["C3_demand_violation_removed"] != c3:
        bad.append("C3 不符")
    if C["C4_rms_weakly_monotone_in_gain"] != c4:
        bad.append("C4 不符")
    if C["C5_sat_frac_non_decreasing_in_gain"] != c5:
        bad.append("C5 不符")

    # boundary_any 由 per_freq within_tol_1cm 重算（不读存储的 max_trackable 字段，
    # 否则篡改单个 within_tol 格不会被捕获）
    any_recomp = []
    for row in B["rows"].values():
        ok = [float(f[:-2]) for f, cell in row["per_freq"].items() if cell["within_tol_1cm"]]
        any_recomp.append(max(ok) if ok else None)
    any_b = max([x for x in any_recomp if x is not None], default=None)
    uns_b = max([v["max_trackable_hz_unsaturated"] for v in B["rows"].values()
                 if v["max_trackable_hz_unsaturated"]], default=None)
    if B["boundary_any_hz"] != any_b:
        bad.append("boundary_any 不符")
    if B["boundary_unsaturated_hz"] != uns_b:
        bad.append("boundary_unsat 不符")
    if rep.get("verdict_pass") != (c1 and c2 and c3):
        bad.append("verdict 不符")
    return (len(bad) == 0), bad, (c1, c2, c3, c4, c5)


def main():
    print("=" * 92)
    print("E71 第三方复刻者排查（纯数据、零作者模块 import）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    A = rep["A_trajectory_budget"]
    B = rep["B_capability_curve"]
    C = rep["C_consistency"]
    tb = A["trajectory_budget"]

    # ================= 内部自洽 =================
    print("\n[内部自洽] C1–C5 / 边界 / verdict 从原始字段独立重算")
    ok_inv, bad, _ = _recompute_invariants(rep)
    _chk("T1 报告三段 + frac/tau_lim 顶层 + verdict 字段齐全",
         all(k in rep for k in ("A_trajectory_budget", "B_capability_curve", "C_consistency",
                                "frac", "tau_lim", "verdict_pass", "verdict_note")))
    _chk("T2 C1–C5 布尔 / B 段边界 / verdict 与原始字段重算完全一致（无自述造假）",
         ok_inv, "违反=%s" % (bad or "无"))

    # T3 before 确失败（动机真实）：越限>0 且 τmax > 预算上限
    frac = rep["frac"]
    tau_lim = rep["tau_lim"]
    budget = [frac * t for t in tau_lim]
    s3 = (A["before"]["violation_steps_demand"] > 0
          and A["before"]["tau_max_demand"] > max(budget) + 1e-6)
    _chk("T3 before 确失败（越限>0 且 τmax>预算上限）—— 预算化必要性成立",
         s3, "越限=%d τmax=%.4f 预算max=%.4f" % (A["before"]["violation_steps_demand"],
                                               A["before"]["tau_max_demand"], max(budget)))

    # T4 after 确修复：越限=0 且 端点≤1cm
    _chk("T4 after 确修复（越限=0 且 端点误差≤1cm）—— 精度未被刚度代价牺牲",
         A["after"]["violation_steps_demand"] == 0
         and A["after"]["errL_m"] <= TOL_CM and A["after"]["errR_m"] <= TOL_CM,
         "越限=%d errL=%.5f errR=%.5f" % (A["after"]["violation_steps_demand"],
                                          A["after"]["errL_m"], A["after"]["errR_m"]))

    # T5 无饱和边界诚实：没有任何 within_tol∧unsaturated 格 + boundary_unsaturated_hz is None
    cnt = 0
    for row in B["rows"].values():
        for cell in row["per_freq"].values():
            if cell["within_tol_1cm"] and cell["unsaturated"]:
                cnt += 1
    _chk("T5 无饱和边界诚实：无 within_tol∧unsaturated 格 且 boundary_unsaturated_hz=None",
         cnt == 0 and B["boundary_unsaturated_hz"] is None,
         "within_tol∧unsaturated 格数=%d 边界=%s" % (cnt, B["boundary_unsaturated_hz"]))

    # T6 能力数值带 wn 增益口径（非裸硬件上限）
    has_wn = "wn_rad_s" in next(iter(B["rows"].values()))
    design = str(B.get("design", ""))
    has_label = ("ωn" in design) or ("wn" in design.lower()) or ("增益" in design)
    note_ok = ("unsaturated" in str(B.get("note", "")).lower()
               and "saturat" in str(B.get("note", "")).lower())
    _chk("T6 B 段每行带 wn 增益口径 + design/note 注明饱和不作为能力主张",
         has_wn and has_label and note_ok,
         "wn_rad_s=%s design带宽标记=%s note口径=%s" % (has_wn, has_label, note_ok))

    # T7 刚度代价诚实：α<1 ⇒ ωn_after<ωn_before 且 α 与 ωn 代价闭式一致
    pred_wn = (tb["alpha"] ** 0.5) * tb["wn_before_rad_s"]
    _chk("T7 刚度代价诚实（α<1 ⇒ ωn_after<ωn_before；√α·ωn_b≈ωn_a）",
         tb["alpha"] < 1.0 - 1e-9
         and tb["wn_after_rad_s"] < tb["wn_before_rad_s"]
         and abs(pred_wn - tb["wn_after_rad_s"]) < 2e-3,
         "α=%.4f ωn %.3f→%.3f ｜ √α·ωn_b=%.4f" % (tb["alpha"], tb["wn_before_rad_s"],
                                                   tb["wn_after_rad_s"], pred_wn))

    # T8 门控前需求口径：before/after 的 violation_steps_demand 是门控前（执行=0 已被 gate 吸收）
    _chk("T8 越限口径是门控前需求（executed=0 ≠ demand>0，未用门控掩盖）",
         A["before"]["violation_steps_executed"] == 0
         and A["before"]["violation_steps_demand"] > 0
         and A["after"]["violation_steps_executed"] == 0
         and A["after"]["violation_steps_demand"] == 0,
         "before exec=%d demand=%d ｜ after exec=%d demand=%d"
         % (A["before"]["violation_steps_executed"], A["before"]["violation_steps_demand"],
            A["after"]["violation_steps_executed"], A["after"]["violation_steps_demand"]))

    # ================= 篡改必报 =================
    print("\n[篡改必报] 构造改写 → 必有检查捕获")
    def _tamp(name, mut, expect):
        bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
        mut(bad)
        ok_b, bad_b, _ = _recompute_invariants(bad)
        caught = (not ok_b)  # 篡改破坏了内部自洽
        _n[0] += 1
        _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught),
                               "detail": "期望 %s" % expect})
        print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                      ("  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!")))
        return bool(caught)

    _tamp("T9 翻转 verdict_pass → T2 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]),
          "T2 重算 verdict 不符")
    _tamp("T10 篡改 after 越限步 0→100 → T2/T4 必报",
          lambda r: r["A_trajectory_budget"]["after"].__setitem__("violation_steps_demand", 100),
          "T2 重算 C3 不符 / T4 失败")
    _tamp("T11 篡改 boundary_unsaturated_hz 2.0 → T2/T5 必报",
          lambda r: r["B_capability_curve"].__setitem__("boundary_unsaturated_hz", 2.0),
          "T2 边界不符 / T5 失败")
    _tamp("T12 篡改 α=1.1>1 → T2/T7 必报",
          lambda r: r["A_trajectory_budget"]["trajectory_budget"].__setitem__("alpha", 1.1),
          "T2 重算 C1 不符 / T7 失败")
    # T13：翻转绑定边界的 within_tol 格（最高 within_tol 频率处）→ T2 必报
    def _flip_tol(r):
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
    _tamp("T13 翻转绑定边界 within_tol 格 → T2 必报",
          _flip_tol, "T2 重算 boundary_any 不符")

    nchk = sum(1 for c in _res["checks"] if c["pass"])
    ntam = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (nchk == len(_res["checks"])) and (ntam == len(_res["tamper"]))
    _res["n_pass"] = nchk
    _res["n_total"] = len(_res["checks"])
    _res["n_tamper_pass"] = ntam
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["thirdparty_pass"] = bool(allpass)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, ensure_ascii=False, indent=2)
    print("\n第三方排查：内部自洽 %d/%d ｜ 篡改 %d/%d ｜ thirdparty_pass = %s"
          % (nchk, len(_res["checks"]), ntam, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
