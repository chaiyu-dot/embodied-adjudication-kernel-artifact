# -*- coding: utf-8 -*-
"""_audit_e69_strict.py —— E69 的**严格审核（L3，审前提 + 独立物理）**。

不 import 任何 e69 / control_strategies / trackability / e58 模块；审计者**自写**平面 2R 的
mass_matrix / gravity_vec / 载荷惯量（与 CS 同一套代数，独立推导），按 E69 的
``achievable_omega_n`` 验收配方（head=frac·(τ_lim−τ_ff), kp=head/|Δq|, ωn=min sqrt(kp/M_ii)）
逐格重算 V0/V1 全部 48 行，核对可达带宽曲线、V2 经验标定关系、V3 盲区逻辑、V4 结论自洽，并篡改。

注：V3 盲区计数依赖 E58.traj + trackability_margin（二者由 E58 L1–L7 + trackability 审核背书），
本件只验证其**结果的逻辑自洽**（假设口径盲区 < 预算口径盲区、两者均 >0、H58-3 翻转），
不重算轨迹/判据本身（属已审模块）。

产物：_audit_e69_strict.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT = os.path.join(EVAL, "e69_bandwidth_report.json")
OUT = os.path.join(EVAL, "_audit_e69_strict.json")

_res = {"experiment": "E69 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：自写 mass_matrix/gravity_vec/载荷惯量 + E69 验收配方，不 import e69/CS/trackability/e58",
        "checks": [], "tamper": []}

# ---- 审计者自写的平面 2R 动力学（与 CS 同一套代数，独立推导）----
L1, L2 = 0.40, 0.30
M1, M2 = 0.60, 0.35
LC1, LC2 = L1 / 2.0, L2 / 2.0
I1 = M1 * L1 * L1 / 12.0
I2 = M2 * L2 * L2 / 12.0
G = 9.81
TAU_LIM = (5.0, 1.8)
FRAC = 0.9
Q_REF = [0.60, -0.90]


def _l2_params(mp):
    m2 = M2 + mp
    lc2 = (M2 * LC2 + mp * L2) / m2 if m2 > 1e-12 else LC2
    i2 = I2 + M2 * (lc2 - LC2) ** 2 + mp * (L2 - lc2) ** 2
    return m2, lc2, i2


def _mass_matrix(q, mp=0.0):
    m2, lc2, i2 = _l2_params(mp)
    c2 = math.cos(q[1])
    m11 = M1 * LC1 * LC1 + m2 * (L1 * L1 + lc2 * lc2 + 2 * L1 * lc2 * c2) + I1 + i2
    m12 = m2 * (lc2 * lc2 + L1 * lc2 * c2) + i2
    m22 = m2 * lc2 * lc2 + i2
    return [[m11, m12], [m12, m22]]


def _gravity_vec(q, mp=0.0):
    m2, lc2, _ = _l2_params(mp)
    g1 = (M1 * LC1 + m2 * L1) * G * math.cos(q[0]) + m2 * lc2 * G * math.cos(q[0] + q[1])
    g2 = m2 * lc2 * G * math.cos(q[0] + q[1])
    return [g1, g2]


def _achievable_wn(q, dq, mp=0.0, tau_ff=None, frac=FRAC, tau_lim=TAU_LIM):
    """独立复刻 E69/V0/V1 的 achievable_omega_n 验收配方。"""
    q = list(q); dq = list(dq)
    if tau_ff is None:
        tau_ff = [abs(x) for x in _gravity_vec(q, mp)]
    else:
        tau_ff = [abs(x) for x in tau_ff]
    M = _mass_matrix(q, mp)
    wn = []
    for i in range(2):
        head = frac * (tau_lim[i] - tau_ff[i])
        if head <= 0.0 or M[i][i] <= 0.0:
            wn.append(0.0)
        else:
            kp = head / max(abs(dq[i]), 1e-3)
            wn.append(math.sqrt(kp / M[i][i]))
    return min(wn)


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    import numpy as np
    ch = []
    # --- S1：V0 ok 自洽（E66 预算 wn == E69 逐关节 wn == 7.758）---
    v0 = rep["V0_consistency_with_E66"]
    s1 = (v0["kp_match"] is True and abs(v0["E66_budget_gains_wn"] - v0["E69_per_joint_wn"]) < 1e-6)
    ch.append(_chk("S1_V0_ok_selfconsistent", v0["ok"] is bool(s1),
                   "E66_wn=%.4f E69_wn=%.4f kp_match=%s ok=%s" % (v0["E66_budget_gains_wn"], v0["E69_per_joint_wn"], v0["kp_match"], v0["ok"])))

    # --- S2：V0 配方独立重算（tau_ff=0，与 E66.budget_gains 同口径）→ 7.758 ---
    wn_qref = _achievable_wn(Q_REF, Q_REF, mp=0.0, tau_ff=[0.0, 0.0])
    ch.append(_chk("S2_V0_recipe_recomputed_at_QREF", abs(wn_qref - v0["E69_per_joint_wn"]) < 1e-3,
                   "recomputed=%.4f stored=%.4f" % (wn_qref, v0["E69_per_joint_wn"])))

    # --- S3：V1 曲线全 48 行逐格重算（tau_ff=重力，默认口径）---
    v1 = rep["V1_curve"]
    rows = v1["rows"]
    rec = [_achievable_wn(r["q"], [r["delta_q"], r["delta_q"]], mp=r["mp_kg"]) for r in rows]
    stored = [r["wn_rad_s"] for r in rows]
    maxabs = max(abs(a - b) for a, b in zip(rec, stored))
    rec_min, rec_med, rec_max = min(rec), float(np.median(rec)), max(rec)
    ch.append(_chk("S3_V1_all48_rows_recomputed", maxabs < 1e-3,
                   "max|Δwn|=%.4f n=%d" % (maxabs, len(rows))))
    ch.append(_chk("S3b_V1_aggregate_matches", abs(rec_min - v1["wn_min"]) < 1e-3
                   and abs(rec_med - v1["wn_median"]) < 1e-3 and abs(rec_max - v1["wn_max"]) < 1e-3,
                   "re_min=%.4f re_med=%.4f re_max=%.4f stored=%.4f/%.4f/%.4f"
                   % (rec_min, rec_med, rec_max, v1["wn_min"], v1["wn_median"], v1["wn_max"])))
    # 全部 < 假设 40
    ch.append(_chk("S3c_V1_all_below_assumed40",
                   v1["all_below_assumed"] is True and rec_max < 40.0,
                   "max_recomputed=%.4f assumed=40" % rec_max))

    # --- S4：V2 经验标定关系自洽 ---
    v2 = rep["V2_empirical"]
    meff = v2["M_eff"]
    wn_f_re = math.sqrt(v2["formula_kp_max"] / meff)   # 注意：公式 wn 用逐关节 min，此处用 meff 仅验量级
    wn_e_re = math.sqrt(v2["empirical_kp_max"] / meff)
    # 独立重算 formula_wn（逐关节 min，tau_ff=0 口径）= 应等于 V0 的 7.758
    fwn = _achievable_wn(Q_REF, Q_REF, mp=0.0, tau_ff=[0.0, 0.0])
    ch.append(_chk("S4_V2_empirical_wn_selfconsistent", abs(wn_e_re - v2["empirical_wn"]) < 1e-2
                   and abs(fwn - v2["formula_wn"]) < 1e-3
                   and v2["formula_over_empirical_wn"] > 1.0,
                   "emp_wn_re=%.4f stored=%.4f form_wn_re=%.4f stored=%.4f ratio=%.3f"
                   % (wn_e_re, v2["empirical_wn"], fwn, v2["formula_wn"], v2["formula_over_empirical_wn"])))
    # 公式偏保守（ratio>1 = 公式 wn > 实测 wn）
    ch.append(_chk("S4b_formula_conservative_not_optimistic", v2["formula_over_empirical_wn"] > 1.0
                   and v2["formula_over_empirical_kp"] < 1.0,
                   "wn_ratio=%.3f kp_ratio=%.3f" % (v2["formula_over_empirical_wn"], v2["formula_over_empirical_kp"])))

    # --- S5：V3 盲区逻辑自洽（假设 < 预算，均 >0；H58-3 翻转）---
    v3 = rep["V3_e58_refill"]
    s5a = (v3["assumed"]["blind_spot_vs_D"] > 0 and v3["budget"]["blind_spot_vs_D"] > 0
           and v3["assumed"]["blind_spot_vs_D"] < v3["budget"]["blind_spot_vs_D"])
    ch.append(_chk("S5_V3_blindspot_logic", s5a,
                   "assumed=%d budget=%d" % (v3["assumed"]["blind_spot_vs_D"], v3["budget"]["blind_spot_vs_D"])))
    s5b = (v3["H58_3_slow_fa_lt_0.10_assumed"] is True and v3["H58_3_slow_fa_lt_0.10_budget"] is False)
    ch.append(_chk("S5b_V3_H58_3_flips_real_bandwidth", s5b,
                   "assumed_PASS=%s budget_FAIL=%s" % (v3["H58_3_slow_fa_lt_0.10_assumed"], v3["H58_3_slow_fa_lt_0.10_budget"])))

    # --- S6：V3b 建议 k_omega 自洽（扫描内无任何 k_omega 把慢档误报压到 <0.10 → None）---
    v3b = rep["V3b_komega_recalibration"]
    all_fa_ge_01 = all((v["slow_false_alarm"] or 0) >= 0.10 for v in v3b["scan"].values())
    ch.append(_chk("S6_V3b_recommendation_consistent", (v3b["recommended_k_omega"] is None) == all_fa_ge_01,
                   "rec=%s all_fa>=0.10=%s" % (v3b["recommended_k_omega"], all_fa_ge_01)))

    # --- S7：V4 结论引用数字与 V2 一致 ---
    v4 = rep["V4_conclusion"]
    s7 = ("%.4f" % v2["empirical_kp_max"] in v4["finding_4"]) and ("%.3f" % v2["empirical_wn"] in v4["finding_4"])
    ch.append(_chk("S7_V4_finding4_matches_V2", s7,
                   "V4 cites kp=%.4f wn=%.3f" % (v2["empirical_kp_max"], v2["empirical_wn"])))

    # --- S8：verdict 自洽 ---
    s8 = rep["verdict_pass"] is (v0["ok"] and v1["all_below_assumed"]
                                 and v3["H58_1_blind_exists_assumed"] and v3["budget"]["blind_spot_vs_D"] > 0)
    ch.append(_chk("S8_verdict_self_consistent", s8, "verdict_pass=%s" % rep["verdict_pass"]))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E69 严格审核（L3 独立物理，不 import e69/CS/trackability/e58）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 V0 ok=false → S1 必报", lambda r: r["V0_consistency_with_E66"].__setitem__("ok", False), ["S1"])
    _tamp("R2 篡改 V1 一行 wn=99 → S3c 必报",
          lambda r: r["V1_curve"]["rows"].__setitem__(0, {**r["V1_curve"]["rows"][0], "wn_rad_s": 99.0}), ["S3", "S3c"])
    _tamp("R3 篡改 V2 empirical_kp=0.5 → S4 必报",
          lambda r: r["V2_empirical"].__setitem__("empirical_kp_max", 0.5), ["S4"])
    _tamp("R4 篡改 V3 budget blind=0 → S5 必报",
          lambda r: r["V3_e58_refill"]["budget"].__setitem__("blind_spot_vs_D", 0), ["S5"])
    _tamp("R5 篡改 verdict_pass=false → S8 必报", lambda r: r.__setitem__("verdict_pass", False), ["S8"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
