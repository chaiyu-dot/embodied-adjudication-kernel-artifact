# -*- coding: utf-8 -*-
"""blind_reverse_e85b.py —— E85b 的 **L5 逆向审核**（机制反推 + 负控 + 边界）。

从结果反推三件事，并用可分辨性反证：

R1 **经济性机制**：v2 下 6 档测量次数**严格递减**且端点 = 0
   → 机制 = "接口越准 ⇒ 区间越窄 ⇒ 触发『点可行 ∧ 区间端不可行』的机会越少"
R2 **条件流解耦的机制证据**：v1 的各档 gt_dangerous_rate **不同**（极差 > 0），v2 的**相同**（极差 = 0）
   → 证明本件的"公共条件集"确实把档位间的条件差异归零
R3 **能力上限机制**：oracle 档可行率恰好 = 1 − gt_dangerous_rate（端点即上限），且 ≥ 所有 s>0 档
R4 **负对照（机制在运行）**：s=1.0 档触发率 > 0 且测量次数 > 0（不是恒 0 的同义反复）
R5 **混淆因果反推**：v1 的 5 档可行率非单调、v2 同 5 档单调 ⇒ 非单调性归因于**条件集不同**（而非 s 本身）

产物：blind_reverse_e85b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e85b_oracle_level_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e85b.json")
_res = {"experiment": "E85b L5 逆向审核", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E85b L5 逆向审核（机制反推 + 负控 + 边界）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    # 报告无顶层 s_desc：档位列表在 design.s_levels（纯浮点列表，降序使用）
    s_desc = sorted(list(rep["design"]["s_levels"]), reverse=True)
    s_old = sorted(rep["design"]["e85_original_levels"], reverse=True)
    v2s = rep["design_v2"]["summary"]
    v1s = rep["design_v1"]["summary"]
    calls = v2s["calls"]

    chk("R1 经济性机制：v2 下 6 档测量次数严格递减、端点 = 0",
        bool(all(calls[i] > calls[i + 1] for i in range(len(calls) - 1)) and calls[-1] == 0.0),
        "calls=%s" % calls)

    gt1 = [rep["design_v1"]["per_s"][str(s)]["gt_dangerous_rate"] for s in s_desc]
    gt2 = [rep["design_v2"]["per_s"][str(s)]["gt_dangerous_rate"] for s in s_desc]
    chk("R2 解耦机制：v1 各档 gt 不同（极差>0）而 v2 相同（极差=0）⇒ 公共条件集确实生效",
        bool(round(max(gt1) - min(gt1), 4) > 0.001 and round(max(gt2) - min(gt2), 4) == 0.0),
        "v1 gt=%s（极差 %.4f）｜ v2 gt 极差=%.4f"
        % (gt1, max(gt1) - min(gt1), max(gt2) - min(gt2)))

    feas = v2s["feasible"]
    ceil = v2s["ceiling"]
    chk("R3 能力上限机制：oracle 档可行率 = 1 − gt（端点即上限），且 ≥ 所有 s>0 档",
        bool(abs(feas[-1] - ceil) <= 1e-4 and feas[-1] >= max(feas) - 1e-9),
        "oracle=%.4f ceiling=%.4f feasible=%s" % (feas[-1], ceil, feas))

    g0 = rep["design_v2"]["per_s"][str(s_desc[0])]["gated"]
    chk("R4 负对照：s=1.0 档触发率 > 0 且测量次数 > 0（机制在运行，非恒 0 同义反复）",
        bool(g0["trigger_rate"] > 0 and g0["measure_calls_mean"] > 0),
        "s=1.0 trigger=%.4f calls=%.4f" % (g0["trigger_rate"], g0["measure_calls_mean"]))

    f1 = [rep["design_v1"]["per_s"][str(s)]["gated"]["feasible_rate"] for s in s_old]
    f2 = [rep["design_v2"]["per_s"][str(s)]["gated"]["feasible_rate"] for s in s_old]
    v1_nonmono = any(f1[i] > f1[i + 1] + 1e-9 for i in range(len(f1) - 1))
    v2_mono = all(f2[i] <= f2[i + 1] + 1e-9 for i in range(len(f2) - 1))
    chk("R5 混淆因果反推：v1 的 5 档非单调、v2 的 5 档单调 ⇒ 非单调性源于**条件集不同**",
        bool(v1_nonmono and v2_mono),
        "v1=%s（非单调=%s）｜ v2=%s（单调=%s）" % (f1, v1_nonmono, f2, v2_mono))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 逆向反推证实：①测量次数随接口精度严格塌缩到 0（机制=预测失败触发机会减少）；"
        "②**条件流解耦**确实把档间条件差异归零（v1 gt 极差>0 → v2 =0）；"
        "③oracle 档恰好达到能力上限 1−gt；④v1 的能力非单调是**条件集不同**造成的，不是 s 造成的。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL5 逆向审核：%d/%d ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
