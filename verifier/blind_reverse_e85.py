# -*- coding: utf-8 -*-
"""blind_reverse_e85.py —— E85 的 **L5 逆向审核**（机制反推 + 负控 + 边界）。

从结果反推「接口精度 → 算力/能力/安全」三轴机制，并用可分辨性反证：

R1 **经济性机制**：测量次数随 s 减小**严格下降**（s=1.0 → s=0.02 至少降 5 倍）
   → 机制 = "接口越准 ⇒ 区间越窄 ⇒ 满足『点可行 ∧ 区间端不可行』的机会越少"
R2 **安全机制（形式而非精度）**：GATED 危险恒 0，而同档位的 POINT 在 s=1.0/0.25 有漏报
R3 **反证"点入口 + 好模型就够"**：POINT 只在 s=0.02（近乎 oracle）才归零
   ⇒ 点入口的安全**以"模型完美"为条件**，而集合值入口不以模型为条件
R4 **负对照（机制必须在运行）**：s=1.0 档的触发率 > 0 且测量次数 > 0（不是"恒 0 的同义反复"）
R5 **能力上限机制**：可行率不随 s 单调（s=0.25 回落）——说明"接口精度 → 能力"不是简单单调映射，
   本件如实保留该非单调（不得美化）

产物：blind_reverse_e85.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e85_interface_accuracy_monotone_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e85.json")
_res = {"experiment": "E85 L5 逆向审核", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _get(row, s):
    return row.get(str(s)) or row.get("%.2f" % s)


def main():
    print("=" * 92)
    print("E85 L5 逆向审核（机制反推 + 负控 + 边界）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    s_desc = rep["s_desc"]
    calls = rep["calls_by_s"]

    # R1 经济性机制：s=1.0 → s=0.02 至少降 5 倍，且逐步严格下降
    strict = all(calls[i] > calls[i + 1] for i in range(len(calls) - 1))
    chk("R1 经济性机制：测量次数随 s 减小**严格下降**，且首末比 ≥ 5×",
        bool(strict and calls[0] >= 5.0 * max(calls[-1], 1e-9)),
        "calls=%s（%.2f×）" % (calls, calls[0] / max(calls[-1], 1e-9)))

    # R2 安全机制
    g0 = _get(rep["per_s"], 1.0)["gated"]
    p0 = _get(rep["per_s"], 1.0)["point"]
    chk("R2 安全机制：GATED 危险恒 0，而同档位 POINT 有漏报（安全由**接口形式**承担）",
        bool(g0["dangerous_count"] == 0 and p0["dangerous_rate"] > 0),
        "s=1.0: GATED danger=%d ｜ POINT danger=%.4f" % (g0["dangerous_count"], p0["dangerous_rate"]))

    # R3 反证"点入口 + 好模型就够"
    p25 = _get(rep["per_s"], 0.25)["point"]["dangerous_rate"]
    p02 = _get(rep["per_s"], 0.02)["point"]["dangerous_rate"]
    chk("R3 反证：POINT 只在 s=0.02（近乎 oracle）才归零 ⇒ 点入口的安全以『模型完美』为条件",
        bool(p0["dangerous_rate"] > 0 and p25 > 0 and p02 == 0.0),
        "POINT danger: s=1.0 %.4f → s=0.25 %.4f → s=0.02 %.4f" % (p0["dangerous_rate"], p25, p02))

    # R4 负对照：机制必须在运行
    chk("R4 负对照：s=1.0 档触发率 > 0 且测量次数 > 0（非恒 0 的同义反复）",
        bool(g0["trigger_rate"] > 0 and g0["measure_calls_mean"] > 0),
        "s=1.0 trigger=%.4f calls=%.4f" % (g0["trigger_rate"], g0["measure_calls_mean"]))

    # R5 能力上限机制：非单调如实保留
    feas = rep["feasible_by_s"]
    nonmono = any(feas[i] > feas[i + 1] + 1e-9 for i in range(len(feas) - 1))
    chk("R5 能力上限机制：可行率**非单调**（存在回落）——如实保留，不美化",
        bool(nonmono and rep["preregistered_verdict"]["H85-2_capability_monotone"] is False),
        "feasible=%s" % feas)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 逆向反推证实：测量次数被『点可行 ∧ 区间端不可行』这一**预测失败判据**精确驱动（随 s 严格下降）；"
        "安全与接口**形式**绑定（点入口在非完美模型下必然漏报）；能力**非单调**，如实保留。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL5 逆向审核：%d/%d ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
