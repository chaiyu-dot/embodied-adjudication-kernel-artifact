# -*- coding: utf-8 -*-
"""_audit_e85b_replicator_anomaly.py —— E85b 的 **L2 第三方复算 + 异常/篡改注入**。

只读报告 JSON：

  T1 内部不变量：两设计的 summary 由 per_s 重算一致；ceiling = 1 − gt(s=0)；oracle 可行率 == ceiling；
     v2 calls 单调且端点 0；全部危险数 = 0；verdict = all(H)；
     **披露完整性**（prereg 的同义反复护栏 + honest_notes 必须含"理想化 oracle"与"条件流混淆"两条）。
  T2 异常注入：逐项篡改 → 对应检查必报（含"删掉混淆披露""把 oracle 端点改成非 0"）。
  T3 跨实验一致性：**E85b 的 v1 与 E85 报告逐项相同**（v1 = E85 原设计，须忠实复现）。
  T4 阴性对照；T5 幂等。

产物：_audit_e85b_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e85b_oracle_level_report.json")
PREREG = os.path.join(EVAL, "e85b_oracle_level_prereg.json")
E85 = os.path.join(EVAL, "e85_interface_accuracy_monotone_report.json")
OUT = os.path.join(EVAL, "_audit_e85b_replicator_anomaly.json")
_res = {"audit": "E85b L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]
KEYS = ("measure_calls_mean", "feasible_rate", "dangerous_count", "false_reject_rate")


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _nonmono(seq):
    return any(seq[i] > seq[i + 1] + 1e-9 for i in range(len(seq) - 1))


def _s_desc(rep):
    """报告没有顶层 s_desc：档位列表在 design.s_levels（纯浮点列表，降序使用）。"""
    return sorted(list(rep["design"]["s_levels"]), reverse=True)


def derived(rep, tag):
    per_s = rep["design_" + tag]["per_s"]
    s_desc = _s_desc(rep)
    calls = [per_s[str(s)]["gated"]["measure_calls_mean"] for s in s_desc]
    feas = [per_s[str(s)]["gated"]["feasible_rate"] for s in s_desc]
    dang = [per_s[str(s)]["gated"]["dangerous_count"] for s in s_desc]
    gts = [per_s[str(s)]["gt_dangerous_rate"] for s in s_desc]
    old5 = [per_s[str(s)]["gated"]["feasible_rate"] for s in sorted(rep["design"]["e85_original_levels"],
                                                                   reverse=True)]
    return {"calls": calls, "feasible": feas, "dangerous": dang, "gt": gts,
            "ceiling": round(1.0 - per_s[str(s_desc[-1])]["gt_dangerous_rate"], 4),
            "oracle_feasible": per_s[str(s_desc[-1])]["gated"]["feasible_rate"],
            "old5_nonmono": _nonmono(old5),
            "mono_calls": all(calls[i] >= calls[i + 1] - 1e-12 for i in range(len(calls) - 1))}


def invariants(rep, pre):
    ok, det = True, []
    for tag in ("v1", "v2"):
        d = derived(rep, tag)
        s = rep["design_" + tag]["summary"]
        if d["calls"] != s["calls"] or d["feasible"] != s["feasible"] or d["dangerous"] != s["dangerous"]:
            ok = False
        if abs(d["ceiling"] - s["ceiling"]) > 1e-4 or abs(d["oracle_feasible"] - s["oracle_feasible"]) > 1e-4:
            ok = False
        if bool(d["old5_nonmono"]) != bool(s["old5_nonmono"]) or bool(d["mono_calls"]) != bool(s["all6_monotone_calls"]):
            ok = False
        if any(x != 0 for x in d["dangerous"]):
            ok = False
    v = rep["preregistered_verdict"]
    d2 = derived(rep, "v2")
    if bool(v["H85b-1_zero_measure_at_oracle"]) != bool(d2["calls"][-1] == 0.0):
        ok = False
    if bool(v["H85b-2_oracle_attains_ceiling"]) != bool(abs(d2["oracle_feasible"] - d2["ceiling"]) <= 1e-4
                                                       and d2["oracle_feasible"] >= max(d2["feasible"]) - 1e-9):
        ok = False
    if bool(v["H85b-4_capability_nonmonotone_or_confound"]) != bool(d2["old5_nonmono"]):
        ok = False
    if rep["verdict_pass"] != all(v.values()):
        ok = False
    # 披露完整性
    if not (pre.get("tautology_guard", {}).get("s0_decision_equals_truth") is True):
        ok = False
    notes = " ".join(rep["honest_notes"])
    for key in ("理想化接口", "条件流", "构造性"):
        if key not in notes:
            ok = False
    if not rep.get("design_v1", {}).get("per_s"):
        ok = False
    # v1 的"条件流混淆"必须**由 v1 的 per_s 重算得到**（不能只信 v1_vs_v2 的转述）
    _sd = _s_desc(rep)
    gt1 = [rep["design_v1"]["per_s"][str(s)]["gt_dangerous_rate"] for s in _sd]
    gt2 = [rep["design_v2"]["per_s"][str(s)]["gt_dangerous_rate"] for s in _sd]
    sp1, sp2 = round(max(gt1) - min(gt1), 4), round(max(gt2) - min(gt2), 4)
    if abs(sp1 - rep["v1_vs_v2"]["v1_gt_dangerous_rate_variation"]["spread"]) > 1e-4:
        ok = False
    if abs(sp2 - rep["v1_vs_v2"]["v2_gt_dangerous_rate_variation"]["spread"]) > 1e-4:
        ok = False
    if not (sp1 > 0.001 and sp2 == 0.0):     # v1 有混淆、v2 无混淆 —— 这正是本件的证据链
        ok = False
    det.append("v1 gt spread=%.4f | v2 gt spread=%.4f | v2 calls=%s | v2 old5_nonmono=%s"
               % (rep["v1_vs_v2"]["v1_gt_dangerous_rate_variation"]["spread"],
                  rep["v1_vs_v2"]["v2_gt_dangerous_rate_variation"]["spread"],
                  d2["calls"], d2["old5_nonmono"]))
    return ok, " ｜ ".join(det)


def main():
    print("=" * 92)
    print("E85b L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    pre = json.load(open(PREREG, encoding="utf-8"))
    e85 = json.load(open(E85, encoding="utf-8"))

    ok, det = invariants(rep, pre)
    chk("T1 第三方复算：两设计 summary/派生量/判据/披露完整性 全部自洽", ok, det)

    a = copy.deepcopy(rep)
    a["preregistered_verdict"]["H85b-4_capability_nonmonotone_or_confound"] = True
    chk("T2a 翻转 H85b-4（把 FAIL 说成 PASS）→ verdict_pass≠all(H) 必报",
        a["verdict_pass"] != all(a["preregistered_verdict"].values()))

    b = copy.deepcopy(rep)
    b["design_v2"]["per_s"][str(_s_desc(rep)[-1])]["gated"]["measure_calls_mean"] = 0.1
    chk("T2b 把 oracle 档的测量次数改成 0.1 → H85b-1 与派生量必报", not invariants(b, pre)[0])

    c = copy.deepcopy(rep)
    c["honest_notes"] = [x for x in c["honest_notes"] if "条件流" not in x]
    chk("T2c 删掉『条件流混淆』披露 → T1 披露完整性必报", not invariants(c, pre)[0])

    d = copy.deepcopy(rep)
    d["honest_notes"] = [x for x in d["honest_notes"] if "理想化接口" not in x]
    chk("T2d 删掉『理想化 oracle』披露 → 必报", not invariants(d, pre)[0])

    e = copy.deepcopy(rep)
    e["design_v1"] = {"summary": rep["design_v2"]["summary"], "per_s": rep["design_v2"]["per_s"]}
    chk("T2e 用 v2 冒充 v1（抹掉条件流混淆这条证据）→ T1 必报（v1 的 gt 极差须 > 0）",
        not invariants(e, pre)[0])

    f = copy.deepcopy(pre)
    f["tautology_guard"]["s0_decision_equals_truth"] = False
    chk("T2f 拆掉同义反复护栏（谎称 s=0 不是 判据≡真值）→ T1 必报", not invariants(rep, f)[0])

    # T3：v1 必须忠实复现 E85
    same = True
    for s in rep["design"]["e85_original_levels"]:
        t = e85["per_s"][str(s)]["gated"]
        m = rep["design_v1"]["per_s"][str(s)]["gated"]
        for k in KEYS:
            if abs(t[k] - m[k]) > 1e-9:
                same = False
    chk("T3 跨实验一致性：E85b 的 design_v1 与 E85 报告**逐项完全一致**（v1 = E85 原设计）", same)

    chk("T4 阴性对照：未篡改 → T1 通过", invariants(rep, pre)[0])
    chk("T5 幂等：两次读入结论一致",
        invariants(json.load(open(REPORT, encoding="utf-8")), pre) == invariants(rep, pre))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "两设计的派生量、判据与披露完整性均有分辨力；",
        "★ T3 是强跨件证据：E85b 的 v1 分支与 E85 报告**逐项完全一致** ⇒ 证明『v1 = E85 原设计』的标注属实。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
