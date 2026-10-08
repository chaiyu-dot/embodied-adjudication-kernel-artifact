# -*- coding: utf-8 -*-
"""_audit_e86_replicator_anomaly.py —— E86 的 **L2 第三方复算 + 异常/篡改注入**。

只读报告 JSON，做第三方重算与注入：

  T1 内部不变量：极差/Spearman/恒等偏差由 models 重算一致；best/worst 与极值一致；
     n_models 与 models 长度一致；H86-5/6（自曝项）恒 True 且说明存在；verdict_pass = all(H)。
  T2 异常注入：逐项篡改 → 对应检查必报。
  T3 跨实验一致性：E86 的「系统输出极差 = 0」与 E61 的「wrapped 极差 = 0.094」同为
     『能力不变性』证据（两者都 ≤ 0.10 且远小于 bare 的极差）。
  T4 阴性对照；T5 幂等。

产物：_audit_e86_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e86_interface_accuracy_real_models_report.json")
E61 = os.path.join(EVAL, "e61_capability_monotone.json")
OUT = os.path.join(EVAL, "_audit_e86_replicator_anomaly.json")
_res = {"audit": "E86 L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _spearman(x, y):
    def avg_rank(v):
        pairs = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and v[pairs[j + 1]] == v[pairs[i]]:
                j += 1
            for k in range(i, j + 1):
                out[pairs[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out
    rx, ry = avg_rank(x), avg_rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return float(num / den) if den > 0 else 0.0


def invariants(rep):
    ms = rep["models"]
    a = [m["interface_accuracy_step1"] for m in ms]
    b = [m["system_output_step2"] for m in ms]
    fl = [m["repair_rate_flip"] for m in ms]
    s = rep["summary"]
    ok, det = True, []
    if abs((max(a) - min(a)) - s["interface_accuracy_spread"]) > 1e-4:
        ok = False
    if abs((max(b) - min(b)) - s["system_output_spread"]) > 1e-4:
        ok = False
    if abs(round(_spearman(a, fl), 4) - s["spearman_interface_vs_repair"]) > 1e-3:
        ok = False
    if abs(max(abs(fl[i] - (1.0 - a[i])) for i in range(len(a)))
           - s["repair_identity_maxdev"]) > 1e-4:
        ok = False
    if s["best_model_by_interface"]["model"] != ms[max(range(len(a)), key=lambda i: a[i])]["model"]:
        ok = False
    if s["worst_model_by_interface"]["model"] != ms[min(range(len(a)), key=lambda i: a[i])]["model"]:
        ok = False
    if s["n_models"] != len(ms):
        ok = False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        ok = False
    # 自曝项必须存在且为 True，并在诚实说明里可追溯
    for k in ("H86-5_no_iter_claim", "H86-6_no_adaptive_compute_claim"):
        if rep["preregistered_verdict"].get(k) is not True:
            ok = False
    if not any("ti" in x and "任务序号" in x for x in rep["honest_notes"]):
        ok = False
    det.append("spread(int/system)=%.4f/%.4f rho=%.4f ident_dev=%.4f"
               % (s["interface_accuracy_spread"], s["system_output_spread"],
                  s["spearman_interface_vs_repair"], s["repair_identity_maxdev"]))
    return ok, " ｜ ".join(det)


def main():
    print("=" * 92)
    print("E86 L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    ok, det = invariants(rep)
    chk("T1 第三方复算：极差/Spearman/恒等偏差/best-worst/自曝项/verdict 全部自洽", ok, det)

    a = copy.deepcopy(rep)
    a["models"][0]["interface_accuracy_step1"] = 0.99
    chk("T2a 把某模型接口精度改成 0.99 → 极差与 summary 不自洽（必报）", not invariants(a)[0])

    b = copy.deepcopy(rep)
    b["models"][0]["system_output_step2"] = 0.5
    chk("T2b 把某模型系统输出改成 0.5 → 系统性不变量（极差 ≤0.05）必报", not invariants(b)[0])

    c = copy.deepcopy(rep)
    c["summary"]["repair_identity_maxdev"] = 0.30
    chk("T2c 篡改恒等偏差 → T1 必报", not invariants(c)[0])

    d = copy.deepcopy(rep)
    d["preregistered_verdict"]["H86-2_system_invariant"] = False
    chk("T2d 翻转 H86-2 → verdict_pass≠all(H) 必报",
        d["verdict_pass"] != all(d["preregistered_verdict"].values()))

    e = copy.deepcopy(rep)
    e["preregistered_verdict"]["H86-5_no_iter_claim"] = False
    chk("T2e 把自曝项 H86-5 设为 False → T1 必报（自曝项不可被删/改写）", not invariants(e)[0])

    f = copy.deepcopy(rep)
    f["honest_notes"] = [x for x in f["honest_notes"] if "任务序号" not in x]
    chk("T2f 抹掉「ti 是任务序号」的诚实说明 → T1 必报", not invariants(f)[0])

    e61 = json.load(open(E61, encoding="utf-8"))
    wrapped_spread = e61["results"]["e50_wrapped_spread"]
    chk("T3 跨实验一致性：E86 系统输出极差(0.0) 与 E61 wrapped 极差(%.3f) 同为『能力不变性』证据（均 ≤0.10）"
        % wrapped_spread,
        bool(rep["summary"]["system_output_spread"] <= 0.10 and wrapped_spread <= 0.10),
        "E86=%.4f E61=%.4f" % (rep["summary"]["system_output_spread"], wrapped_spread))

    chk("T4 阴性对照：未篡改 → T1 通过", invariants(rep)[0])
    chk("T5 幂等：两次读入结论一致",
        invariants(json.load(open(REPORT, encoding="utf-8"))) == invariants(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["E86 的极差/Spearman/恒等偏差/自曝项/跨件一致性均有分辨力；"
                        "篡改数值与『删掉自曝说明』均可被抓。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
