# -*- coding: utf-8 -*-
"""_audit_e85_replicator_anomaly.py —— E85 的 **L2 第三方复算 + 异常/篡改注入**。

只读报告 JSON，做第三方重算与注入：

  T1 内部不变量：由 per_s 重算 calls/feasible/dangerous 序列、ceiling、单调性、7 条判据；
     口径不变量 trigger ≤ calls ≤ k_max×trigger；危险仅出现在点入口臂；verdict_pass = all(H)。
  T2 异常注入：逐项篡改 → 对应检查必报（含"把诚实 FAIL 悄悄改成 PASS"这种篡改）。
  T3 跨实验一致性：E85 的 s=1.0 档与 E84（同台架、同门控臂、n 不同）在**方向**上一致
     （POINT 漏报 > 0；GATED 危险 0；GATED 测量 > 0）。
  T4 阴性对照；T5 幂等。

产物：_audit_e85_replicator_anomaly.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e85_interface_accuracy_monotone_report.json")
E84 = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
OUT = os.path.join(EVAL, "_audit_e85_replicator_anomaly.json")
_res = {"audit": "E85 L2 replicator + anomaly injection", "checks": [], "findings": []}
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


def derived(rep):
    """从 per_s 重算全部派生量与判据。"""
    s_desc = sorted([float(k) for k in rep["per_s"]], reverse=True)
    calls = [rep["per_s"][str(s) if str(s) in rep["per_s"] else ("%.2f" % s)]["gated"]["measure_calls_mean"]
             for s in s_desc]
    feas = [rep["per_s"][str(s) if str(s) in rep["per_s"] else ("%.2f" % s)]["gated"]["feasible_rate"]
            for s in s_desc]
    dang = [rep["per_s"][str(s) if str(s) in rep["per_s"] else ("%.2f" % s)]["gated"]["dangerous_count"]
            for s in s_desc]
    ceiling = feas[-1]
    econ = all(calls[i] >= calls[i + 1] - 1e-12 for i in range(len(calls) - 1))
    cap = all(feas[i] <= feas[i + 1] + 1e-12 for i in range(len(feas) - 1))
    coarse = any(feas[i] >= 0.95 * ceiling for i, s in enumerate(s_desc) if s >= 0.25)

    def pt(s, k, d=None):
        row = rep["per_s"].get(str(s)) or rep["per_s"].get("%.2f" % s)
        return (row or {}).get("point", {}).get(k, d)
    v = {
        "H85-1_economy_monotone": bool(econ and calls[0] > calls[-1]),
        "H85-2_capability_monotone": bool(cap),
        "H85-3_safety_flat": bool(all(d == 0 for d in dang)),
        "H85-4_near_oracle_zero_measure": bool(calls[-1] == 0.0),
        "H85-5_coarse_gets_ceiling": bool(coarse),
        "H85-6_point_form_still_leaks": bool(pt(1.0, "dangerous_rate", 0.0) > 0.0
                                               and pt(0.25, "dangerous_rate", 0.0) > 0.0),
        "H85-7_point_needs_oracle": bool(pt(0.02, "dangerous_rate", None) == 0.0),
    }
    extra = {"s_desc": s_desc, "calls": calls, "feasible": feas, "dangerous": dang,
             "ceiling": ceiling, "rho_calls": round(_spearman(s_desc, calls), 4),
             "rho_feas": round(_spearman(s_desc, feas), 4)}
    return v, extra


def invariants(rep):
    v, ex = derived(rep)
    ok, det = True, []
    if v != rep["preregistered_verdict"]:
        ok = False
    if abs(ex["rho_calls"] - rep["spearman_s_vs_calls"]) > 1e-3:
        ok = False
    if [round(x, 4) for x in ex["calls"]] != [round(x, 4) for x in rep["calls_by_s"]]:
        ok = False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        ok = False
    # 口径不变量：trigger ≤ calls ≤ k_max × trigger
    kmax = 3
    for k, row in rep["per_s"].items():
        g = row["gated"]
        if not (g["trigger_rate"] - 1e-9 <= g["measure_calls_mean"] <= kmax * g["trigger_rate"] + 1e-9):
            ok = False
        if g["dangerous_count"] != 0:
            ok = False
    # 三档 POINT 的危险率关系：s=1.0 > s=0.25 > s=0.02
    def pr(s):
        row = rep["per_s"].get(str(s)) or rep["per_s"].get("%.2f" % s)
        return (row or {}).get("point", {}).get("dangerous_rate", None)
    if not (pr(1.0) is not None and pr(0.25) is not None and pr(0.02) is not None
            and pr(1.0) > pr(0.25) > pr(0.02)):
        ok = False
    det.append("calls=%s feas=%s dang=%s rho(calls)=%.4f" % (ex["calls"], ex["feasible"],
                                                             ex["dangerous"], ex["rho_calls"]))
    return ok, " ｜ ".join(det)


def main():
    print("=" * 92)
    print("E85 L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    ok, det = invariants(rep)
    chk("T1 第三方复算：派生序列/单调性/7 判据/口径不变量/点入口危险递减 全部自洽", ok, det)

    a = copy.deepcopy(rep)
    a["per_s"]["1.0"]["gated"]["measure_calls_mean"] = 0.05     # 粗接口却最省算力
    chk("T2a 把 s=1.0 的测量次数压到 0.05 → H85-1 必报", derived(a)[0] != rep["preregistered_verdict"])

    b = copy.deepcopy(rep)
    b["per_s"]["0.25"]["gated"]["dangerous_count"] = 1
    chk("T2b 让某档 GATED 出现危险 → H85-3 与不变量必报", not invariants(b)[0])

    c = copy.deepcopy(rep)
    c["per_s"]["0.02"]["gated"]["measure_calls_mean"] = 0.0
    chk("T2c 把 s=0.02 测量次数改成 0（把诚实 FAIL 悄悄改成 PASS）→ 判据重算必与报告不符",
        derived(c)[0] != rep["preregistered_verdict"])

    d = copy.deepcopy(rep)
    d["per_s"]["0.5"]["gated"]["feasible_rate"] = 0.99        # 破坏能力单调
    chk("T2d 破坏可行率单调 → H85-2 必报", derived(d)[0] != rep["preregistered_verdict"])

    e = copy.deepcopy(rep)
    e["per_s"]["1.0"]["point"]["dangerous_rate"] = 0.0
    chk("T2e 抹掉点入口在粗接口下的漏报 → H85-6 必报", derived(e)[0] != rep["preregistered_verdict"])

    f = copy.deepcopy(rep)
    for k in ("H85-2_capability_monotone", "H85-4_near_oracle_zero_measure",
              "H85-5_coarse_gets_ceiling"):
        f["preregistered_verdict"][k] = True      # 一次性抹掉全部诚实 FAIL
    chk("T2f 把诚实 FAIL 全部翻成 PASS → verdict_pass≠all(H) 必报",
        f["verdict_pass"] != all(f["preregistered_verdict"].values()))

    e84 = json.load(open(E84, encoding="utf-8"))
    mine_pt = rep["per_s"]["1.0"]["point"]["dangerous_rate"]
    mine_g = rep["per_s"]["1.0"]["gated"]
    e84_pt = e84["aggregate"]["POINT"]["dangerous_rate"]
    e84_g = e84["aggregate"]["GATED"]
    chk("T3 跨实验一致性：E85 的 s=1.0 档与 E84 同台架同臂，**方向**一致（n 不同故不逐位相等）",
        bool(mine_pt > 0 and e84_pt > 0 and mine_g["dangerous_count"] == 0
             and e84_g["dangerous"] == 0 and mine_g["measure_calls_mean"] > 0
             and abs(mine_pt - e84_pt) < 0.05),
        "E85 s=1.0 POINT=%.4f GATED meas=%.4f ｜ E84 POINT=%.4f GATED meas=%.4f"
        % (mine_pt, mine_g["measure_calls_mean"], e84_pt, e84_g["measure_calls_mean"]))

    chk("T4 阴性对照：未篡改 → T1 通过", invariants(rep)[0])
    chk("T5 幂等：两次读入结论一致",
        invariants(json.load(open(REPORT, encoding="utf-8"))) == invariants(rep))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["E85 的派生序列/单调性/口径不变量/跨件方向一致性均有分辨力；"
                        "包括『把诚实 FAIL 改成 PASS』在内的篡改均可被抓。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
