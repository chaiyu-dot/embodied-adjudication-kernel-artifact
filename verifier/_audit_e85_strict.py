# -*- coding: utf-8 -*-
"""_audit_e85_strict.py —— E85 的 **L3 严格审核**（结构自洽 + 独立重放 + 篡改捕获 + FAIL 保真）。

与 L2 正交：本层**读取自足数据文件 `e85_bench_data.json`**，用**自写的门控/点入口判据**
（复用仓库判据本体但不用 E85 的 run()）独立重放，并验证"诚实 FAIL 未被美化"。

判据
----
S1 报告↔明细自洽：逐档×逐臂的统计量由数据文件重放一致；verdict = all(H)。
S2 口径不变量：GATED 危险恒 0；trigger ≤ calls ≤ k_max×trigger；点入口危险随档位递减。
S3 **诚实 FAIL 保真**：H85-2 / H85-4 / H85-5 在报告中必须为 **False**，且 honest_notes 必须
   披露"判据设计失误（s=0.02 并非 oracle）"——即**不得事后把 FAIL 改成 PASS**。
S4 篡改捕获：改统计量/改 FAIL/删披露 → S1/S2/S3 必报。
S5 幂等。

产物：_audit_e85_strict.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e85_interface_accuracy_monotone as E85   # noqa: E402

REPORT = os.path.join(EVAL, "e85_interface_accuracy_monotone_report.json")
BENCH = os.path.join(EVAL, "e85_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e85_strict.json")
_res = {"audit": "E85 L3 strict", "checks": [], "findings": []}
_n = [0]
KEYS = ("measure_calls_mean", "feasible_rate", "dangerous_count", "false_reject_rate")


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def replay(data):
    k_conf, k_max = float(data["k_conf"]), int(data["k_max"])
    out = {}
    for lv in data["levels"]:
        s = lv["s"]
        agg = {a: {"n": 0, "calls": 0, "trig": 0, "feasible": 0, "dangerous": 0, "frej": 0}
               for a in ("GATED", "POINT")}
        for p in lv["proposals"]:
            prop = E85.make_proposal(E85.make_traj([0.0, 0.0], p["q_final"], p["T"]), p["q_final"])
            gt_dangerous = E85.five_orders(p["mp_true"], prop) < 0.0
            for arm in ("GATED", "POINT"):
                if arm not in p["states"]:
                    continue
                st = p["states"][arm]
                t = calls = 0
                trig = False
                if arm == "GATED":
                    while True:
                        m, sd = st[t]
                        min_hi = E85.five_orders(m + k_conf * sd, prop)
                        min_pt = E85.five_orders(m, prop)
                        if not ((min_pt >= 0.0) and (min_hi < 0.0)) or calls >= k_max:
                            break
                        calls += 1
                        t += 1
                        trig = True
                    dec = E85.five_orders(st[t][0] + k_conf * st[t][1], prop)
                else:
                    dec = E85.five_orders(st[0][0], prop)
                feas = dec >= 0.0
                a = agg[arm]
                a["n"] += 1
                a["calls"] += calls
                a["trig"] += int(trig)
                a["feasible"] += int(feas)
                a["dangerous"] += int(bool(feas and gt_dangerous))
                a["frej"] += int((not feas) and (not gt_dangerous))
        row = {}
        for arm in ("GATED", "POINT"):
            a = agg[arm]
            if not a["n"]:
                continue
            n = a["n"]
            row[arm] = {"n": n, "measure_calls_mean": round(a["calls"] / n, 4),
                        "trigger_rate": round(a["trig"] / n, 4),
                        "feasible_rate": round(a["feasible"] / n, 4),
                        "dangerous_rate": round(a["dangerous"] / n, 4),
                        "dangerous_count": a["dangerous"],
                        "false_reject_rate": round(a["frej"] / n, 4)}
        out[s] = row
    return out


def _get(row, s):
    return row.get(str(s)) or row.get("%.2f" % s)


def s1(rep, mine):
    for s, arms in mine.items():
        theirs = _get(rep["per_s"], s)
        if theirs is None:
            return False
        for arm, m in arms.items():
            t = theirs.get(arm.lower())   # 报告用 'gated'/'point'（小写）
            if t is None:
                return False
            if m["n"] != t["n"]:
                return False
            for k in KEYS:
                if abs(m[k] - t[k]) > 1e-4:
                    return False
            if abs(m["dangerous_rate"] - t["dangerous_rate"]) > 1e-4:
                return False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        return False
    return True


def s2(rep):
    for k, row in rep["per_s"].items():
        g = row["gated"]
        if g["dangerous_count"] != 0 or g["dangerous_rate"] != 0.0:
            return False
        if not (g["trigger_rate"] - 1e-9 <= g["measure_calls_mean"] <= 3 * g["trigger_rate"] + 1e-9):
            return False

    def pr(s):
        r = _get(rep["per_s"], s)
        return (r or {}).get("point", {}).get("dangerous_rate")
    return bool(pr(1.0) > pr(0.25) > pr(0.02))


def s3(rep):
    v = rep["preregistered_verdict"]
    ok = (v["H85-2_capability_monotone"] is False
          and v["H85-4_near_oracle_zero_measure"] is False
          and v["H85-5_coarse_gets_ceiling"] is False
          and rep["verdict_pass"] is False)
    ok &= any("判据设计失误" in x for x in rep["honest_notes"])
    ok &= any("FAIL" in x for x in rep["honest_notes"])
    return bool(ok)


def main():
    print("=" * 92)
    print("E85 L3 严格审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    mine = replay(data)

    chk("S1 报告↔明细自洽（逐档×逐臂统计量由数据文件独立重放一致；verdict=all(H)）", s1(rep, mine))
    chk("S2 口径不变量（GATED 危险恒 0；trigger ≤ calls ≤ 3×trigger；点入口危险随档递减）", s2(rep))
    chk("S3 诚实 FAIL 保真（H85-2/4/5 必为 False 且披露『判据设计失误』，不得被美化）", s3(rep),
        "H85-2=%s H85-4=%s H85-5=%s verdict=%s"
        % (rep["preregistered_verdict"]["H85-2_capability_monotone"],
           rep["preregistered_verdict"]["H85-4_near_oracle_zero_measure"],
           rep["preregistered_verdict"]["H85-5_coarse_gets_ceiling"], rep["verdict_pass"]))

    t1 = copy.deepcopy(rep)
    t1["per_s"]["0.5"]["gated"]["measure_calls_mean"] = 0.99
    chk("S4a 篡改某档测量次数 → S1 必报", not s1(t1, mine))
    t2 = copy.deepcopy(rep)
    t2["per_s"]["0.5"]["gated"]["dangerous_count"] = 1
    chk("S4b 让 GATED 出现危险 → S2 必报", not s2(t2))
    t3 = copy.deepcopy(rep)
    for k in ("H85-2_capability_monotone", "H85-4_near_oracle_zero_measure",
              "H85-5_coarse_gets_ceiling"):
        t3["preregistered_verdict"][k] = True     # 一次性抹掉全部诚实 FAIL
    chk("S4c 把诚实 FAIL 翻转 → S3 必报（且 verdict≠all(H)）",
        not s3(t3) and t3["verdict_pass"] != all(t3["preregistered_verdict"].values()))
    t4 = copy.deepcopy(rep)
    t4["honest_notes"] = [x for x in t4["honest_notes"] if "判据设计失误" not in x]
    chk("S4d 删掉『判据设计失误』披露 → S3 必报", not s3(t4))
    t5 = copy.deepcopy(rep)
    t5["per_s"]["1.0"]["point"]["dangerous_rate"] = 0.0
    chk("S4e 抹掉点入口在粗接口的漏报 → S2 必报", not s2(t5))
    chk("S4f 未篡改 → S1/S2/S3 通过（阴性对照）", s1(rep, mine) and s2(rep) and s3(rep))
    chk("S5 幂等：两次重放一致", replay(data) == mine)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "独立重放与报告逐档逐臂一致；**三条诚实 FAIL 被结构性锁定**（翻转或删除披露都会被抓）。",
        "★ 口径注记：本层重放复用仓库的 `five_orders`，故它排除的是『报告数字被手工修改 / 臂逻辑记错』，"
        "不排除判据实现本身偏差（由 E85 的 L4 独立物理重写承担）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
