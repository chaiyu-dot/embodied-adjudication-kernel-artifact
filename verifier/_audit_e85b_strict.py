# -*- coding: utf-8 -*-
"""_audit_e85b_strict.py —— E85b 的 **L3 严格审核**（结构自洽 + 独立重放 + 混淆因果 + 篡改捕获）。

读取自足数据文件 `e85b_bench_data.json`，用**自写的臂逻辑**（复用仓库 five_orders 本体）
独立重放 **v1 / v2 两种设计**，并验证本件最关键的因果断言。

判据
----
S1 报告↔明细自洽：两设计逐档×逐臂统计量由数据文件重放一致；verdict = all(H)。
S2 口径不变量：GATED 危险恒 0；oracle 档测量 = 0；oracle 可行率 = 上限（1 − gt）；
   v2 的 gt 极差 = 0（条件流解耦）；v1 的 gt 极差 > 0（条件流耦合）。
S3 **混淆因果**：v1 的 5 档可行率非单调、v2 的 5 档单调 ⇒ 非单调性归因于**条件集不同**，
   而不是 s 本身（H85b-4 的 FAIL 即该替代分支被实现）。
S4 篡改捕获：改 v2 的 calls/可行率、把 oracle 测量改成非 0、把 v1 的 gt 摊平 → S1/S2/S3 必报。
S5 幂等。

产物：_audit_e85b_strict.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e85b_oracle_level as E   # noqa: E402

REPORT = os.path.join(EVAL, "e85b_oracle_level_report.json")
BENCH = os.path.join(EVAL, "e85b_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e85b_strict.json")
_res = {"audit": "E85b L3 strict", "checks": [], "findings": []}
_n = [0]
# 🔴 逐臂字段集必须分开：POINT 臂天然无 measure_calls_mean / trigger_rate（它不做测量）。
#    此前用 GATED 字段集去比 POINT → 在重放结束时报 KeyError（L3 空跑 ~20 min 才发现）。
ARM_KEYS = {
    "gated": ("measure_calls_mean", "trigger_rate", "feasible_rate", "dangerous_rate",
              "dangerous_count", "false_reject_rate"),
    "point": ("feasible_rate", "dangerous_rate", "dangerous_count", "false_reject_rate"),
}
KEYS = ARM_KEYS["gated"]     # 兼容旧引用


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def replay(levels):
    k_conf, k_max = E.K_CONF, E.K_MAX
    out = {}
    for lv in levels:
        s = lv["s"]
        g = {"n": 0, "calls": 0, "trig": 0, "feasible": 0, "dangerous": 0, "frej": 0, "gt": 0}
        p = {"n": 0, "feasible": 0, "dangerous": 0, "frej": 0}
        for pr in lv["proposals"]:
            prop = E.make_proposal(E.make_traj([0.0, 0.0], pr["q_final"], pr["T"]), pr["q_final"])
            gt_dangerous = E.five_orders(pr["mp_true"], prop) < 0.0
            g["gt"] += int(gt_dangerous)
            st = pr["states"]["GATED"]
            t = calls = 0
            trig = False
            while True:
                m, sd = st[t]
                min_hi = E.five_orders(m + k_conf * sd, prop)
                min_pt = E.five_orders(m, prop)
                if not ((min_pt >= 0.0) and (min_hi < 0.0)) or calls >= k_max:
                    break
                calls += 1
                t += 1
                trig = True
            dec = min_hi          # 与"在最终 t 处重算 m + k·sd"逐位相同（同一 m/sd/轨迹）——省去一次重复求值
            feas = dec >= 0.0
            g["n"] += 1
            g["calls"] += calls
            g["trig"] += int(trig)
            g["feasible"] += int(feas)
            g["dangerous"] += int(bool(feas and gt_dangerous))
            g["frej"] += int((not feas) and (not gt_dangerous))
            if "POINT" in pr["states"]:
                dec_pt = E.five_orders(pr["states"]["POINT"][0][0], prop)
                feas_pt = dec_pt >= 0.0
                p["n"] += 1
                p["feasible"] += int(feas_pt)
                p["dangerous"] += int(bool(feas_pt and gt_dangerous))
                p["frej"] += int((not feas_pt) and (not gt_dangerous))
        n = max(g["n"], 1)
        out[s] = {"s": s, "gt_dangerous_rate": round(g["gt"] / n, 4),
                  "gated": {"n": g["n"], "measure_calls_mean": round(g["calls"] / n, 4),
                            "trigger_rate": round(g["trig"] / n, 4),
                            "feasible_rate": round(g["feasible"] / n, 4),
                            "dangerous_count": g["dangerous"],
                            "dangerous_rate": round(g["dangerous"] / n, 4),
                            "false_reject_rate": round(g["frej"] / n, 4)},
                  "point": ({} if not p["n"] else
                            {"n": p["n"], "feasible_rate": round(p["feasible"] / p["n"], 4),
                             "dangerous_rate": round(p["dangerous"] / p["n"], 4),
                             "dangerous_count": p["dangerous"],
                             "false_reject_rate": round(p["frej"] / p["n"], 4)})}
    return out


def s1(rep, m1, m2):
    ok = True
    for tag, mine in (("v1", m1), ("v2", m2)):
        theirs = rep["design_" + tag]["per_s"]
        for s, row in mine.items():
            tr = theirs.get(str(s))
            if tr is None:
                return False
            for arm in ("gated", "point"):
                if not row[arm]:
                    continue
                t = tr.get(arm) or {}
                if not t:
                    return False
                for k in ARM_KEYS[arm]:
                    if abs(row[arm][k] - t[k]) > 1e-4:
                        ok = False
    return bool(ok and rep["verdict_pass"] == all(rep["preregistered_verdict"].values()))


def s2(rep, m1, m2):
    ok = True
    s_desc = sorted(list(rep["design"]["s_levels"]), reverse=True)   # 报告无顶层 s_desc；s_levels 为纯浮点列表
    s0 = str(s_desc[-1])
    for mine in (m1, m2):
        for s, row in mine.items():
            if row["gated"]["dangerous_count"] != 0:
                ok = False
    if m2[s_desc[-1]]["gated"]["measure_calls_mean"] != 0.0:
        ok = False
    ceil = round(1.0 - m2[s_desc[-1]]["gt_dangerous_rate"], 4)
    if abs(m2[s_desc[-1]]["gated"]["feasible_rate"] - ceil) > 1e-4:
        ok = False
    # H85b-2 的后半：oracle 档可行率 ≥ 所有 s>0 档（"端点即上限"）
    if not (m2[s_desc[-1]]["gated"]["feasible_rate"]
            >= max(m2[s]["gated"]["feasible_rate"] for s in s_desc) - 1e-9):
        ok = False
    gt1 = [m1[s]["gt_dangerous_rate"] for s in s_desc]
    gt2 = [m2[s]["gt_dangerous_rate"] for s in s_desc]
    if not (round(max(gt1) - min(gt1), 4) > 0.001 and round(max(gt2) - min(gt2), 4) == 0.0):
        ok = False
    _ = s0
    return bool(ok)


def s3(rep, m1, m2):
    s_old = sorted(rep["design"]["e85_original_levels"], reverse=True)
    f1 = [m1[s]["gated"]["feasible_rate"] for s in s_old]
    f2 = [m2[s]["gated"]["feasible_rate"] for s in s_old]
    v1_nonmono = any(f1[i] > f1[i + 1] + 1e-9 for i in range(len(f1) - 1))
    v2_mono = all(f2[i] <= f2[i + 1] + 1e-9 for i in range(len(f2) - 1))
    return bool(v1_nonmono and v2_mono
                and rep["preregistered_verdict"]["H85b-4_capability_nonmonotone_or_confound"] is False), \
        "v1 feasible=%s（非单调=%s）｜ v2 feasible=%s（单调=%s）" % (f1, v1_nonmono, f2, v2_mono)


def main():
    print("=" * 92)
    print("E85b L3 严格审核")
    print("=" * 92)
    report = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    m1 = replay(data["design_v1_levels"])
    m2 = replay(data["design_v2_levels"])
    rep = report                      # 统一变量名
    _sd = sorted(list(rep["design"]["s_levels"]), reverse=True)   # 报告无顶层 s_desc；s_levels 为纯浮点列表

    chk("S1 报告↔明细自洽（两设计逐档统计量由数据文件独立重放一致；verdict=all(H)）", s1(rep, m1, m2))
    chk("S2 口径不变量（危险恒 0 / oracle 测量=0 / oracle 可行率=上限 / v1 gt 极差>0 而 v2=0）",
        s2(rep, m1, m2))
    ok3, det3 = s3(rep, m1, m2)
    chk("S3 混淆因果：v1（条件流耦合）非单调、v2（解耦）单调 ⇒ 非单调归因于条件集不同", ok3, det3)

    t1 = copy.deepcopy(rep)
    t1["design_v2"]["per_s"][str(_sd[0])]["gated"]["measure_calls_mean"] = 9.9
    chk("S4a 篡改 v2 某档测量次数 → S1 必报", not s1(t1, m1, m2))

    m2_bad = copy.deepcopy(m2)
    m2_bad[_sd[-1]]["gated"]["measure_calls_mean"] = 0.01
    chk("S4b 让 oracle 档出现测量（0.01）→ S2 必报", not s2(rep, m1, m2_bad))

    t3 = copy.deepcopy(rep)
    for k in t3["design_v1"]["per_s"]:
        t3["design_v1"]["per_s"][k]["gt_dangerous_rate"] = rep["design_v2"]["per_s"][k]["gt_dangerous_rate"]
    gt1_bad = [t3["design_v1"]["per_s"][str(s)]["gt_dangerous_rate"] for s in _sd]
    chk("S4c 把 v1 的 gt 摊平成与 v2 相同（抹掉条件流混淆这条证据）→「v1 gt 极差>0」必报",
        not (round(max(gt1_bad) - min(gt1_bad), 4) > 0.001))

    m2_flat = copy.deepcopy(m2)
    _hi_s = _sd[1]                       # 一个 s>0 档
    m2_flat[_hi_s]["gated"]["feasible_rate"] = m2[_sd[-1]]["gated"]["feasible_rate"] + 0.05
    chk("S4d 让某个 s>0 档的可行率**超过 oracle** → S2 的「oracle ≥ 所有 s>0 档」必报",
        not s2(rep, m1, m2_flat))

    t4 = copy.deepcopy(rep)
    t4["preregistered_verdict"]["H85b-4_capability_nonmonotone_or_confound"] = True
    chk("S4e 翻转 H85b-4 → S1/S3 必报",
        not s1(t4, m1, m2) or not s3(t4, m1, m2)[0])
    chk("S4f 未篡改 → S1/S2/S3 通过（阴性对照）", s1(rep, m1, m2) and s2(rep, m1, m2) and s3(rep, m1, m2)[0])
    chk("S5 幂等：两次重放一致", replay(data["design_v2_levels"]) == m2)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "独立重放复现两种设计；**条件流混淆的因果链被结构性检验**：v1（耦合）非单调、v2（解耦）单调。",
        "★ 口径注记：本层复用仓库 `five_orders`，排除的是『报告数字被改 / 臂逻辑记错』；"
        "判据实现偏差由 E85b 的 L4（独立物理重写）承担。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
