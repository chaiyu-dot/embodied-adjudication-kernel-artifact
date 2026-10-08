# -*- coding: utf-8 -*-
"""_audit_e75_replicator_anomaly.py —— **第三方复刻者视角**的异常排查审核（零 API）。

模拟不信任作者代码的外部复刻者：只凭 e75 报告里的数字，**自己独立实现能量层**
（功共轭残差、精确 m_E、E 阶分档、序数一致率、误档集）重算报告每一个关键结论（T1–T6），
再构造篡改证明审核敏感（T7–T9）。**不 import planning.* / e66 / e75 / standalone**。

产物：_audit_e75_replicator_anomaly.json
"""
import copy
import json
import math
import os

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e75_energy_margin_exact_report.json")
BAND_EDGES = (-0.05, 0.25, 0.60)
SEV = {"unsafe": 0, "tight": 1, "medium": 2, "loose": 3}


def _band(m):
    if m < BAND_EDGES[0]:
        return "unsafe"
    if m < BAND_EDGES[1]:
        return "tight"
    if m < BAND_EDGES[2]:
        return "medium"
    return "loose"


def _margin(dE, e_avail):
    e_scale = max(abs(e_avail), abs(dE), 1e-12)
    return (e_avail - dE) / e_scale


def _ordinal(m_arr, d_arr):
    n = len(m_arr); num = den = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(d_arr[i] - d_arr[j]) < 1e-12:
                continue
            den += 1
            if (d_arr[i] - d_arr[j]) * (m_arr[j] - m_arr[i]) > 0:
                num += 1
    return (num / den) if den else 1.0


def _chk(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": str(detail)[:400]}


def _recompute(rep):
    """从报告数字独立重算全部能量层结论。"""
    A = rep["A_work_conjugacy"]; B = rep["B_ordinal"]; C = rep["C_misband"]
    out = {}
    out["resid"] = {k: (v["W_J"] - v["D_J"]) - (v["dKE_J"] + v["dPE_J"]) for k, v in A["rows"].items()}
    out["maxresid"] = max(abs(x) for x in out["resid"].values())
    out["mE"] = [_margin(r["dE_mech_J"], B["e_avail_J"]) for r in B["rows"]]
    out["mK"] = [_margin(r["KE_peak_J"], B["e_avail_J"]) for r in B["rows"]]
    out["bandE"] = [_band(x) for x in out["mE"]]
    out["bandK"] = [_band(x) for x in out["mK"]]
    demand = [r["dPE_J"] for r in B["rows"]]
    out["ocE"] = _ordinal(out["mE"], demand)
    out["ocK"] = _ordinal(out["mK"], demand)
    out["mis"] = [{"mp": r["mp"], "band_exact": be, "band_kinetic_only": bk}
                  for r, be, bk in zip(B["rows"], out["bandE"], out["bandK"])
                  if SEV[be] <= 1 and SEV[bk] >= 2]
    return out


def main():
    rep = json.load(open(REPORT, encoding="utf-8"))
    A, B, C = rep["A_work_conjugacy"], rep["B_ordinal"], rep["C_misband"]
    rec = _recompute(rep)
    checks = []

    # T1 A 段功共轭残差独立重算（由报告 W/D/ΔKE/ΔPE 反算）
    t1 = all(abs(v) < 1e-8 for v in rec["resid"].values())
    checks.append(_chk("T1_work_conjugacy_independent", t1,
                       "max|(W−D)−(ΔKE+ΔPE)|=%.3e" % rec["maxresid"]))

    # T2 B 段 m_E 独立重算
    t2 = all(abs(rec["mE"][i] - B["rows"][i]["m_E_exact"]) < 1e-12
             and abs(rec["mK"][i] - B["rows"][i]["m_E_kinetic_only"]) < 1e-12
             for i in range(len(B["rows"])))
    checks.append(_chk("T2_margin_independent", t2, "m_E_exact/kinetic 逐行"))

    # T3 E 阶分档独立重算
    t3 = all(rec["bandE"][i] == B["rows"][i]["band_exact"]
             and rec["bandK"][i] == B["rows"][i]["band_kinetic_only"] for i in range(len(B["rows"])))
    checks.append(_chk("T3_band_independent", t3, "band 逐行"))

    # T4 序数一致率独立重算
    t4 = (abs(rec["ocE"] - B["ordinal_consistency_exact"]) < 1e-12
          and abs(rec["ocK"] - B["ordinal_consistency_kinetic_only"]) < 1e-12)
    checks.append(_chk("T4_ordinal_independent", t4, "精确 %.3f 动能 %.3f" % (rec["ocE"], rec["ocK"])))

    # T5 误档集独立重算（数与明细）
    t5 = (len(rec["mis"]) == C["n_misbanded"]
          and [m["mp"] for m in rec["mis"]] == [m["mp"] for m in C["misbanded"]])
    checks.append(_chk("T5_misband_independent", t5, "误档 %d 条" % len(rec["mis"])))

    # T6 内部自洽（verdict 与判据一致 + 精确核不劣于动能-only）
    c = rep["criteria"]
    inv = bool(c["C1_work_conjugacy_lt_1e-8"] and A["all_ok"] and c["C2_ordinal_not_degraded"]
               and B["all_ok"] and c["C3_kinetic_only_misbands"] and C["all_ok"])
    t6 = (inv == bool(rep["verdict_pass"])) and (rec["ocE"] >= rec["ocK"] - 1e-12)
    checks.append(_chk("T6_self_consistent_and_not_degraded", t6,
                       "inv=%s verdict=%s ocE=%.3f ocK=%.3f" % (inv, rep["verdict_pass"], rec["ocE"], rec["ocK"])))

    # T7–T9 篡改
    bad = copy.deepcopy(rep)
    bad["A_work_conjugacy"]["rows"]["s0_mp0.00"]["W_J"] += 1.0
    checks.append(_chk("T7_tamper_work_detected", max(abs(v) for v in _recompute(bad)["resid"].values()) > 1e-8,
                       "改 W → 功共轭残差重算必报"))
    bad2 = copy.deepcopy(rep)
    bad2["B_ordinal"]["rows"][0]["dE_mech_J"] += 1.0
    checks.append(_chk("T8_tamper_dEmech_detected",
                       abs(_recompute(bad2)["mE"][0] - bad2["B_ordinal"]["rows"][0]["m_E_exact"]) > 1e-12,
                       "改 dE_mech → m_E 重算必报"))
    bad3 = copy.deepcopy(rep)
    bad3["B_ordinal"]["rows"][2]["band_kinetic_only"] = "medium"
    r3 = _recompute(bad3)
    t9 = not all(r3["bandK"][i] == bad3["B_ordinal"]["rows"][i]["band_kinetic_only"]
                 for i in range(len(bad3["B_ordinal"]["rows"])))
    checks.append(_chk("T9_tamper_band_detected", t9,
                       "改存储 band_kinetic_only → 独立重算 band 与之不符（T3 类）必报"))

    npass = sum(x["pass"] for x in checks)
    out = {"audit": "_audit_e75_replicator_anomaly", "experiment": "e75_energy_margin_exact",
           "pass": npass, "total": len(checks), "verdict": bool(npass == len(checks)),
           "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(os.path.join(EVAL, "_audit_e75_replicator_anomaly.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly audit pass = %d/%d  verdict=%s" % (npass, len(checks), out["verdict"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
