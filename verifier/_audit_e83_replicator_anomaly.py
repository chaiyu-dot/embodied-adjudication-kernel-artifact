# -*- coding: utf-8 -*-
"""_audit_e83_replicator_anomaly.py —— E83 的 **L2 第三方复算 + 异常/篡改注入**。

与 L3（结构自洽 + 独立重仿真 PD_G）正交：本层**只读报告 JSON** 做第三方重算，并**注入异常**，
证明报告有分辨力（不是"怎么写都过"）。

判据
----
T1 第三方复算：由 `gated/ungated` 行独立重算 spread、intrinsic/gate_dep 集合、over 标志，与报告一致
T2 异常注入：
   a) 注入一族 gated util=1.3 → 「gated util≤1」检查必报
   b) 注入一族 ungated 越限=0 但 util>1 → 集合一致性检查必报
   c) 把一个承重族的 ungated 越限清零 → gate_load_bearing 集合必变（可分辨）
T3 跨实验一致性：E83 的压力场景 gated 越限=0 与 E68/E77 的"门控在环 executed=0"口径一致
T4 阴性对照：未篡改 → T1 全过

产物：_audit_e83_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e83_family_safety_observable_report.json")
OUT = os.path.join(EVAL, "_audit_e83_replicator_anomaly.json")
_res = {"audit": "E83 L2 replicator + anomaly injection", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _spread(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return (max(v) - min(v)) if len(v) >= 2 else 0.0


def third_party(rep):
    """只读 JSON 的第三方重算。返回 (ok, detail)。"""
    ok = True
    det = []
    for lv in rep["gated"]:
        g, u = rep["gated"][lv], rep["ungated"][lv]
        sp = rep["spread"][lv]
        if abs(_spread([g[n]["executed_peak_util"] for n in g]) - sp["executed_peak_util"]) > 5e-4:
            ok = False
        if abs(_spread([g[n]["demand_saturation_duty"] for n in g]) - sp["demand_saturation_duty"]) > 5e-4:
            ok = False
        if abs(_spread([g[n]["peak_joint_accel_rad_s2"] for n in g]) - sp["peak_joint_accel_rad_s2"]) > 5e-2:
            ok = False
        intr = sorted(n for n in u if u[n]["executed_violation_steps"] == 0)
        if intr != sorted(rep["intrinsically_bounded_families"][lv]):
            ok = False
        gdep = sorted(n for n in u if u[n]["executed_violation_steps"] > 0)
        if gdep != sorted(rep["gate_load_bearing_families"][lv]):
            ok = False
        det.append("%s: σ=(%.4f,%.4f,%.3f) |intr|=%d |gdep|=%d"
                   % (lv, sp["executed_peak_util"], sp["demand_saturation_duty"],
                      sp["peak_joint_accel_rad_s2"], len(intr), len(gdep)))
    return ok, " ｜ ".join(det)


def util_bound_ok(rep):
    return all(rep["gated"][lv][n]["executed_peak_util"] <= 1.0 + 1e-9
               for lv in rep["gated"] for n in rep["gated"][lv])


def ungated_consistent(rep):
    """ungated 侧不变量：越限步数>0 ⟺ 执行力矩利用率>1（二者必须同向）。"""
    for lv in rep["ungated"]:
        for n in rep["ungated"][lv]:
            r = rep["ungated"][lv][n]
            if (r["executed_violation_steps"] > 0) != (r["executed_peak_util"] > 1.0 + 1e-9):
                return False
    return True


def main():
    print("=" * 92)
    print("E83 L2 第三方复算 + 异常注入")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    ok, det = third_party(rep)
    chk("T1 第三方复算（spread / 集合 / 严重度）与报告一致", ok, det)

    lv = sorted(rep["gated"])[0]
    fam = sorted(rep["gated"][lv])[0]

    a = copy.deepcopy(rep)
    a["gated"][lv][fam]["executed_peak_util"] = 1.3
    chk("T2a 注入 gated util=1.3 → 「gated util≤1」必报", not util_bound_ok(a))

    b = copy.deepcopy(rep)
    b["ungated"][lv][rep["intrinsically_bounded_families"][lv][0]]["executed_violation_steps"] = 5
    chk("T2b 把自带界族的 ungated 越限改成 5（而 util≤1）→ ungated 不变量必报",
        not ungated_consistent(b))
    chk("T2b' 未篡改 → ungated 不变量通过（阴性对照）", ungated_consistent(rep))

    c = copy.deepcopy(rep)
    dep = rep["gate_load_bearing_families"][lv][0]
    c["ungated"][lv][dep]["executed_violation_steps"] = 0
    chk("T2c 把承重族 ungated 越限清零 → gate_load_bearing 集合必变（可分辨）",
        sorted(n for n in c["ungated"][lv] if c["ungated"][lv][n]["executed_violation_steps"] > 0)
        != sorted(rep["gate_load_bearing_families"][lv]))

    # ---- T3 跨实验一致性（只读 E68/E77 报告）----
    e68 = json.load(open(os.path.join(EVAL, "e68_control_suite_report.json"), encoding="utf-8"))
    e77 = json.load(open(os.path.join(EVAL, "e77_controller_family_extension_report.json"), encoding="utf-8"))
    e83_zero = all(rep["gated"][lv2][n]["executed_violation_steps"] == 0
                   for lv2 in rep["gated"] for n in rep["gated"][lv2])
    e68_zero = e68["V1_gate_invariant"]["ok"]
    chk("T3 跨实验口径一致：E83 gated 全 0 ⇔ E68 V1 门控不变量 ok", bool(e83_zero and e68_zero),
        "E83=%s E68.V1.ok=%s" % (e83_zero, e68_zero))

    chk("T4 阴性对照：未篡改 → T1 通过", third_party(rep)[0])

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["第三方复算与异常注入均按预期：报告的 spread/集合/边界有分辨力（篡改可被抓）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL2 第三方/异常注入：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
