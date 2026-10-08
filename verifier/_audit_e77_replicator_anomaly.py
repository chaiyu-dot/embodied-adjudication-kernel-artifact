# -*- coding: utf-8 -*-
"""_audit_e77_replicator_anomaly.py —— **第三方复刻者视角**异常排查（零 API）。

只凭 e77 报告数字，独立重算派生结论（ss_bounded / 门控承重 / verdict 自洽 / 接口完整性）；
再构造篡改。**不 import planning.* / e66 / e68 / e77 / standalone**。
产物：_audit_e77_replicator_anomaly.json
"""
import copy
import json
import math
import os

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e77_controller_family_extension_report.json")


def _chk(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": str(detail)[:400]}


def _derived(rep):
    cfg = rep["config"]; tol = cfg["ss_tol_rad"]; new = cfg["new_families"]
    A, C, D = rep["A_metrics"], rep["C_capability"], rep["D_determinism_and_ablation"]
    out = {}
    out["ss_bounded"] = {n: (C["rows"][n]["ss_err_nominal"] is not None
                             and C["rows"][n]["ss_err_nominal"] <= tol) for n in new}
    out["danger_zero"] = all(A["rows"][n][sc]["executed_violation_steps"] == 0
                             and not A["rows"][n][sc]["diverged"] for n in new for sc in cfg["scenarios"])
    abl = {k: v["executed_violation_steps_no_gate"] for k, v in D["gate_off_ablation"].items()}
    out["gate_load_bearing"] = any(v > 0 for v in abl.values())
    out["det"] = bool(D["deterministic"])
    c = rep["criteria"]
    out["inv"] = bool(c["C1_interface_consistent"] and rep["B_interface"]["all_ok"]
                      and c["C2_danger_zero_executed"] and out["danger_zero"]
                      and c["C3_capability_reportable_and_ss_bounded"] and C["all_ok"]
                      and c["C4_deterministic"] and out["det"])
    return out


def main():
    rep = json.load(open(REPORT, encoding="utf-8"))
    d = _derived(rep)
    new = rep["config"]["new_families"]
    checks = []

    checks.append(_chk("T1_ss_bounded_independent",
                       all(d["ss_bounded"][n] == rep["C_capability"]["rows"][n]["ss_bounded_nominal"] for n in new),
                       "独立判 ss_bounded == 报告"))
    checks.append(_chk("T2_danger_zero_independent", d["danger_zero"],
                       "全部新族 x 场景 executed 越限=0 且无发散"))
    checks.append(_chk("T3_gate_load_bearing_independent", d["gate_load_bearing"],
                       "至少一族 gate-off 执行越限>0 ⇒ 门控承重"))
    checks.append(_chk("T4_verdict_self_consistent", d["inv"] == bool(rep["verdict_pass"]),
                       "重算 inv=%s 报告 verdict=%s" % (d["inv"], rep["verdict_pass"])))
    checks.append(_chk("T5_interface_complete",
                       all(set(rep["B_interface"]["rows"][n]).issuperset(
                           {"capability_has_required", "saturation_policy_is_dict", "output_gated_ok"}) for n in new),
                       "3 族接口字段完整"))
    checks.append(_chk("T6_reference_families_present",
                       all(k in rep["A_ref_metrics"] for k in rep["config"]["ref_families"]),
                       "参照族 %s 存在" % rep["config"]["ref_families"]))

    # 篡改
    bad1 = copy.deepcopy(rep); bad1["C_capability"]["rows"]["FLC"]["ss_err_nominal"] = 0.5
    checks.append(_chk("T7_tamper_ss_detected",
                       _derived(bad1)["ss_bounded"]["FLC"] != bad1["C_capability"]["rows"]["FLC"]["ss_bounded_nominal"],
                       "改 ss_nominal → 派生 ss_bounded 不符必报"))
    bad2 = copy.deepcopy(rep)
    for k in bad2["D_determinism_and_ablation"]["gate_off_ablation"]:
        bad2["D_determinism_and_ablation"]["gate_off_ablation"][k]["executed_violation_steps_no_gate"] = 0
    checks.append(_chk("T8_tamper_ablation_detected", not _derived(bad2)["gate_load_bearing"],
                       "抹消融越限 → 门控承重前提消失必报"))
    bad3 = copy.deepcopy(rep); bad3["verdict_pass"] = False
    checks.append(_chk("T9_tamper_verdict_detected", _derived(bad3)["inv"] != bool(bad3["verdict_pass"]),
                       "翻转 verdict → 自洽检查必报"))

    npass = sum(x["pass"] for x in checks)
    out = {"audit": "_audit_e77_replicator_anomaly", "experiment": "e77_controller_family_extension",
           "pass": npass, "total": len(checks), "verdict": bool(npass == len(checks)),
           "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(os.path.join(EVAL, "_audit_e77_replicator_anomaly.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly audit pass = %d/%d  verdict=%s" % (npass, len(checks), out["verdict"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
