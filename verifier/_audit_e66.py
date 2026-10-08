# -*- coding: utf-8 -*-
"""_audit_e66.py —— E66 的**正向审核（L1，同码重跑）**。

独立性：低（同码重跑）。仅证明"报告与源码一致、非陈旧、字段齐全、内部自洽"。
不证明物理正确性（那要 L3/L4）。

产物：_audit_e66.json
"""
import json
import os
import subprocess
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
sys.path.insert(0, SRC)
EXP = os.path.join(EVAL, "e66_closed_loop_control.py")
REPORT = os.path.join(EVAL, "e66_closed_loop_report.json")
OUT = os.path.join(EVAL, "_audit_e66.json")

_res = {"experiment": "E66 正向审核(L1 同码重跑)", "independence_scope": "同码重跑（最低独立性）", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 88)
    print("E66 正向审核（L1 同码重跑）：重跑实验，证明报告与源码一致、非陈旧")
    print("=" * 88)
    # 重跑实验（同码），生成最新报告
    rc = subprocess.run([sys.executable, EXP], cwd=EVAL, capture_output=True, text=True, timeout=600)
    ok = chk("L1_rerun_exit0", rc.returncode == 0, rc.stderr[-400:] if rc.returncode else "rc=0")
    if not ok:
        json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return 1
    rep = json.load(open(REPORT, encoding="utf-8"))

    # 三个核心 verdict 必须为真（与协议基线/记忆一致）
    chk("L1_verdict_pass", rep.get("verdict_pass") is True)
    chk("L1_V1_MCg_ok", rep.get("V1_MCg_decomposition_ok") is True)
    chk("L1_V2_gate_ok", rep.get("V2_gate_zero_violation") is True)
    chk("L1_V3_energy_ok", rep.get("V3_energy_balance_ok") is True)

    # 关键数字与协议基线一致（非陈旧：achievable_wn=7.758, static_payload=0.3972）
    chk("L1_achievable_wn_7p758", abs(rep["budget_design"]["achievable_wn_rad_s"] - 7.758) < 1e-3,
        rep["budget_design"]["achievable_wn_rad_s"])
    chk("L1_static_payload_0p397", abs(rep["static_payload_limit_kg"] - 0.3972) < 1e-3,
        rep["static_payload_limit_kg"])
    chk("L1_energy_residual_lt_1e8", abs(rep["energy_balance"]["residual_J"]) < 1e-8,
        rep["energy_balance"]["residual_J"])

    # 门控：执行值 τmax 必须 == τ_lim（恒被 clip），demand 越限非零（真有需求越限，证明"零执行越限"是 clip 构造的，不是控制自然无越限）
    ga = rep["gate_ablation"]
    chk("L1_gate_PD_exec_tau_max_eq_lim", abs(ga["PD_with_gate"]["tau_max"] - 5.0) < 1e-9)
    chk("L1_gate_PD_demand_violation_nonzero", ga["PD_with_gate"]["demand_violation_steps"] > 0,
        ga["PD_with_gate"]["demand_violation_steps"])
    chk("L1_gate_CT_demand_violation_nonzero", ga["CT_with_gate"]["demand_violation_steps"] > 0,
        ga["CT_with_gate"]["demand_violation_steps"])

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["l1_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nL1: %d/%d pass = %s" % (npass, len(_res["checks"]), _res["l1_pass"]))
    print("wrote", OUT)
    return 0 if _res["l1_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
