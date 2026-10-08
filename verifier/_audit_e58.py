# -*- coding: utf-8 -*-
"""_audit_e58.py —— E58 的**正向审核（L1，同码重跑）**。
独立性：低（同码重跑）。证明报告与源码一致、非陈旧、字段齐全。
产物：_audit_e58.json
"""
import json
import os
import subprocess
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
sys.path.insert(0, SRC)
EXP = os.path.join(EVAL, "e58_trackability_gate.py")
REPORT = os.path.join(EVAL, "e58_trackability_report.json")
OUT = os.path.join(EVAL, "_audit_e58.json")

_res = {"experiment": "E58 正向审核(L1 同码重跑)", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 88)
    print("E58 正向审核（L1 同码重跑）")
    print("=" * 88)
    rc = subprocess.run([sys.executable, EXP], cwd=EVAL, capture_output=True, text=True, timeout=600)
    chk("L1_rerun_exit0", rc.returncode == 0, rc.stderr[-400:] if rc.returncode else "rc=0")
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("L1_verdict_pass", rep.get("verdict_pass") is True)
    H = rep["preregistered_verdict"]
    chk("L1_H58_1_blind_spot_exists", H["H58-1_blind_spot_exists"] is True)
    chk("L1_H58_2_all_T_le_0p5", H["H58-2_blind_spot_all_T_le_0.5s"] is True)
    chk("L1_H58_3_slow_false_alarm_lt_0p10", H["H58-3_slow_false_alarm_lt_0.10"] is True)
    chk("L1_blind_spot_nonzero", rep["results"]["blind_spot_vs_D"] > 0,
        rep["results"]["blind_spot_vs_D"])
    # 慢档误报率应小（盲区集中在快轨迹）
    sfa = rep["results"]["slow_T_ge_0.8s_false_alarm_rate"]
    chk("L1_slow_false_alarm_rate_small", sfa is not None and sfa < 0.10, sfa)
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
