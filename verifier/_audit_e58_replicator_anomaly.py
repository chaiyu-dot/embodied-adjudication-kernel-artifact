# -*- coding: utf-8 -*-
"""_audit_e58_replicator_anomaly.py —— E58 的**第三方/盲复刻（L2/L4，纯数据）**。
不 import 任何实验模块；只读 e58_trackability_report.json，验证内部算术自洽、篡改必被捕获。
产物：_audit_e58_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e58_trackability_report.json")
OUT = os.path.join(EVAL, "_audit_e58_replicator_anomaly.json")

_res = {"experiment": "E58 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    res = rep["results"]
    rows = rep["rows"]
    # 计数自洽
    ch.append(chk("D1_n_conditions_consistent",
                  res["n_conditions"] == len(rows)
                  and res["n_conditions"] == sum(v["n"] for v in res["C_infeasible_by_T"].values())))
    # 盲区求和自洽
    ch.append(chk("D2_blind_spot_sum_consistent",
                  res["blind_spot_vs_D"]
                  == sum(1 for r in rows if r["D_ok"] and not r["C_ok"])
                  == sum(res["blind_spot_by_T"].values())))
    # H58-2：盲区全在 T≤0.5
    blind = [r for r in rows if r["D_ok"] and not r["C_ok"]]
    ch.append(chk("D3_H58_2_blind_all_T_le_0p5",
                  rep["preregistered_verdict"]["H58-2_blind_spot_all_T_le_0.5s"] is True
                  and len(blind) > 0 and all(r["T_s"] <= 0.5 for r in blind),
                  "max_T=%.2f n=%d" % (max(r["T_s"] for r in blind), len(blind))))
    # H58-3：慢档误报率 < 0.10
    sfa = res["slow_T_ge_0.8s_false_alarm_rate"]
    ch.append(chk("D4_H58_3_slow_false_alarm_lt_0p10",
                  rep["preregistered_verdict"]["H58-3_slow_false_alarm_lt_0.10"] is True
                  and sfa is not None and sfa < 0.10, "rate=%.4f" % sfa))
    # H58-1：盲区存在
    ch.append(chk("D5_H58_1_blind_exists",
                  rep["preregistered_verdict"]["H58-1_blind_spot_exists"] is True and res["blind_spot_vs_D"] > 0,
                  "n=%d" % res["blind_spot_vs_D"]))
    # by_T 内部：D_ok&¬C_ok 计数 == 盲区中该 T 的数量
    for T, v in res["C_infeasible_by_T"].items():
        n_blind_T = sum(1 for r in blind if abs(r["T_s"] - float(T)) < 1e-9)
        ch.append(chk("D6_by_T_%s_Dok_nC_consistent" % T, v["D_ok_C_infeasible"] == n_blind_T,
                      "by_T=%d recompute=%d" % (v["D_ok_C_infeasible"], n_blind_T)))
    # verdict 自洽
    H = rep["preregistered_verdict"]
    ch.append(chk("D7_verdict_self_consistent",
                  rep["verdict_pass"] == (H["H58-1_blind_spot_exists"]
                                          and H["H58-2_blind_spot_all_T_le_0.5s"]
                                          and H["H58-3_slow_false_alarm_lt_0.10"])))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name, "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E58 第三方/盲复刻审核（L2/L4 纯数据）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 H58-2=false → D3 必报",
          lambda r: r["preregistered_verdict"].__setitem__("H58-2_blind_spot_all_T_le_0.5s", False), ["D3"])
    _tamp("R2 篡改 blind_spot_vs_D=0 → D2/D5 必报",
          lambda r: r["results"].__setitem__("blind_spot_vs_D", 0), ["D2", "D5"])
    _tamp("R3 篡改 slow_false_alarm=0.99 → D4 必报",
          lambda r: r["results"].__setitem__("slow_T_ge_0.8s_false_alarm_rate", 0.99), ["D4"])
    _tamp("R4 篡改 verdict=false → D7 必报", lambda r: r.__setitem__("verdict_pass", False), ["D7"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["replicator_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n第三方审核：算术 %d/%d ｜ 篡改 %d/%d ｜ pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
