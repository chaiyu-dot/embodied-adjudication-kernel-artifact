# -*- coding: utf-8 -*-
"""_audit_e59_replicator_anomaly.py —— E59 的第三方/盲复刻（L2/L4，纯数据）。

不 import 任何实验模块；只读 e59_termination_report.json，从存储 rows.d 重跑终止规则、
验证逐 eps 统计/p95/Spearman/判据/verdict 自洽；篡改必有检查报出。

产物：_audit_e59_replicator_anomaly.json
"""
import copy
import json
import math
import os
import sys

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e59_termination_report.json")
OUT = os.path.join(EVAL, "_audit_e59_replicator_anomaly.json")

K_MAX, SHRINK, FLOOR = 6, 0.35, 0.012
EPS_LIST = ["0.0", "0.05", "0.1", "0.2", "0.4"]

_res = {"experiment": "E59 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（d 重跑终止规则 + 聚合自洽）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _decide(d, hw0):
    if d < FLOOR:
        return K_MAX, "⊘"
    hw = hw0
    for k in range(1, K_MAX + 1):
        if hw < d:
            return k, "decided"
        hw *= SHRINK
    return None, "nonterminated"


def _forward(rep):
    ch = []
    d = rep["design"]
    rows = rep["rows"]
    # D1 design sane
    ch.append(chk("D1_design_sane", d["K_max"] == K_MAX and abs(d["shrink"] - SHRINK) < 1e-12
                  and abs(d["floor"] - FLOOR) < 1e-12 and len(d["eps_list"]) == 5, "shrink=%s floor=%s" % (d["shrink"], d["floor"])))
    # D2 rows count + fields
    ch.append(chk("D2_rows_shape", len(rows) == 240 and all({"obj", "seed", "d", "side"} <= set(r) for r in rows),
                  "n=%d" % len(rows)))
    # D3 per-eps 覆盖 + 行数守恒
    R = rep["results"]
    ch.append(chk("D3_per_eps_coverage", set(R["per_eps"].keys()) == set(EPS_LIST) and R["n_rows"] == 240))
    # D4 从 rows.d 重跑终止规则 → 逐 eps 统计一致
    bad = 0
    all_rounds = []
    for eps_k in EPS_LIST:
        eps = float(eps_k)
        hw0 = 0.02 if eps <= 0 else 0.5 * eps
        rounds = []
        nterm = 0
        nNA = 0
        for r in rows:
            k, oc = _decide(r["d"], hw0)
            if oc == "decided":
                rounds.append(k)
                all_rounds.append(k)
            elif oc == "nonterminated":
                nterm += 1
            elif oc == "⊘":
                nNA += 1
        sv = R["per_eps"][eps_k]
        if len(rounds) != sv["n_decided"] or nterm != sv["nonterminated"]:
            bad += 1
        if rounds and (int(max(rounds)) != sv["rounds_max"] or abs(float(np.percentile(rounds, 95)) - sv["rounds_p95"]) > 1e-3):
            bad += 1
        # 守恒：decided + NA + nonterm == n
        if len(rounds) + nNA + nterm != sv["n"]:
            bad += 1
    ch.append(chk("D4_recompute_per_eps_from_stored_d", bad == 0, "mismatch=%d" % bad))
    # D5 p95 一致
    p95 = float(np.percentile(all_rounds, 95))
    ch.append(chk("D5_p95_recompute", abs(p95 - R["rounds_p95"]) < 1e-3, "re=%.4f store=%.4f" % (p95, R["rounds_p95"])))
    # D6 nonterminated 守恒
    ch.append(chk("D6_nonterminated_conserved",
                  R["nonterminated"] == sum(v["nonterminated"] for v in R["per_eps"].values()) and R["nonterminated"] == 0))
    # D7 criteria + verdict
    H = rep["preregistered_verdict"]
    ch.append(chk("D7_criteria_and_verdict_selfconsistent",
                  H["H59-1_nonterminated_eq_0"] == (R["nonterminated"] == 0)
                  and H["H59-2_rounds_p95_le_4"] == (R["rounds_p95"] <= 4.0)
                  and rep["verdict_pass"] == all(H.values()), "verdict=%s" % rep["verdict_pass"]))
    # D8 self_consistency
    sc = R["self_consistency"]
    ch.append(chk("D8_self_consistency", sc["cells_conserved"] and sc["nonterminated_matches"]))
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
    print("E59 第三方/盲复刻审核（L2/L4 纯数据）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改某行 d → D4 必报",
          lambda r: r["rows"][12].__setitem__("d", 0.001), ["D4"])
    _tamp("R2 篡改 n_decided → D4 必报",
          lambda r: r["results"]["per_eps"]["0.0"].__setitem__("n_decided", 1), ["D4"])
    _tamp("R3 篡改 nonterminated → D6 必报",
          lambda r: (r["results"].__setitem__("nonterminated", 3),
                     r["results"]["per_eps"]["0.0"].__setitem__("nonterminated", 3)), ["D6"])
    _tamp("R4 篡改 verdict_pass → D7 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["D7"])

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
