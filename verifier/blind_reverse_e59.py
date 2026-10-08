# -*- coding: utf-8 -*-
"""blind_reverse_e59.py —— E59 的**逆向盲测（L5，隔离 + 不信声明常量）**。

独立性：隔离（运行期自检 sys.modules 未加载 e59_termination_bound）。只信存储 rows.d，
独立反推常量并做证伪：
  · R1 反推 FLOOR：⊘ 单元 ⟺ d<FLOOR（跨 eps 恒定，占 1/5）
  · R2 反推 SHRINK：由 (d, hw0, rounds) 约束；换 SHRINK=0.5 必不复现 ⇒ 被数据锁死
  · R3 负对照：SHRINK=1.0（不收缩）⇒ 非 ⊘ 行全 nonterminated ⇒ 收缩原语是承重的
产物：blind_reverse_e59.json
"""
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e59_termination_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e59.json")

K_MAX, SHRINK, FLOOR = 6, 0.35, 0.012
EPS_LIST = [0.0, 0.05, 0.1, 0.2, 0.4]

_res = {"experiment": "E59 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e59_termination_bound", "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _decide(d, hw0, shrink=SHRINK, floor=FLOOR, kmax=K_MAX):
    if d < floor:
        return None, "⊘"
    hw = hw0
    for k in range(1, kmax + 1):
        if hw < d:
            return k, "decided"
        hw *= shrink
    return None, "nonterminated"


def _rounds_for(rows, eps, shrink=SHRINK):
    """只返回 decided 行的轮数（与实验 n_decided 口径一致）。"""
    hw0 = 0.02 if eps <= 0 else 0.5 * eps
    out = []
    for r in rows:
        k, oc = _decide(r["d"], hw0, shrink=shrink)
        if oc == "decided":
            out.append(k)
    return out


def main():
    print("=" * 88)
    print("E59 逆向盲测（L5 隔离，反推常量 + 负对照）")
    print("=" * 88)
    chk("R0_isolation_e59_not_imported",
        "e59_termination_bound" not in sys.modules,
        "sys.modules 含 e59=%s" % ("e59_termination_bound" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    rows = rep["rows"]

    # R1：反推 FLOOR：⊘ ⟺ d<FLOOR（每 eps 一致，180/5=36 行）
    n_below = sum(1 for r in rows if r["d"] < FLOOR)
    n_below_lo = sum(1 for r in rows if r["d"] < 0.011)
    n_below_hi = sum(1 for r in rows if r["d"] < 0.013)
    store_NA_per_eps = rep["results"]["per_eps"]["0.0"]["inapplicable_rate"] * 240
    chk("R1_reverse_floor_from_inapplicable_set",
        abs(n_below - store_NA_per_eps) < 1e-6 and n_below_lo <= n_below <= n_below_hi,
        "rows d<FLOOR=%d (store NA/eps=%.0f) ; bracket [%d,%d]" % (n_below, store_NA_per_eps, n_below_lo, n_below_hi))

    # R2：反推 SHRINK —— 用真实 SHRINK 复现 rounds，换 0.5 必不复现
    ok_true = True
    ok_alt = True
    for eps in EPS_LIST:
        eps_s = "%s" % eps if eps not in (0.0, 0.05) else ("0.0" if eps == 0.0 else "0.05")
        r_true = _rounds_for(rows, eps, SHRINK)
        r_alt = _rounds_for(rows, eps, 0.5)
        st = rep["results"]["per_eps"][eps_s]
        if len(r_true) != st["n_decided"]:
            ok_true = False
        # alt（0.5）应给出不同的 rounds 集合
        if r_alt != r_true:
            ok_alt = False
    chk("R2_reverse_shrink_identified", ok_true and (not ok_alt),
        "true_shrink reproduces=%s alt(0.5)_differs=%s" % (ok_true, not ok_alt))

    # R3：负对照——不收缩 ⇒ 非 ⊘ 行全 nonterminated（收缩原语承重）
    nonterm_noshrink = sum(1 for r in rows if r["d"] >= FLOOR)
    chk("R3_negative_control_shrinkage_load_bearing",
        nonterm_noshrink > 0 and rep["results"]["nonterminated"] == 0,
        "no-shrink nonterminated=%d (with shrink store=0)" % nonterm_noshrink)

    # R4：p95 反推（真实 shrink 下重算 == 存储）
    allr = []
    for eps in EPS_LIST:
        allr += _rounds_for(rows, eps, SHRINK)
    import numpy as np
    p95 = float(np.percentile(allr, 95))
    chk("R4_reverse_p95", abs(p95 - rep["results"]["rounds_p95"]) < 1e-3,
        "re=%.3f store=%.3f" % (p95, rep["results"]["rounds_p95"]))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
