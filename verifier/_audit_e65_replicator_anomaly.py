# -*- coding: utf-8 -*-
"""_audit_e65_replicator_anomaly.py —— E65 的第三方/盲复刻（L2/L4，纯数据 + 篡改必报）。

不 import 任何实验模块；只读 e65_inapplicable_guard_report.json，从存储 rows 重算逐臂统计
（漏报率/Wilson95/⊘选择性）、Newcombe 配对差 CI、判据 H65-1..6 与 verdict；篡改必有检查报出。

产物：_audit_e65_replicator_anomaly.json
"""
import copy
import hashlib
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e65_inapplicable_guard_report.json")
PREREG = os.path.join(EVAL, "e65_inapplicable_guard_prereg.json")
OUT = os.path.join(EVAL, "_audit_e65_replicator_anomaly.json")

ARMS = ["G3_RETRY", "G3_ABSTAIN", "G3_SHRINK", "G4_OURS"]
_res = {"experiment": "E65 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（逐臂统计重算 + 判据）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _newcombe(k1, n1, k2, n2, z=1.959964):
    l1, u1 = _wilson(k1, n1, z); l2, u2 = _wilson(k2, n2, z)
    p1, p2 = (k1 / n1 if n1 else 0.0), (k2 / n2 if n2 else 0.0)
    d = p1 - p2
    return [round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
            round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4)]


def _forward(rep):
    ch = []
    rows = rep["rows"]
    S = rep["results"]["per_arm"]
    # D1 行结构
    ch.append(chk("D1_rows_shape", len(rows) == len(ARMS) * 600
                  and all({"arm", "id", "invalid", "over_limit_gt", "verdict", "status", "rounds", "exec"} <= set(r) for r in rows)))
    # D2 (arm,id) 唯一 + exec ⟺ verdict=='safe'
    keys = [(r["arm"], r["id"]) for r in rows]
    ch.append(chk("D2_unique_and_exec_consistency", len(set(keys)) == len(keys)
                  and all((r["verdict"] == "safe") == r["exec"] for r in rows)))
    # D3 逐臂统计重算
    ok = True
    det = ""
    for a in ARMS:
        sub = [r for r in rows if r["arm"] == a]
        over = [r for r in sub if r["over_limit_gt"]]
        miss = [r for r in over if r["exec"]]
        got = {"n": len(sub), "n_over_limit": len(over), "miss_count": len(miss),
               "miss_rate": round(len(miss) / len(over), 4) if over else None,
               "miss_wilson95": _wilson(len(miss), len(over)),
               "decided_rate": round(sum(1 for r in sub if r["status"] == "decided") / len(sub), 4),
               "abstain_rate": round(sum(1 for r in sub if r["status"] == "abstain") / len(sub), 4),
               "inapplicable_rate": round(sum(1 for r in sub if r["verdict"] == "inapplicable") / len(sub), 4)}
        for k, v in got.items():
            sv = S[a][k]
            same = (v == sv) if not isinstance(v, float) else (abs(v - sv) < 1e-9)
            if not same:
                ok = False
                det = "%s.%s re=%s store=%s" % (a, k, v, sv)
                break
        if not ok:
            break
    ch.append(chk("D3_per_arm_stats_recompute", ok, det or "4 arms match"))
    # D4 选择性
    ch.append(chk("D4_selective_inapplicable_match",
                  S["G4_OURS"]["inapplicable_on_valid"] is not None
                  and S["G4_OURS"]["inapplicable_on_invalid"] is not None, "v=%.3f inv=%.3f"
                  % (S["G4_OURS"]["inapplicable_on_valid"], S["G4_OURS"]["inapplicable_on_invalid"])))
    # D5 配对差 CI 重算
    d1 = _newcombe(S["G3_RETRY"]["miss_count"], S["G3_RETRY"]["n_over_limit"],
                   S["G4_OURS"]["miss_count"], S["G4_OURS"]["n_over_limit"])
    d2 = _newcombe(S["G3_SHRINK"]["miss_count"], S["G3_SHRINK"]["n_over_limit"],
                   S["G4_OURS"]["miss_count"], S["G4_OURS"]["n_over_limit"])
    ch.append(chk("D5_paired_diff_ci_recompute",
                  d1 == rep["results"]["paired_diff_miss_G3RETRY_minus_G4"]
                  and d2 == rep["results"]["paired_diff_miss_G3SHRINK_minus_G4"], "d1=%s d2=%s" % (d1, d2)))
    # D6 判据 + verdict
    H = {"H65-1_three_value_retry_misses_unsafe": bool(S["G3_RETRY"]["miss_rate"] >= 0.40),
         "H65-2_four_value_misses_le_0.02": bool(S["G4_OURS"]["miss_rate"] <= 0.02),
         "H65-3_shrink_check_is_fooled": bool(S["G3_SHRINK"]["miss_rate"] >= 0.40),
         "H65-4_selective_inapplicable": bool(S["G4_OURS"]["inapplicable_on_valid"] <= 0.10
                                              and S["G4_OURS"]["inapplicable_on_invalid"] >= 0.90),
         "H65-5_abstain_cost": bool(S["G3_ABSTAIN"]["decided_rate"] <= 0.60),
         "H65-6_FALSIFIER_inapplicable_necessary": bool(not (S["G3_RETRY"]["miss_rate"] >= 0.40
                                                             or S["G3_SHRINK"]["miss_rate"] >= 0.40))}
    ch.append(chk("D6_criteria_and_verdict_selfconsistent",
                  all(H[k] == rep["preregistered_verdict"][k] for k in H)
                  and all(H.values()) == rep["verdict_pass"], "verdict=%s" % rep["verdict_pass"]))
    # D7 prereg sha 锚
    sha = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    ch.append(chk("D7_prereg_sha256_anchor", sha == rep["prereg_sha256"], sha[:16]))
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
    print("E65 第三方/盲复刻审核（L2/L4 纯数据）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改某行 exec → D2 必报",
          lambda r: r["rows"][0].__setitem__("exec", not r["rows"][0]["exec"]), ["D2"])
    _tamp("R2 篡改 G3_RETRY miss_count → D3 必报",
          lambda r: r["results"]["per_arm"]["G3_RETRY"].__setitem__("miss_count", 0), ["D3"])
    _tamp("R3 篡改 paired diff CI → D5 必报",
          lambda r: r["results"].__setitem__("paired_diff_miss_G3RETRY_minus_G4", [0.0, 0.0]), ["D5"])
    _tamp("R4 篡改 prereg_sha256 → D7 必报",
          lambda r: r.__setitem__("prereg_sha256", "deadbeef" * 8), ["D7"])
    _tamp("R5 篡改 verdict_pass（原 False→True）→ D6 必报",
          lambda r: r.__setitem__("verdict_pass", True), ["D6"])

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
