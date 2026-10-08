# -*- coding: utf-8 -*-
"""_audit_e80_replicator_anomaly.py —— E80 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e80_three_paradigm_dynamics_report.json，验证**报告内部算术
自洽**、派生量（max_abs / max_pairwise / 各判据 ok 标志 / verdict）可从存储字段重算、
bench_sha256 完整性锚，篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。能证明"报告内部一致"，不能证明"存储字段物理正确"
（那要 L3 独立物理 / L5 逆向）。

产物：_audit_e80_replicator_anomaly.json
"""
import copy
import hashlib
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e80_three_paradigm_dynamics_report.json")
OUT = os.path.join(EVAL, "_audit_e80_replicator_anomaly.json")

_res = {"experiment": "E80 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    bench = rep["bench"]
    A, B, C = rep["A_three_paradigm"], rep["B_falsification_witness"], rep["C_ke_crosscheck"]
    tol_C1, tol_C2, tol_C3 = bench["tol_C1"], bench["tol_C2"], bench["tol_C3"]

    # --- D1：bench 完整且参数合理 ---
    need = {"models", "n_states_per_model", "seed", "q_range", "qd_range",
            "qdd_range", "g", "tol_C1", "tol_C2", "tol_C3"}
    ok_bench = need.issubset(bench.keys()) and len(bench["models"]) == 3 \
        and bench["n_states_per_model"] == 8 and bench["g"] == 9.81
    ch.append(chk("D1_bench_complete_and_sane", ok_bench, "models=%d n=%d g=%s"
                  % (len(bench["models"]), bench["n_states_per_model"], bench["g"])))

    # --- D2：A 行数 = 3×8=24；tau_rne==tau_cf（两解析路）；存储偏差字段自洽 ---
    rows = A["rows"]
    n_ok = (len(rows) == 24)
    two_analytic_ok = True
    dev_self_ok = True
    for r in rows:
        tr, tc, tl = r["tau_rne"], r["tau_cf"], r["tau_lag"]
        # 两解析路径须到机器精度一致
        two_analytic_ok &= (max(abs(tr[0] - tc[0]), abs(tr[1] - tc[1])) < 1e-9)
        # 存储偏差字段须等于 |τ_a − τ_b|
        dev_self_ok &= (abs(r["d_rne_cf"] - max(abs(tr[0] - tc[0]), abs(tr[1] - tc[1]))) < 1e-6
                        and abs(r["d_rne_lag"] - max(abs(tr[0] - tl[0]), abs(tr[1] - tl[1]))) < 1e-6
                        and abs(r["d_cf_lag"] - max(abs(tc[0] - tl[0]), abs(tc[1] - tl[1]))) < 1e-6)
    ch.append(chk("D2_rows_count_and_two_analytic_paths_agree", n_ok and two_analytic_ok,
                  "n=%d two_analytic_ok=%s" % (len(rows), two_analytic_ok)))
    ch.append(chk("D2b_stored_deviation_fields_selfconsistent", dev_self_ok))

    # --- D3：A 聚合 max_abs / max_pairwise 从行重算一致；< tol_C1；all_ok 一致 ---
    mx = max(max(r["d_rne_cf"], r["d_rne_lag"], r["d_cf_lag"]) for r in rows)
    mx_pair = {"rne_vs_cf": max(r["d_rne_cf"] for r in rows),
               "rne_vs_lag": max(r["d_rne_lag"] for r in rows),
               "cf_vs_lag": max(r["d_cf_lag"] for r in rows)}
    agg_ok = (abs(mx - A["max_abs"]) < 1e-10
              and abs(mx_pair["rne_vs_cf"] - A["max_pairwise"]["rne_vs_cf"]) < 1e-10
              and abs(mx_pair["rne_vs_lag"] - A["max_pairwise"]["rne_vs_lag"]) < 1e-10
              and abs(mx_pair["cf_vs_lag"] - A["max_pairwise"]["cf_vs_lag"]) < 1e-10
              and A["max_abs"] < tol_C1
              and A["all_ok"] is (A["max_abs"] < tol_C1))
    ch.append(chk("D3_max_abs_and_pairwise_recomputed_and_all_ok_consistent", agg_ok,
                  "recomp=%.3e stored=%.3e tol=%.0e" % (mx, A["max_abs"], tol_C1)))

    # --- D4：B 证伪见证自洽 ---
    b_ok = (B["max_bad_vs_good"] > tol_C2
            and B["detected"] is (B["max_bad_vs_good"] > tol_C2)
            and B["all_ok"] is B["detected"])
    ch.append(chk("D4_falsification_witness_selfconsistent", b_ok,
                  "max_bad=%.3e tol=%.0e detected=%s" % (B["max_bad_vs_good"], tol_C2, B["detected"])))

    # --- D5：C KE 独立一致自洽 ---
    c_ok = (C["max_rel_dev"] < tol_C3 and C["all_ok"] is (C["max_rel_dev"] < tol_C3))
    ch.append(chk("D5_ke_crosscheck_selfconsistent", c_ok,
                  "max_rel_dev=%.3e tol=%.0e" % (C["max_rel_dev"], tol_C3)))

    # --- D6：criteria 与 verdict 自洽 ---
    cr = rep["criteria"]
    verdict_ok = (cr["C1_three_paradigm_agree"] is A["all_ok"]
                  and cr["C2_broken_paradigm_detected"] is B["all_ok"]
                  and cr["C3_ke_independent_agree"] is C["all_ok"]
                  and rep["verdict_pass"] is (cr["C1_three_paradigm_agree"]
                                             and cr["C2_broken_paradigm_detected"]
                                             and cr["C3_ke_independent_agree"]))
    ch.append(chk("D6_criteria_and_verdict_selfconsistent", verdict_ok,
                  "C1=%s C2=%s C3=%s verdict=%s"
                  % (cr["C1_three_paradigm_agree"], cr["C2_broken_paradigm_detected"],
                     cr["C3_ke_independent_agree"], rep["verdict_pass"])))

    # --- D7：bench_sha256 完整性锚 ---
    sha = hashlib.sha256(json.dumps(bench, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ch.append(chk("D7_bench_sha256_integrity_anchor", rep["bench_sha256"] == sha,
                  "stored=%s recomp=%s" % (rep["bench_sha256"], sha)))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name,
                             "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E80 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 A.all_ok=False → D3 必报",
          lambda r: r["A_three_paradigm"].__setitem__("all_ok", False), ["D3"])
    _tamp("R2 篡改 B.detected=False → D4 必报",
          lambda r: r["B_falsification_witness"].__setitem__("detected", False), ["D4"])
    _tamp("R3 篡改 C.max_rel_dev=9.9 → D5 必报",
          lambda r: r["C_ke_crosscheck"].__setitem__("max_rel_dev", 9.9), ["D5"])
    _tamp("R4 篡改第一行 tau_lag 数据 → D2b 必报",
          lambda r: r["A_three_paradigm"]["rows"][0]["tau_lag"].__setitem__(0,
                  r["A_three_paradigm"]["rows"][0]["tau_lag"][0] + 100.0), ["D2b"])
    _tamp("R5 篡改 verdict_pass=False → D6 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["D6"])
    _tamp("R6 篡改 bench_sha256 → D7 必报",
          lambda r: r.__setitem__("bench_sha256", "deadbeefdeadbeef"), ["D7"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["replicator_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n第三方审核：算术 %d/%d ｜ 篡改 %d/%d ｜ pass = %s"
          % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
