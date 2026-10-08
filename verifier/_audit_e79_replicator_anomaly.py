# -*- coding: utf-8 -*-
"""_audit_e79_replicator_anomaly.py —— E79 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e79_momentum_impulse_report.json，验证**报告内部算术自洽**、
派生量（各判据 ok 标志 / n_missed / v_star / bench_sha256）可从存储字段 + bench 重算、
篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON）。能证明"报告内部一致"，不能证明"存储字段物理正确"
（那要 L3 独立物理 / L5 逆向）。

产物：_audit_e79_replicator_anomaly.json
"""
import copy
import hashlib
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
OUT = os.path.join(EVAL, "_audit_e79_replicator_anomaly.json")

_res = {"experiment": "E79 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）", "checks": [], "tamper": []}

_TOL = 1e-12


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    b = rep["bench"]
    A, B, C, D = rep["A_closedform"], rep["B_impulse_momentum"], rep["C_verdict_coupling"], rep["D_cop_vs_static"]
    m, v, e, amax = b["m_payload_kg"], b["v_cap_mps"], b["restitution"], b["decel_max_mps2"]
    dtc, Fallow = b["dt_contact_s"], b["F_allow_N"]

    # --- D1：bench 完整且参数合理 ---
    ok_bench = (m > 0 and v > 0 and amax > 0 and dtc > 0 and 0 <= e <= 1 and Fallow > 0
                and b["com_z_m"] > 0 and len(b["support_poly"]) == 4
                and len(b["v_cap_sweep"]) >= 2)
    ch.append(chk("D1_bench_complete_and_sane", ok_bench,
                  "m=%g v=%g a=%g dt=%g e=%g F=%g" % (m, v, amax, dtc, e, Fallow)))

    # --- D2：A 闭式一致（存储 impl == 解析）---
    J_ana = (1.0 + e) * m * v
    KE_ana = 0.5 * m * v * v
    d_ana = v * v / (2.0 * amax)
    F_ana = m * amax
    devA = max(abs(J_ana - A["rows"]["A1_impulse"]["impl"]),
               abs(KE_ana - A["rows"]["A2_braking"]["kinetic_energy_J"]["impl"]),
               abs(d_ana - A["rows"]["A2_braking"]["braking_distance_m"]["impl"]),
               abs(F_ana - A["rows"]["A2_braking"]["braking_force_N"]["impl"]),
               abs(m * v - A["rows"]["A3_momentum"]["impl"]))
    ch.append(chk("D2_A_closedform_stored_eq_analytic", devA < 1e-9 and A["max_abs_err"] < _TOL
                  and A["all_ok"] is (A["max_abs_err"] < _TOL), "max_dev=%.2e" % devA))

    # --- D3：B 冲量-动量定理（存储 witnesses == 0；闭合关系）---
    Brows = B["rows"]
    v1p, v2p = Brows["v1_prime"], Brows["v2_prime"]
    dp1, dp2 = Brows["dp1"], Brows["dp2"]
    Jc = Brows["J_closure"]
    devB = max(abs(Brows["momentum_conserved_err"]), abs(Brows["dp1_vs_J_err"]),
               abs(Brows["e0_check_err"]), abs(dp1 + dp2), abs(dp1 - Jc))
    ch.append(chk("D3_B_impulse_momentum_witnesses_zero_and_closure", devB < _TOL and B["all_ok"] is True,
                  "max_dev=%.2e" % devB))

    # --- D4：C 冲量进裁决（存储 verdicts 自洽；n_missed 重算；v_star）---
    n_miss_re = 0
    v_star_re = Fallow * dtc / ((1.0 + e) * m)
    c_ok = True
    for vk, vr in C["rows"].items():
        if not vk.startswith("v="):
            continue
        vv = vr["v_mps"]
        F_peak = (1.0 + e) * m * vv / dtc
        F_static = m * amax
        if abs(F_peak - vr["F_peak_N"]) >= 1e-9 or abs(F_static - vr["F_static_N"]) >= 1e-9:
            c_ok = False
        if vr["verdict_static"] and not vr["verdict_impulse"]:
            n_miss_re += 1
    ch.append(chk("D4_C_verdict_coupling_selfconsistent", c_ok and n_miss_re == C["n_missed_by_static"]
                  and abs(v_star_re - C["rows"]["_v_star_impulse_mps"]) < 1e-9
                  and C["all_ok"] is (C["n_missed_by_static"] >= 1),
                  "n_miss_re=%d stored=%d v_star=%.4f" % (n_miss_re, C["n_missed_by_static"], v_star_re)))

    # --- D5：D 动态 CoP（cop_x 公式 + 矩形支撑域 + n 重算）---
    g = 9.81
    xs = [p[0] for p in b["support_poly"]]
    ys = [p[1] for p in b["support_poly"]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    n_dyn_re = 0
    d_ok = True
    for ak, ar in D["rows"].items():
        if not ak.startswith("ax="):
            continue
        ax = ar["com_acc_x"]
        cop_x_re = b["com_xy"][0] - (b["com_z_m"] / g) * ax   # CoP 公式（纯代数）
        if abs(cop_x_re - ar["cop_x"]) >= 1e-9:
            d_ok = False
        in_rect = (xmin - 1e-9 <= ar["cop_x"] <= xmax + 1e-9) and (ymin - 1e-9 <= b["com_xy"][1] <= ymax + 1e-9)
        if in_rect != ar["dynamic_cop_in_support"]:
            d_ok = False
        if ar["static_com_in_support"] and not ar["dynamic_cop_in_support"]:
            n_dyn_re += 1
    ch.append(chk("D5_D_cop_selfconsistent", d_ok and n_dyn_re == D["n_static_only_missed"]
                  and D["all_ok"] is (D["n_static_only_missed"] >= 1),
                  "n_dyn_re=%d stored=%d" % (n_dyn_re, D["n_static_only_missed"])))

    # --- D6：criteria + verdict 自洽 ---
    cr = rep["criteria"]
    verdict_ok = (cr["C1_closedform_consistent"] is A["all_ok"]
                  and cr["C2_impulse_momentum_theorem"] is B["all_ok"]
                  and cr["C3_impulse_arm_effective"] is C["all_ok"]
                  and cr["C4_dynamic_cop_arm_effective"] is D["all_ok"]
                  and rep["verdict_pass"] is (cr["C1_closedform_consistent"] and cr["C2_impulse_momentum_theorem"]
                                             and cr["C3_impulse_arm_effective"] and cr["C4_dynamic_cop_arm_effective"]))
    ch.append(chk("D6_criteria_and_verdict_selfconsistent", verdict_ok,
                  "verdict=%s" % rep["verdict_pass"]))

    # --- D7：bench_sha256 完整性锚 ---
    sha = hashlib.sha256(json.dumps(b, sort_keys=True).encode("utf-8")).hexdigest()[:16]
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
    print("E79 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 A.max_abs_err=9.9 → D2 必报",
          lambda r: r["A_closedform"].__setitem__("max_abs_err", 9.9), ["D2"])
    _tamp("R2 篡改 B.momentum_conserved_err=9.9 → D3 必报",
          lambda r: r["B_impulse_momentum"]["rows"].__setitem__("momentum_conserved_err", 9.9), ["D3"])
    _tamp("R3 篡改 C.n_missed_by_static=0 → D4 必报",
          lambda r: r["C_verdict_coupling"].__setitem__("n_missed_by_static", 0), ["D4"])
    _tamp("R4 篡改 D.n_static_only_missed=0 → D5 必报",
          lambda r: r["D_cop_vs_static"].__setitem__("n_static_only_missed", 0), ["D5"])
    _tamp("R5 篡改 verdict_pass=False → D6 必报", lambda r: r.__setitem__("verdict_pass", False), ["D6"])
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
