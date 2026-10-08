# -*- coding: utf-8 -*-
"""_audit_e81_replicator_anomaly.py —— E81 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e81_dynamic_speed_ceiling_report.json（+ E73/E79 作外部数据交叉），
验证报告内部算术自洽、派生量（v_dynamic_cap / binding / criteria / verdict）可从存储字段重算、
bench_sha256 完整性锚，篡改任一字段必有检查报出。

独立性：第三方（纯数据，只读 JSON + 外部报告交叉）。能证明"报告内部一致 + 与 E73/E79 外部一致"，
但不能证明"科氏扫掠物理正确"（那要 L3 独立物理 / L5 逆向）。

产物：_audit_e81_replicator_anomaly.json
"""
import copy
import hashlib
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e81_dynamic_speed_ceiling_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
E79_REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
OUT = os.path.join(EVAL, "_audit_e81_replicator_anomaly.json")

_res = {"experiment": "E81 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）+ E73/E79 外部交叉", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    bench = rep["bench"]
    A, B, C = rep["A_kinematic"], rep["B_impulse"], rep["C_coriolis"]

    # --- D1：bench 完整且参数合理 ---
    need = {"L1", "L2", "M1", "M2", "TAU_LIM", "g", "m_payload_kg", "restitution",
            "dt_contact_s", "F_allow_N", "sigma_min_min", "n_q1", "n_q2", "n_dir", "qdd_zero"}
    ok_bench = (need.issubset(bench.keys()) and len(bench["TAU_LIM"]) == 2
                and bench["g"] == 9.81 and bench["L1"] > 0 and bench["L2"] > 0
                and bench["m_payload_kg"] > 0 and 0 <= bench["restitution"] < 1)
    ch.append(chk("D1_bench_complete_and_sane", ok_bench, "taulim=%s g=%s m=%s e=%s"
                  % (bench["TAU_LIM"], bench["g"], bench["m_payload_kg"], bench["restitution"])))

    # --- D2：A.v_kinematic == E73 外部交叉 + source 标记 ---
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin_e73 = float(r73["B_envelope_curve"]["v_max_on_path"])
    ch.append(chk("D2_A_kinematic_matches_E73", abs(A["v_kinematic_mps"] - v_kin_e73) < 1e-9
                  and A["source"].startswith("E73"), "stored=%.4f e73=%.4f" % (A["v_kinematic_mps"], v_kin_e73)))

    # --- D3：B 内部自洽：v_impulse 重算==存储；abs_err_vs_e79==重算 ---
    v_imp_re = bench["F_allow_N"] * bench["dt_contact_s"] / ((1.0 + bench["restitution"]) * bench["m_payload_kg"])
    r79 = json.load(open(E79_REPORT, encoding="utf-8"))
    v_imp_e79 = float(r79["C_verdict_coupling"]["rows"]["_v_star_impulse_mps"])
    b_ok = (abs(v_imp_re - B["v_impulse_mps"]) < 1e-9
            and abs(v_imp_re - v_imp_e79) < 1e-9
            and abs(B["abs_err_vs_e79"] - abs(v_imp_re - v_imp_e79)) < 1e-12)
    ch.append(chk("D3_B_impulse_selfconsistent", b_ok, "recomp=%.6f stored=%.6f e79=%.6f err=%.2e"
                  % (v_imp_re, B["v_impulse_mps"], v_imp_e79, B["abs_err_vs_e79"])))

    # --- D4：C 内部自洽：有限正数 + n_valid>0 + worst_config 良构（单位方向）---
    wc = C["worst_config"]
    c_ok = (C["v_coriolis_mps"] is not None and C["v_coriolis_mps"] > 0 and math.isfinite(C["v_coriolis_mps"])
            and isinstance(C["n_valid_configs_dirs"], int) and C["n_valid_configs_dirs"] > 0
            and isinstance(wc["q"], list) and len(wc["q"]) == 2
            and isinstance(wc["dir"], list) and len(wc["dir"]) == 2
            and abs(math.hypot(wc["dir"][0], wc["dir"][1]) - 1.0) < 1e-6)
    ch.append(chk("D4_C_coriolis_selfconsistent", c_ok, "v_cor=%.4f nvalid=%d"
                  % (C["v_coriolis_mps"], C["n_valid_configs_dirs"])))

    # --- D5：动态天花板 = min(三者) + 绑定（tie-break 顺序 kinematic→impulse→coriolis）---
    v_kin, v_imp, v_cor = A["v_kinematic_mps"], B["v_impulse_mps"], C["v_coriolis_mps"]
    v_dyn_re = min(v_kin, v_imp, v_cor)
    bind_re = ("kinematic" if v_dyn_re == v_kin else "impulse" if v_dyn_re == v_imp else "coriolis")
    ch.append(chk("D5_dynamic_cap_and_binding", abs(v_dyn_re - rep["v_dynamic_cap_mps"]) < 1e-9
                  and bind_re == rep["binding_constraint"]
                  and rep["v_dynamic_cap_mps"] < v_kin - 1e-9,
                  "recomp=%.4f stored=%.4f bind=%s" % (v_dyn_re, rep["v_dynamic_cap_mps"], bind_re)))

    # --- D6：criteria 与 verdict 自洽 ---
    cr = rep["criteria"]
    verdict_ok = (cr["C1_dynamic_tighter_than_kinematic"] is (rep["v_dynamic_cap_mps"] < A["v_kinematic_mps"] - 1e-9)
                  and cr["C2_impulse_matches_e79"] is (B["abs_err_vs_e79"] < 1e-9)
                  and cr["C3_coriolis_sweep_converged"] is (math.isfinite(C["v_coriolis_mps"]) and C["n_valid_configs_dirs"] > 0)
                  and rep["verdict_pass"] is (cr["C1_dynamic_tighter_than_kinematic"]
                                              and cr["C2_impulse_matches_e79"]
                                              and cr["C3_coriolis_sweep_converged"]))
    ch.append(chk("D6_criteria_and_verdict_selfconsistent", verdict_ok,
                  "C1=%s C2=%s C3=%s verdict=%s" % (cr["C1_dynamic_tighter_than_kinematic"],
                  cr["C2_impulse_matches_e79"], cr["C3_coriolis_sweep_converged"], rep["verdict_pass"])))

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
    print("E81 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON + E73/E79 交叉）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 A v_kinematic=9.9 → D2 必报",
          lambda r: r["A_kinematic"].__setitem__("v_kinematic_mps", 9.9), ["D2"])
    _tamp("R2 篡改 B v_impulse=9.9 → D3 必报",
          lambda r: r["B_impulse"].__setitem__("v_impulse_mps", 9.9), ["D3"])
    _tamp("R3 篡改 C n_valid_configs_dirs=0 → D4 必报",
          lambda r: r["C_coriolis"].__setitem__("n_valid_configs_dirs", 0), ["D4"])
    _tamp("R4 篡改 v_dynamic_cap=9.9 → D5 必报",
          lambda r: r.__setitem__("v_dynamic_cap_mps", 9.9), ["D5"])
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
