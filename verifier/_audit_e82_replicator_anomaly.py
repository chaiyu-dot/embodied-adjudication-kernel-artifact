# -*- coding: utf-8 -*-
"""_audit_e82_replicator_anomaly.py —— E82 的**第三方/盲复刻（L2/L4，纯数据）**。

不 import 任何实验模块；只读 e82_material_domain_report.json（+ E73 外部交叉），验证报告内部
算术自洽、保守/名义点从存储 band 重算、四裁决点保守≤名义、单调标记一致、bench_sha256 锚，
篡改必有检查报出。

产物：_audit_e82_replicator_anomaly.json
"""
import copy
import hashlib
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e82_material_domain_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
OUT = os.path.join(EVAL, "_audit_e82_replicator_anomaly.json")

_res = {"experiment": "E82 第三方/盲复刻(L2/L4 纯数据)", "independence_scope":
        "不 import 被审模块，只读 JSON（第三方）+ E73 外部交叉", "checks": [], "tamper": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    b = rep["bench"]

    # D1 bench 完整
    need = {"m_payload_kg", "A_contact_m2", "delta_contact_m", "d_stop_m", "F_allow_N", "dt_contact_s", "q_friction"}
    ok_bench = need.issubset(b.keys()) and b["m_payload_kg"] > 0 and b["F_allow_N"] > 0
    ch.append(chk("D1_bench_complete_and_sane", ok_bench))

    # D2 本体 8 类齐全
    ch.append(chk("D2_ontology_eight_categories", len(rep["ontology_categories"]) == 8,
                  "n=%d" % len(rep["ontology_categories"])))

    # D3 保守点 = band lo/hi/hi/lo
    s3 = True
    for r in rep["rows"]:
        cp, bd = r["conservative_point"], r["band"]
        s3 &= (abs(cp["mu"] - bd["mu"][0]) < 1e-12 and abs(cp["k"] - bd["k"][1]) < 1e-12
               and abs(cp["e"] - bd["e"][1]) < 1e-12 and abs(cp["sigma"] - bd["sigma"][0]) < 1e-12)
    ch.append(chk("D3_conservative_point_matches_band", s3))

    # D4 名义点 = 中点
    s4 = True
    for r in rep["rows"]:
        np_, bd = r["nominal_point"], r["band"]
        s4 &= (abs(np_["mu"] - 0.5 * (bd["mu"][0] + bd["mu"][1])) < 1e-12
               and abs(np_["k"] - 0.5 * (bd["k"][0] + bd["k"][1])) < 1e-12
               and abs(np_["e"] - 0.5 * (bd["e"][0] + bd["e"][1])) < 1e-12
               and abs(np_["sigma"] - 0.5 * (bd["sigma"][0] + bd["sigma"][1])) < 1e-12)
    ch.append(chk("D4_nominal_point_matches_midpoint", s4))

    # D5 E72 保守≤名义（纯算术：存储两值比较；μ→力物理链由 L3 承担）
    s5 = True
    for r in rep["rows"]:
        ec_v = r["E72_max_tangential_N"]["conservative"]; en_v = r["E72_max_tangential_N"]["nominal"]
        s5 &= (ec_v <= en_v + 1e-3) and math.isfinite(ec_v) and math.isfinite(en_v) and ec_v > 0
    ch.append(chk("D5_E72_conservative_le_nominal", s5))

    # D6 E79 保守≤名义
    s6 = True
    for r in rep["rows"]:
        ec_v = r["E79_v_star_mps"]["conservative"]; en_v = r["E79_v_star_mps"]["nominal"]
        s6 &= (ec_v <= en_v + 1e-4) and math.isfinite(ec_v) and ec_v > 0
    ch.append(chk("D6_E79_conservative_le_nominal", s6))

    # D7 E76 保守≤名义
    s7 = True
    for r in rep["rows"]:
        ec_v = r["E76_max_internal_force_N"]["conservative"]; en_v = r["E76_max_internal_force_N"]["nominal"]
        s7 &= (ec_v <= en_v + 1e-3) and math.isfinite(ec_v) and ec_v > 0
    ch.append(chk("D7_E76_conservative_le_nominal", s7))

    # D8 E73 保守≤名义 且 ≤ 运动学
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin = float(r73["B_envelope_curve"]["v_max_on_path"])
    s8 = True
    for r in rep["rows"]:
        ec_v = r["E73_impact_speed_cap_mps"]["conservative"]; en_v = r["E73_impact_speed_cap_mps"]["nominal"]
        s8 &= (ec_v <= en_v + 1e-4) and (ec_v <= v_kin + 1e-4) and math.isfinite(ec_v) and ec_v > 0
    ch.append(chk("D8_E73_conservative_le_nominal_and_kinematic", s8, "v_kin=%.4f" % v_kin))

    # D9 单调标记 == 重算
    s9 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        mono_re = (r["E72_max_tangential_N"]["conservative"] <= r["E72_max_tangential_N"]["nominal"] + 1e-3
                   and r["E79_v_star_mps"]["conservative"] <= r["E79_v_star_mps"]["nominal"] + 1e-4
                   and r["E76_max_internal_force_N"]["conservative"] <= r["E76_max_internal_force_N"]["nominal"] + 1e-3
                   and r["E73_impact_speed_cap_mps"]["conservative"] <= r["E73_impact_speed_cap_mps"]["nominal"] + 1e-4
                   and r["E73_impact_speed_cap_mps"]["conservative"] <= v_kin + 1e-4)
        s9 &= (mono_re is r["monotone_conservative"])
    ch.append(chk("D9_monotonicity_flags_selfconsistent", s9))

    # D10 criteria + verdict + bench_sha256
    cr = rep["criteria"]
    verd = (cr["C1_monotone_conservative"] and cr["C2_all_four_verdict_points_fed"]
            and cr["C3_zero_api_classifier_runnable"])
    sha = hashlib.sha256(json.dumps(b, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    s10 = (cr["C1_monotone_conservative"] is s9 and cr["C2_all_four_verdict_points_fed"] is (len(rep["rows"]) == 8)
           and cr["C3_zero_api_classifier_runnable"] is True and rep["verdict_pass"] is verd
           and rep["bench_sha256"] == sha)
    ch.append(chk("D10_criteria_verdict_and_sha", s10))
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
    print("E82 第三方/盲复刻审核（L2/L4 纯数据，只读 JSON + E73 交叉）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("内部算术判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 steel 保守 μ 乐观 1.5 → D3 必报",
          lambda r: r["rows"].__setitem__(0, dict(r["rows"][0], conservative_point=dict(r["rows"][0]["conservative_point"], mu=1.5))), ["D3"])
    _tamp("R2 篡改 steel E72 保守虚高 9.9 → D5 必报",
          lambda r: r["rows"][0]["E72_max_tangential_N"].__setitem__("conservative", 9.9), ["D5"])
    _tamp("R3 篡改 row[1] monotone=False → D9 必报",
          lambda r: r["rows"][1].__setitem__("monotone_conservative", False), ["D9"])
    _tamp("R4 篡改 verdict_pass=False → D10 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["D10"])
    _tamp("R5 篡改 bench_sha256 → D10 必报",
          lambda r: r.__setitem__("bench_sha256", "deadbeefdeadbeef"), ["D10"])

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
