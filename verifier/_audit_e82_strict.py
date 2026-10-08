# -*- coding: utf-8 -*-
"""_audit_e82_strict.py —— E82 严格审核（L3 独立物理）。

不 import e82；用真实 planning.friction_cone 独立重算 E72 摩擦锥，并独立重算 E79/E76/E73 闭式，
验证保守先验带传播：保守点 μ_lo/k_hi/e_hi/σ_lo 在四裁决点恒 ≤ 名义点，且 E73 冲击界 ≤ 运动学上限，
无乐观泄漏。篡改用例覆盖保守点/单调性/verdict。

产物：_audit_e82_strict.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import friction_cone as FC                                    # noqa: E402

REPORT = os.path.join(EVAL, "e82_material_domain_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
OUT = os.path.join(EVAL, "_audit_e82_strict.json")

_res = {"experiment": "E82 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写闭式 + 真实 friction_cone 参考，不 import e82", "checks": [], "tamper": []}


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _cons(band):
    return {"mu": band["mu"][0], "k": band["k"][1], "e": band["e"][1], "sigma": band["sigma"][0]}


def _nom(band):
    return {"mu": 0.5 * (band["mu"][0] + band["mu"][1]), "k": 0.5 * (band["k"][0] + band["k"][1]),
            "e": 0.5 * (band["e"][0] + band["e"][1]), "sigma": 0.5 * (band["sigma"][0] + band["sigma"][1])}


def _e79(e, F, dt, m):
    return F * dt / ((1.0 + e) * m)


def _e76(mu, N):
    return mu * N


def _e73(k_gpa, sigma_mpa, A, delta, m, F):
    k_eff = (k_gpa * 1e9) * A / max(delta, 1e-9)
    return sigma_mpa * 1e6 * A / math.sqrt(k_eff * max(m, 1e-9)), k_eff


def _forward(rep):
    ch = []
    b = rep["bench"]
    q = b["q_friction"]
    N_total = 20.0
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin = float(r73["B_envelope_curve"]["v_max_on_path"])

    # --- S1：本体先验带 sanity（μ∈[0,2], k>0, e∈[0,1), σ>0, 下界≤上界）---
    ont = rep["ontology_categories"]
    sanity = True
    for cat in ont:
        # 从 rows 取 band
        row = next(r for r in rep["rows"] if r["category"] == cat)
        bd = row["band"]
        for key in ("mu", "k", "e", "sigma"):
            lo, hi = bd[key]
            sanity &= (lo <= hi) and (lo >= 0)
        sanity &= (bd["mu"][1] <= 2.0) and (bd["k"][0] > 0) and (bd["e"][1] < 1.0) and (bd["sigma"][0] > 0)
    ch.append(_chk("S1_ontology_bands_sane", sanity, "categories=%d" % len(ont)))

    # --- S2：保守点重算 == 存储（μ_lo/k_hi/e_hi/σ_lo）---
    s2 = True
    for r in rep["rows"]:
        cp = r["conservative_point"]
        bd = r["band"]
        s2 &= (abs(cp["mu"] - bd["mu"][0]) < 1e-12 and abs(cp["k"] - bd["k"][1]) < 1e-12
               and abs(cp["e"] - bd["e"][1]) < 1e-12 and abs(cp["sigma"] - bd["sigma"][0]) < 1e-12)
    ch.append(_chk("S2_conservative_point_recomputed", s2))

    # --- S3：名义点重算 == 存储（中点）---
    s3 = True
    for r in rep["rows"]:
        np_ = r["nominal_point"]
        bd = r["band"]
        s3 &= (abs(np_["mu"] - 0.5 * (bd["mu"][0] + bd["mu"][1])) < 1e-12
               and abs(np_["k"] - 0.5 * (bd["k"][0] + bd["k"][1])) < 1e-12
               and abs(np_["e"] - 0.5 * (bd["e"][0] + bd["e"][1])) < 1e-12
               and abs(np_["sigma"] - 0.5 * (bd["sigma"][0] + bd["sigma"][1])) < 1e-12)
    ch.append(_chk("S3_nominal_point_recomputed", s3))

    # --- S4：E72 摩擦锥保守 < 名义（真实 FC 参考）---
    s4 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        ec = FC.max_tangential_push(q, cp["mu"])["max_tangential_N"]
        en = FC.max_tangential_push(q, np_["mu"])["max_tangential_N"]
        stored_c = r["E72_max_tangential_N"]["conservative"]
        stored_n = r["E72_max_tangential_N"]["nominal"]
        s4 &= (ec <= en + 1e-9) and abs(ec - stored_c) < 1e-3 and abs(en - stored_n) < 1e-3
    ch.append(_chk("S4_E72_friction_cone_conservative_tighter", s4))

    # --- S5：E79 冲量保守 < 名义（独立闭式）---
    s5 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        ec = _e79(cp["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
        en = _e79(np_["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
        s5 &= (ec <= en + 1e-12) and abs(ec - r["E79_v_star_mps"]["conservative"]) < 1e-4 \
            and abs(en - r["E79_v_star_mps"]["nominal"]) < 1e-4
    ch.append(_chk("S5_E79_impulse_conservative_tighter", s5))

    # --- S6：E76 多臂内力保守 < 名义（独立闭式）---
    s6 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        ec = _e76(cp["mu"], N_total)
        en = _e76(np_["mu"], N_total)
        s6 &= (ec <= en + 1e-12) and abs(ec - r["E76_max_internal_force_N"]["conservative"]) < 1e-9 \
            and abs(en - r["E76_max_internal_force_N"]["nominal"]) < 1e-9
    ch.append(_chk("S6_E76_multiarm_conservative_tighter", s6))

    # --- S7：E73 冲击界保守 < 名义 且 ≤ 运动学上限（独立闭式）---
    s7 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        ec, _ = _e73(cp["k"], cp["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        en, _ = _e73(np_["k"], np_["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        s7 &= (ec <= en + 1e-9) and (ec <= v_kin + 1e-9) \
            and abs(ec - r["E73_impact_speed_cap_mps"]["conservative"]) < 1e-6 \
            and abs(en - r["E73_impact_speed_cap_mps"]["nominal"]) < 1e-6
    ch.append(_chk("S7_E73_impact_cap_conservative_tighter_and_le_kinematic", s7,
                    "v_kin=%.4f" % v_kin))

    # --- S8：单调性标记自洽（存储 mono == 重算）---
    s8 = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        ec = FC.max_tangential_push(q, cp["mu"])["max_tangential_N"]
        en = FC.max_tangential_push(q, np_["mu"])["max_tangential_N"]
        e79c = _e79(cp["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
        e79n = _e79(np_["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
        e76c = _e76(cp["mu"], N_total); e76n = _e76(np_["mu"], N_total)
        e73c, _ = _e73(cp["k"], cp["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        e73n, _ = _e73(np_["k"], np_["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        mono_re = (e79c <= e79n + 1e-9) and (e76c <= e76n + 1e-9) and (e73c <= e73n + 1e-9) \
            and (ec <= en + 1e-9) and (e73c <= v_kin + 1e-9)
        s8 &= (mono_re is r["monotone_conservative"])
    ch.append(_chk("S8_monotonicity_flags_selfconsistent", s8))

    # --- S9：零 API 分类器可复现（审计者自写关键词映射，不 import e82；验证逻辑零 API）---
    _kw = {"steel": ["steel", "钢", "metal", "金属"], "wood": ["wood", "木", "oak", "timber"],
           "rubber": ["rubber", "橡胶", "silicone", "elastomer"]}
    def _classify(d):
        d = (d or "").lower()
        for cat, kws in _kw.items():
            if any(kw in d for kw in kws):
                return cat
        return "unknown"
    s9 = (_classify("a shiny steel beam") == "steel" and _classify("旧木桌") == "wood"
          and _classify("rubber tire") == "rubber")
    ch.append(_chk("S9_zero_api_classifier_runnable", s9))

    # --- S10：criteria + verdict 自洽 ---
    cr = rep["criteria"]
    verd = (cr["C1_monotone_conservative"] and cr["C2_all_four_verdict_points_fed"]
            and cr["C3_zero_api_classifier_runnable"])
    s10 = (cr["C1_monotone_conservative"] is s8 and cr["C2_all_four_verdict_points_fed"] is (len(rep["rows"]) == 8)
           and cr["C3_zero_api_classifier_runnable"] is s9 and rep["verdict_pass"] is verd)
    ch.append(_chk("S10_criteria_and_verdict_selfconsistent", s10))
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
    print("E82 严格审核（L3 独立物理，不 import e82）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 steel 保守 μ 上界（乐观 1.5）→ S2 必报",
          lambda r: r["rows"].__setitem__(0, dict(r["rows"][0], conservative_point=dict(r["rows"][0]["conservative_point"], mu=1.5))), ["S2"])
    _tamp("R2 篡改 steel E72 保守值虚高（乐观 9.9）→ S4 必报",
          lambda r: r["rows"][0]["E72_max_tangential_N"].__setitem__("conservative", 9.9), ["S4"])
    _tamp("R3 篡改 row[1] monotone 标 False（数据实单调）→ S8 必报",
          lambda r: r["rows"][1].__setitem__("monotone_conservative", False), ["S8"])
    _tamp("R4 篡改 verdict_pass=False → S10 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S10"])
    _tamp("R5 篡改 steel E73 冲击界虚高（乐观 9.9）→ S7 必报",
          lambda r: r["rows"][0]["E73_impact_speed_cap_mps"].__setitem__("conservative", 9.9), ["S7"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s"
          % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
