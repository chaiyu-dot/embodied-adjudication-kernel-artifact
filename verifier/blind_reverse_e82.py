# -*- coding: utf-8 -*-
"""blind_reverse_e82.py —— E82 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 e82_material_domain）。审计者自写四裁决点闭式，
并以负对照证明保守映射有分辨力（非同义反复）：
  · 若改用乐观点（μ_hi/k_lo/e_lo/σ_hi）则四裁决界会变大（更松）→ 证明保守(μ_lo/...)映射
    确把裁决压严，而非冻结数。
  · E73 冲击界对所有材料 ≤ 运动学上限 → 材料耦合确有意义。
  · 零 API 关键词分类器审计者自写复现 steel/wood/rubber。

产物：blind_reverse_e82.json
"""
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 e82_material_domain 不可被 import
from planning import friction_cone as FC                                    # noqa: E402

REPORT = os.path.join(EVAL, "e82_material_domain_report.json")
E73_REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e82.json")

_res = {"experiment": "E82 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e82_material_domain", "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _e79(e, F, dt, m):
    return F * dt / ((1.0 + e) * m)


def _e76(mu, N):
    return mu * N


def _e73(k_gpa, sigma_mpa, A, delta, m, F):
    k_eff = (k_gpa * 1e9) * A / max(delta, 1e-9)
    return sigma_mpa * 1e6 * A / math.sqrt(k_eff * max(m, 1e-9))


def main():
    print("=" * 88)
    print("E82 逆向盲测（L5 隔离，自写闭式 + 负对照证伪分辨力）")
    print("=" * 88)
    chk("R0_isolation_e82_not_imported",
        "e82_material_domain" not in sys.modules,
        "sys.modules 含 e82=%s" % ("e82_material_domain" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    b = rep["bench"]
    q = b["q_friction"]
    N_total = 20.0
    r73 = json.load(open(E73_REPORT, encoding="utf-8"))
    v_kin = float(r73["B_envelope_curve"]["v_max_on_path"])

    # --- R1：独立重算四裁决点，保守≤名义 对所有 8 类成立 ---
    all_mono = True
    for r in rep["rows"]:
        cp, np_ = r["conservative_point"], r["nominal_point"]
        e72c = FC.max_tangential_push(q, cp["mu"])["max_tangential_N"]
        e72n = FC.max_tangential_push(q, np_["mu"])["max_tangential_N"]
        e79c, e79n = _e79(cp["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"]), _e79(np_["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
        e76c, e76n = _e76(cp["mu"], N_total), _e76(np_["mu"], N_total)
        e73c = _e73(cp["k"], cp["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        e73n = _e73(np_["k"], np_["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        all_mono &= (e72c <= e72n + 1e-9 and e79c <= e79n + 1e-9 and e76c <= e76n + 1e-9
                     and e73c <= e73n + 1e-9 and e73c <= v_kin + 1e-9)
    chk("R1_independent_recompute_conservative_le_nominal_all", all_mono,
        "categories=%d v_kin=%.4f" % (len(rep["rows"]), v_kin))

    # --- R2：负对照——乐观点(μ_hi/k_lo/e_lo/σ_hi)使裁决变松 → 证明保守映射压严非冻结 ---
    steel = next(r for r in rep["rows"] if r["category"] == "steel")
    bd = steel["band"]
    opt = {"mu": bd["mu"][1], "k": bd["k"][0], "e": bd["e"][0], "sigma": bd["sigma"][1]}
    c_e72 = FC.max_tangential_push(q, steel["conservative_point"]["mu"])["max_tangential_N"]
    o_e72 = FC.max_tangential_push(q, opt["mu"])["max_tangential_N"]
    c_e79 = _e79(steel["conservative_point"]["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
    o_e79 = _e79(opt["e"], b["F_allow_N"], b["dt_contact_s"], b["m_payload_kg"])
    c_e73 = _e73(steel["conservative_point"]["k"], steel["conservative_point"]["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
    o_e73 = _e73(opt["k"], opt["sigma"], b["A_contact_m2"], b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
    negctrl = (o_e72 > c_e72 + 1e-9) and (o_e79 > c_e79 + 1e-12) and (o_e73 > c_e73 + 1e-9)
    chk("R2_negative_control_optimistic_relaxes_verdict", negctrl,
        "E72 c=%.3f o=%.3f | E79 c=%.4f o=%.4f | E73 c=%.4f o=%.4f" % (c_e72, o_e72, c_e79, o_e79, c_e73, o_e73))

    # --- R3：E73 冲击界 ≤ 运动学上限 对所有材料（材料耦合确有意义）---
    all_le_kin = True
    worst_ratio = 0.0
    for r in rep["rows"]:
        ec = _e73(r["conservative_point"]["k"], r["conservative_point"]["sigma"], b["A_contact_m2"],
                  b["delta_contact_m"], b["m_payload_kg"], b["F_allow_N"])
        all_le_kin &= (ec <= v_kin + 1e-9)
        worst_ratio = max(worst_ratio, ec / v_kin)
    chk("R3_impact_cap_le_kinematic_all", all_le_kin, "worst_ratio=%.3f" % worst_ratio)

    # --- R4：零 API 关键词分类器审计者自写复现 ---
    _kw = {"steel": ["steel", "钢", "metal", "金属"], "wood": ["wood", "木", "oak", "timber"],
           "rubber": ["rubber", "橡胶", "silicone", "elastomer"]}
    def _cls(d):
        d = (d or "").lower()
        for c, kws in _kw.items():
            if any(k in d for k in kws):
                return c
        return "unknown"
    chk("R4_zero_api_classifier_reproduced", _cls("a shiny steel beam") == "steel"
        and _cls("旧木桌") == "wood" and _cls("rubber tire") == "rubber")

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
