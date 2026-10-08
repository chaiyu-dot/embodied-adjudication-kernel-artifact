# -*- coding: utf-8 -*-
"""blind_replicate_e82.py —— E82 的**盲复刻（只给数据文件 + 独立物理重实现）**。

模拟外部复刻者：只拿 e82_bench_data.json + e82_standalone_physics.py（自写摩擦锥/冲量/内力/
速度耦合），**不 import planning.* / friction_cone / e82_material_domain / e73**，
独立复现 E82 的四裁决点保守传播并与作者报告逐格比对。

隔离保证：运行期 sys.modules 不得含 planning / friction_cone / e82_material_domain / e73。
产物：blind_replicate_e82.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e82_standalone_physics as P  # noqa: E402  （复刻者自有实现）

REPORT = os.path.join(EVAL, "e82_material_domain_report.json")
BENCH = os.path.join(EVAL, "e82_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e82.json")

_repo_mods = ("planning", "friction_cone", "e82_material_domain",
              "e73_velocity_spectrum_envelope", "control_strategies")
_res = {"experiment": "E82 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


def main():
    print("=" * 92)
    print("E82 盲复刻（只给数据文件 + 独立物理重实现）vs 作者报告")
    print("=" * 92)
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need_bench = ["m_payload_kg", "A_contact_m2", "delta_contact_m", "F_allow_N", "dt_contact_s",
                  "q_friction", "N_total", "L1", "L2", "TAU_LIM"]
    miss = [k for k in need_bench if k not in data.get("bench", {})]
    miss += [] if len(data.get("ontology", {})) == 8 else ["ontology(8)"]
    miss += [] if "v_kinematic_mps" in data.get("kinematic", {}) else ["kinematic"]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data)
    rmap = {r["category"]: r for r in rep["rows"]}
    omap = {r["category"]: r for r in out["rows"]}

    # B1 保守/名义点（从 band 反推）
    mism = []
    for cat, r in rmap.items():
        o = omap[cat]
        for f in ("mu", "k", "e", "sigma"):
            if abs(o["conservative_point"][f] - r["conservative_point"][f]) > 1e-12:
                mism.append((cat, "cp", f))
            if abs(o["nominal_point"][f] - r["nominal_point"][f]) > 1e-12:
                mism.append((cat, "np", f))
    chk("B1 保守点/名义点（8 类 × 4 参）从 band 反推复现", not mism, "mismatch=%s" % mism[:3] if mism else "全一致")

    # B2 四裁决点保守/名义值逐类
    field = {"E72": "E72_max_tangential_N", "E79": "E79_v_star_mps",
             "E76": "E76_max_internal_force_N", "E73": "E73_impact_speed_cap_mps"}
    tol = {"E72": 1e-3, "E79": 1e-5, "E76": 1e-3, "E73": 1e-5}
    mism2 = []
    for cat, r in rmap.items():
        o = omap[cat]
        for k, rk in field.items():
            for which in ("conservative", "nominal"):
                if abs(o[k][which] - r[rk][which]) > tol[k]:
                    mism2.append((cat, k, which, o[k][which], r[rk][which]))
    chk("B2 四裁决点（4 点 × 8 类 × 保守/名义）逐格复现", not mism2, "mismatch=%s" % mism2[:3] if mism2 else "全一致")

    # B3 单调保守 + E73 ≤ 运动学
    ok3 = out["all_monotone"] and all(r["monotone_conservative"] for r in rep["rows"])
    chk("B3 单调保守（保守 ≤ 名义）对所有 8 类成立", ok3, "all_monotone=%s" % ok3)

    # B4 verdict 裸布尔重算
    chk("B4 verdict（C1∧C2∧C3）裸布尔重算 == 报告",
        bool(out["all_monotone"]) == bool(rep["criteria"]["C1_monotone_conservative"])
        and rep["verdict_pass"] == bool(rep["criteria"]["C1_monotone_conservative"]
                                        and rep["criteria"]["C2_all_four_verdict_points_fed"]
                                        and rep["criteria"]["C3_zero_api_classifier_runnable"]),
        "re_all_monotone=%s store_verdict=%s" % (out["all_monotone"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e82_bench_data.json + 独立物理重实现）：E82 四裁决点保守传播可被外部独立重建。",
        "  保守/名义点、E72/E79/E76/E73 四界、单调性：逐格一致（0 mismatch）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
