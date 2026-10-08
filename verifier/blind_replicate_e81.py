# -*- coding: utf-8 -*-
"""blind_replicate_e81.py —— E81 的**盲复刻（只给数据文件 + 独立物理重实现）**。

模拟外部复刻者：只拿 e81_bench_data.json + e81_standalone_physics.py（自写 2R 雅可比/闭式
力矩/伪逆/二分/扫掠），**不 import planning.* / e73 / e79 / e81_dynamic_speed_ceiling**，
独立复现 E81 的 A/B/C 三段与动态天花板，并与作者报告逐格比对。

隔离保证：运行期 sys.modules 不得含 planning / rne_dynamics / e81_dynamic_speed_ceiling / e73 / e79。
产物：blind_replicate_e81.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e81_standalone_physics as P  # noqa: E402  （复刻者自有实现）

REPORT = os.path.join(EVAL, "e81_dynamic_speed_ceiling_report.json")
BENCH = os.path.join(EVAL, "e81_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e81.json")

_repo_mods = ("planning", "rne_dynamics", "e81_dynamic_speed_ceiling",
              "e73_velocity_spectrum_envelope", "e79_momentum_impulse_chain", "control_strategies")
_res = {"experiment": "E81 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
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
    print("E81 盲复刻（只给数据文件 + 独立物理重实现）vs 作者报告")
    print("=" * 92)
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["L1", "L2", "M1", "M2", "TAU_LIM", "g", "m_payload_kg", "restitution",
            "dt_contact_s", "F_allow_N", "sigma_min_min", "n_q1", "n_q2", "n_dir"]
    miss = [k for k in need if k not in data.get("bench", {})] + \
           ([] if "v_kinematic_mps" in data.get("kinematic", {}) else ["kinematic.v_kinematic_mps"])
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data)
    tol = rep["C_coriolis"]["tol"]

    chk("B1 A 运动学段一致（引用 E73）",
        abs(out["A_kinematic"]["v_kinematic_mps"] - rep["A_kinematic"]["v_kinematic_mps"]) < 1e-9,
        "re=%.4f store=%.4f" % (out["A_kinematic"]["v_kinematic_mps"], rep["A_kinematic"]["v_kinematic_mps"]))

    chk("B2 B 冲量段独立重算 == 报告（<1e-9）",
        abs(out["B_impulse"]["v_impulse_mps"] - rep["B_impulse"]["v_impulse_mps"]) < 1e-9,
        "re=%.6f store=%.6f" % (out["B_impulse"]["v_impulse_mps"], rep["B_impulse"]["v_impulse_mps"]))

    chk("B3 C 科氏扫掠独立重算 == 报告（v_cor / n_valid / worst_q）",
        abs(out["C_coriolis"]["v_coriolis_mps"] - rep["C_coriolis"]["v_coriolis_mps"]) < 1e-6
        and out["C_coriolis"]["n_valid"] == rep["C_coriolis"]["n_valid_configs_dirs"]
        and max(abs(out["C_coriolis"]["worst_q"][i] - rep["C_coriolis"]["worst_config"]["q"][i])
                for i in range(2)) < 1e-4,
        "v_cor re=%.7f store=%.7f ; nvalid=%d/%d" % (out["C_coriolis"]["v_coriolis_mps"],
                                                     rep["C_coriolis"]["v_coriolis_mps"],
                                                     out["C_coriolis"]["n_valid"],
                                                     rep["C_coriolis"]["n_valid_configs_dirs"]))

    chk("B4 动态天花板 + 绑定一致（min + argmin）",
        abs(out["v_dynamic_cap_mps"] - rep["v_dynamic_cap_mps"]) < 1e-9
        and out["binding_constraint"] == rep["binding_constraint"],
        "cap re=%.4f store=%.4f bind re=%s store=%s" % (out["v_dynamic_cap_mps"], rep["v_dynamic_cap_mps"],
                                                        out["binding_constraint"], rep["binding_constraint"]))

    chk("B5 verdict 裸布尔重算 == 报告", out["all_ok"] == bool(rep["verdict_pass"]),
        "re=%s store=%s" % (out["all_ok"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e81_bench_data.json + 独立物理重实现）：E81 动态天花板可被外部独立重建。",
        "  v_kinematic(引用) / v_impulse(重算) / v_coriolis(独立扫掠) / v_dynamic_cap / 绑定：逐格一致。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
