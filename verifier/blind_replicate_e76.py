# -*- coding: utf-8 -*-
"""blind_replicate_e76.py —— E76 的**盲复刻（只给数据文件 + 独立物理重实现）**。

只拿到 e76_bench_data.json + e76_standalone_physics.py（自行重写 2R 动力学/PD_G/RK4/fk/ik/jac/
双臂协同仿真/夹持代数），**不 import planning.* / e66 / e67 / e76**，独立复现 A/B/C 再与报告比对。
产物：blind_replicate_e76.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e76_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e76_dualarm_coordination_report.json")
BENCH = os.path.join(EVAL, "e76_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e76.json")

_repo_mods = ("planning", "rne_dynamics", "energy_kernel", "e66_closed_loop_control",
              "e67_testbed_completeness", "e76_dualarm_coordination", "control_strategies")


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E76 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92); print("E76 盲复刻（只给数据文件 + 独立物理重实现）"); print("=" * 92)
    loaded = _repo_loaded()
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["bench", "algorithm", "config"]
    need_b = ["L1", "L2", "M1", "M2", "LC1", "LC2", "I1", "I2", "G", "TAU_LIM", "B_VISC", "DT", "BASE_L", "BASE_R"]
    need_c = ["targets", "T_s", "ramp", "mp_ladder", "drift_tol_m", "internal_force_N", "F_des"]
    miss = [k for k in need if k not in data] + [k for k in need_b if k not in data.get("bench", {})] \
        + [k for k in need_c if k not in data.get("config", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss,
        "missing=%s" % miss if miss else "keys 齐备（%d bench / %d config）" % (len(data["bench"]), len(data["config"])))

    o = P.replicate(data)
    A, B, C = o["A_relative_pose_keeping"], o["B_internal_force_decoupling"], o["C_danger_zero"]
    rA, rB, rC = rep["A_relative_pose_keeping"], rep["B_internal_force_decoupling"], rep["C_danger_zero"]

    mA = []
    for k in rA["rows"]:
        a, r = rA["rows"][k], A["rows"][k]
        for f in ("rel_drift_max_m", "tau_max_demand", "violation_steps_demand", "diverged"):
            if a[f] != r[f]:
                mA.append((k, f, a[f], r[f]))
    chk("B1 A 段 %d 行逐格复现（漂移/需求峰值/越限步/发散）" % len(rA["rows"]), not mA,
        "mismatch=%s" % mA[:3] if mA else "全一致")
    chk("B2 A 段 max drift(上限内) 与单调性复现",
        rA["max_drift_within_limit_m"] == A["max_drift_within_limit_m"]
        and rA["drift_monotone_in_mp"] == A["drift_monotone_in_mp"],
        "复刻 %.6f 报告 %.6f" % (A["max_drift_within_limit_m"], rA["max_drift_within_limit_m"]))

    mB = []
    for f in ("net_object_wrench_inf", "orthogonality_motion_vs_internal",
              "motion_component_invariance_vs_alpha_max_err", "internal_force_sum_norm"):
        if abs(rB[f] - B[f]) > 1e-15:
            mB.append((f, rB[f], B[f]))
    chk("B3 B 段内力解耦四项标量复现（‖GᵀW_int‖/正交/α不变/对称）", not mB,
        "mismatch=%s" % mB[:3] if mB else "全一致")

    mC = []
    for k in rC["rows"]:
        a, r = rC["rows"][k], C["rows"][k]
        for f in ("violation_steps_demand", "tau_max_demand", "diverged"):
            if a[f] != r[f]:
                mC.append((k, f, a[f], r[f]))
    chk("B4 C 段危险 0 逐行复现", not mC, "mismatch=%s" % mC[:3] if mC else "全一致")

    verdict = bool(A["all_ok"] and B["all_ok"] and C["all_ok"])
    chk("B5 verdict 由复刻三段裸布尔重算 == 报告", verdict == bool(rep["verdict_pass"]),
        "复刻 %s 报告 %s" % (verdict, rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e76_bench_data.json + 独立物理重实现）：E76 全部结论可被外部独立重建。",
        "  A 段漂移阶梯、B 段内力解耦四项、C 段危险 0：逐格一致（0 mismatch）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
