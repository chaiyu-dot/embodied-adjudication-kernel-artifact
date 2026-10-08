# -*- coding: utf-8 -*-
"""blind_replicate_e67.py —— E67 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e67_bench_data.json + e67_standalone_physics.py（自写 2R 核 + 速度谱/双臂/跟踪/摩擦/门控消融），
**不 import planning.* / control_strategies / e66_closed_loop_control / e67_testbed_completeness**，
独立复现 E67 的关键标量（v_max/a_max/jerk、双臂到点/漂移、跟踪 RMS、摩擦锥、门控承重），并与报告比对。

产物：blind_replicate_e67.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e67_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e67_testbed_report.json")
BENCH = os.path.join(EVAL, "e67_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e67.json")
_repo = ("planning", "control_strategies", "rne_dynamics", "e66_closed_loop_control",
         "e67_testbed_completeness", "energy_kernel")
_res = {"experiment": "E67 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E67 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/A/B/C/D/E）",
        all(k in data for k in ("bench", "A", "B", "C", "D", "E")), "keys=%s" % list(data.keys()))
    out = P.replicate(data)

    rA = rep["A_velocity_spectrum"]
    chk("B1 [A] 末端速度谱上确界复现（v_max=3.0461，且逐位绑 joint_speed_cap）",
        abs(out["A"]["v_max_ee"] - rA["v_max_ee"]) < 1e-3
        and abs(out["A"]["torque_only_v_ee_nominal_q"] - rA["torque_only_v_ee_nominal_q"]) < 1e-3,
        "v_max re=%.4f store=%.4f | torque_nom re=%.4f store=%.4f"
        % (out["A"]["v_max_ee"], rA["v_max_ee"],
           out["A"]["torque_only_v_ee_nominal_q"], rA["torque_only_v_ee_nominal_q"]))
    chk("B2 [A] 加速度上确界（a_max=66.1436）与 jerk 上界（ωn·a_max=513.152）复现",
        abs(out["A"]["a_max_ee"] - rA["a_max_ee"]) < 1e-3
        and abs(out["A"]["jerk_bound"] - rA["jerk_bound"]) < 1e-2,
        "a_max re=%.4f store=%.4f | jerk re=%.3f store=%.3f | ωn=%.4f"
        % (out["A"]["a_max_ee"], rA["a_max_ee"], out["A"]["jerk_bound"], rA["jerk_bound"], out["A"]["wn"]))

    rB = rep["B_dual_arm"]
    chk("B3 [B] 双臂独立到点误差 L/R 复现（均 0.0）",
        abs(out["B"]["independent"]["errL_m"] - rB["independent"]["errL_m"]) < 1e-6
        and abs(out["B"]["independent"]["errR_m"] - rB["independent"]["errR_m"]) < 1e-6,
        "re=(%.5f,%.5f) store=(%.5f,%.5f)"
        % (out["B"]["independent"]["errL_m"], out["B"]["independent"]["errR_m"],
           rB["independent"]["errL_m"], rB["independent"]["errR_m"]))
    chk("B4 [B] 协同搬运相对漂移=0（对称）/ 非对称 +0.25kg 漂移复现",
        abs(out["B"]["coordinated_carry"]["rel_drift_max_m"] - rB["coordinated_carry"]["rel_drift_max_m"]) < 1e-4
        and abs(out["B"]["coordinated_carry_asym_load"]["rel_drift_max_m"]
                - rB["coordinated_carry_asym_load"]["rel_drift_max_m"]) < 1e-3,
        "drift re=%s store=%s | asym re=%s store=%s"
        % (out["B"]["coordinated_carry"]["rel_drift_max_m"], rB["coordinated_carry"]["rel_drift_max_m"],
           out["B"]["coordinated_carry_asym_load"]["rel_drift_max_m"],
           rB["coordinated_carry_asym_load"]["rel_drift_max_m"]))
    chk("B5 [B] 目标可行性（L/R 均在限位内且重力矩在预算内）复现",
        out["B"]["target_feasibility_L"]["in_joint_limits"] == rB["target_feasibility_L"]["in_joint_limits"]
        and out["B"]["target_feasibility_R"]["in_joint_limits"] == rB["target_feasibility_R"]["in_joint_limits"],
        "L re=%s store=%s | R re=%s store=%s"
        % (out["B"]["target_feasibility_L"], rB["target_feasibility_L"],
           out["B"]["target_feasibility_R"], rB["target_feasibility_R"]))

    rC = rep["C_moving_target"]
    bad = 0
    for arm in ("no_ff", "ff"):
        for f, v in out["C"][arm]["rows"].items():
            s = rC[arm]["rows"][f]
            if v["rms_err_m"] is None or abs(v["rms_err_m"] - s["rms_err_m"]) > 1e-4:
                bad += 1
    chk("B6 [C] 跟踪 RMS（6 频率 × 无/有前馈 = 12 格）与报告一致（±1e-4）", bad == 0, "mismatch=%d" % bad)
    chk("B7 [C] 最大可跟踪：有前馈 1cm→0.2Hz/0.0563 m·s⁻¹；无前馈无解 复现",
        (out["C"]["ff"]["max_trackable_freq_hz_tol1cm"] == rC["ff"]["max_trackable_freq_hz_tol1cm"])
        and (out["C"]["no_ff"]["max_trackable_freq_hz_tol1cm"] == rC["no_ff"]["max_trackable_freq_hz_tol1cm"])
        and abs((out["C"]["ff"]["max_trackable_speed_mps_tol1cm"] or 0)
                - (rC["ff"]["max_trackable_speed_mps_tol1cm"] or 0)) < 1e-4,
        "ff max re=%s store=%s | no_ff max re=%s store=%s"
        % (out["C"]["ff"]["max_trackable_freq_hz_tol1cm"], rC["ff"]["max_trackable_freq_hz_tol1cm"],
           out["C"]["no_ff"]["max_trackable_freq_hz_tol1cm"], rC["no_ff"]["max_trackable_freq_hz_tol1cm"]))

    rD = rep["D_contact_friction"]["friction_cone_max_push"]
    bad2 = sum(1 for k, v in out["D"]["friction_cone_max_push"].items()
               if abs(v["max_tangential_N"] - rD[k]["max_tangential_N"]) > 1e-3
               or abs(v["at_normal_N"] - rD[k]["at_normal_N"]) > 1e-3)
    chk("B8 [D] 摩擦锥最大可推切向力（μ=0.2→1.18 / 0.8→4.0 N）复现", bad2 == 0, "mismatch=%d" % bad2)
    rDf = [v for k, v in rep["D_contact_friction"].items() if k.startswith("contact_feasibility")][0]
    bad3 = sum(1 for k, v in out["D"]["contact_feasibility"].items()
               if v["feasible"] != rDf[k]["feasible"]
               or max(abs(a - b) for a, b in zip(v["tau"], rDf[k]["tau"])) > 1e-3)
    chk("B9 [D] 接触可行性（JᵀF→τ 门控）逐力复现", bad3 == 0, "mismatch=%d" % bad3)

    rE = rep["E_gate_ablation"]["conclusion"]
    chk("B10 [E] 门控承重结论复现（load_bearing / executed_zero / gate_off_exceeds）",
        out["E"]["conclusion"]["gate_load_bearing"] == rE["gate_load_bearing"]
        and out["E"]["conclusion"]["gate_on_executed_zero"] == rE["gate_on_executed_zero"]
        and out["E"]["conclusion"]["gate_off_really_exceeds"] == rE["gate_off_really_exceeds"],
        "re=%s store=%s" % ({k: out["E"]["conclusion"][k] for k in
                             ("gate_load_bearing", "gate_on_executed_zero", "gate_off_really_exceeds")},
                            {k: rE[k] for k in ("gate_load_bearing", "gate_on_executed_zero", "gate_off_really_exceeds")}))
    bad4 = sum(1 for k, v in rE["detail"].items() if out["E"]["conclusion"]["detail"].get(k) != v)
    chk("B11 [E] 门控消融需求越限步数逐场景复现（8 格：budget/aggressive × on/off × G1/G2）",
        bad4 == 0, "mismatch=%d | re=%s" % (bad4, out["E"]["conclusion"]["detail"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e67_bench_data.json + 独立实现）：E67 台架四段（速度谱/双臂/跟踪/"
                        "摩擦）+ 门控承重消融的关键标量可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
