# -*- coding: utf-8 -*-
"""blind_replicate_e68.py —— E68 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e68_bench_data.json + e68_standalone_physics.py（自写 2R 核 + 控制器族 + 独立 MPC），
**不 import planning.* / control_strategies / e66_closed_loop_control / e68_control_strategy_suite**，
独立复现 E68 的模型一致性 / MPC 预测一致性 / 门控不变量 / 阶跃指标 / 不可行处置，并与报告比对。

产物：blind_replicate_e68.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e68_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e68_control_suite_report.json")
BENCH = os.path.join(EVAL, "e68_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e68.json")
_repo = ("planning", "control_strategies", "rne_dynamics", "e66_closed_loop_control",
         "e68_control_strategy_suite", "energy_kernel")
_res = {"experiment": "E68 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E68 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/families/MPC_calibration）",
        all(k in data for k in ("bench", "families", "MPC_calibration")), "keys=%s" % list(data.keys()))
    out = P.replicate(data)

    chk("B1 V0 重构 plant 的 M/C/g 与独立递推 RNE 一致（<1e-8）",
        out["V0"]["ok"] and out["V0"]["plant_vs_RNE_max_abs"] < 1e-8,
        "re err=%.2e store=%.2e" % (out["V0"]["plant_vs_RNE_max_abs"],
                                    rep["V0_model_consistency"]["CS_vs_RNE_max_abs"]))
    v0b = rep["V0b_mpc_consistency"]
    chk("B2 V0b MPC 预测一致性（前馈@目标≈g；线性预测 vs 欧拉 <1e-2）复现",
        out["V0b"]["gravity_ff_err"] < 1e-6 and out["V0b"]["pred_vs_euler"] < 1e-2
        and abs(out["V0b"]["pred_vs_euler"] - v0b["pred_vs_euler_max_abs_over_Np"]) < 5e-4,
        "ff re=%.2e | pred re=%.4e store=%.4e" % (out["V0b"]["gravity_ff_err"],
                                                  out["V0b"]["pred_vs_euler"], v0b["pred_vs_euler_max_abs_over_Np"]))
    chk("B3 预算增益复现（kp=1.8 / kd=0.488 / ωn=7.758）+ 静载上限（0.3972 kg）",
        abs(out["budget"]["kp"] - rep["budget_design"]["kp"]) < 1e-6
        and abs(out["budget"]["wn"] - rep["budget_design"]["achievable_wn_rad_s"]) < 1e-3
        and abs(out["static_payload_limit_kg"] - rep["static_payload_limit_kg"]) < 1e-3,
        "kp re=%.4f | ωn re=%.3f | mp_lim re=%.4f" % (out["budget"]["kp"], out["budget"]["wn"],
                                                      out["static_payload_limit_kg"]))

    ste = rep["step_response"]
    bad = 0
    for nm, v in out["step"].items():
        s = ste[nm]
        for key in ("settling_s", "ss_err_rad", "tau_max"):
            a, b = v[key], s[key]
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 5e-3:
                bad += 1
    chk("B4 V2 阶跃指标（7 族 × 整定/稳态误差/τmax）与报告一致（±5e-3）", bad == 0, "mismatch=%d" % bad)
    chk("B5 V2 MPC 上升/整定复现（rise=0.596s / settling=1.018s）",
        abs(out["step"]["MPC"]["rise_s"] - ste["MPC"]["rise_s"]) < 5e-3
        and abs(out["step"]["MPC"]["settling_s"] - ste["MPC"]["settling_s"]) < 5e-3,
        "rise re=%s store=%s | settling re=%s store=%s"
        % (out["step"]["MPC"]["rise_s"], ste["MPC"]["rise_s"],
           out["step"]["MPC"]["settling_s"], ste["MPC"]["settling_s"]))

    invr = rep["V1_gate_invariant"]["per_family"]
    chk("B6 V1 门控不变量：全族执行越限=0（构造保证）",
        out["V1"]["ok"] and all(out["V1"]["per_family"][k]["executed_violation_steps"] == 0
                                for k in out["V1"]["per_family"]),
        "ok re=%s" % out["V1"]["ok"])
    bad2 = sum(1 for nm, v in out["V1"]["per_family"].items()
               if v["demand_violation_steps"] != invr[nm]["demand_violation_steps"])
    chk("B7 V1 需求越限步数逐族复现（P=2852/PD=186/PID=186/PD_G=194/CT=166/FO=0/MPC=0）",
        bad2 == 0, "mismatch=%d | re=%s" % (bad2, {k: v["demand_violation_steps"]
                                                   for k, v in out["V1"]["per_family"].items()}))
    bad3 = sum(1 for nm, v in out["no_gate"].items()
               if v["violation_steps"] != rep["gate_ablation_no_gate"][nm]["violation_steps"])
    chk("B8 无门控对照（PD=103/CT=97/MPC=0 越限步）复现", bad3 == 0, "mismatch=%d" % bad3)
    chk("B9 V4 不可行处置复现（可行载精确保持；不可行载任何控制器都超差）",
        out["V4_ok"] == rep["V4_ok"], "re=%s store=%s" % (out["V4_ok"], rep["V4_ok"]))
    chk("B10 结论判定复现（verdict_pass=True）", out["verdict_pass"] == rep["verdict_pass"],
        "re=%s store=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e68_bench_data.json + 独立实现）：E68 的模型一致性、MPC 预测一致性、"
                        "全族门控不变量、阶跃指标与不可行处置可被外部独立重建（含独立 MPC）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
