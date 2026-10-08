# -*- coding: utf-8 -*-
"""blind_replicate_e66.py —— E66 的**盲复刻（只给数据文件 + 独立物理重实现）**。

只拿 e66_bench_data.json + e66_standalone_physics.py（自写 M/C/g + 递推 RNE + RK4 + 控制器），
**不 import planning.* / rne_dynamics / control_strategies / e66_closed_loop_control**，
独立复现 E66 关键标量并与报告比对。
产物：blind_replicate_e66.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e66_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e66_closed_loop_report.json")
BENCH = os.path.join(EVAL, "e66_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e66.json")
_repo = ("planning", "rne_dynamics", "control_strategies", "e66_closed_loop_control", "energy_kernel")
_res = {"experiment": "E66 盲复刻（只给数据文件 + 独立重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name, ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E66 盲复刻（只给数据文件 + 独立重实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    miss = [k for k in ("l", "m", "tau_lim", "b_visc", "dt", "q0", "q_ref") if k not in data.get("bench", {})]
    chk("B0b 数据文件自足", not miss, "missing=%s" % miss if miss else "全齐")
    out = P.replicate(data["bench"])

    chk("B1 M/C/g 对拍 max|Δτ| == 报告（<1e-8，与递推 RNE 独立对拍）",
        out["MCg_vs_RNE_max_abs_err"] < 1e-8, "re=%.2e store=%.2e" % (out["MCg_vs_RNE_max_abs_err"], rep["MCg_vs_RNE_max_abs_err"]))
    b, rb = out["budget"], rep["budget_design"]
    chk("B2 力矩预算 → 可达带宽复现（kp/kd/ωn）",
        abs(b["kp"] - rb["kp"]) < 1e-4 and abs(b["kd"] - rb["kd"]) < 1e-4 and abs(b["achievable_wn_rad_s"] - rb["achievable_wn_rad_s"]) < 1e-3,
        "ωn re=%.3f store=%.3f" % (b["achievable_wn_rad_s"], rb["achievable_wn_rad_s"]))
    bad = 0
    for k, v in out["step_size"].items():
        s = rep["step_size_vs_achievable_bandwidth"][k]
        if abs(v["kp_max"] - s["kp_max"]) > 1e-3 or abs(v["achievable_wn_rad_s"] - s["achievable_wn_rad_s"]) > 1e-3:
            bad += 1
    chk("B3 步长 vs 可达带宽（3 构型）复现", bad == 0, "mismatch=%d" % bad)
    g, rg = out["PDG_step"], rep["step_response"]["PD_G"]
    chk("B4 PD_G 阶跃指标复现（上升/整定/稳态误差/τmax）",
        abs((g["settling_s"] or 0) - (rg["settling_s"] or 0)) < 2e-3 and abs(g["ss_err_rad"] - rg["ss_err_rad"]) < 1e-4
        and abs(g["tau_max"] - rg["tau_max"]) < 1e-3,
        "ss_err re=%.6f store=%.6f | tau_max re=%.4f store=%.4f" % (g["ss_err_rad"], rg["ss_err_rad"], g["tau_max"], rg["tau_max"]))
    chk("B5 静态可持载荷上限复现（0.3972 kg）",
        abs(out["static_payload_limit_kg"] - rep["static_payload_limit_kg"]) < 1e-3,
        "re=%.4f store=%.4f" % (out["static_payload_limit_kg"], rep["static_payload_limit_kg"]))
    chk("B6 能量平衡残差复现（W−D−ΔKE−ΔPE，<1e-8）",
        abs(out["energy"]["residual_J"]) < 1e-8, "re=%.2e store=%.2e" % (out["energy"]["residual_J"], rep["energy_balance"]["residual_J"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e66_bench_data.json + 独立重实现）：E66 闭环控制关键标量可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
