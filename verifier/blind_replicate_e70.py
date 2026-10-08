# -*- coding: utf-8 -*-
"""blind_replicate_e70.py —— E70 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e70_bench_data.json + e70_standalone_physics.py（自写 2R 核 + 端到端残差环），
**不 import planning.* / control_strategies / e66_closed_loop_control / e70_task_residual_loop**，
独立复现 E70 的到点率 / 安全不变量 / ⊘ 处置 / 夹取对照 / 迭代预算敏感性，并与报告比对。

产物：blind_replicate_e70.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e70_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e70_task_residual_loop_report.json")
BENCH = os.path.join(EVAL, "e70_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e70.json")
_repo = ("planning", "rne_dynamics", "control_strategies", "e66_closed_loop_control",
         "e70_task_residual_loop", "energy_kernel")
_res = {"experiment": "E70 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E70 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    miss = [k for k in ("bench", "goals_A", "coarse_qref_A", "goals_B", "coarse_qref_B", "clamp_qref_B")
            if k not in data]
    chk("B0b 数据文件自足（目标集 + 粗提案/夹取参考已外化）", not miss,
        "missing=%s" % miss if miss else "全齐 n A=%d B=%d" % (len(data["goals_A"]), len(data["goals_B"])))
    out = P.replicate(data)
    ra = rep["SET_A_admissible"]; rb = rep["SET_B_inadmissible"]

    chk("B1 SET_A COARSE 到点率=0 & 误差中位复现（0.255 m）",
        abs(out["SET_A_coarse"]["reach_1cm_rate"] - ra["coarse"]["reach_1cm_rate"]) < 1e-9
        and abs(out["SET_A_coarse"]["median_err_m"] - ra["coarse"]["median_err_m"]) < 1e-3,
        "reach re=%.4f store=%.4f | med re=%.4f store=%.4f"
        % (out["SET_A_coarse"]["reach_1cm_rate"], ra["coarse"]["reach_1cm_rate"],
           out["SET_A_coarse"]["median_err_m"], ra["coarse"]["median_err_m"]))
    chk("B2 SET_A RESID 到点率复现（0.9375）& 误差中位（0.0043 m）& 状态计数",
        abs(out["SET_A_resid"]["reach_1cm_rate"] - ra["resid"]["reach_1cm_rate"]) < 1e-9
        and abs(out["SET_A_resid"]["median_err_m"] - ra["resid"]["median_err_m"]) < 1e-3
        and out["SET_A_resid"]["status_counts"] == ra["resid"]["status_counts"],
        "reach re=%.4f store=%.4f | status=%s vs %s"
        % (out["SET_A_resid"]["reach_1cm_rate"], ra["resid"]["reach_1cm_rate"],
           out["SET_A_resid"]["status_counts"], ra["resid"]["status_counts"]))
    chk("B3 SET_B（不可达）RESID 到点率=0 & 状态 {budget_exhausted:6,reject:9} 复现",
        abs(out["SET_B_resid"]["reach_1cm_rate"] - rb["resid"]["reach_1cm_rate"]) < 1e-9
        and out["SET_B_resid"]["status_counts"] == rb["resid"]["status_counts"],
        "reach re=%.4f store=%.4f | status=%s vs %s"
        % (out["SET_B_resid"]["reach_1cm_rate"], rb["resid"]["reach_1cm_rate"],
           out["SET_B_resid"]["status_counts"], rb["resid"]["status_counts"]))
    chk("B4 CLAMP 对照：到点率=0 & 误差中位复现（0.2803 m）",
        abs(out["SET_B_clamp"]["reach_1cm_rate"] - rb["clamp"]["reach_1cm_rate"]) < 1e-9
        and abs(out["SET_B_clamp"]["median_err_m"] - rb["clamp"]["median_err_m"]) < 1e-3,
        "reach re=%.4f store=%.4f | med re=%.4f store=%.4f"
        % (out["SET_B_clamp"]["reach_1cm_rate"], rb["clamp"]["reach_1cm_rate"],
           out["SET_B_clamp"]["median_err_m"], rb["clamp"]["median_err_m"]))
    chk("B5 V1 配对：仅 COARSE 到点=%d ｜ 仅 RESID 到点=%d 复现" % (rep["V1_paired"]["coarse_only"],
                                                              rep["V1_paired"]["resid_only"]),
        out["V1_paired"]["coarse_only"] == rep["V1_paired"]["coarse_only"]
        and out["V1_paired"]["resid_only"] == rep["V1_paired"]["resid_only"],
        "re=(%d,%d) store=(%d,%d)" % (out["V1_paired"]["coarse_only"], out["V1_paired"]["resid_only"],
                                      rep["V1_paired"]["coarse_only"], rep["V1_paired"]["resid_only"]))
    chk("B6 安全不变量：执行越限=0（门控构造保证）& 需求越限=1119 复现",
        out["V2"]["exec_violation_steps_total"] == 0
        and out["V2"]["demand_violation_steps_total"] == rep["V2_safety_invariant"]["demand_violation_steps_total"],
        "exec re=%d store=%d | demand re=%d store=%d"
        % (out["V2"]["exec_violation_steps_total"], rep["V2_safety_invariant"]["exec_violation_steps_total"],
           out["V2"]["demand_violation_steps_total"], rep["V2_safety_invariant"]["demand_violation_steps_total"]))
    chk("B7 V3/V4 判定复现（不可达判 reject/⊘；夹取非正确处置）",
        out["V3_ok"] == rep["V3_ok"] and out["V4_ok"] == rep["V4_ok"],
        "V3 re=%s store=%s | V4 re=%s store=%s" % (out["V3_ok"], rep["V3_ok"], out["V4_ok"], rep["V4_ok"]))
    v5 = rep["V5_kmax_sensitivity"]
    chk("B8 V5 迭代预算 16：到点率不变（0.9375）→ 残余失败非预算问题 复现",
        abs(out["V5_kmax16"]["reach_1cm_rate"] - v5["reach_1cm_rate"]) < 1e-9
        and abs(out["V5_kmax16"]["median_err_m"] - v5["median_err_m"]) < 1e-4,
        "reach re=%.4f store=%.4f | med_err re=%.6f store=%.6f"
        % (out["V5_kmax16"]["reach_1cm_rate"], v5["reach_1cm_rate"],
           out["V5_kmax16"]["median_err_m"], v5["median_err_m"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e70_bench_data.json + 独立实现）：E70 端到端残差环的"
                        "到点率/安全不变量/⊘ 处置/夹取对照/迭代预算敏感性可被外部独立重建。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
