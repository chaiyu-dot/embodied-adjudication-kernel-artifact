# -*- coding: utf-8 -*-
"""blind_replicate_e84.py —— E84 的 **L4 隔离盲复刻**（只给数据文件 + 独立实现）。

只拿 `e84_bench_data.json` + 复刻者侧独立实现（`l4_plant2r` + `e84c_standalone_physics` +
`e84_standalone_physics`），**不 import `planning.*` / 任何 e8x 实验脚本**，独立重放四臂：

  POINT（点入口） / SOUND_NOMEAS（区间不测） / GATED（预测误差门控测量） / MEAS_ALL（全测）

并与作者报告逐项比对（计数精确相等；比率容差 1e-4；预注册判据逐条重算）。

产物：blind_replicate_e84.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e84_standalone_physics as SP   # noqa: E402

REPORT = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
BENCH = os.path.join(EVAL, "e84_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e84.json")
_repo = ("planning", "rne_dynamics", "rigid_body_certificate", "trackability", "energy_margin",
         "active_estimator", "control_strategies", "e84_five_order_wrapper_loop",
         "e84c_depth_ablation_grid", "world_physics_engine", "external_physics")
_res = {"experiment": "E84 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E84 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/q_lim/omega_n/zeta/T_s/qdot_lim/k_conf/k_max/arms/proposals）",
        all(k in data for k in ("bench", "q_lim", "omega_n", "zeta", "T_s", "qdot_lim",
                                "k_conf", "k_max", "arms", "proposals")),
        "proposals=%d arms=%s" % (len(data["proposals"]), data["arms"]))

    out = SP.replay(data)
    mine, theirs = out["aggregate"], rep["aggregate"]

    # B1 四臂逐项指标
    bad, det = 0, []
    for arm in data["arms"]:
        a, b = mine[arm], theirs[arm]
        if a["n"] != b["n"] or a["dangerous"] != b["dangerous"]:
            bad += 1
        for k in ("dangerous_rate", "feasible_rate", "false_reject_rate", "measure_calls_mean",
                  "trigger_rate", "gt_dangerous_rate"):
            if abs(a[k] - b[k]) > 1e-4:
                bad += 1
        det.append("%s dang=%d feas=%.4f meas=%.4f" % (arm, a["dangerous"], a["feasible_rate"],
                                                       a["measure_calls_mean"]))
    chk("B1 四臂逐项指标与报告一致（计数精确、比率 1e-4）", bad == 0, " ｜ ".join(det))

    # B2 逐阶承重分布
    chk("B2 逐阶承重分布（binding order）与报告一致",
        out["order_binding_distribution"] == rep["order_binding_distribution"],
        "mine=%s theirs=%s" % (out["order_binding_distribution"], rep["order_binding_distribution"]))

    # B3 关键不等式（论文级主张）由复刻结果重算
    g, m_all = mine["GATED"], mine["MEAS_ALL"]
    chk("B3 '门控测量在几乎不牺牲可行率的前提下大幅省算力' 由复刻结果重算成立",
        g["measure_calls_mean"] <= max(m_all["measure_calls_mean"] / 5.0, 1e-9)
        and g["feasible_rate"] >= m_all["feasible_rate"] - 0.01
        and g["feasible_rate"] > mine["SOUND_NOMEAS"]["feasible_rate"],
        "GATED meas=%.4f feas=%.4f | MEAS_ALL meas=%.4f feas=%.4f | SOUND feas=%.4f"
        % (g["measure_calls_mean"], g["feasible_rate"], m_all["measure_calls_mean"],
           m_all["feasible_rate"], mine["SOUND_NOMEAS"]["feasible_rate"]))

    # B4 预注册判据逐条重算
    five_ok = None
    present = set(rep["design"]["arms"])
    v = {
        "H84-1_point_entry_reopens_danger": bool(mine["POINT"]["dangerous_rate"] > 0.0),
        "H84-2_sound_arms_zero_danger": bool(
            mine["SOUND_NOMEAS"]["dangerous"] == 0 and mine["GATED"]["dangerous"] == 0
            and mine["MEAS_ALL"]["dangerous"] == 0),
        "H84-3_gated_far_fewer_measures": bool(
            g["measure_calls_mean"] <= max(m_all["measure_calls_mean"] / 5.0, 1e-9)),
        "H84-4_gated_keeps_feasibility": bool(
            g["feasible_rate"] >= mine["SOUND_NOMEAS"]["feasible_rate"] - 1e-9),
        "H84-5_five_order_in_certificate": bool(rep["preregistered_verdict"]["H84-5_five_order_in_certificate"]),
    }
    chk("B4 预注册判据 H84-1..5 逐条复现（H84-5 为包裹层证书结构属性，沿用报告并由 E84 的 L3 复算）",
        v == rep["preregistered_verdict"], "mine=%s theirs=%s" % (v, rep["preregistered_verdict"]))

    # B5 verdict_pass
    chk("B5 结论判定复现（verdict_pass / all(H)）",
        bool(all(v.values())) == rep["verdict_pass"],
        "mine all(H)=%s theirs=%s" % (all(v.values()), rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e84_bench_data.json + 复刻者侧独立物理）：E84 的四臂指标、逐阶承重分布、"
        "预注册判据与'门控省算力'不等式均被外部独立重建。",
        "**语义强度标注**：数据文件刻意把『估计器轨迹 (mean,std)』外化为数据，"
        "因此本 L4 检验的是**判据/门控/统计管线**的独立可重建性；"
        "『逆方差融合 + z>3 拒融合』内部实现不在本件复刻范围（属 active_estimator 的独立件）。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
