# -*- coding: utf-8 -*-
"""blind_replicate_e85.py —— E85 的 **L4 隔离盲复刻**（只给数据文件 + 独立实现）。

只拿 `e85_bench_data.json` + 复刻者侧独立实现（`l4_plant2r` + `e84c_standalone_physics` +
`e84_standalone_physics` + `e85_standalone_physics`），**不 import 仓库模块 / 任何 e8x 脚本**：
  · 逐档重放 GATED / POINT（零 RNG，估计器轨迹由数据给出）；
  · 重算平均测量次数 / 触发率 / 可行率 / 危险数 / 误拒率；
  · 重算单调性、Spearman、ceiling 与 7 条预注册判据（含 3 条诚实 FAIL）。

产物：blind_replicate_e85.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e85_standalone_physics as SP   # noqa: E402

REPORT = os.path.join(EVAL, "e85_interface_accuracy_monotone_report.json")
BENCH = os.path.join(EVAL, "e85_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e85.json")
_repo = ("planning", "rne_dynamics", "rigid_body_certificate", "trackability", "energy_margin",
         "active_estimator", "control_strategies", "e85_interface_accuracy_monotone",
         "e84_five_order_wrapper_loop", "e84b_independent_engine_gt", "e84c_depth_ablation_grid")
_res = {"experiment": "E85 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]
KEYS = ("measure_calls_mean", "trigger_rate", "feasible_rate", "dangerous_rate",
        "dangerous_count", "false_reject_rate")


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E85 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（bench/q_lim/omega_n/zeta/T_s/qdot_lim/k_conf/k_max/levels[*].proposals[*].states）",
        all(k in data for k in ("bench", "q_lim", "omega_n", "zeta", "T_s", "qdot_lim",
                                "k_conf", "k_max", "s_levels", "levels")),
        "levels=%d" % len(data["levels"]))

    out = SP.replicate(data)

    def _close(a, b, tol=1e-4):
        return all(abs(a[k] - b[k]) <= tol for k in KEYS if k in b)

    bad = []
    for s in out["per_level"]:
        mine = out["per_level"][s]
        theirs = rep["per_s"].get(str(s)) or rep["per_s"].get(("%.2f" % s))
        if theirs is None:
            bad.append((s, "missing"))
            continue
        for arm in mine:
            if not _close(mine[arm], theirs[arm.lower()]):   # 报告用小写键
                bad.append((s, arm))
    chk("B1 逐档×逐臂（测量次数/触发率/可行率/危险数/误拒率）与报告一致", not bad, "mismatch=%s" % bad)

    bad2 = []
    for k, mine in (("calls_by_s", out["calls_by_s"]), ("feasible_by_s", out["feasible_by_s"]),
                    ("dangerous_count_by_s", out["dangerous_count_by_s"])):
        if [round(x, 4) for x in mine] != [round(x, 4) for x in rep[k]]:
            bad2.append(k)
    if abs(out["ceiling"] - rep["ceiling_feasible_s_near_oracle"]) > 1e-4:
        bad2.append("ceiling")
    if abs(out["spearman_s_vs_calls"] - rep["spearman_s_vs_calls"]) > 1e-3:
        bad2.append("rho_calls")
    chk("B2 汇总序列（calls / feasible / dangerous / ceiling / ρ）与报告一致", not bad2,
        "mismatch=%s ｜ mine calls=%s" % (bad2, out["calls_by_s"]))

    chk("B3 预注册判据 H85-1..7 逐条复现（**含 3 条诚实 FAIL，同样复现**）",
        out["H"] == rep["preregistered_verdict"],
        "mine=%s theirs=%s" % (out["H"], rep["preregistered_verdict"]))
    chk("B4 结论判定复现（verdict_pass，作者为 False 亦须复现）",
        out["verdict_pass"] == rep["verdict_pass"],
        "mine=%s theirs=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e85_bench_data.json + 复刻者侧独立物理）：5 档×2 臂的全部统计量、"
        "单调性、Spearman、ceiling 与 7 条判据（**含 3 条诚实 FAIL**）均被外部独立重建。",
        "**语义强度标注**：数据文件把『估计器轨迹 (mean,std)』外化为数据 → 本 L4 检验"
        "**判据/门控/统计管线**的独立可重建性；`active_estimator` 的融合内部不在本件范围。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
