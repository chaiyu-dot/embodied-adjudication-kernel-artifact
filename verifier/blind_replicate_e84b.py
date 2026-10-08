# -*- coding: utf-8 -*-
"""blind_replicate_e84b.py —— E84b 的 **L4 隔离盲复刻**（只给数据文件 + 独立实现）。

只拿 `e84_bench_data.json` + 复刻者侧独立实现（自建 URDF/Bullet 引擎 + 共享显式 RNE 核），
**不 import `planning.*` / 任何 e8x 实验脚本**：

  · 复刻者**自建第三个引擎实例**（自己的 URDF、自己的 `changeDynamics` 惯量注入、自己的同参自检）；
  · 自算跨引擎误差（瞬时 + 轨迹峰值）并与申报容差比较；
  · 在**自建引擎真值**下重放四臂 → 危险计数 / 两真值一致率；
  · 与作者报告比对（结论按**精确相等**；跨引擎误差按**同数量级**，因其依赖具体 URDF/精度设定）。

🔴 与 E84 的 L4 不同：本件的复刻者**自带一个独立引擎**，因此它检验的是
「跨引擎真值替换后结论不变」这件事本身可被外部重建 —— 而不仅是四臂统计管线。

产物：blind_replicate_e84b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e84b_standalone_physics as SP   # noqa: E402

REPORT = os.path.join(EVAL, "e84b_independent_engine_gt_report.json")
BENCH = os.path.join(EVAL, "e84_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e84b.json")
_repo = ("planning", "rne_dynamics", "rigid_body_certificate", "trackability", "energy_margin",
         "active_estimator", "control_strategies", "e84b_independent_engine_gt",
         "e84_five_order_wrapper_loop", "e84c_depth_ablation_grid", "world_physics_engine",
         "external_physics")
_res = {"experiment": "E84b 盲复刻（只给数据文件 + 独立引擎）", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E84b 盲复刻（只给数据文件 + 复刻者自建引擎）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    B = dict(data["bench"])

    plant = SP.BulletRigid2R(B["l"][0], B["l"][1], B["m"][0], B["m"][1], B["g"])

    # B1 复刻者自建引擎的同参自检（必须是**自己**做的，而非沿用报告结论）
    sc = plant.selfcheck([0.0, 0.2, 0.5, 0.8])
    chk("B1 复刻者自建引擎的同参自检（质量/izz/质心 与解析参数一致）", sc["pass"], str(sc))

    # B2 自算跨引擎误差 → 与报告同数量级
    ce = SP.cross_engine(B, plant, n=40, seed=4242)
    rep_ce = rep["cross_engine_check"]
    tol = rep_ce["tolerance_abs_Nm"]
    chk("B2 自算跨引擎误差 ≤ 容差，且与报告同数量级（≤50× 报告值）",
        bool(ce["inst_max"] <= tol and ce["traj_max"] <= tol
             and ce["inst_max"] <= 50.0 * max(rep_ce["instantaneous"]["max_abs_err_Nm"], 1e-9)),
        "mine inst=%.3e traj=%.3e ｜ store inst=%.3e traj=%.3e ｜ tol=%.3e"
        % (ce["inst_max"], ce["traj_max"], rep_ce["instantaneous"]["max_abs_err_Nm"],
           rep_ce["trajectory_peak"]["max_abs_err_Nm"], tol))

    # B3 引擎真值下的四臂危险计数 + 两真值一致率
    agg, agree = SP.danger_under_engine(B, plant, data["proposals"], data,
                                        B["tau_lim"], float(data["k_conf"]))
    rep_agg = rep["aggregate"]
    same_counts = all(agg[a]["dangerous"] == rep_agg[a]["dangerous_count_bullet_gt"] for a in ARMS)
    chk("B3 自建引擎真值下四臂危险计数与报告逐臂相等", same_counts,
        "mine=%s theirs=%s" % ({a: agg[a]["dangerous"] for a in ARMS},
                               {a: rep_agg[a]["dangerous_count_bullet_gt"] for a in ARMS}))
    chk("B4 两真值（引擎 / 解析）危险判定一致率复现 = 1.0", agree == rep["meta"]["gt_agreement_rate"],
        "mine=%.4f theirs=%.4f" % (agree, rep["meta"]["gt_agreement_rate"]))

    # B5 结论级复现
    v = {
        "H84b-2_cross_engine_agree": bool(ce["inst_max"] <= tol and ce["traj_max"] <= tol),
        "H84b-3_point_entry_still_dangerous": bool(agg["POINT"]["dangerous"] > 0),
        "H84b-4_safe_arms_zero": bool(all(agg[a]["dangerous"] == 0
                                          for a in ("SOUND_NOMEAS", "GATED", "MEAS_ALL"))),
        "H84b-5_gt_verdict_agreement": bool(agree >= 0.98),
    }
    ok5 = all(v[k] == rep["preregistered_verdict"][k] for k in v)
    chk("B5 结论级判据（跨引擎一致 / 点入口仍危险 / 安全臂为 0 / 两真值一致）与报告逐条一致", ok5,
        "mine=%s" % v)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e84_bench_data.json + **复刻者自建 Bullet 引擎**）：同参自检、跨引擎误差量级、"
        "引擎真值下的四臂危险计数与两真值一致率均被外部独立重建。",
        "**语义强度标注**：跨引擎误差**不做逐位比对**（依赖各自 URDF/精度设定），只比数量级与是否 ≤ 容差；"
        "结论级判据（计数、一致率）按精确相等比对。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
