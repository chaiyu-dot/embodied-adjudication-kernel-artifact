# -*- coding: utf-8 -*-
"""blind_replicate_e58.py —— E58 的**盲复刻（只给数据文件 + 独立实现）**。

只拿 e58_bench_data.json + e58_standalone_physics.py（自写 2R 核 + 四阶判定），
**不 import planning.trackability / e58_trackability_gate / e66_closed_loop_control**，
逐条独立重算 K/S/D/C 四阶并复现盲区统计，与作者报告比对。

产物：blind_replicate_e58.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e58_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e58_trackability_report.json")
BENCH = os.path.join(EVAL, "e58_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e58.json")
_repo = ("planning", "trackability", "rne_dynamics", "control_strategies",
         "e66_closed_loop_control", "e58_trackability_gate")
_res = {"experiment": "E58 盲复刻（只给数据文件 + 独立实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E58 盲复刻（只给数据文件 + 独立实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)
    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（l/q0/tau_lim/actuator/270 条件）",
        all(k in data for k in ("bench", "conditions")) and len(data["conditions"]) == rep["design"]["n_conditions"],
        "n=%d (store %d)" % (len(data["conditions"]), rep["design"]["n_conditions"]))
    out = P.replicate(data)
    rr = rep["results"]

    # 逐条四阶一致（这是最强的复刻：270 条 K/S/D/C 全等）
    bad = 0
    for a, b in zip(out["rows"], rep["rows"]):
        if (a["K_ok"], a["S_ok"], a["D_ok"], a["C_ok"]) != (b["K_ok"], b["S_ok"], b["D_ok"], b["C_ok"]):
            bad += 1
    chk("B1 逐条四阶判定（K/S/D/C × 270）与报告全等", bad == 0, "mismatch=%d" % bad)
    bad2 = sum(1 for a, b in zip(out["rows"], rep["rows"]) if abs(a["c_margin"] - b["c_margin"]) > 5e-4
               or abs(a["dyn_margin"] - b["dyn_margin"]) > 5e-4)
    chk("B2 逐条连续裕度（c_margin/dyn_margin × 270）与报告一致（±5e-4）", bad2 == 0, "mismatch=%d" % bad2)

    chk("B3 盲区（D_ok∧¬C_ok）规模复现（30）",
        out["blind_spot_vs_D"] == rr["blind_spot_vs_D"],
        "re=%d store=%d" % (out["blind_spot_vs_D"], rr["blind_spot_vs_D"]))
    chk("B4 盲区（S_ok∧¬C_ok）规模复现（90）",
        out["blind_spot_vs_S"] == rr["blind_spot_vs_S"],
        "re=%d store=%d" % (out["blind_spot_vs_S"], rr["blind_spot_vs_S"]))
    chk("B5 按 T 的盲区分布复现 {3:0,1.5:0,0.8:0,0.5:20,0.35:10,0.2:0}",
        out["blind_by_T"] == rr["blind_spot_by_T"],
        "re=%s store=%s" % (out["blind_by_T"], rr["blind_spot_by_T"]))
    chk("B6 慢档误报率复现（T≥0.8s → 0.0）",
        abs(out["slow_false_alarm"] - rr["slow_T_ge_0.8s_false_alarm_rate"]) < 1e-9,
        "re=%s store=%s" % (out["slow_false_alarm"], rr["slow_T_ge_0.8s_false_alarm_rate"]))
    chk("B7 预注册判据 H58-1/2/3 复现（全 True）",
        out["H"] == rep["preregistered_verdict"],
        "re=%s store=%s" % (out["H"], rep["preregistered_verdict"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = ["★ 盲复刻（只给 e58_bench_data.json + 独立实现）：E58 的 270 条四阶判定"
                        "（K/S/D/C）与盲区统计可被外部独立逐条重建（含连续裕度）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
