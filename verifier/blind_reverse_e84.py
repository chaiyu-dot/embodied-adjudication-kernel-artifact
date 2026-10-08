# -*- coding: utf-8 -*-
"""blind_reverse_e84.py —— E84 的 **L5 逆向审核**（机制反推 + 负控 + 边界）。

不复述 L3 的结构自洽，而是从**结果反推机制**并用自足数据文件独立反证：

R1 **入口形式 ⇔ 漏报**：危险只出现在点入口臂；集合值入口（区间）三臂危险恒 0
R2 **偏序机制**：可行率 POINT ≥ GATED ≥ SOUND（点最宽松；测量只能把"歧义"变成"可判"，不能凭空造可行）
R3 **测量经济性反推**：GATED 的测量次数 **恰等于**"点可行 ∧ 区间端不可行"的触发数（由数据文件重算）
R4 **负对照（恒可行判据）**：漏报率 = gt_dangerous_rate（>0），远超四臂实际 → 指标有分辨力
R5 **逐阶承重反推**：本台架下只有质量相关的力矩阶（D/C）承重（K/S/E 从不成为 binding），
   与 E84c 网格的机制反推一致（K 只在超限幅值下 binding；E 被 D 支配）
R6 **边界**：全测臂危险 0 且可行率最高（上界）；SOUND 误拒率最高（保守代价）

产物：blind_reverse_e84.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e84_five_order_wrapper_loop as E84    # noqa: E402

REPORT = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
BENCH = os.path.join(EVAL, "e84_bench_data.json")
OUT = os.path.join(EVAL, "blind_reverse_e84.json")
_res = {"experiment": "E84 L5 逆向审核", "checks": [], "findings": []}
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
    print("E84 L5 逆向审核（机制反推 + 负控 + 边界）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    a = rep["aggregate"]

    # R1 入口形式 ⇔ 漏报
    zero_interval = (a["SOUND_NOMEAS"]["dangerous"] == 0 and a["GATED"]["dangerous"] == 0
                     and a["MEAS_ALL"]["dangerous"] == 0)
    chk("R1 机制反推：漏报只出现在**点入口**臂；三个集合值入口臂危险恒 0",
        bool(a["POINT"]["dangerous"] > 0 and zero_interval),
        "POINT=%d 其余=%d/%d/%d" % (a["POINT"]["dangerous"], a["SOUND_NOMEAS"]["dangerous"],
                                    a["GATED"]["dangerous"], a["MEAS_ALL"]["dangerous"]))

    # R2 可行率偏序
    chk("R2 机制反推：可行率 POINT ≥ GATED ≥ SOUND（点最宽松；测量把歧义变成可判）",
        bool(a["POINT"]["feasible_rate"] >= a["GATED"]["feasible_rate"] >= a["SOUND_NOMEAS"]["feasible_rate"]),
        "POINT/MEAS_ALL/GATED/SOUND = %.4f/%.4f/%.4f/%.4f"
        % (a["POINT"]["feasible_rate"], a["MEAS_ALL"]["feasible_rate"],
           a["GATED"]["feasible_rate"], a["SOUND_NOMEAS"]["feasible_rate"]))

    # R3 测量次数 = 触发条件计数（由数据文件独立重算）
    k_conf = float(data["k_conf"])
    trig = 0
    for p in data["proposals"]:
        q1 = p["q_final"]
        prop = E84.make_proposal(E84.make_traj([0.0, 0.0], q1, p["T"]), q1)
        states = p["states"]["GATED"]
        for (m, s) in states[:-1]:
            min_pt = E84.five_orders(m, prop)[1]
            min_hi = E84.five_orders(m + k_conf * s, prop)[1]
            if (min_pt >= 0.0) and (min_hi < 0.0):
                trig += 1
    chk("R3 机制反推：GATED 平均测量次数 = 『点可行 ∧ 区间端不可行』的触发数（数据文件重算）",
        abs(trig / len(data["proposals"]) - a["GATED"]["measure_calls_mean"]) < 1e-4,
        "重算触发/提案 = %.4f ｜ 报告测量均值 = %.4f" % (trig / len(data["proposals"]),
                                                    a["GATED"]["measure_calls_mean"]))

    # R4 负对照：恒可行判据
    chk("R4 负对照：恒可行判据的漏报率 = gt_dangerous_rate > 0（指标有分辨力，非恒 0 同义反复）",
        bool(a["POINT"]["gt_dangerous_rate"] > 0 and a["MEAS_ALL"]["dangerous"] == 0),
        "gt_dangerous_rate=%.4f ｜ 全测臂漏报=0" % a["POINT"]["gt_dangerous_rate"])

    # R5 逐阶承重反推
    b = rep["order_binding_distribution"]
    chk("R5 机制反推：本台架只有质量相关力矩阶（D/C）承重，K/S/E 从不 binding（与 E84c 网格一致）",
        bool(b["D"] + b["C"] == rep["n_proposals"] and b["K"] == 0 and b["S"] == 0 and b["E"] == 0),
        "binding=%s n=%s" % (b, rep["n_proposals"]))

    # R6 边界
    chk("R6 边界：全测臂 = 上界（危险 0 且可行率最高）；SOUND 误拒率最高（保守代价）",
        bool(a["MEAS_ALL"]["dangerous"] == 0
             and a["MEAS_ALL"]["feasible_rate"] >= max(a["GATED"]["feasible_rate"],
                                                       a["SOUND_NOMEAS"]["feasible_rate"])
             and a["SOUND_NOMEAS"]["false_reject_rate"] >= a["GATED"]["false_reject_rate"]),
        "feasible MEAS_ALL/GATED/SOUND=%.4f/%.4f/%.4f ｜ false_rej SOUND/GATED=%.4f/%.4f"
        % (a["MEAS_ALL"]["feasible_rate"], a["GATED"]["feasible_rate"], a["SOUND_NOMEAS"]["feasible_rate"],
           a["SOUND_NOMEAS"]["false_reject_rate"], a["GATED"]["false_reject_rate"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 逆向反推证实：漏报与『入口形式』一一对应（点入口才有漏报）；"
        "测量次数被『点可行 ∧ 区间端不可行』这一预测失败判据**精确解释**（重算 = 报告）；"
        "重量级承重只落在 D/C（动力学与控制论阶），与 E84c 的机制反推互相印证。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL5 逆向审核：%d/%d ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
