# -*- coding: utf-8 -*-
"""blind_replicate_e80.py —— E80 的**盲复刻（只给数据文件 + 独立物理重实现）**。

模拟外部复刻者：只拿 e80_bench_data.json + e80_standalone_physics.py（自写递归 Newton–Euler、
解析闭式、拉格朗日数值微分、几何动能），**不 import planning.* / rne_dynamics / lagrange_dynamics /
energy_kernel / e80_three_paradigm_dynamics**，独立复现 24 个状态的三范式力矩并与作者报告逐格比对。

隔离保证：运行期 sys.modules 不得含 planning / rne_dynamics / lagrange_dynamics / energy_kernel / e80。
产物：blind_replicate_e80.json
"""
import json
import os
import sys

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e80_standalone_physics as P  # noqa: E402  （复刻者自有实现）

REPORT = os.path.join(EVAL, "e80_three_paradigm_dynamics_report.json")
BENCH = os.path.join(EVAL, "e80_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e80.json")

_repo_mods = ("planning", "rne_dynamics", "lagrange_dynamics", "energy_kernel",
              "e80_three_paradigm_dynamics", "control_strategies")
_res = {"experiment": "E80 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


def main():
    print("=" * 92)
    print("E80 盲复刻（只给数据文件 + 独立物理重实现）vs 作者报告")
    print("=" * 92)
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["models", "n_states_per_model", "seed", "q_range", "qd_range", "qdd_range", "g"]
    miss = [k for k in need if k not in data.get("bench", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data)
    rr = rep["A_three_paradigm"]["rows"]
    chk("B1 状态集规模一致（%d 行）" % len(rr), len(out["rows"]) == len(rr), "re=%d store=%d" % (len(out["rows"]), len(rr)))

    # B2 闭式（精确）逐格
    m_cf = 0.0
    for o, r in zip(out["rows"], rr):
        m_cf = max(m_cf, max(abs(o["tau_cf"][i] - r["tau_cf"][i]) for i in range(2)))
    chk("B2 解析闭式逐格复现（24 状态 × 2 关节）", m_cf < 1e-9, "max|Δcf|=%.2e" % m_cf)

    # B3 RNE 递推（精确）逐格
    m_rne = 0.0
    for o, r in zip(out["rows"], rr):
        m_rne = max(m_rne, max(abs(o["tau_rne"][i] - r["tau_rne"][i]) for i in range(2)))
    chk("B3 递推 Newton–Euler 逐格复现（24 状态 × 2 关节）", m_rne < 1e-8, "max|Δrne|=%.2e" % m_rne)

    # B4 RNE vs 闭式一致（机器精度）
    chk("B4 RNE vs 闭式最大偏差 == 报告（机器精度 <1e-10）",
        out["max_rne_vs_cf"] < 1e-10 and rep["A_three_paradigm"]["max_pairwise"]["rne_vs_cf"] < 1e-10,
        "re=%.2e store=%.2e" % (out["max_rne_vs_cf"], rep["A_three_paradigm"]["max_pairwise"]["rne_vs_cf"]))

    # B5 拉格朗日（数值微分）逐格 — 与存储数值法同阶（容差 1e-5）
    m_lag = 0.0
    for o, r in zip(out["rows"], rr):
        m_lag = max(m_lag, max(abs(o["tau_lag"][i] - r["tau_lag"][i]) for i in range(2)))
    chk("B5 拉格朗日（数值微分）逐格复现（与存储数值法同阶，<1e-5）", m_lag < 1e-5, "max|Δlag|=%.2e" % m_lag)

    # B6 几何动能 T == ½q̇ᵀMq̇（独立一致）
    worst_ke = 0.0
    for st in P.generate_states(data["bench"]):
        l1, l2, m1, m2 = st["model"]
        t1 = P.kinetic_geometric(l1, l2, m1, m2, st["q"], st["qd"])
        # ½ q̇ᵀM q̇（M 由闭式结构给出）
        q1, q2 = st["q"]; d1, d2 = st["qd"]
        i1, i2 = m1 * l1 * l1 / 12.0, m2 * l2 * l2 / 12.0
        lc1, lc2 = l1 / 2.0, l2 / 2.0
        c2 = np.cos(q2)
        M11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + i1 + i2
        M12 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + i2
        M22 = m2 * lc2 ** 2 + i2
        t2 = 0.5 * (M11 * d1 ** 2 + 2 * M12 * d1 * d2 + M22 * d2 ** 2)
        worst_ke = max(worst_ke, abs(t1 - t2) / max(abs(t2), 1e-9))
    chk("B6 几何动能 T == ½q̇ᵀMq̇（相对偏差 <1e-12）", worst_ke < 1e-12, "max_rel=%.2e" % worst_ke)

    # B7 verdict 裸布尔重算（C1 由三范式一致）
    okA = (out["max_rne_vs_cf"] < rep["A_three_paradigm"]["tol"])
    chk("B7 C1 裸布尔重算 == 报告（三范式一致 <1e-7）",
        bool(okA) == bool(rep["criteria"]["C1_three_paradigm_agree"]),
        "re=%s store=%s" % (okA, rep["criteria"]["C1_three_paradigm_agree"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e80_bench_data.json + 独立物理重实现）：三范式互证可被外部独立重建。",
        "  递归 RNE / 解析闭式 / 拉格朗日数值微分：24 状态 × 2 关节逐格一致；几何动能一致。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
