# -*- coding: utf-8 -*-
"""blind_reverse_e58.py —— E58 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载被审模块）。只信"数据文件"（报告里的逐行
f_traj_hz / c_margin / binding / 执行器参数）。**反推**执行器带宽 ωn，并做证伪：
若 ωn→∞（无限执行器带宽），盲区应消失 —— 证明盲区是**带宽现象**而非判据缺陷（可证伪性见证）。

产物：blind_reverse_e58.json
"""
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 刻意不把 EVAL 加入 sys.path
import numpy as np                                                          # noqa: E402

REPORT = os.path.join(EVAL, "e58_trackability_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e58.json")

_res = {"experiment": "E58 逆向盲测(L5 隔离)", "independence_scope": "隔离：sys.modules 不含 e58_trackability_gate",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _c_margin(omega_n, f_traj_hz, T_s=0.001, k_omega=2.0, k_margin=2.0):
    """审计者自写 C 阶闭式（与 trackability 同式），仅用于逆推与证伪复算。"""
    w = 2.0 * math.pi * f_traj_hz
    m_omega = 1.0 - (k_omega * w) / omega_n if omega_n > 0 else -1.0
    t_react = 1.0 / omega_n if omega_n > 0 else float("inf")
    t_budget = (1.0 / (k_margin * f_traj_hz)) if f_traj_hz > 0 else float("inf")
    m_tau = 1.0 - t_react / t_budget if math.isfinite(t_budget) else 1.0
    m_s = 1.0 - 10.0 * T_s * f_traj_hz
    return min(m_omega, m_tau, m_s)


def main():
    print("=" * 88)
    print("E58 逆向盲测（L5 隔离，反推 ωn + 带宽证伪）")
    print("=" * 88)
    chk("R0_isolation_e58_not_imported", "e58_trackability_gate" not in sys.modules)

    rep = json.load(open(REPORT, encoding="utf-8"))
    rows = rep["rows"]
    K_OMEGA, K_MARGIN, T_S = 2.0, 2.0, 0.001

    # --- 反推 ωn：仅对 binding=='omega' 的行（margin 由 m_ω 主导）---
    rec = []
    for r in rows:
        if r["binding"] == "omega" and r["f_traj_hz"] > 0 and r["c_margin"] < 1.0:
            w = 2.0 * math.pi * r["f_traj_hz"]
            # m_ω = 1 - k_ω·w/ωn  ⇒ ωn = k_ω·w/(1-m_ω)
            wn = K_OMEGA * w / (1.0 - r["c_margin"])
            rec.append(wn)
    if rec:
        med = float(np.median(rec))
        # 设计值 ωn=40；反推应接近 40（允许 5% 容差，因舍入/多绑定近似）
        chk("R1_reverse_omega_n_matches_40", abs(med - 40.0) < 2.0,
            "median_recovered_wn=%.3f (design=40)" % med)
    else:
        chk("R1_reverse_omega_n_matches_40", False, "no omega-binding rows")

    # --- 证伪 A：ωn→∞（无限执行器带宽）→ 盲区应消失 ---
    blind_orig = [r for r in rows if r["D_ok"] and not r["C_ok"]]
    blind_inf = [r for r in rows if r["D_ok"] and _c_margin(1e9, r["f_traj_hz"]) < 0.0]
    chk("R2_falsify_blind_spot_vanishes_at_infinite_bw",
        len(blind_orig) > 0 and len(blind_inf) == 0,
        "orig_blind=%d infinite_bw_blind=%d" % (len(blind_orig), len(blind_inf)))

    # --- 证伪 B：ωn 下降 → 盲区扩大（单调性，证明盲区对执行器带宽敏感）---
    # 在原始盲区行里，ωn=20 时应比 ωn=40 产生更多不可追踪
    blind_lo = [r for r in rows if r["D_ok"] and _c_margin(20.0, r["f_traj_hz"]) < 0.0]
    chk("R3_falsify_lower_bw_more_blind", len(blind_lo) >= len(blind_orig),
        "wn40_blind=%d wn20_blind=%d" % (len(blind_orig), len(blind_lo)))

    # --- 证伪 C：若用"纯动力学阶"（只看 D）判定可行 → 会漏报这些盲区（证明 C 阶必要）---
    # 即：存在 D_ok 但 C 不可行的条件（已被 R2/R3 支撑）；此处补一个"对照臂"：
    # 若去掉 C 阶判据（只按 D 判可行），可行率虚高 → 这些盲区被漏报
    nDok = sum(1 for r in rows if r["D_ok"])
    nCok_given_Dok = sum(1 for r in rows if r["D_ok"] and r["C_ok"])
    leak = nDok - nCok_given_Dok
    chk("R4_C_order_necessary_leak_exists", leak > 0 and leak == len(blind_orig),
        "D_ok_but_C_infeasible=%d" % leak)

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
