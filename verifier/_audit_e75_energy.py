# -*- coding: utf-8 -*-
"""_audit_e75_energy.py —— e75 的正向（F1-F9）+ 反向（R1-R6）双向审核。

纪律（承自 e71/e72 审计）：
  · 正向：用 `planning.energy_kernel` 原语 + e66 仿真**独立驱动**重算报告每个数值，不信任报告自带的布尔；
  · 反向：构造篡改，证明审计确实敏感（改报告 → 必有检查报错）。

产物：_audit_e75_energy.json
"""
import copy
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.energy_kernel import (mechanical_energy, work_conjugacy_residual,   # noqa: E402
                                    energy_margin_exact, e_order_band)
from planning.energy_margin import margin_to_band                                  # noqa: E402
from planning.rne_dynamics import quintic_trajectory                              # noqa: E402
import e66_closed_loop_control as E66                                             # noqa: E402
import e75_energy_margin_exact as E75                                             # noqa: E402

REPORT = E75.REPORT
_res = {"audit": "_audit_e75_energy", "checks": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return {"name": name, "pass": bool(ok)}


def _P(mp):
    return {"l1": E75.L1, "l2": E75.L2, "m1": E75.M1, "m2": E75.M2,
            "lc1": E75.LC1, "lc2": E75.LC2, "i1": E75.I1, "i2": E75.I2, "g": E75.G, "mp": mp}


def forward(rep):
    ch = []
    # F1 结构
    need = ["A_work_conjugacy", "B_ordinal", "C_misband", "criteria", "verdict_pass"]
    ch.append(_chk("F1_structure", all(k in rep for k in need),
                   "missing=%s" % [k for k in need if k not in rep]))

    # F2 A 段独立重算（e66 仿真 + energy_kernel 原语）
    a_ok = True; mx = 0.0
    for idx, (q0, q_ref, mp) in enumerate(E75.SCN_A):
        kp, kd, wn = E66.budget_gains(q_ref)
        log = E66.simulate(E66.Ctrl("PD_G", kp, kd), q_ref, q0=q0, T=4.0, dt=E75.DT, mp=mp, with_aux=True)
        q, qd = log["q"], log["qd"]; p = _P(mp)
        me0 = mechanical_energy(q[0], qd[0], p); me1 = mechanical_energy(q[-1], qd[-1], p)
        resid = work_conjugacy_residual(float(log["W"][-1]), float(log["D"][-1]),
                                        me1["ke"] - me0["ke"], me1["pe"] - me0["pe"])
        st = rep["A_work_conjugacy"]["rows"]["s%d_mp%.2f" % (idx, mp)]
        if abs(resid - st["residual_J"]) > 1e-9:
            a_ok = False
        mx = max(mx, abs(resid))
    ch.append(_chk("F2_A_work_conjugacy_recompute", a_ok and abs(mx - rep["A_work_conjugacy"]["max_abs_residual"]) < 1e-12,
                   "重算 max|resid|=%.3e 报告 %.3e" % (mx, rep["A_work_conjugacy"]["max_abs_residual"])))

    # F3 B 段 m_E 独立重算
    b_ok = True
    for r in rep["B_ordinal"]["rows"]:
        me = energy_margin_exact(r["dE_mech_J"], rep["B_ordinal"]["e_avail_J"])
        mk = energy_margin_exact(r["KE_peak_J"], rep["B_ordinal"]["e_avail_J"])
        if abs(me - r["m_E_exact"]) > 1e-9 or abs(mk - r["m_E_kinetic_only"]) > 1e-9:
            b_ok = False
        if e_order_band(me) != r["band_exact"]:
            b_ok = False
    ch.append(_chk("F3_B_margin_recompute", b_ok, "m_E/band 逐行重算"))

    # F4 序数一致率独立重算
    demand = [r["dPE_J"] for r in rep["B_ordinal"]["rows"]]
    mE = [r["m_E_exact"] for r in rep["B_ordinal"]["rows"]]
    oc = E75._ordinal_consistency(mE, demand)
    ch.append(_chk("F4_ordinal_recompute", abs(oc - rep["B_ordinal"]["ordinal_consistency_exact"]) < 1e-12,
                   "重算 %.4f 报告 %.4f" % (oc, rep["B_ordinal"]["ordinal_consistency_exact"])))

    # F5 C 段误档独立重算
    SEV = {"unsafe": 0, "tight": 1, "medium": 2, "loose": 3}
    cnt = sum(1 for r in rep["B_ordinal"]["rows"]
              if SEV[r["band_exact"]] <= 1 and SEV[r["band_kinetic_only"]] >= 2)
    ch.append(_chk("F5_misband_recompute", cnt == rep["C_misband"]["n_misbanded"],
                   "重算 %d 报告 %d" % (cnt, rep["C_misband"]["n_misbanded"])))

    # F5b 逐行误档明细重算（防"改非计数行/改档名而计数不变"漏报）
    recomputed = [{"mp": r["mp"], "band_exact": r["band_exact"], "band_kinetic_only": r["band_kinetic_only"]}
                  for r in rep["B_ordinal"]["rows"]
                  if SEV[r["band_exact"]] <= 1 and SEV[r["band_kinetic_only"]] >= 2]
    stored = [{"mp": m["mp"], "band_exact": m["band_exact"], "band_kinetic_only": m["band_kinetic_only"]}
              for m in rep["C_misband"]["misbanded"]]
    ch.append(_chk("F5b_misband_detail_recompute", recomputed == stored,
                   "逐行明细 %s" % ("一致" if recomputed == stored else "%s vs %s" % (recomputed, stored))))

    # F6 不变量自洽
    c = rep["criteria"]
    inv = bool(c["C1_work_conjugacy_lt_1e-8"] and rep["A_work_conjugacy"]["all_ok"]
               and c["C2_ordinal_not_degraded"] and rep["B_ordinal"]["all_ok"]
               and c["C3_kinetic_only_misbands"] and rep["C_misband"]["all_ok"])
    ch.append(_chk("F6_invariants_self_consistent", inv == bool(rep["verdict_pass"]),
                   "inv=%s rep=%s" % (inv, rep["verdict_pass"])))

    # F7 档位口径一致：energy_kernel.e_order_band ≡ energy_margin.margin_to_band
    band_ok = all(e_order_band(v) == margin_to_band(v)
                  for v in (-0.5, -0.05, 0.0, 0.24, 0.25, 0.59, 0.60, 0.9))
    ch.append(_chk("F7_band_口径一致", band_ok, "e_order_band ≡ margin_to_band"))

    # F8 确定性
    det = (E66.simulate(E66.Ctrl("PD_G", 30.0, 6.0), [0.6, -0.9], T=1.0, dt=1e-3, with_aux=True)["W"][-1]
           == E66.simulate(E66.Ctrl("PD_G", 30.0, 6.0), [0.6, -0.9], T=1.0, dt=1e-3, with_aux=True)["W"][-1])
    ch.append(_chk("F8_determinism", bool(det), "same input same output"))

    # F9 精确核与动能-only 的分歧在 PE 主导场景确实存在（非编码伪影）
    pe_dom = [r for r in rep["B_ordinal"]["rows"] if r["dPE_J"] > r["KE_peak_J"]]
    diverged = any(r["m_E_exact"] < r["m_E_kinetic_only"] - 0.1 for r in pe_dom)
    ch.append(_chk("F9_exact_vs_kinetic_divergence_real", len(pe_dom) >= 3 and diverged,
                   "PE 主导 %d 行，其中 exact 显著更保守=%s" % (len(pe_dom), diverged)))
    return ch


def _tamp(name, mut, expect_prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = forward(bad)
    caught = any((not c["pass"]) and c["name"].startswith(tuple(expect_prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res.setdefault("tamper", []).append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E75 作者双向审核"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = forward(rep)
    np_ = sum(c["pass"] for c in fwd)
    print("正向：%d/%d" % (np_, len(fwd)))
    print("[R] 篡改：")
    _tamp("R1 改 A 段残差 → F2 必报",
          lambda r: r["A_work_conjugacy"]["rows"]["s0_mp0.00"].__setitem__("residual_J", 1e-3), ["F2"])
    _tamp("R2 改 B 段 m_E_exact → F3 必报",
          lambda r: r["B_ordinal"]["rows"][0].__setitem__("m_E_exact", r["B_ordinal"]["rows"][0]["m_E_exact"] + 0.2), ["F3"])
    _tamp("R3 改 e_avail → F3 必报",
          lambda r: r["B_ordinal"].__setitem__("e_avail_J", r["B_ordinal"]["e_avail_J"] * 2.0), ["F3"])
    _tamp("R4 改误档行 band_kinetic_only（loose→medium，计数不变）→ F5b 必报",
          lambda r: r["B_ordinal"]["rows"][2].__setitem__("band_kinetic_only", "medium"), ["F5b"])
    _tamp("R5 翻转 verdict_pass → F6 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["F6"])
    _tamp("R6 改 max_abs_residual → F2 必报",
          lambda r: r["A_work_conjugacy"].__setitem__("max_abs_residual", 1e-3), ["F2"])
    ntp = sum(1 for t in _res.get("tamper", []) if t["caught"])
    ntt = len(_res.get("tamper", []))
    _res["n_pass"], _res["n_total"] = np_, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["audit_pass"] = bool(np_ == len(fwd) and ntp == ntt)
    _res["findings"] = ["★ 作者双向审核：正向 %d/%d + 篡改 %d/%d。" % (np_, len(fwd), ntp, ntt)]
    with open(os.path.join(EVAL, "_audit_e75_energy.json"), "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n审核：正向 %d/%d ｜ 篡改 %d/%d ｜ audit_pass = %s"
          % (np_, len(fwd), ntp, ntt, _res["audit_pass"]))
    return _res


if __name__ == "__main__":
    main()
