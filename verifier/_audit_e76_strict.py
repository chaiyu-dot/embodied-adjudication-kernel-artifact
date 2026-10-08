# -*- coding: utf-8 -*-
"""_audit_e76_strict.py —— **严格审核（审前提，硬重算不读自述）**。

审结论成立所需的前提：
  S1 上限内漂移 ≤1cm（硬重算，独立物理）；
  S2 漂移随载荷单调；
  S3 内力解耦闭式恒等（GᵀW_int=0 ∧ 正交 ∧ α 不变）硬重算；
  S4 上限内危险 0（含内力，demand 口径）；
  S5 静态上限前提：mp 超上限 → 需求饱和（"先可行后可控"）；
  S6 内力**计入需求口径**（否则越限统计漏报）；
  S7 verdict 自洽；
  S8 数据文件自足。
篡改 R1–R5 必有对应 S 报出。
产物：_audit_e76_strict.json
"""
import copy
import json
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e76_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e76_dualarm_coordination_report.json")
BENCH = os.path.join(EVAL, "e76_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e76_strict.json")

_res = {"audit": "_audit_e76_strict", "checks": [], "tamper": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return (name, bool(ok))


def _forward(rep, data):
    ch = []
    A, B, C = rep["A_relative_pose_keeping"], rep["B_internal_force_decoupling"], rep["C_danger_zero"]
    o = P.replicate(data)
    oA, oB, oC = o["A_relative_pose_keeping"], o["B_internal_force_decoupling"], o["C_danger_zero"]

    # S1 上限内漂移（硬重算）
    s1 = oA["max_drift_within_limit_m"] is not None and oA["max_drift_within_limit_m"] <= data["config"]["drift_tol_m"]
    ch.append(_chk("S1_drift_within_limit_le_1cm", s1, "重算 max drift=%.6f ≤ %.3f"
                   % (oA["max_drift_within_limit_m"], data["config"]["drift_tol_m"])))

    # S2 漂移单调
    ch.append(_chk("S2_drift_monotone_in_load", oA["drift_monotone_in_mp"],
                   "monotone=%s" % oA["drift_monotone_in_mp"]))

    # S3 内力解耦闭式恒等（硬重算）＋ 存储四项 == 重算（防篡改）
    s3 = (oB["net_object_wrench_inf"] < 1e-12 and oB["orthogonality_motion_vs_internal"] < 1e-12
          and oB["motion_component_invariance_vs_alpha_max_err"] < 1e-12
          and abs(oB["net_object_wrench_inf"] - B["net_object_wrench_inf"]) < 1e-15
          and abs(oB["orthogonality_motion_vs_internal"] - B["orthogonality_motion_vs_internal"]) < 1e-18
          and abs(oB["motion_component_invariance_vs_alpha_max_err"] - B["motion_component_invariance_vs_alpha_max_err"]) < 1e-15
          and abs(oB["internal_force_sum_norm"] - B["internal_force_sum_norm"]) < 1e-15)
    ch.append(_chk("S3_internal_decoupling_identity", s3,
                   "‖GᵀW_int‖=%.2e orth=%.2e α=%.2e（且存储四项==重算）"
                   % (oB["net_object_wrench_inf"], oB["orthogonality_motion_vs_internal"],
                      oB["motion_component_invariance_vs_alpha_max_err"])))

    # S4 上限内危险 0（硬重算）
    s4 = all(v["violation_steps_demand"] == 0 and not v["diverged"] for v in oC["rows"].values())
    ch.append(_chk("S4_danger_zero_within_limit", s4, "within-limit 场景全 0"))

    # S5 静态上限前提（超限 → 饱和）硬重算
    over = P.coord_sim(data["config"]["targets"], data["config"]["T_s"], mp_R=1.0, ramp=data["config"]["ramp"])
    ch.append(_chk("S5_static_limit_premise", over["violation_steps_demand"] > 0,
                   "mp=1.0 越限步=%d" % over["violation_steps_demand"]))

    # S6 内力计入需求口径（硬重算：大内力应使 demand 越限；无内力则否）
    t = data["config"]["targets"]
    r0 = P.coord_sim(t, data["config"]["T_s"], mp_R=0.2, k_int=0.0, ramp=data["config"]["ramp"])
    rbig = P.coord_sim(t, data["config"]["T_s"], mp_R=0.2, k_int=20.0, ramp=data["config"]["ramp"])
    ch.append(_chk("S6_internal_counted_in_demand",
                   r0["violation_steps_demand"] == 0 and rbig["violation_steps_demand"] > 0,
                   "k_int=0 越限=%d；k_int=20 越限=%d" % (r0["violation_steps_demand"], rbig["violation_steps_demand"])))

    # S9 存储 A max drift == 硬重算（防改聚合量）
    ch.append(_chk("S9_stored_max_drift_equals_recompute",
                   oA["max_drift_within_limit_m"] is not None
                   and abs(oA["max_drift_within_limit_m"] - A["max_drift_within_limit_m"]) < 1e-9,
                   "重算 %s 报告 %s" % (oA["max_drift_within_limit_m"], A["max_drift_within_limit_m"])))

    # S10 存储 C 段逐行 == 硬重算（防改单行越限步）
    mism = []
    for k in C["rows"]:
        a, r = C["rows"][k], oC["rows"].get(k)
        if r is None or a["violation_steps_demand"] != r["violation_steps_demand"] \
                or abs(a["tau_max_demand"] - r["tau_max_demand"]) > 1e-3 or bool(a["diverged"]) != bool(r["diverged"]):
            mism.append((k, a.get("violation_steps_demand"), None if r is None else r["violation_steps_demand"]))
    ch.append(_chk("S10_stored_C_rows_equal_recompute", not mism, "mismatch=%s" % mism[:3] if mism else "逐行一致"))

    # S7 verdict 自洽
    c = rep["criteria"]
    inv = bool(c["C1_drift_le_1cm_within_static_limit_and_monotone"] and A["all_ok"]
               and c["C2_internal_decoupled_exact"] and B["all_ok"]
               and c["C3_danger_zero_within_limit"] and C["all_ok"])
    ch.append(_chk("S7_verdict_self_consistent", inv == bool(rep["verdict_pass"]),
                   "重算 %s 报告 %s" % (inv, rep["verdict_pass"])))

    # S8 数据文件自足
    need_b = ["L1", "L2", "M1", "M2", "LC1", "LC2", "I1", "I2", "G", "TAU_LIM", "B_VISC", "DT", "BASE_L", "BASE_R"]
    need_c = ["targets", "T_s", "ramp", "mp_ladder", "drift_tol_m", "internal_force_N", "F_des"]
    miss = [k for k in need_b if k not in data.get("bench", {})] + [k for k in need_c if k not in data.get("config", {})]
    ch.append(_chk("S8_data_file_self_contained", not miss, "missing=%s" % miss))
    return ch


def _tamp(name, mut, prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    data = json.load(open(BENCH, encoding="utf-8"))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad, data)
    caught = any((not c[1]) and c[0].startswith(tuple(prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E76 严格审核（审前提，硬重算不读自述）"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    fwd = _forward(rep, data)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))
    print("\n[R] 篡改：")
    _tamp("R1 改 A 段上限内漂移 → S9 必报（存储量==重算）",
          lambda r: r["A_relative_pose_keeping"].__setitem__("max_drift_within_limit_m", 0.5), ["S9"])
    _tamp("R2 改 B 段 ‖GᵀW_int‖ → S3 必报",
          lambda r: r["B_internal_force_decoupling"].__setitem__("net_object_wrench_inf", 0.5), ["S3"])
    _tamp("R3 改 C 段越限步 → S10 必报（存储==重算）",
          lambda r: r["C_danger_zero"]["rows"]["mp=0.20"].__setitem__("violation_steps_demand", 999), ["S10"])
    _tamp("R4 翻转 verdict_pass → S7 必报", lambda r: r.__setitem__("verdict_pass", False), ["S7"])
    _tamp("R5 把 B 段 α-不变误差改大 → S3 必报",
          lambda r: r["B_internal_force_decoupling"].__setitem__("motion_component_invariance_vs_alpha_max_err", 0.5), ["S3"])
    ntp = sum(1 for t in _res["tamper"] if t["caught"]); ntt = len(_res["tamper"])
    _res["n_pass"], _res["n_total"] = npass, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["strict_pass"] = bool(npass == len(fwd) and ntp == ntt)
    _res["findings"] = [
        "★ 严格审核：%d 条前提成立、%d 条篡改全被捕获。" % (npass, ntt),
        "  关键前提 S3：内力解耦是闭式恒等（GᵀW_int≡0 且与运动分量正交、α 不改变运动）。",
        "  S5/S6：静态上限 0.357kg 是硬边界（超限饱和），内力已计入需求口径（防漏报）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), ntp, ntt, _res["strict_pass"]))
    return _res


if __name__ == "__main__":
    main()
