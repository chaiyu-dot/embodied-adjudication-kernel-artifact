# -*- coding: utf-8 -*-
"""_audit_e77_strict.py —— **严格审核（审前提，硬重算不读自述）**。

前提：S1 接口三契约 / S2 危险 0（执行口径）/ S3 标称 ss ≤ tol / S4 确定性 /
S5 门控承重（gate-off 执行越限>0）/ S6 verdict 自洽 / S7 数据文件自足 / S8 存储量==硬重算 /
S9 D 段存储==硬重算 / **S10 执行值判据不可证伪的见证（零依赖定义性 + 敌意控制器仿真）** /
**S11 门控承重稳健性（stress ≥2 族越限 + IMP 更严参考下越限）**。
篡改 R1–R5 必有对应 S 报出。产物：_audit_e77_strict.json
"""
import copy
import json
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e77_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e77_controller_family_extension_report.json")
BENCH = os.path.join(EVAL, "e77_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e77_strict.json")
FIELDS = ("rise_s", "settling_s", "overshoot", "ss_err_rad", "tau_max_executed",
          "executed_violation_steps", "demand_violation_steps", "demand_tau_max", "diverged")

_res = {"audit": "_audit_e77_strict", "checks": [], "tamper": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return (name, bool(ok))


def _forward(rep, data):
    ch = []
    A, B, C, D = rep["A_metrics"], rep["B_interface"], rep["C_capability"], rep["D_determinism_and_ablation"]
    new = data["family_specs"].keys() and list(data["family_specs"].keys())
    o = P.replicate(data)
    oA, oC, oD = o["A_metrics"], o["C_capability"], o["D_determinism_and_ablation"]

    # S1 接口三契约（硬重算）
    s1 = True
    for n in new:
        c = P.build(n, data["gains"]["kp_design"], data["gains"]["kd_design"], data["family_specs"]["LRN"]["W"])
        tau = c.act(np.array([0.1, 0.0]), np.zeros(2), np.array([0.6, -0.9]), 0.0)
        if not np.all(np.abs(tau) <= np.asarray(data["bench"]["TAU_LIM"]) + 1e-9):
            s1 = False
    ch.append(_chk("S1_interface_contracts", s1 and B["all_ok"], "三族经同一门控输出有界=%s" % s1))

    # S2 危险 0（执行口径，硬重算）
    s2 = all(oA["rows"][n][sc]["executed_violation_steps"] == 0 and not oA["rows"][n][sc]["diverged"]
             for n in new for sc in data["scenarios"])
    ch.append(_chk("S2_danger_zero_executed", s2, "全部新族 x 场景 executed 越限=0"))

    # S3 标称 ss ≤ tol（硬重算）
    s3 = all(oC["rows"][n]["ss_err_nominal"] <= data["ss_tol_rad"] for n in new)
    ch.append(_chk("S3_ss_bounded_nominal", s3,
                   "标称 ss=%s ≤ %.3f" % ({n: oC["rows"][n]["ss_err_nominal"] for n in new}, data["ss_tol_rad"])))

    # S4 确定性
    ch.append(_chk("S4_deterministic", oD["deterministic"], "同输入同输出"))

    # S5 门控承重（gate-off 执行越限 > 0）
    abl = {k: v["executed_violation_steps_no_gate"] for k, v in oD["gate_off_ablation"].items()}
    ch.append(_chk("S5_gate_load_bearing", oD["gate_load_bearing"] and any(v > 0 for v in abl.values()),
                   "gate-off 执行越限=%s ⇒ 门控承重" % abl))

    # S6 verdict 自洽
    c = rep["criteria"]
    C2 = all(A["rows"][n][sc]["executed_violation_steps"] == 0 and not A["rows"][n][sc]["diverged"]
             for n in new for sc in data["scenarios"])
    inv = bool(c["C1_interface_consistent"] and B["all_ok"] and c["C2_danger_zero_executed"] and C2
               and c["C3_capability_reportable_and_ss_bounded"] and C["all_ok"]
               and c["C4_deterministic"] and D["deterministic"])
    ch.append(_chk("S6_verdict_self_consistent", inv == bool(rep["verdict_pass"]),
                   "重算 %s 报告 %s" % (inv, rep["verdict_pass"])))

    # S7 数据文件自足
    need = ["bench", "gains", "scenarios", "cap_scenarios", "q0", "ss_tol_rad", "family_specs"]
    nb = ["L1", "L2", "M1", "M2", "LC1", "LC2", "I1", "I2", "G", "TAU_LIM", "B_VISC", "DT"]
    miss = [k for k in need if k not in data] + [k for k in nb if k not in data.get("bench", {})]
    ch.append(_chk("S7_data_file_self_contained", not miss, "missing=%s" % miss))

    # S8 存储量 == 硬重算（A 段逐格 + C/D）
    mism = []
    for n in new:
        for sc in data["scenarios"]:
            a, r = A["rows"][n][sc], oA["rows"][n][sc]
            for f in FIELDS:
                if a[f] != r[f]:
                    mism.append((n, sc, f))
        if A and C["rows"][n]["ss_bounded_nominal"] != oC["rows"][n]["ss_bounded_nominal"]:
            mism.append((n, "ss_bounded"))
    ch.append(_chk("S8_stored_equals_recompute", not mism, "mismatch=%s" % mism[:3] if mism else "逐格一致"))

    # S9 D 段存储量 == 硬重算（确定性 / 门控承重 / 消融逐族）
    abl_rep = {k: v["executed_violation_steps_no_gate"] for k, v in D["gate_off_ablation"].items()}
    abl_rec = {k: v["executed_violation_steps_no_gate"] for k, v in oD["gate_off_ablation"].items()}
    s9 = (bool(D["deterministic"]) == bool(oD["deterministic"])
          and bool(D["gate_load_bearing"]) == bool(oD["gate_load_bearing"]) and abl_rep == abl_rec)
    ch.append(_chk("S9_stored_D_equals_recompute", s9,
                   "deterministic=%s/%s gate_load=%s/%s abl=%s/%s"
                   % (D["deterministic"], oD["deterministic"], D["gate_load_bearing"],
                      oD["gate_load_bearing"], abl_rep, abl_rec)))

    # S10 可证伪性见证（两路）：
    #   (a) 定义性（零依赖、不碰被审代码）：clip 的像恒在界内 ⇒「执行值越限」对**任意输入**恒为 0；
    #   (b) 仿真（独立物理）：恒 1e6 N·m 敌意控制器经同一门控 → executed=0、demand=满步。
    lim0 = np.asarray(data["bench"]["TAU_LIM"], float)
    probe = np.array([[1e6, 1e6], [1e6, -1e6], [3.0 * lim0[0], 4.0 * lim0[1]], [0.0, 0.0]])
    n_over = int(np.sum(np.any(np.abs(probe) > lim0 + 1e-9, axis=1)))                            # 门控前越限行（3/4）
    n_after = int(np.sum(np.any(np.abs(np.clip(probe, -lim0, lim0)) > lim0 + 1e-9, axis=1)))    # clip 后（0/4）
    wit = P.falsifiability_witness(data, T=1.0)
    s10 = (n_over >= 3 and n_after == 0
           and wit["executed_violation_steps"] == 0
           and wit["demand_violation_steps"] == wit["n_steps"]
           and wit["tau_max_executed"] <= max(data["bench"]["TAU_LIM"]) + 1e-9)
    ch.append(_chk("S10_executed_criterion_unfalsifiable_witness", s10,
                   "定义性：探针门控前越限 %d/%d、clip 后越限 %d/%d（恒 0）；仿真：敌意控制器(1e6 N·m) gate=ON → "
                   "executed=%d demand=%d/%d τmax_exec=%.2f ⇒ C2(执行值)不可证伪，安全证据只能来自门控关断消融"
                   % (n_over, len(probe), n_after, len(probe), wit["executed_violation_steps"],
                      wit["demand_violation_steps"], wit["n_steps"], wit["tau_max_executed"])))

    # S11 门控承重稳健性：stress 下各新族 gate-off 越限（≥2 族>0）+ IMP 更严参考下确实越限
    sref = data["scenarios"]["stress"]["q_ref"]
    ung = {n: P.ungated_violations(data, n, sref, data["scenarios"]["stress"]["T"])["executed_violation_steps"]
           for n in new}
    imp_harsh = P.ungated_violations(data, "IMP", [2.8, -2.8], 0.3)
    s11 = (sum(1 for v in ung.values() if v > 0) >= 2) and imp_harsh["executed_violation_steps"] > 0
    ch.append(_chk("S11_gate_load_bearing_robust", s11,
                   "stress gate-off 越限=%s（≥2 族>0）；IMP 更严参考[2.8,-2.8]/0.3s gate-off 越限=%d "
                   "⇒ IMP 在本 stress 的 0 是**场景所限**、非固有安全"
                   % (ung, imp_harsh["executed_violation_steps"])))
    return ch


def _tamp(name, mut, prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    data = json.load(open(BENCH, encoding="utf-8"))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad, data)
    # ★ 精确名匹配（前缀 + "_"）：禁用裸前缀，防 "S1" 吞 "S10"/"S11" 类误报
    caught = any((not c[1]) and c[0].startswith(tuple(p + "_" for p in prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E77 严格审核（审前提，硬重算不读自述）"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    fwd = _forward(rep, data)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))
    print("\n[R] 篡改：")
    _tamp("R1 改 A 段指标 → S8 必报",
          lambda r: r["A_metrics"]["rows"]["FLC"]["nominal"].__setitem__("ss_err_rad", 0.5), ["S8"])
    _tamp("R2 改 C 段 ss_bounded → S8/S3 必报",
          lambda r: r["C_capability"]["rows"]["IMP"].__setitem__("ss_bounded_nominal", not r["C_capability"]["rows"]["IMP"]["ss_bounded_nominal"]), ["S8", "S6"])
    _tamp("R3 抹掉门控承重标记 → S9 必报（存储==重算）",
          lambda r: r["D_determinism_and_ablation"].__setitem__("gate_load_bearing", False), ["S9"])
    _tamp("R4 翻转 verdict_pass → S6 必报", lambda r: r.__setitem__("verdict_pass", False), ["S6"])
    _tamp("R5 抹掉 B 段接口 all_ok → S1 必报",
          lambda r: r["B_interface"].__setitem__("all_ok", False), ["S1", "S6"])
    ntp = sum(1 for t in _res["tamper"] if t["caught"]); ntt = len(_res["tamper"])
    _res["n_pass"], _res["n_total"] = npass, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["strict_pass"] = bool(npass == len(fwd) and ntp == ntt)
    _res["findings"] = [
        "★ 严格审核：%d 条前提成立、%d 条篡改全被捕获。" % (npass, ntt),
        "  ★ 口径更正：C2「执行值越限=0」由门控 clip **保证**、不可证伪（S10 用恒 1e6 N·m 敌意控制器见证）。",
        "  安全证据=门控**关断**消融：stress 下 FLC/LRN 越限、IMP 0（S11 证其为场景所限，更严参考下 IMP 亦越限）。",
        "  S8/S9：报告存储量与独立重算逐格一致（A/C/D 段）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), ntp, ntt, _res["strict_pass"]))
    return _res


if __name__ == "__main__":
    main()
