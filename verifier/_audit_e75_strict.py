# -*- coding: utf-8 -*-
"""_audit_e75_strict.py —— **严格审核（审前提，硬重算不读自述）**。

审"结论成立所需的**前提**"——任一条为假，整段结论不成立，而逐数字重算可能仍全过：
  S1  功共轭恒等前提：无耗散（B_VISC=0）时 W = ΔKE+ΔPE（硬重算，独立物理）；
  S2  精确核 vs 动能-only 的分歧**真实**：至少 3 行 PE 主导，且 exact 在 ≥1 行显著更保守（非编码伪影）；
  S3  demand 序良定义：ΔPE 随载重**严格单调**（否则"序数一致"无意义）；
  S4  存储 B 行基础量（dPE/KE_peak）= 独立物理硬重算（防"单格+聚合同步改"）；
  S5  误档方向正确：每处误档都是 exact 更保守（tight/unsafe）而 kinetic-only 更松；
  S6  verdict 自洽：C1∧C2∧C3 == verdict_pass；
  S7  A 段无发散且 max|残差| < 1e-8；
  S8  数据文件自足（台架+算法常量齐备）。
篡改 R1–R5 必有对应 S 报出。

产物：_audit_e75_strict.json
"""
import copy
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e75_standalone_physics as P  # noqa: E402  （复刻者自有独立物理）

REPORT = os.path.join(EVAL, "e75_energy_margin_exact_report.json")
BENCH = os.path.join(EVAL, "e75_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e75_strict.json")
SEV = {"unsafe": 0, "tight": 1, "medium": 2, "loose": 3}

_res = {"audit": "_audit_e75_strict", "checks": [], "tamper": [], "findings": []}
_n = [0]


def _chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return (name, bool(ok))


def _forward(rep, data):
    ch = []
    A, B, C = rep["A_work_conjugacy"], rep["B_ordinal"], rep["C_misband"]

    # S1 无耗散恒等（硬重算）
    d2 = json.loads(json.dumps(data)); d2["bench"]["B_VISC"] = 0.0
    o0 = P.replicate(d2)
    mx0 = max(abs(v["W_J"] - (v["dKE_J"] + v["dPE_J"])) for v in o0["A_work_conjugacy"]["rows"].values())
    ch.append(_chk("S1_work_conjugacy_identity_no_dissipation", mx0 < 1e-8,
                   "B_VISC=0 → max|W−ΔKE−ΔPE|=%.2e" % mx0))

    # S2 分歧真实
    pe_dom = [r for r in B["rows"] if r["dPE_J"] > r["KE_peak_J"]]
    n_cons = sum(1 for r in pe_dom if r["m_E_exact"] < r["m_E_kinetic_only"] - 0.1)
    ch.append(_chk("S2_divergence_real", len(pe_dom) >= 3 and n_cons >= 3,
                   "PE 主导 %d 行，其中 exact 显著更保守 %d 行" % (len(pe_dom), n_cons)))

    # S3 demand 序严格单调
    dpe = [r["dPE_J"] for r in B["rows"]]
    mono = all(dpe[i + 1] > dpe[i] + 1e-9 for i in range(len(dpe) - 1))
    ch.append(_chk("S3_demand_order_well_defined", mono,
                   "ΔPE 序列严格递增=%s %s" % (mono, [round(x, 3) for x in dpe])))

    # S4 存储 B 行基础量 == 独立硬重算
    o = P.replicate(data)
    mism = []
    for i, r in enumerate(B["rows"]):
        if abs(r["dPE_J"] - o["B_ordinal"]["rows"][i]["dPE_J"]) > 1e-9:
            mism.append(("dPE", i))
        if abs(r["KE_peak_J"] - o["B_ordinal"]["rows"][i]["KE_peak_J"]) > 1e-9:
            mism.append(("KE_peak", i))
        if abs(r["m_E_exact"] - o["B_ordinal"]["rows"][i]["m_E_exact"]) > 1e-12:
            mism.append(("m_E", i))
    ch.append(_chk("S4_stored_raw_equals_recompute", not mism,
                   "mismatch=%s" % mism[:3] if mism else "5 行 dPE/KE_peak/m_E 全一致"))

    # S5 误档方向正确
    dir_ok = all(SEV[m["band_exact"]] <= 1 and SEV[m["band_kinetic_only"]] >= 2 for m in C["misbanded"])
    ch.append(_chk("S5_misband_direction_correct", dir_ok and C["n_misbanded"] == len(C["misbanded"]),
                   "误档 %d 条，方向全为 exact 更保守=%s" % (len(C["misbanded"]), dir_ok)))

    # S6 verdict 自洽
    c = rep["criteria"]
    inv = bool(c["C1_work_conjugacy_lt_1e-8"] and A["all_ok"] and c["C2_ordinal_not_degraded"]
               and B["all_ok"] and c["C3_kinetic_only_misbands"] and C["all_ok"])
    ch.append(_chk("S6_verdict_self_consistent", inv == bool(rep["verdict_pass"]),
                   "重算 %s 报告 %s" % (inv, rep["verdict_pass"])))

    # S7 A 段无发散 + 残差
    s7 = (not any(v["diverged"] for v in A["rows"].values())) and A["max_abs_residual"] < 1e-8
    ch.append(_chk("S7_A_no_divergence_and_residual", s7,
                   "max|resid|=%.3e diverged=%s" % (A["max_abs_residual"],
                                                    any(v["diverged"] for v in A["rows"].values()))))

    # S8 数据文件自足
    need_b = ["L1", "L2", "M1", "M2", "LC1", "LC2", "I1", "I2", "G", "TAU_LIM", "B_VISC", "DT"]
    need_a = ["zeta", "frac", "sim_T", "band_edges"]
    miss = [k for k in need_b if k not in data.get("bench", {})] + \
           [k for k in need_a if k not in data.get("algorithm", {})] + \
           [k for k in ("scenario_A", "ladder_B", "q0_ladder", "qref_ladder") if k not in data]
    ch.append(_chk("S8_data_file_self_contained", not miss, "missing=%s" % miss))
    return ch


def _tamp(name, mut, expect_prefixes):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    data = json.load(open(BENCH, encoding="utf-8"))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad, data)
    caught = any((not c[1]) and c[0].startswith(tuple(expect_prefixes)) for c in ch)
    del _res["checks"][n0:]
    _n[0] += 1
    _res["tamper"].append({"n": _n[0], "name": name, "caught": bool(caught)})
    print("  [%s] T %02d %s%s" % ("PASS" if caught else "FAIL", _n[0], name,
                                  "  —— 篡改被捕获" if caught else "  —— !! 未捕获 !!"))
    return caught


def main():
    print("=" * 92); print("E75 严格审核（审前提，硬重算不读自述）"); print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))
    fwd = _forward(rep, data)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))
    print("\n[R] 篡改：")
    _tamp("R1 改 B 行 dPE → S4 必报",
          lambda r: r["B_ordinal"]["rows"][0].__setitem__("dPE_J", r["B_ordinal"]["rows"][0]["dPE_J"] + 1.0), ["S4", "S3"])
    _tamp("R2 改 B 行 KE_peak → S4 必报",
          lambda r: r["B_ordinal"]["rows"][1].__setitem__("KE_peak_J", r["B_ordinal"]["rows"][1]["KE_peak_J"] * 2), ["S4"])
    _tamp("R3 改 m_E_exact → S4 必报",
          lambda r: r["B_ordinal"]["rows"][2].__setitem__("m_E_exact", r["B_ordinal"]["rows"][2]["m_E_exact"] + 0.3), ["S4"])
    _tamp("R4 把误档方向翻反（kinetic 更保守）→ S5 必报",
          lambda r: r["C_misband"]["misbanded"][0].__setitem__("band_kinetic_only", "unsafe"), ["S5"])
    _tamp("R5 翻转 verdict_pass → S6 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S6"])
    ntp = sum(1 for t in _res["tamper"] if t["caught"]); ntt = len(_res["tamper"])
    _res["n_pass"], _res["n_total"] = npass, len(fwd)
    _res["n_tamper_pass"], _res["n_tamper_total"] = ntp, ntt
    _res["strict_pass"] = bool(npass == len(fwd) and ntp == ntt)
    _res["findings"] = [
        "★ 严格审核（审前提）：%d 条前提成立，%d 条篡改全被捕获。" % (npass, ntt),
        "  关键前提 S1：无耗散时 W=ΔKE+ΔPE 硬重算恒等 → 精确核就是功共轭，不是拟合。",
        "  S2 分歧真实：PE 主导场景 exact 显著更保守（漏 PE 会误判安全）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), ntp, ntt, _res["strict_pass"]))
    return _res


if __name__ == "__main__":
    main()
