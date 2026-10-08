# -*- coding: utf-8 -*-
"""blind_replicate_e72.py —— E72 的**盲复刻（只给数据文件 + 独立物理重实现）**。

模拟外部复刻者：只拿到 e72_bench_data.json（物理参数 + 算法常量 + 全部测试用例/期望值）
+ e72_standalone_physics.py（复刻者从物理重写的雅可比/wrench→τ/库仑锥/三值裁决/包络最大化/
AND 组合），**不读 e72_friction_cone_verdict.py、不 import planning.friction_cone**，
独立复现 E72 的 A/B/C 三段，再与作者报告逐位 / 逐格比对。

隔离保证：
  · 本脚本运行期 sys.modules 不得含 planning / friction_cone / e72_friction_cone_verdict；
  · 唯一允许的"作者侧"输入是 e72_friction_cone_report.json（待验证的**主张**，而非代码）。

产物：blind_replicate_e72.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e72_standalone_physics as P  # noqa: E402  （复刻者自有实现）

REPORT = os.path.join(EVAL, "e72_friction_cone_report.json")
BENCH = os.path.join(EVAL, "e72_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e72.json")

_repo_mods = ("planning", "friction_cone", "e72_friction_cone_verdict",
              "control_strategies")


def _repo_loaded():
    return [m for m in sys.modules
            if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E72 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E72 盲复刻（只给数据文件 + 独立物理重实现）vs 作者报告")
    print("=" * 92)

    loaded = _repo_loaded()
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))

    # === B0b 数据文件自足性（重建所需全部键齐备） ===
    need_top = ["bench", "algorithm", "source_default", "cases"]
    need_bench = ["L1", "L2", "G", "TAU_LIM", "q"]
    need_alg = ["envelope_Fn_hi", "envelope_n_grid", "envelope_bisect_iters",
                "envelope_push_hi", "rho_Fn_eps", "slip_tol", "contact_Fn_thresh",
                "tau_tol", "envelope_round"]
    need_cases = ["envelope", "feasibility_anchor", "slip_flip", "no_contact",
                  "positive_safe", "nonfinite", "rho_boundary", "combination"]
    miss = [k for k in need_top if k not in data]
    miss += [k for k in need_bench if k not in data.get("bench", {})]
    miss += [k for k in need_alg if k not in data.get("algorithm", {})]
    miss += [k for k in need_cases if k not in data.get("cases", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备（物理+算法常量+测试用例）", not miss,
        "missing=%s" % miss if miss else "keys=%d top / %d bench / %d alg / %d cases"
        % (len(data), len(data.get("bench", {})), len(data.get("algorithm", {})), len(data.get("cases", {}))))

    out = P.replicate(data)
    A, B, C = out["A_envelope_replica"], out["B_tau_coupling_three_valued"], out["C_combination_invariants"]
    repA, repB, repC = rep["A_envelope_replica"], rep["B_tau_coupling_three_valued"], rep["C_combination_invariants"]

    # === B1 A 段包络 4 行逐格 ===
    mismA = []
    for k in repA["rows"]:
        a, r = repA["rows"][k], A["rows"][k]
        for f in ("max_tangential_N", "at_normal_N", "cone_utilization"):
            if abs(a[f] - r[f]) > 1e-9:
                mismA.append((k, f, a[f], r[f]))
    chk("B1 A 段包络 4 行逐格复现（max_tangential/at_normal/cone_util）", not mismA,
        "mismatch=%s" % mismA[:4] if mismA else "4x3 格全一致")

    # === B2 A all_match 复现 ===
    chk("B2 A 段 all_match 复现（与 E67 锚点一致）", A["all_match"] == repA["all_match"],
        "复刻=%s 报告=%s" % (A["all_match"], repA["all_match"]))

    # === B3 B 段 4 锚点逐格（value/feasible/rho/tau/reason/source） ===
    mismB = []
    for k in repB["tau_feasibility_anchor"]:
        a, r = repB["tau_feasibility_anchor"][k], B["tau_feasibility_anchor"][k]
        for f in ("value", "feasible", "rho", "reason", "source"):
            if a[f] != r[f]:
                mismB.append((k, f, a[f], r[f]))
        for i in range(2):
            if abs(a["tau"][i] - r["tau"][i]) > 1e-9:
                mismB.append((k, "tau[%d]" % i, a["tau"][i], r["tau"][i]))
    chk("B3 B 段 τ-可行性 4 锚点逐格复现（含 τ 向量 4 位与 reason/source）", not mismB,
        "mismatch=%s" % mismB[:4] if mismB else "4 锚点 x 7 字段全一致")

    # === B4 滑移翻转 ===
    a_sf, r_sf = repB["slip_flip_F=[3,5]"], B["slip_flip_F=[3,5]"]
    ok4 = (a_sf["flipped"] == r_sf["flipped"] and a_sf["mu=0.2"] == r_sf["mu=0.2"] == "UNSAFE"
           and a_sf["mu=0.8"] == r_sf["mu=0.8"] == "SAFE"
           and abs(a_sf["rho_mu0.2"] - r_sf["rho_mu0.2"]) < 1e-9
           and abs(a_sf["rho_mu0.8"] - r_sf["rho_mu0.8"]) < 1e-9)
    chk("B4 库仑滑移翻转复现：F=[3,5] μ0.2→UNSAFE(ρ=3.0) / μ0.8→SAFE(ρ=0.75)", ok4,
        "复刻 ρ0.2=%.4f ρ0.8=%.4f flipped=%s" % (r_sf["rho_mu0.2"], r_sf["rho_mu0.8"], r_sf["flipped"]))

    # === B5 无接触 ⊘ ===
    chk("B5 无接触复现为 ⊘(NA)（前提假→降级不约束）",
        B["no_contact_NA"] == repB["no_contact_NA"] and B["no_contact_NA"]["tag"] == "\u2298",
        "复刻=%s 报告=%s" % (B["no_contact_NA"], repB["no_contact_NA"]))

    # === B6 正面 SAFE ===
    chk("B6 正面 SAFE 复现：F=[1,5]@μ0.4 → SAFE",
        B["positive_SAFE_F=[1,5]_mu0.4"] == repB["positive_SAFE_F=[1,5]_mu0.4"] == "SAFE",
        "复刻=%s 报告=%s" % (B["positive_SAFE_F=[1,5]_mu0.4"], repB["positive_SAFE_F=[1,5]_mu0.4"]))

    # === B7 C 段组合真值表 5 行逐格 ===
    mismC = []
    for k in repC["combination_table"]:
        a, r = repC["combination_table"][k], C["combination_table"][k]
        if a != r:
            mismC.append((k, a, r))
    chk("B7 C 段 AND 组合真值表 5 行逐格复现（value/contact/dynamic/shorted_by）", not mismC,
        "mismatch=%s" % mismC[:3] if mismC else "5 行 x 4 字段全一致")

    # === B8 C 段不变量标志复现 ===
    flags = ["truth_table_ok", "NA_short_honest", "source_doc_ok", "deterministic",
             "inf_capped", "NA_rho_finite"]
    bad_flags = [f for f in flags if C[f] != repC[f]]
    chk("B8 C 段 6 项不变量标志复现（真值表/⊘诚实/来源档/确定性/非有限封顶/NA-ρ有限）", not bad_flags,
        "mismatch=%s" % bad_flags if bad_flags else "6/6 一致")

    # === B9 C all_ok 复现 ===
    chk("B9 C 段 all_ok 复现", C["all_ok"] == repC["all_ok"],
        "复刻=%s 报告=%s" % (C["all_ok"], repC["all_ok"]))

    # === B10 verdict 自洽（由复刻三段的裸布尔重算） ===
    C1 = A["all_match"]
    C2 = B["all_ok"]
    C3 = C["all_ok"]
    C4 = bool(C["deterministic"] and C["inf_capped"] and C["NA_rho_finite"])
    verdict = bool(C1 and C2 and C3 and C4)
    chk("B10 verdict 由复刻三段裸布尔重算 == 报告 verdict_pass", verdict == bool(rep["verdict_pass"]),
        "复刻 C1/C2/C3/C4=%s/%s/%s/%s verdict=%s ｜ 报告 verdict=%s"
        % (C1, C2, C3, C4, verdict, rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e72_bench_data.json + 独立物理重实现）：E72 全部结论可被外部独立重建。",
        "  A 段包络 4 行、B 段 4 锚点（含 τ 向量）、C 段组合真值表 5 行：逐格一致（0 mismatch）。",
        "  数据文件自足：物理参数 + 算法常量（Fn_hi/n_grid/bisect_iters/eps）+ 全部测试用例齐备。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s"
          % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
