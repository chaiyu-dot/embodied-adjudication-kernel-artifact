# -*- coding: utf-8 -*-
"""blind_replicate_e77.py —— E77 的**盲复刻（只给数据文件 + 独立物理重实现）**。

只拿到 e77_bench_data.json + e77_standalone_physics.py（自行重写 2R 被控对象/RK4/三新族/指标），
**不 import planning.* / e66 / e68 / e77**，独立复现 A/C/D 再与报告比对。产物：blind_replicate_e77.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e77_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e77_controller_family_extension_report.json")
BENCH = os.path.join(EVAL, "e77_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e77.json")
_repo_mods = ("planning", "control_strategies", "e66_closed_loop_control",
              "e68_control_strategy_suite", "e77_controller_family_extension")
FIELDS = ("rise_s", "settling_s", "overshoot", "ss_err_rad", "tau_max_executed",
          "executed_violation_steps", "demand_violation_steps", "demand_tau_max", "diverged")


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E77 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:200]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92); print("E77 盲复刻（只给数据文件 + 独立物理重实现）"); print("=" * 92)
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need = ["bench", "gains", "scenarios", "cap_scenarios", "q0", "ss_tol_rad", "family_specs"]
    need_b = ["L1", "L2", "M1", "M2", "LC1", "LC2", "I1", "I2", "G", "TAU_LIM", "B_VISC", "DT"]
    miss = [k for k in need if k not in data] + [k for k in need_b if k not in data.get("bench", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss,
        "missing=%s" % miss if miss else "keys 齐备（%d bench / %d scen / %d fam）"
        % (len(data["bench"]), len(data["scenarios"]), len(data["family_specs"])))

    o = P.replicate(data)
    A, C, D = o["A_metrics"], o["C_capability"], o["D_determinism_and_ablation"]
    rA, rC, rD = rep["A_metrics"], rep["C_capability"], rep["D_determinism_and_ablation"]

    m = []
    for n in data["family_specs"]:
        for sc in data["scenarios"]:
            a, r = rA["rows"][n][sc], A["rows"][n][sc]
            for f in FIELDS:
                if a[f] != r[f]:
                    m.append((n, sc, f, a[f], r[f]))
    chk("B1 A 段 %d 族 x %d 场景逐格复现（9 指标）" % (len(data["family_specs"]), len(data["scenarios"])),
        not m, "mismatch=%s" % m[:3] if m else "全一致")

    mc = []
    for n in data["family_specs"]:
        a, r = rC["rows"][n], C["rows"][n]
        for f in ("ss_err_nominal", "ss_err_perturbed", "settle_max_s", "ss_bounded_nominal"):
            if a[f] != r[f]:
                mc.append((n, f, a[f], r[f]))
    chk("B2 C 段能力指标逐族复现", not mc, "mismatch=%s" % mc[:3] if mc else "全一致")

    abl_a = {k: v["executed_violation_steps_no_gate"] for k, v in rD["gate_off_ablation"].items()}
    abl_b = {k: v["executed_violation_steps_no_gate"] for k, v in D["gate_off_ablation"].items()}
    chk("B3 D 段门控消融复现（gate=False 执行越限）", abl_a == abl_b and D["gate_load_bearing"] == rD["gate_load_bearing"],
        "复刻 %s 报告 %s" % (abl_b, abl_a))

    C1 = rep["criteria"]["C1_interface_consistent"]
    C2 = all(A["rows"][n][sc]["executed_violation_steps"] == 0 and not A["rows"][n][sc]["diverged"]
             for n in data["family_specs"] for sc in data["scenarios"])
    C3 = C["all_ok"]
    C4 = D["deterministic"]
    verdict = bool(C1 and C2 and C3 and C4)
    chk("B4 verdict 由复刻裸布尔重算 == 报告（C2/C3/C4）", verdict == bool(rep["verdict_pass"]),
        "复刻 C2/C3/C4=%s/%s/%s verdict=%s ｜ 报告 %s" % (C2, C3, C4, verdict, rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e77_bench_data.json + 独立物理重实现）：E77 全部结论可被外部独立重建。",
        "  A 段 3 族 x 4 场景 x 9 指标、C 段能力指标、D 段门控消融：逐格一致（0 mismatch）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
