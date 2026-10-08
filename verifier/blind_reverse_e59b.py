# -*- coding: utf-8 -*-
"""blind_reverse_e59b.py —— E59b 的 L5 逆向盲测（隔离 + 覆盖结构性反推 + 负控）。

独立性：隔离（自检 sys.modules 不含 e59b / world_physics_engine）。报告只落聚合（无 rows）→
本层只做**结构性反推 + 负控**（逐项复算不可行，已记 finding）：
  · R1 覆盖按 R 的条目齐全且各 R 计数一致；
  · R2 σ 比分档守恒（三档 n 之和 = 5×决定数）；
  · R3 负控：规格不符档（σ 比 >1.25）覆盖率 <0.90 ⇒ 覆盖度量有鉴别力（非恒真）。
产物：blind_reverse_e59b.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e59b_termination_real_loop_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e59b.json")

_res = {"experiment": "E59b 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e59b/world_physics_engine；报告无 rows → 结构性反推", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def main():
    print("=" * 88)
    print("E59b 逆向盲测（L5 隔离，结构性反推 + 负控）")
    print("=" * 88)
    chk("R0_isolation_e59b_not_imported",
        "e59b_termination_real_loop" not in sys.modules and "world_physics_engine" not in sys.modules,
        "sys.modules 含 e59b=%s wpe=%s" % ("e59b_termination_real_loop" in sys.modules, "world_physics_engine" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    R = rep["results"]; D = rep["design"]

    # R1：覆盖按 R 条目齐全 + 各 R n 一致
    cov = R["coverage"]
    chk("R1_coverage_per_R_complete",
        set(cov.keys()) == set(str(r) for r in D["check_R"]) and len({cov[k]["n"] for k in cov}) == 1,
        "R keys=%s n_set=%s" % (sorted(cov.keys()), {cov[k]["n"] for k in cov}))

    # R2：σ 比分档守恒
    sr = R["coverage_by_sigma_ratio"]
    nsum = sum(sr[k]["n"] for k in sr)
    chk("R2_sigma_ratio_conservation",
        set(sr.keys()) == {"<0.8", "0.8-1.25", ">1.25"}
        and nsum == len(D["check_R"]) * R["n_decided"],
        "n_sum=%d expect=%d" % (nsum, len(D["check_R"]) * R["n_decided"]))

    # R3：负控——规格不符档覆盖 <0.90，规格相符档 ≥0.90（覆盖度量有鉴别力）
    chk("R3_negative_control_coverage_has_teeth",
        sr[">1.25"]["coverage"] < 0.90 and sr["0.8-1.25"]["coverage"] >= 0.90,
        ">1.25 cov=%.4f ; 0.8-1.25 cov=%.4f" % (sr[">1.25"]["coverage"], sr["0.8-1.25"]["coverage"]))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass; _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
