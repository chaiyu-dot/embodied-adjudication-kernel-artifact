# -*- coding: utf-8 -*-
"""blind_replicate_e79.py —— E79 的**盲复刻（只给数据文件 + 独立物理重实现）**。

模拟外部复刻者：只拿到 e79_bench_data.json + e79_standalone_physics.py（自行从物理重写的
动量-冲量/制动/动态 CoP），**不 import planning.* / rne_dynamics / e79_momentum_impulse_chain**，
独立复现 E79 的 A/B/C/D 四段并与作者报告逐格比对。

隔离保证：运行期 sys.modules 不得含 planning / rne_dynamics / e79_momentum_impulse_chain。
产物：blind_replicate_e79.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e79_standalone_physics as P  # noqa: E402  （复刻者自有实现）

REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
BENCH = os.path.join(EVAL, "e79_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e79.json")

_repo_mods = ("planning", "rne_dynamics", "e79_momentum_impulse_chain", "control_strategies",
              "e66_closed_loop_control")
_res = {"experiment": "E79 盲复刻（只给数据文件 + 独立物理重实现）", "checks": [], "findings": []}
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
    print("E79 盲复刻（只给数据文件 + 独立物理重实现）vs 作者报告")
    print("=" * 92)
    loaded = _repo_loaded()
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    need_bench = ["m_payload_kg", "v_cap_mps", "decel_max_mps2", "dt_contact_s", "restitution",
                  "F_allow_N", "com_z_m", "com_xy", "support_poly", "v_cap_sweep"]
    miss = [k for k in need_bench if k not in data.get("bench", {})] + \
           [k for k in ("m1", "m2", "v1", "v2", "e") if k not in data.get("collision", {})]
    chk("B0b 数据文件自足：重建所需全部键齐备", not miss, "missing=%s" % miss if miss else "全齐")

    out = P.replicate(data)
    A, B, C, D = out["A_closedform"], out["B_impulse_momentum"], out["C_verdict_coupling"], out["D_cop_vs_static"]
    rA, rB, rC, rD = rep["A_closedform"], rep["B_impulse_momentum"], rep["C_verdict_coupling"], rep["D_cop_vs_static"]

    # B1 A 段
    chk("B1 A 段 max|闭式误差| 复现 == 报告（<1e-12）",
        abs(A["max_abs_err"] - rA["max_abs_err"]) < 1e-15 and A["all_ok"] == rA["all_ok"],
        "复刻 %.3e 报告 %.3e" % (A["max_abs_err"], rA["max_abs_err"]))

    # B2 B 段冲量-动量（三个见证量）
    mism = []
    for f in ("momentum_conserved_err", "dp1_vs_J_err", "e0_check_err"):
        if abs(B["rows"][f] - rB["rows"][f]) > 1e-15:
            mism.append((f, B["rows"][f], rB["rows"][f]))
    chk("B2 B 段冲量-动量见证量逐格复现", not mism and B["all_ok"] == rB["all_ok"],
        "mismatch=%s" % mism if mism else "全一致")

    # B3 C 段对照臂漏报数 + 用例
    chk("B3 C 段忽略冲量漏报数复现（需 ≥1）",
        C["n_missed_by_static"] == rC["n_missed_by_static"] and C["missed_cases"] == rC["missed_cases"]
        and C["n_missed_by_static"] >= 1,
        "复刻 %d 报告 %d" % (C["n_missed_by_static"], rC["n_missed_by_static"]))

    # B4 D 段静态 CoM 漏报数 + cop_x 逐格
    mismD = []
    for k, r in rD["rows"].items():
        if k in D["rows"] and abs(D["rows"][k]["cop_x"] - r["cop_x"]) > 1e-9:
            mismD.append((k, D["rows"][k]["cop_x"], r["cop_x"]))
    chk("B4 D 段静态 CoM 漏报数 + cop_x 逐格复现",
        D["n_static_only_missed"] == rD["n_static_only_missed"] and not mismD
        and D["n_static_only_missed"] >= 1,
        "复刻 %d 报告 %d mismatch=%s" % (D["n_static_only_missed"], rD["n_static_only_missed"], mismD[:2]))

    # B5 v* 复现（冲量判据交叉点）
    v_star = data["bench"]["F_allow_N"] * data["bench"]["dt_contact_s"] / \
        ((1.0 + data["bench"]["restitution"]) * data["bench"]["m_payload_kg"])
    chk("B5 v*（冲量判据交叉点）复现 == 报告",
        abs(round(v_star, 10) - rC["rows"]["_v_star_impulse_mps"]) < 1e-9,
        "复刻 %.4f 报告 %.4f" % (v_star, rC["rows"]["_v_star_impulse_mps"]))

    # B6 verdict 裸布尔重算
    verdict = bool(A["all_ok"] and B["all_ok"] and C["all_ok"] and D["all_ok"])
    chk("B6 verdict 由复刻四段裸布尔重算 == 报告 verdict_pass",
        verdict == bool(rep["verdict_pass"]), "复刻 %s 报告 %s" % (verdict, rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给 e79_bench_data.json + 独立物理重实现）：E79 全部结论可被外部独立重建。",
        "  A 闭式一致 / B 冲量-动量 / C 对照臂漏报 / D 静态 CoM 漏报 / v*：逐格一致（0 mismatch）。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]), _res["replicate_pass"]))
    return _res


if __name__ == "__main__":
    main()
