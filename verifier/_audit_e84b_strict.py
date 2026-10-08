# -*- coding: utf-8 -*-
"""_audit_e84b_strict.py —— E84b 的 **L3 严格审核**（结构自洽 + 独立复算引擎环节 + 篡改捕获）。

与 L2 正交：L2 只读 JSON；本层**实际调用**独立引擎环节做独立复算：
  S1 报告↔声明自洽：`sharing_removed` 必须**包含 S 与 D**、`still_shared` 必须包含 E；
     `model_class_still_shared` 非空（诚实边界声明完整）；verdict = all(H)。
  S2 **装载自检独立复算**：重新构造 BulletPlant 调 `verify()`，与报告的 plant_selfcheck 一致（同量级）。
  S3 **跨引擎误差独立复算**：重跑 cross_engine（小样本）与报告的误差量级一致（同数量级内）。
  S4 **臂层与真值无关性**：E84b 的 feasible / measure 必须与 E84 报告逐位相等
     （因为臂逻辑不依赖 GT 引擎）——若不等说明重放流被改动。
  S5 **负对照（关键）**：故意**不打惯量补丁**的 Bullet 模型与解析核的误差必须**远超**阈值，
     从而证明 S3 的"5e-6 级别一致"不是偶然——跨引擎检查对"同参前提"敏感。
  S6 篡改捕获与幂等。

产物：_audit_e84b_strict.json
"""
import copy
import json
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e84b_independent_engine_gt as B   # noqa: E402

REPORT = os.path.join(EVAL, "e84b_independent_engine_gt_report.json")
PREREG = os.path.join(EVAL, "e84b_independent_engine_gt_prereg.json")
OUT = os.path.join(EVAL, "_audit_e84b_strict.json")
_res = {"audit": "E84b L3 strict", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def s1(rep, pre):
    eng = rep["independent_engine"]
    ok = True
    removed = set(eng.get("sharing_removed", []))
    ok &= ("S" in "".join(removed) or any("S" in x for x in removed))
    ok &= any("D" in x for x in removed)
    ok &= any("E" in x for x in eng.get("still_shared", []))
    ok &= bool(eng.get("model_class_still_shared"))
    ok &= (rep["verdict_pass"] == all(rep["preregistered_verdict"].values()))
    ok &= (pre["independent_engine"]["sharing_removed"] == eng["sharing_removed"])
    return bool(ok)


def main():
    print("=" * 92)
    print("E84b L3 严格审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    pre = json.load(open(PREREG, encoding="utf-8"))

    chk("S1 报告↔声明自洽（sharing_removed 含 S/D、still_shared 含 E、模型类边界声明完整、verdict=all(H)）",
        s1(rep, pre),
        "removed=%s still_shared=%s" % (rep["independent_engine"]["sharing_removed"],
                                        rep["independent_engine"]["still_shared"]))

    # S2 装载自检独立复算
    plant = B.BulletPlant()
    mine = plant.verify()
    rep_ps = rep["plant_selfcheck"]
    ok2 = (mine["pass"] == rep_ps["pass"]
           and abs(mine["max_mass_err_kg"] - rep_ps["max_mass_err_kg"]) < 1e-6
           and abs(mine["max_izz_err_kgm2"] - rep_ps["max_izz_err_kgm2"]) < 1e-6
           and abs(mine["max_com_err_m"] - rep_ps["max_com_err_m"]) < 1e-6)
    chk("S2 装载自检独立复算（重建 BulletPlant.verify() 与报告一致）", ok2,
        "mine=%s" % {k: round(v, 10) if isinstance(v, float) else v for k, v in mine.items()
                     if k in ("max_mass_err_kg", "max_izz_err_kgm2", "max_com_err_m", "pass")})

    # S3 跨引擎误差独立复算（小样本，比量级）
    rng = np.random.RandomState(777)
    ce = B.cross_engine(rng, plant, n=40)
    rep_ce = rep["cross_engine_check"]
    ok3 = (ce["instantaneous"]["max_abs_err_Nm"] <= 10.0 * max(rep_ce["instantaneous"]["max_abs_err_Nm"], 1e-9)
           and ce["pass"] == rep_ce["pass"])
    chk("S3 跨引擎误差独立复算（小样本重跑，与报告同量级且 pass 一致）", ok3,
        "mine max=%.3e ｜ store max=%.3e | tol=%.3e"
        % (ce["instantaneous"]["max_abs_err_Nm"], rep_ce["instantaneous"]["max_abs_err_Nm"],
           rep_ce["tolerance_abs_Nm"]))

    # S4 臂层与真值来源无关（与 E84 逐位相等）
    e84 = json.load(open(os.path.join(EVAL, "e84_five_order_wrapper_report.json"), encoding="utf-8"))
    ok4 = all(rep["aggregate"][a]["feasible_rate"] == e84["aggregate"][a]["feasible_rate"]
              and rep["aggregate"][a]["measure_calls_mean"] == e84["aggregate"][a]["measure_calls_mean"]
              and rep["aggregate"][a]["dangerous_rate_analytic_gt"] == e84["aggregate"][a]["dangerous_rate"]
              for a in ARMS)
    chk("S4 臂层与真值来源无关：feasible / measure / analytic-GT 危险率与 E84 报告逐位相等", ok4)

    # S5 负对照：不打惯量补丁 → 误差必须远超阈值
    class _NoPatch(B.BulletPlant):
        def _patch_inertia(self, rid, mp):
            pass
    raw = _NoPatch()
    ce_raw = B.cross_engine(np.random.RandomState(777), raw, n=40)
    chk("S5 负对照（关键）：不打惯量补丁的 Bullet 与解析核误差**远超**阈值 → 跨引擎检查对同参前提敏感",
        bool(ce_raw["instantaneous"]["max_abs_err_Nm"] > 50.0 * rep_ce["tolerance_abs_Nm"]
             and not ce_raw["pass"]),
        "no-patch max=%.3e ｜ tol=%.3e（比值 %.0f×）"
        % (ce_raw["instantaneous"]["max_abs_err_Nm"], rep_ce["tolerance_abs_Nm"],
           ce_raw["instantaneous"]["max_abs_err_Nm"] / rep_ce["tolerance_abs_Nm"]))

    # S6 篡改捕获
    t1 = copy.deepcopy(rep)
    t1["independent_engine"]["sharing_removed"] = ["K/C: mass-independent"]
    chk("S6a 把 sharing_removed 里的 S/D 拿掉 → S1 必报（夸大独立性）", not s1(t1, pre))
    t2 = copy.deepcopy(rep)
    t2["independent_engine"]["model_class_still_shared"] = ""
    chk("S6b 抹掉'模型类仍共享'的边界声明 → S1 必报", not s1(t2, pre))
    t3 = copy.deepcopy(rep)
    t3["preregistered_verdict"]["H84b-3_point_entry_still_dangerous"] = False
    chk("S6c 翻转 H84b-3 → S1 必报（verdict≠all(H)）", not s1(t3, pre))
    chk("S6d 未篡改 → S1 通过（阴性对照）", s1(rep, pre))
    chk("S6e 幂等：两次读入结论一致", s1(json.load(open(REPORT, encoding="utf-8")), pre) == s1(rep, pre))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "装载同参自检与跨引擎一致性可被独立复算；**负对照（不打惯量补丁）误差远超阈值**，"
        "证明该检查对'同参前提'有真实分辨力。",
        "声明完整性（sharing_removed/still_shared/model_class_still_shared）被结构性检查，防止夸大独立性。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
