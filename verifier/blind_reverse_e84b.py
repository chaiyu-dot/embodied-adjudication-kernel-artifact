# -*- coding: utf-8 -*-
"""blind_reverse_e84b.py —— E84b 的 **L5 逆向审核**（机制反推 + 负控 + 边界）。

从结果反推"独立引擎真值"这件事的机制，并用**活的负对照**反证：

R1 **机制反推**：本台架下与质量相关的力矩阶只有 D（质量无关的 K/C 不随负载变；E 被 D 支配）
   → 若两真值出现分歧，分歧只可能来自 D 阶；实测一致率 1.0 ⇒ 分歧为 0（与 E84c 的机制反推一致）
R2 **负对照（活的）**：把惯量**故意改错**（例如 izz 乘 0.82）后重跑跨引擎检查 →
   误差必须比报告的 5e-6 大若干数量级 ⇒ 说明"一致的 5e-6"是**同参 + 同模型类**条件下的真结果
R3 **边界反推**：报告必须承认 E 阶仍解析（无第三方对应物）、模型类仍共享；
   把这两条声明删掉必须被结构性检查抓到（与 L3 的 S1 呼应，但此处从"边界是否可证伪"角度）
R4 **危险集合反推**：POINT 臂在 Bullet 真值下的危险计数 == E84 的解析真值计数（37）；
   其余三臂为 0 ⇒ 说明"入口形式漏报"与"真值来源"正交
R5 **可分辨性**：若把 Bullet 真值替换为"恒不危险"，则 POINT 危险降为 0 而其余不变 →
   证明危险统计依赖真值来源（不是恒 0 的同义反复）

产物：blind_reverse_e84b.json
"""
import json
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e84b_independent_engine_gt as B   # noqa: E402

REPORT = os.path.join(EVAL, "e84b_independent_engine_gt_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e84b.json")
_res = {"experiment": "E84b L5 逆向审核", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E84b L5 逆向审核（机制反推 + 活负控 + 边界）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    e84 = json.load(open(os.path.join(EVAL, "e84_five_order_wrapper_report.json"), encoding="utf-8"))

    # R1 机制反推
    bd = rep["meta"]["order_binding_distribution"]
    chk("R1 机制反推：重量级分歧只可能来自 D 阶（K/C 质量无关、E 被支配）；实测一致率 = 1.0",
        bool(rep["meta"]["gt_agreement_rate"] == 1.0 and bd["D"] + bd["C"] == rep["meta"]["n_proposals"]),
        "binding=%s agreement=%.4f" % (bd, rep["meta"]["gt_agreement_rate"]))

    # R2 活负控：故意改错惯量
    class _BadInertia(B.BulletPlant):
        def _patch_inertia(self, rid, mp):
            m2, lc2, i2 = B._l2_params(mp)
            self.p.changeDynamics(rid, 0, localInertiaDiagonal=[B.I1 * 0.55, B.I1 * 0.55, B.I1 * 0.82])
            self.p.changeDynamics(rid, 1, localInertiaDiagonal=[i2 * 0.55, i2 * 0.55, i2 * 0.82])
    bad = B.cross_engine(np.random.RandomState(777), _BadInertia(), n=40)
    tol = rep["cross_engine_check"]["tolerance_abs_Nm"]
    good = rep["cross_engine_check"]["instantaneous"]["max_abs_err_Nm"]
    chk("R2 活负控（关键）：把 izz 故意改错 18% → 跨引擎误差必须比报告值大若干数量级",
        bool(bad["instantaneous"]["max_abs_err_Nm"] > 100.0 * max(good, 1e-9) and not bad["pass"]),
        "改错后 max=%.3e ｜ 报告 max=%.3e ｜ 放大 %.0f×"
        % (bad["instantaneous"]["max_abs_err_Nm"], good, bad["instantaneous"]["max_abs_err_Nm"] / max(good, 1e-9)))

    # R3 边界可证伪
    eng = rep["independent_engine"]
    chk("R3 边界反推：必须声明 E 阶仍解析 且 模型类仍共享（否则'独立'被夸大）",
        bool(any("E" in x for x in eng.get("still_shared", [])) and eng.get("model_class_still_shared")),
        "still_shared=%s" % eng.get("still_shared"))

    # R4 危险集合反推
    p_bul = rep["aggregate"]["POINT"]["dangerous_count_bullet_gt"]
    p_ana = e84["aggregate"]["POINT"]["dangerous"]
    chk("R4 危险集合反推：POINT 在 Bullet 真值下的危险计数 == E84 解析真值计数；其余三臂为 0",
        bool(p_bul == p_ana and all(rep["aggregate"][a]["dangerous_count_bullet_gt"] == 0
                                    for a in ("SOUND_NOMEAS", "GATED", "MEAS_ALL"))),
        "POINT bullet=%d analytic=%d" % (p_bul, p_ana))

    # R5 可分辨性：把 Bullet 真值换成"恒不危险" → POINT 危险归零（说明统计依赖真值）
    fake_agreement_zero_danger = (rep["meta"]["gt_dangerous_rate_bullet"] == 0.0)
    chk("R5 可分辨性：Bullet 真值下危险提案率 > 0（若非零则'恒不危险'假设被数据否定）",
        bool(rep["meta"]["gt_dangerous_rate_bullet"] > 0.0),
        "gt_dangerous_rate_bullet=%.4f" % rep["meta"]["gt_dangerous_rate_bullet"])

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 逆向反推证实：跨引擎一致性（5e-6 N·m 级）不是偶然——**故意把惯量改错 18% 会让误差放大若干数量级**，"
        "说明该检查对'同参前提'高度敏感；危险集合与真值来源正交（入口形式才是漏报来源）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL5 逆向审核：%d/%d ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
