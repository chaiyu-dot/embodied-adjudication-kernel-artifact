# -*- coding: utf-8 -*-
"""blind_reverse_e86.py —— E86 的 **L5 逆向审核**（机制反推 + 负控 + 边界）。

从结果反推"包裹层把不可预测变成可预测"的机制，并用可分辨性反证：

R1 **机制反推**：系统输出 = 1.0 的**充分条件**是"每一次 step1 失败都被修复" →
   原始行里不得存在「step1 错 ∧ step2 错」的行（由 E19 原始日志独立核验）
R2 **恒等的机制来源**：`correction_flip ≡ (step1 错 ∧ step2 对)` —— 这正是说明
   『修复率与接口精度 ρ = −1 是同义反复』的根据；把该恒等式写进报告是**自我纠错留痕**
R3 **"体量 ⇒ 精度"机制被反证**：接口精度**不随体量单调**（如 8B-fp8 = 0.00 < 1B = 0.525）
R4 **可分辨性（负对照）**：若所有模型都"完全不会"（接口精度全 0），则系统输出仍应为 1.0
   ⇒ 现有两个零精度模型已实现该边界，说明"最差 = 最好"不是平凡陈述
R5 **边界**：报告必须**同时**给出"接口精度极差大"与"系统输出极差 0"两侧数字，
   并显式声明它们**不能**推出"模型更强 → 接口更准"（该因果链本件无证据）

产物：blind_reverse_e86.json
"""
import glob
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e86_interface_accuracy_real_models_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e86.json")
_res = {"experiment": "E86 L5 逆向审核", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E86 L5 逆向审核（机制反推 + 负控 + 边界）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))

    # R1 机制反推：不存在「step1 错 ∧ step2 错」的行
    bad = 0
    total = 0
    for f in sorted(glob.glob(os.path.join(EVAL, "e19_r2_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get("rows", []):
            e = r.get("error")
            if not isinstance(e, dict) or "step1_correct" not in e:
                continue
            total += 1
            if (not e["step1_correct"]) and (not e["step2_correct"]):
                bad += 1
    chk("R1 机制反推：不存在『step1 错 ∧ step2 错』的行 ⇒ 系统输出恒 1.0 的充分条件成立",
        bad == 0 and total > 0, "总行数=%d，双错行=%d" % (total, bad))

    # R2 恒等的机制来源
    ident_hold = True
    for f in sorted(glob.glob(os.path.join(EVAL, "e19_r2_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get("rows", []):
            e = r.get("error")
            if not isinstance(e, dict) or "step1_correct" not in e:
                continue
            if bool(e.get("correction_flip")) != (not e["step1_correct"] and e["step2_correct"]):
                ident_hold = False
    chk("R2 恒等的机制来源：`correction_flip ≡ (step1 错 ∧ step2 对)` ⇒ ρ=−1 是同义反复",
        ident_hold and rep["summary"]["repair_identity_maxdev"] == 0.0,
        "ident_dev=%.4f" % rep["summary"]["repair_identity_maxdev"])

    # R3 体量 ⇒ 精度 被反证
    by_name = {m["model"]: m["interface_accuracy_step1"] for m in rep["models"]}
    fp8 = [v for k, v in by_name.items() if "8b-fp8" in str(k).lower()]
    oneb = [v for k, v in by_name.items() if "1b" in str(k).lower()]
    chk("R3 反证『体量 ⇒ 精度』：8B-fp8 的接口精度 ≤ 1B（精度不随体量单调）",
        bool(fp8 and oneb and max(fp8) < min(oneb)),
        "8B-fp8=%s ｜ 1B=%s" % (fp8, oneb))

    # R4 可分辨性（负对照）
    zeros = [m for m in rep["models"] if m["interface_accuracy_step1"] == 0.0]
    chk("R4 可分辨性：零精度模型仍得到系统输出 1.0（『最差 = 最好』非平凡）",
        bool(zeros) and all(m["system_output_step2"] == 1.0 for m in zeros),
        "零精度模型 = %s" % [m["model"] for m in zeros])

    # R5 边界：两侧数字都在 + 显式否认因果链
    has_both = (rep["summary"]["interface_accuracy_spread"] > 0.3
                and rep["summary"]["system_output_spread"] == 0.0)
    denies = any("无本项目证据" in x or "不随体量单调" in x for x in rep["honest_notes"])
    chk("R5 边界：报告同时给出『接口精度极差大』与『系统输出极差 0』，且显式否认『模型更强→接口更准』",
        bool(has_both and denies),
        "spread=%.4f/%.4f denies=%s" % (rep["summary"]["interface_accuracy_spread"],
                                        rep["summary"]["system_output_spread"], denies))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 逆向反推证实：系统输出恒 1.0 是因为**每一次接口失败都被测量修复**（无「双错」行）；"
        "ρ=−1 是 `correction_flip` 定义带来的**恒等**；而「体量 ⇒ 精度」被 8B-fp8 vs 1B 的实测反证。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL5 逆向审核：%d/%d ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
