# -*- coding: utf-8 -*-
"""blind_reverse_e79.py —— E79 的**逆向盲测（L5，隔离 + 负对照证伪）**。

独立性：隔离（运行期自检 sys.modules 未加载 e79_momentum_impulse_chain）。只信"数据文件"
（bench 声明参数 + 存储结论），独立推导冲量/CoP 的**交叉点公式**与**负对照**，证明：

  · C3 对照臂非空：静态判据（F=m·a_max）对该台架**永不拒绝**，故冲量项必须进裁决；
    若用"只按静力"的坏裁决，必漏报 v>v* 的 3 个工况（负对照）。
  · C4 对照臂非空：动态 CoP 交叉点 ax* 可独立解出，且 ax≥ax* 的 5 例静态 CoM 判安全、
    动态 CoP 判失衡；若用"忽略加速度"的坏 CoP，必漏报这 5 例（负对照）。
  · 符号/单调性：CoP 偏移量与加速度反号、随 |ax| 单调，方向正确。

产物：blind_reverse_e79.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 e79_momentum_impulse_chain 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e79_momentum_impulse_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e79.json")

_res = {"experiment": "E79 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e79_momentum_impulse_chain",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def main():
    print("=" * 88)
    print("E79 逆向盲测（L5 隔离，负对照证伪对照臂）")
    print("=" * 88)
    chk("R0_isolation_e79_not_imported",
        "e79_momentum_impulse_chain" not in sys.modules,
        "sys.modules 含 e79=%s" % ("e79_momentum_impulse_chain" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    b = rep["bench"]
    C, D = rep["C_verdict_coupling"], rep["D_cop_vs_static"]
    m, e, amax = b["m_payload_kg"], b["restitution"], b["decel_max_mps2"]
    dtc, Fallow = b["dt_contact_s"], b["F_allow_N"]
    g = 9.81

    # --- R1：静态判据 F=m·a_max 对该台架**永不拒绝** → 冲量臂结构性必要 ---
    F_static = m * amax                       # 2*8 = 16 N
    static_always_ok = all(F_static <= Fallow for _ in b["v_cap_sweep"])
    n_miss_re = sum(1 for vk, vr in C["rows"].items()
                    if vk.startswith("v=") and vr["verdict_static"] and not vr["verdict_impulse"])
    chk("R1_static_verdict_never_rejects_structural", static_always_ok and F_static <= Fallow,
        "F_static=%.1f Fallow=%.1f" % (F_static, Fallow))

    # --- R2：冲量交叉点 v* 独立解出 + 漏报数复算 ---
    v_star_re = Fallow * dtc / ((1.0 + e) * m)     # = 1.875
    n_miss_formula = sum(1 for v in b["v_cap_sweep"] if v > v_star_re)
    chk("R2_v_star_crossover_and_n_missed", abs(v_star_re - C["rows"]["_v_star_impulse_mps"]) < 1e-9
        and n_miss_formula == C["n_missed_by_static"] and n_miss_re == C["n_missed_by_static"],
        "v_star_re=%.4f formula_miss=%d stored=%d" % (v_star_re, n_miss_formula, C["n_missed_by_static"]))

    # --- R3：负对照 —— 坏裁决（只按静力）必漏报 3 个冲量受限工况 ---
    # 用 RNE.contact_impulse 独立算 F_peak 验证存储；坏裁决 = 永远用 F_static
    wrong_miss = 0
    for vk, vr in C["rows"].items():
        if not vk.startswith("v="):
            continue
        F_peak_rne = RNE.contact_impulse(m, vr["v_mps"], e) / dtc
        bad_verdict = (F_static <= Fallow)          # 坏裁决：只看静力
        if bad_verdict and not (F_peak_rne <= Fallow):
            wrong_miss += 1
    chk("R3_falsify_impulse_arm_negative_control", wrong_miss == C["n_missed_by_static"] and wrong_miss >= 1,
        "bad_verdict_missed=%d stored_missed=%d" % (wrong_miss, C["n_missed_by_static"]))

    # --- R4：动态 CoP 交叉点 ax* 独立解出 + 漏报数复算 ---
    xs = [p[0] for p in b["support_poly"]]
    xmin, xmax = min(xs), max(xs)
    # CoP 随正 ax 向 −x 偏移，故从下界 xmin 出域：cop_x = xmin ⇒ ax* = (com_x − xmin)·g/z
    ax_star_re = (b["com_xy"][0] - xmin) * g / b["com_z_m"]   # cop_x = xmin 时的 ax
    n_dyn_formula = sum(1 for ak, ar in D["rows"].items()
                        if ak.startswith("ax=") and ar["com_acc_x"] > ax_star_re)
    n_dyn_re = sum(1 for ak, ar in D["rows"].items()
                   if ak.startswith("ax=") and ar["static_com_in_support"] and not ar["dynamic_cop_in_support"])
    chk("R4_cop_crossover_and_n_missed", abs(n_dyn_formula - D["n_static_only_missed"]) < 1e-9
        and n_dyn_re == D["n_static_only_missed"] and D["n_static_only_missed"] >= 1,
        "ax_star_re=%.4f formula_miss=%d stored=%d" % (ax_star_re, n_dyn_formula, D["n_static_only_missed"]))

    # --- R5：负对照 —— 坏 CoP（忽略加速度，静态）必漏报这 5 例 ---
    wrong_dyn = 0
    for ak, ar in D["rows"].items():
        if not ak.startswith("ax="):
            continue
        bad_in = RNE.cop_in_support(b["com_xy"], b["support_poly"])   # 坏：用静态 CoM
        if bad_in and not ar["dynamic_cop_in_support"]:
            wrong_dyn += 1
    chk("R5_falsify_dynamic_cop_negative_control", wrong_dyn == D["n_static_only_missed"] and wrong_dyn >= 1,
        "bad_cop_missed=%d stored=%d" % (wrong_dyn, D["n_static_only_missed"]))

    # --- R6：CoP 偏移符号/单调性（与加速度反号、随 |ax| 单调）---
    cop_x0 = b["com_xy"][0]
    shift_ok = True
    prev_shift = None
    for ak, ar in sorted(D["rows"].items()):
        if not ak.startswith("ax="):
            continue
        ax = ar["com_acc_x"]
        # 独立重算 cop_x
        cop_x_re = RNE.dynamic_cop(b["com_xy"], b["com_z_m"], [ax, 0.0])[0]
        if abs(cop_x_re - ar["cop_x"]) >= 1e-9:
            shift_ok = False
        shift = cop_x_re - cop_x0                  # 应 = -(z/g)·ax，与 ax 反号
        if abs(shift + (b["com_z_m"] / g) * ax) >= 1e-9:
            shift_ok = False
        if prev_shift is not None and ax > 0:
            if not (shift <= prev_shift):           # ax 增大 → shift 更负（单调）
                shift_ok = False
        prev_shift = shift
    chk("R6_cop_shift_sign_and_monotonic", shift_ok, "shift 与 ax 反号且随 |ax| 单调")

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
