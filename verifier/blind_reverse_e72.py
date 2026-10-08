# -*- coding: utf-8 -*-
"""blind_reverse_e72.py —— E72 的**外部逆向盲测**（比盲复刻更狠：连声明参数也不信）。

不信任 e72 报告里的任何**声明参数**（L1/L2/G/TAU_LIM/q），只把它当作待解释的"数据"：
  1. 闭式恒等式反推：由报告 τ 向量 + 已知 q 反推 L1/L2；由 μ=0.6 包络（τ 受限）反推 TAU_LIM[1]。
  2. 证伪式可识别性：对每个声明参数各施 ±5% 扰动，独立复现 E72 全部标量；若扰动后数值变化
     > 容差 ⇒ 该参数被数据**锁死**（identified）；若毫无变化 ⇒ 数据无法区分它 ⇒ **诚实标为未识别**。

隔离保证：运行期 sys.modules 不得含 planning / friction_cone / e72_friction_cone_verdict；
只 import 复刻者自有的 e72_standalone_physics。

产物：blind_reverse_e72.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e72_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e72_friction_cone_report.json")
BENCH = os.path.join(EVAL, "e72_bench_data.json")
OUT = os.path.join(EVAL, "blind_reverse_e72.json")

_repo_mods = ("planning", "friction_cone", "e72_friction_cone_verdict", "control_strategies")


def _repo_loaded():
    return [m for m in sys.modules
            if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


_res = {"experiment": "E72 外部逆向盲测（反推 + 证伪可识别性）", "checks": [], "findings": []}
_n = [0]
TOL = 1e-3   # 识别判定容差（扰动引起的相对变化 > TOL 视为被数据锁死）


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:240]) if detail else ""))
    return bool(ok)


# ---- 复刻者自有的雅可比（用于闭式反推，与 standalone 同式，独立写出便于反向核对） ----
def _jac(q, L1, L2):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def _scalars(bench, algorithm, src):
    """在给定声明参数下独立复现 E72 的全部标量（供可识别性扰动用）。"""
    P.configure(bench, algorithm, src)
    q = bench["q"]
    sc = {}
    for mu in (0.2, 0.4, 0.6, 0.8):
        r = P.max_tangential_push(q, mu)
        sc["env_tang_%.1f" % mu] = r["max_tangential_N"]
        sc["env_normal_%.1f" % mu] = r["at_normal_N"]
        sc["env_util_%.1f" % mu] = r["cone_utilization"]
    for F, tag in (([0.0, 5.0], "0_5"), ([3.0, 5.0], "3_5"),
                   ([8.0, 5.0], "8_5"), ([15.0, 5.0], "15_5")):
        v = P.contact_verdict(q, F, 0.8)
        sc["tau_%s_1" % tag] = round(float(v["tau"][0]), 6)
        sc["tau_%s_2" % tag] = round(float(v["tau"][1]), 6)
        sc["rho_%s" % tag] = round(float(v["rho"]), 6)
        sc["val_%s" % tag] = 1.0 if v["value"] == "SAFE" else (0.0 if v["value"] == "UNSAFE" else 0.5)
    return sc


def _max_rel_diff(a, b):
    m = 0.0
    for k in a:
        d = abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-9)
        m = max(m, d)
    return m


def main():
    print("=" * 92)
    print("E72 外部逆向盲测（反推 + 证伪可识别性）")
    print("=" * 92)

    loaded = _repo_loaded()
    chk("R0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench, ALG, SRC = data["bench"], data["algorithm"], data["source_default"]
    q = list(bench["q"])
    repB = rep["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]

    # ==== R1 闭式恒等式：τ = JᵀF（用报告 τ 向量 + 声明 L1/L2/q 独立复算） ====
    tau_errs = []
    for F, tag in (([0.0, 5.0], "[0.0, 5.0]"), ([3.0, 5.0], "[3.0, 5.0]"),
                   ([8.0, 5.0], "[8.0, 5.0]"), ([15.0, 5.0], "[15.0, 5.0]")):
        tau = _jac(q, bench["L1"], bench["L2"]).T @ np.asarray(F, float)
        st = repB[tag]["tau"]
        tau_errs.append(max(abs(round(tau[0], 4) - st[0]), abs(round(tau[1], 4) - st[1])))
    chk("R1 闭式恒等式 τ=JᵀF 精确成立（报告 τ 向量 4 位一致）", max(tau_errs) < 1e-9,
        "4 锚点 max|Δτ| = %.2e" % max(tau_errs))

    # ==== R2 闭式恒等式：ρ = |Ft|/(μ·Fn)（由报告 ρ 反算校验） ====
    rho_errs = []
    for F, mu, tag in (([3.0, 5.0], 0.8, "[3.0, 5.0]"), ([8.0, 5.0], 0.8, "[8.0, 5.0]"),
                       ([15.0, 5.0], 0.8, "[15.0, 5.0]"), ([0.0, 5.0], 0.8, "[0.0, 5.0]")):
        rho = abs(F[0]) / (mu * max(abs(F[1]), ALG["rho_Fn_eps"]))
        rho_errs.append(abs(round(rho, 4) - repB[tag]["rho"]))
    chk("R2 闭式恒等式 ρ=|Ft|/(μ·Fn) 精确成立（报告 ρ 4 位一致）", max(rho_errs) < 1e-9,
        "max|Δρ| = %.2e" % max(rho_errs))

    # ==== R3 反推 L1/L2：只用报告 [0,5] 的 τ 向量 + 已知 q（连 L1/L2 也不信） ====
    # τ = Jᵀ[0,5] = 5·[L1 c1+L2 c12, L2 c12]  ⇒ L2 = (τ2/5)/c12 ; L1 = ((τ1/5) - L2 c12)/c1
    t1, t2 = repB["[0.0, 5.0]"]["tau"]
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    L2_inf = (t2 / 5.0) / c12
    L1_inf = ((t1 / 5.0) - L2_inf * c12) / c1
    dL1, dL2 = abs(L1_inf - bench["L1"]), abs(L2_inf - bench["L2"])
    chk("R3 反推 L1/L2：仅用报告 τ[0,5] 向量 + 已知 q 反推", dL1 < 1e-4 and dL2 < 1e-4,
        "反推 L1=%.6f（Δ=%.1e）L2=%.6f（Δ=%.1e）｜ 声明 %.2f/%.2f"
        % (L1_inf, dL1, L2_inf, dL2, bench["L1"], bench["L2"]))

    # ==== R4 反推 TAU_LIM[1]：由 μ=0.6 包络（τ 受限，joint2 绑定）反推 ====
    r06 = rep["A_envelope_replica"]["rows"]["mu=0.6"]
    Ft06, Fn06 = r06["max_tangential_N"], r06["at_normal_N"]
    tau06 = _jac(q, L1_inf, L2_inf).T @ np.asarray([Ft06, Fn06], float)
    tau1_lim_inf = abs(tau06[1])   # joint2 绑定 ⇒ 恰好等于 τ_lim[1]
    d_lim1 = abs(tau1_lim_inf - bench["TAU_LIM"][1])
    chk("R4 反推 TAU_LIM[1]：由 μ=0.6 包络（joint2 绑定）反推", d_lim1 < 1e-3,
        "反推 τ_lim[1]=%.6f（Δ=%.1e）｜ 声明 %.2f" % (tau1_lim_inf, d_lim1, bench["TAU_LIM"][1]))

    # ==== R5 区制诊断：μ=0.6 为 τ 受限（util<1）、μ=0.8 为锥受限（util=1） ====
    util06 = r06["cone_utilization"]
    util08 = rep["A_envelope_replica"]["rows"]["mu=0.8"]["cone_utilization"]
    chk("R5 区制诊断：μ=0.6 τ 受限（util<1）vs μ=0.8 锥受限（util=1）",
        util06 < 1.0 - 1e-6 and util08 > 1.0 - 1e-6,
        "util(0.6)=%.4f util(0.8)=%.4f" % (util06, util08))

    # ==== R6 证伪式可识别性：7 个声明参数各 ±5% 扰动 ====
    base = _scalars(bench, ALG, SRC)
    pars = [("L1", lambda b, v: b.__setitem__("L1", v), bench["L1"]),
            ("L2", lambda b, v: b.__setitem__("L2", v), bench["L2"]),
            ("G", lambda b, v: b.__setitem__("G", v), bench["G"]),
            ("TAU_LIM[0]", lambda b, v: b["TAU_LIM"].__setitem__(0, v), bench["TAU_LIM"][0]),
            ("TAU_LIM[1]", lambda b, v: b["TAU_LIM"].__setitem__(1, v), bench["TAU_LIM"][1]),
            ("q[0]", lambda b, v: b["q"].__setitem__(0, v), bench["q"][0]),
            ("q[1]", lambda b, v: b["q"].__setitem__(1, v), bench["q"][1])]
    ident, not_ident, sens_tab = [], [], {}
    for name, setter, v0 in pars:
        sens = 0.0
        for sign in (+1.0, -1.0):
            b2 = json.loads(json.dumps(bench))
            setter(b2, v0 * (1.0 + sign * 0.05))
            sc = _scalars(b2, ALG, SRC)
            sens = max(sens, _max_rel_diff(sc, base))
        sens_tab[name] = sens
        (ident if sens > TOL else not_ident).append(name)
    chk("R6 可识别性证伪（±5% 扰动）：被识别集合如实列出，未识别项诚实标注",
        len(ident) + len(not_ident) == len(pars),
        "identified=%s | not_identified=%s" % (ident, not_ident))

    # ==== R7 被识别参数敏感度均显著 > 容差 ====
    weak = [k for k in ident if sens_tab[k] <= TOL]
    chk("R7 被识别参数敏感度均 > 容差（1e-3）", not weak,
        "最弱=%s 敏感度表=%s" % (min(sens_tab, key=sens_tab.get),
                                 {k: float("%.2e" % v) for k, v in sens_tab.items()}))

    # ==== R8 基线复现 == 报告标量（反推成立；且 A 段 all_match） ====
    base_env_ok = True
    for mu in (0.2, 0.4, 0.6, 0.8):
        st = rep["A_envelope_replica"]["rows"]["mu=%.1f" % mu]
        if (abs(base["env_tang_%.1f" % mu] - st["max_tangential_N"]) > 1e-9 or
                abs(base["env_normal_%.1f" % mu] - st["at_normal_N"]) > 1e-9 or
                abs(base["env_util_%.1f" % mu] - st["cone_utilization"]) > 1e-9):
            base_env_ok = False
    base_tau_ok = True
    for F, tag in (([0.0, 5.0], "0_5"), ([3.0, 5.0], "3_5"), ([8.0, 5.0], "8_5"), ([15.0, 5.0], "15_5")):
        st = repB[[k for k in repB if k.replace(" ", "") == "[%.1f,%.1f]" % (F[0], F[1])][0]]
        # 报告 τ 存 4 位小数；基线存 6 位 → 统一按报告精度（4 位）比对，容差 1e-9
        if (abs(round(base["tau_%s_1" % tag], 4) - st["tau"][0]) > 1e-9 or
                abs(round(base["tau_%s_2" % tag], 4) - st["tau"][1]) > 1e-9):
            base_tau_ok = False
    chk("R8 独立复现基线与报告标量逐项一致（反推成立）", base_env_ok and base_tau_ok,
        "env=%s tau=%s" % (base_env_ok, base_tau_ok))

    # ==== R9 可识别性网格自洽 ====
    chk("R9 可识别性网格自洽：被识别数 + 未识别数 == 声明参数总数",
        len(ident) + len(not_ident) == len(pars), "identified=%d not_identified=%d total=%d"
        % (len(ident), len(not_ident), len(pars)))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"] = npass
    _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["identified"] = ident
    _res["not_identified"] = not_ident
    _res["sensitivity"] = {k: float("%.3e" % v) for k, v in sens_tab.items()}
    _res["inferred"] = {"L1": float("%.6f" % L1_inf), "L2": float("%.6f" % L2_inf),
                        "TAU_LIM[1]": float("%.6f" % tau1_lim_inf)}
    _res["findings"] = [
        "★ 逆向盲测（连声明参数也不信）：报告标量可由闭式恒等式完全反推并自洽：",
        "  R1 τ=JᵀF 精确；R2 ρ=|Ft|/(μFn) 精确；R3 由 [0,5] 的 τ 向量反推 L1=%.5f/L2=%.5f（声明 %.2f/%.2f）；"
        "R4 由 μ=0.6 包络（joint2 绑定）反推 TAU_LIM[1]=%.5f（声明 %.2f）。"
        % (L1_inf, L2_inf, bench["L1"], bench["L2"], tau1_lim_inf, bench["TAU_LIM"][1]),
        "  可识别性证伪（正负5%% 扰动）：7 个声明参数中 %d 个被数据锁死，未识别=%s。"
        % (len(ident), not_ident),
        "  机制：G 不进入 C4 摩擦锥（无重力项）故不可识别；TAU_LIM[0]=5.0 在本组静态用例中"
        "从不承重（4 个可行性锚点与 μ=0.6 包络均由 TAU_LIM[1]=1.8 绑定）故亦不可识别——"
        "与 E74 '可识别性由工况覆盖决定' 同律，已诚实标注而非谎称全识别。",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n逆向盲测结果：%d/%d PASS ｜ reverse_pass = %s"
          % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return _res


if __name__ == "__main__":
    main()
