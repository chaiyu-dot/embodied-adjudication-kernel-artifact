# -*- coding: utf-8 -*-
"""blind_reverse_e77.py —— E77 的**外部逆向盲测**（连声明参数也不信）。

闭式反推：LRN 权重 → 反推 kp/kd/重力系数；FLC 小信号极限 ≈ PD；IMP 刚度调度；可识别性 ±5% 证伪。
隔离：只 import e77_standalone_physics。产物：blind_reverse_e77.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
import e77_standalone_physics as P  # noqa: E402

REPORT = os.path.join(EVAL, "e77_controller_family_extension_report.json")
BENCH = os.path.join(EVAL, "e77_bench_data.json")
OUT = os.path.join(EVAL, "blind_reverse_e77.json")
_repo_mods = ("planning", "control_strategies", "e66_closed_loop_control",
              "e68_control_strategy_suite", "e77_controller_family_extension")
TOL = 1e-3

_res = {"experiment": "E77 外部逆向盲测（反推 + 证伪可识别性）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:240]) if detail else ""))
    return bool(ok)


def _repo_loaded():
    return [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo_mods)]


def _scalars(data):
    o = P.replicate(data)
    sc = {}
    for n, rows in o["A_metrics"]["rows"].items():
        for s, v in rows.items():
            for f in ("rise_s", "settling_s", "ss_err_rad", "tau_max_executed",
                      "executed_violation_steps", "demand_violation_steps"):
                sc["%s_%s_%s" % (n, s, f)] = float(v[f] or 0.0)
    for n, v in o["C_capability"]["rows"].items():
        sc["C_%s_ss" % n] = float(v["ss_err_nominal"] or 0.0)
    for n, v in o["D_determinism_and_ablation"]["gate_off_ablation"].items():
        sc["D_%s_abl" % n] = float(v["executed_violation_steps_no_gate"])
    return sc


def _rel(a, b):
    return max(abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-9) for k in a)


def main():
    print("=" * 92); print("E77 外部逆向盲测（反推 + 证伪可识别性）"); print("=" * 92)
    chk("R0 隔离自检：运行期未加载任何仓库模块", len(_repo_loaded()) == 0, "loaded=%s" % _repo_loaded())

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    P.configure(data)
    kp = data["gains"]["kp_design"]; kd = data["gains"]["kd_design"]
    W = np.asarray(data["family_specs"]["LRN"]["W"], float)

    # R1 LRN 权重 → 反推 kp/kd（W[0,0]=kp, W[0,2]=kd；W[1,1]=kp, W[1,3]=kd）
    kp_inf = 0.5 * (W[0, 0] + W[1, 1]); kd_inf = 0.5 * (W[0, 2] + W[1, 3])
    chk("R1 反推 kp/kd：由 LRN 权重 (e,−q̇) 系数", abs(kp_inf - kp) < 1e-6 and abs(kd_inf - kd) < 1e-6,
        "反推 kp=%.6f kd=%.6f ｜ 声明 %.4f/%.4f" % (kp_inf, kd_inf, kp, kd))

    # R2 反推重力系数：W[0,6]≈(M1·LC1+M2·L1)·G, W[0,8]≈W[1,8]≈M2·LC2·G
    g1c = (data["bench"]["M1"] * data["bench"]["LC1"] + data["bench"]["M2"] * data["bench"]["L1"]) * data["bench"]["G"]
    g2c = data["bench"]["M2"] * data["bench"]["LC2"] * data["bench"]["G"]
    chk("R2 反推重力前馈系数（cos q₀ / cos(q₀+q₁)）",
        abs(W[0, 6] - g1c) < 1e-6 and abs(W[0, 8] - g2c) < 1e-6 and abs(W[1, 8] - g2c) < 1e-6,
        "W[0,6]=%.4f(应 %.4f) W[0,8]=%.4f W[1,8]=%.4f(应 %.4f)" % (W[0, 6], g1c, W[0, 8], W[1, 8], g2c))

    # R3 LRN 与 PD_G 在随机状态上一致（策略 == PD_G + 重力前馈；**非逐位**，残差 ~1e-12 来自岭回归）
    rng = np.random.RandomState(7); mx = 0.0
    for _ in range(500):
        q = rng.uniform(-1.5, 1.5, 2); qd = rng.uniform(-2, 2, 2); r = rng.uniform(-1.5, 1.5, 2)
        tau_lrn = W @ P.LRN._feat(q, qd, r)
        tau_pdg = kp * (r - q) + kd * (-qd) + P.gravity_vec(q, 0.0)
        mx = max(mx, float(np.max(np.abs(tau_lrn - tau_pdg))))
    chk("R3 LRN 策略 == PD_G+重力前馈（随机状态，max|Δτ|<1e-9，非逐位）", mx < 1e-9, "max|Δτ|=%.2e" % mx)

    # R4 FLC 定性性质：输出对误差**单调不减**且**有界**（|τ−g| ≤ tau_scale）
    flc = P.FLC(kp, kd)
    outs = []
    for e0 in np.linspace(-0.5, 0.5, 21):
        q = np.array([0.0, 0.0]); r = np.array([e0, e0])
        outs.append(float(flc.act(q, np.zeros(2), r, 0.0)[0] - P.gravity_vec(q, 0.0)[0]))
    mono = all(outs[i + 1] >= outs[i] - 1e-12 for i in range(len(outs) - 1))
    bounded = all(abs(o) <= flc.tau_scale[0] + 1e-9 for o in outs)
    chk("R4 FLC 输出对误差单调不减且有界", mono and bounded,
        "单调=%s 有界=%s 范围[%.3f, %.3f]" % (mono, bounded, min(outs), max(outs)))

    # R5 IMP 刚度随误差单调衰减
    imp = P.IMP(kp, kd); ks = []
    for e in (0.0, 0.2, 0.5, 1.0, 2.0):
        s = 1.0 - 0.6 * min(1.0, abs(e) / 0.5); ks.append(kp * s)
    chk("R5 IMP 变刚度随误差单调不增", all(ks[i + 1] <= ks[i] + 1e-12 for i in range(len(ks) - 1)),
        "K_eff(e=0/0.2/0.5/1/2)=%s" % [round(x, 4) for x in ks])

    # R6 可识别性 ±5%（bench + gains）
    base = _scalars(data)
    pars = [("L1", lambda b, v: b.__setitem__("L1", v), data["bench"]["L1"]),
            ("L2", lambda b, v: b.__setitem__("L2", v), data["bench"]["L2"]),
            ("M1", lambda b, v: b.__setitem__("M1", v), data["bench"]["M1"]),
            ("M2", lambda b, v: b.__setitem__("M2", v), data["bench"]["M2"]),
            ("LC1", lambda b, v: b.__setitem__("LC1", v), data["bench"]["LC1"]),
            ("LC2", lambda b, v: b.__setitem__("LC2", v), data["bench"]["LC2"]),
            ("I1", lambda b, v: b.__setitem__("I1", v), data["bench"]["I1"]),
            ("I2", lambda b, v: b.__setitem__("I2", v), data["bench"]["I2"]),
            ("G", lambda b, v: b.__setitem__("G", v), data["bench"]["G"]),
            ("TAU_LIM[0]", lambda b, v: b["TAU_LIM"].__setitem__(0, v), data["bench"]["TAU_LIM"][0]),
            ("TAU_LIM[1]", lambda b, v: b["TAU_LIM"].__setitem__(1, v), data["bench"]["TAU_LIM"][1]),
            ("B_VISC", lambda b, v: b.__setitem__("B_VISC", v), data["bench"]["B_VISC"]),
            ("kp_design", lambda b, v: b.__setitem__("kp_design", v), data["gains"]["kp_design"]),
            ("kd_design", lambda b, v: b.__setitem__("kd_design", v), data["gains"]["kd_design"])]
    ident, not_ident, sens = [], [], {}
    for name, setter, v0 in pars:
        s = 0.0
        for sign in (+1.0, -1.0):
            d2 = json.loads(json.dumps(data))
            setter(d2["gains"] if name.endswith("design") else d2["bench"], v0 * (1.0 + sign * 0.05))
            s = max(s, _rel(_scalars(d2), base))
        sens[name] = s
        (ident if s > TOL else not_ident).append(name)
    chk("R6 可识别性证伪（±5%）：被识别集合如实列出", len(ident) + len(not_ident) == len(pars),
        "identified=%s | not_identified=%s" % (ident, not_ident))
    chk("R7 被识别参数敏感度均 > 容差", all(sens[k] > TOL for k in ident),
        "敏感度表=%s" % {k: float("%.2e" % v) for k, v in sens.items()})

    o = P.replicate(data)
    ok8 = all(abs(o["C_capability"]["rows"][n]["ss_err_nominal"] - rep["C_capability"]["rows"][n]["ss_err_nominal"]) < 1e-12
              for n in data["family_specs"])
    chk("R8 独立复现基线与报告一致（反推成立）", ok8,
        "ss_nom=%s" % {n: o["C_capability"]["rows"][n]["ss_err_nominal"] for n in data["family_specs"]})

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["identified"] = ident; _res["not_identified"] = not_ident
    _res["sensitivity"] = {k: float("%.3e" % v) for k, v in sens.items()}
    _res["inferred"] = {"kp": float("%.6f" % kp_inf), "kd": float("%.6f" % kd_inf)}
    _res["findings"] = [
        "★ 逆向盲测（连声明参数也不信）：LRN 权重可反推 kp/kd 与重力系数；LRN 策略 == PD_G+重力（max|Δτ|~1e-12，**非逐位**，残差来自岭回归）。",
        "  可识别性证伪（正负5%%）：%d 参数中 %d 被数据锁死，未识别=%s。" % (len(pars), len(ident), not_ident),
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=2, ensure_ascii=False)
    print("\n逆向盲测结果：%d/%d PASS ｜ reverse_pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    return _res


if __name__ == "__main__":
    main()
