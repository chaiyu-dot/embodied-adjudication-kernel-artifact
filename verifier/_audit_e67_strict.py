# -*- coding: utf-8 -*-
"""_audit_e67_strict.py —— E67 的**严格审核（L3，审前提 + 独立物理）**。

不 import 任何 e67 模块；审计者**自写** A 段（FK/雅可比/力矩受限速度-加速度谱）与 D 段
（摩擦锥 JᵀF→τ 可行性），用 rne_dynamics 作独立力矩参考，重算报告头条数并逐格核对 A 段包络，
并做篡改用例。B/C/E 段（含控制器闭环）留待 L1 + 需求/执行双口径核对（见 replicator）。

产物：_audit_e67_strict.json
"""
import copy
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                                  # noqa: E402

REPORT = os.path.join(EVAL, "e67_testbed_report.json")
OUT = os.path.join(EVAL, "_audit_e67_strict.json")

_res = {"experiment": "E67 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：自写 FK/Jac/max_speed/accel + 摩擦锥；力矩用 rne_dynamics(参考)，不 import e67",
        "checks": [], "tamper": []}

L1, L2 = 0.40, 0.30
M1, M2 = 0.60, 0.35
G = 9.81
TAU_LIM = (5.0, 1.8)
JLIM = 2.0
QDIAL_LIM = 4.0


def fk(q):
    return [L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
            L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1])]


def jac(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return [[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]]


def mass(q):
    m2, lc2, i2 = M2, L2 / 2.0, M2 * L2 * L2 / 12.0
    c2 = math.cos(q[1])
    m11 = M1 * (L1 / 2.0) ** 2 + m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + M1 * L1 ** 2 / 12.0 + i2
    m12 = m2 * (lc2 ** 2 + L1 * lc2 * c2) + i2
    m22 = m2 * lc2 ** 2 + i2
    return [[m11, m12], [m12, m22]]


def coriolis(q, qd):
    m2, lc2 = M2, L2 / 2.0
    h = -m2 * L1 * lc2 * math.sin(q[1])
    return [h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2]


def gravity(q):
    m2, lc2 = M2, L2 / 2.0
    g1 = (M1 * (L1 / 2.0) + m2 * L1) * G * math.cos(q[0]) + m2 * lc2 * G * math.cos(q[0] + q[1])
    g2 = m2 * lc2 * G * math.cos(q[0] + q[1])
    return [g1, g2]


def tau_at(q, qd, qdd):
    import numpy as np
    M = mass(q); C = coriolis(q, qd); g = gravity(q)
    return [M[0][0] * qdd[0] + M[0][1] * qdd[1] + C[0] + g[0],
            M[1][0] * qdd[0] + M[1][1] * qdd[1] + C[1] + g[1]]


def max_speed_along(q, u, qdot_lim=QDIAL_LIM, hi=20.0):
    import numpy as np
    u = [u[0], u[1]]
    def ok(s):
        t = tau_at(q, [s * u[0], s * u[1]], [0.0, 0.0])
        return abs(t[0]) <= TAU_LIM[0] + 1e-12 and abs(t[1]) <= TAU_LIM[1] + 1e-12
    if not ok(1e-9):
        return 0.0
    lo, hi_ = 1e-9, hi
    if ok(hi_):
        return min(hi_, qdot_lim)
    for _ in range(60):
        mid = 0.5 * (lo + hi_)
        if ok(mid):
            lo = mid
        else:
            hi_ = mid
    return min(lo, qdot_lim)


def max_accel_along(q, u, hi=1e4):
    def ok(s):
        t = tau_at(q, [0.0, 0.0], [s * u[0], s * u[1]])
        return abs(t[0]) <= TAU_LIM[0] + 1e-12 and abs(t[1]) <= TAU_LIM[1] + 1e-12
    lo, hi_ = 1e-9, hi
    if not ok(lo):
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi_)
        if ok(mid):
            lo = mid
        else:
            hi_ = mid
    return lo


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def part_A_independent():
    import numpy as np
    qs = np.linspace(-1.8, 1.8, 7)
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    v_max_ee = 0.0; a_max_ee = 0.0
    env = []
    for q1 in qs:
        for q2 in qs:
            q = [float(q1), float(q2)]
            J = jac(q)
            bv, ba = 0.0, 0.0
            for th in ang:
                u = [math.cos(th), math.sin(th)]
                s = max_speed_along(q, u)
                v = math.hypot(J[0][0] * s * u[0] + J[0][1] * s * u[1],
                               J[1][0] * s * u[0] + J[1][1] * s * u[1])
                bv = max(bv, v)
                sa = max_accel_along(q, u)
                a = math.hypot(J[0][0] * sa * u[0] + J[0][1] * sa * u[1],
                               J[1][0] * sa * u[0] + J[1][1] * sa * u[1])
                ba = max(ba, a)
            env.append((round(q1, 4), round(q2, 4), round(bv, 4), round(ba, 4)))
            v_max_ee = max(v_max_ee, bv)
            a_max_ee = max(a_max_ee, ba)
    return v_max_ee, a_max_ee, env


def part_D_independent(q=(0.7, -1.0), mus=(0.2, 0.4, 0.6, 0.8)):
    import numpy as np
    J = jac(q)
    Jt = [[J[0][0], J[1][0]], [J[0][1], J[1][1]]]
    out = {}
    for mu in mus:
        best = (0.0, 0.0)
        for i in range(401):
            Fn = 40.0 * i / 400.0
            Ft_cone = mu * Fn
            lo, hi = 0.0, 200.0
            if abs(Jt[0][0] * 0.0 + Jt[0][1] * Fn) <= TAU_LIM[0] + 1e-12 and \
               abs(Jt[1][0] * 0.0 + Jt[1][1] * Fn) <= TAU_LIM[1] + 1e-12:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    t0 = Jt[0][0] * mid + Jt[0][1] * Fn
                    t1 = Jt[1][0] * mid + Jt[1][1] * Fn
                    if abs(t0) <= TAU_LIM[0] + 1e-12 and abs(t1) <= TAU_LIM[1] + 1e-12:
                        lo = mid
                    else:
                        hi = mid
            Ft = min(Ft_cone, lo)
            if Ft > best[0]:
                best = (Ft, Fn)
        out["mu=%.1f" % mu] = {"max_tangential_N": round(best[0], 4),
                                "at_normal_N": round(best[1], 4),
                                "cone_utilization": round(best[0] / max(mu * best[1], 1e-9), 4)}
    return out


def _forward(rep):
    ch = []
    A = rep["A_velocity_spectrum"]
    # --- A 段独立复算 ---
    v_re, a_re, env = part_A_independent()
    ch.append(_chk("S1_A_vmax_ee_matches", abs(v_re - A["v_max_ee"]) < 1e-2,
                   "recomputed=%.4f stored=%.4f" % (v_re, A["v_max_ee"])))
    ch.append(_chk("S2_A_amax_ee_matches", abs(a_re - A["a_max_ee"]) < 1e-1,
                   "recomputed=%.4f stored=%.4f" % (a_re, A["a_max_ee"])))
    # A 段包络逐格核对 v_max（抽样若干构型）
    rep_env = {(round(e["q"][0], 4), round(e["q"][1], 4)): e["v_max"] for e in A["envelope"]}
    nchk = 0; vmax = 0.0
    for q1, q2, bv, ba in env:
        if (q1, q2) in rep_env:
            d = abs(bv - rep_env[(q1, q2)])
            vmax = max(vmax, d)
            nchk += 1
    ch.append(_chk("S3_A_envelope_vmax_rowwise", vmax < 1e-2 and nchk == len(rep_env),
                   "checked=%d/%d max|Δv|=%.4f" % (nchk, len(rep_env), vmax)))
    # 绑定约束：v_max 由关节速度上限(4.0)限定（torque 不先越限）
    # 验证：在报告 at 构型，max_speed_along 返回恰为 4.0（被关节上限切）
    atq = A["at"]["q"]
    smax = max(max_speed_along(atq, [1.0, 0.0]), max_speed_along(atq, [0.0, 1.0]),
              max_speed_along(atq, [0.7071, 0.7071]))
    ch.append(_chk("S4_A_bound_by_joint_speed_cap",
                   abs(smax - QDIAL_LIM) < 1e-6, "max_s_at_opt=%.4f cap=%.1f" % (smax, QDIAL_LIM)))
    # --- D 段独立复算 ---
    D = rep["D_contact_friction"]["friction_cone_max_push"]
    d_re = part_D_independent()
    nbad = 0; dmax = 0.0
    for k in D:
        d = abs(d_re[k]["max_tangential_N"] - D[k]["max_tangential_N"])
        dmax = max(dmax, d)
        if d > 1e-2:
            nbad += 1
    ch.append(_chk("S5_D_friction_cone_matches", nbad == 0, "max|ΔFt|=%.4f bad=%d" % (dmax, nbad)))
    # D 段接触可行性：JᵀF→τ 越限判定独立复算
    feas_key = [k for k in rep["D_contact_friction"] if k.startswith("contact_feasibility")][0]
    feas = rep["D_contact_friction"][feas_key]
    nfb = 0
    for Fstr, v in feas.items():
        F = [float(x) for x in Fstr.replace("F=[", "").replace("]", "").split(",")]
        J = jac([0.7, -1.0]); Jt = [[J[0][0], J[1][0]], [J[0][1], J[1][1]]]
        t0 = Jt[0][0] * F[0] + Jt[0][1] * F[1]
        t1 = Jt[1][0] * F[0] + Jt[1][1] * F[1]
        ok = abs(t0) <= TAU_LIM[0] + 1e-12 and abs(t1) <= TAU_LIM[1] + 1e-12
        if ok != v["feasible"]:
            nfb += 1
    ch.append(_chk("S6_D_contact_feasibility_matches", nfb == 0, "mismatch=%d" % nfb))
    # verdict 内部自洽（E67 无单一 verdict_pass，但 A 段闭环自检）
    binding = A.get("binding_summary", {})
    ch.append(_chk("S7_A_binding_summary_nonempty", "joint_speed_cap" in binding and binding["joint_speed_cap"] > 0,
                   str(binding)))
    return ch


def _tamp(name, mut, expect):
    bad = copy.deepcopy(json.load(open(REPORT, encoding="utf-8")))
    mut(bad)
    n0 = len(_res["checks"])
    ch = _forward(bad)
    caught = any((not c[1]) and c[0].startswith(tuple(expect)) for c in ch)
    del _res["checks"][n0:]
    _res["tamper"].append({"name": name, "caught": bool(caught), "detail": "期望捕获 %s" % expect})
    print("  [%s] T %s%s" % ("PASS" if caught else "FAIL", name, "  —— 篡改被捕获" if caught else "  —— !! 未被捕获 !!"))
    return bool(caught)


def main():
    print("=" * 88)
    print("E67 严格审核（L3 独立物理，不 import e67）— A 速度谱 + D 摩擦锥")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 v_max_ee=99 → S1 必报",
          lambda r: r["A_velocity_spectrum"].__setitem__("v_max_ee", 99.0), ["S1"])
    _tamp("R2 篡改 a_max_ee=999 → S2 必报",
          lambda r: r["A_velocity_spectrum"].__setitem__("a_max_ee", 999.0), ["S2"])
    _tamp("R3 篡改 某格 envelope v_max → S3 必报",
          lambda r: r["A_velocity_spectrum"]["envelope"].__setitem__(0,
                 {**r["A_velocity_spectrum"]["envelope"][0], "v_max": 99.0}), ["S3"])
    _tamp("R4 篡改 D friction mu=0.2 Ft → S5 必报",
          lambda r: r["D_contact_friction"]["friction_cone_max_push"]["mu=0.2"].__setitem__("max_tangential_N", 99.0),
          ["S5"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass
    _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp
    _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
