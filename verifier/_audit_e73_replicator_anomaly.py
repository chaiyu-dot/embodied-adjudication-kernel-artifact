# -*- coding: utf-8 -*-
"""_audit_e73_replicator_anomaly.py —— e73 的**第三方复刻者视角**异常排查审核（零 API）。

模拟一个**不信任作者代码**的外部复刻者：只凭论文/台架公开物理参数（2R 连杆质量/惯量、
τ_lim、joint speed cap）与 e73 报告里的数字，**自己独立实现整套 2R 动力学与速度/加速度包络**，
重算报告每一个关键数字（T1–T5），再把作者模块当**黑盒**做异常输入 fuzz（T6），
最后核验报告内部自洽与"能力口径诚实"（T7）。

与 _audit_e73（作者自查）的本质区别：
  · 本脚本 T1–T5 **不 import e66/e67/e73**，纯第三方从零实现（含 M/C/g 与 Jacobian）；
  · T6 故意喂作者测试没覆盖的坏输入，排查鲁棒性异常；
  · 这正是"别人来复刻我实验数据时排查异常"的口径。

产物：_audit_e73_replicator_anomaly.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e73_velocity_spectrum_report.json")

# ---------------- 第三方独立台架参数（与论文/报告公开一致，不 import 作者模块） ----------------
L1, L2 = 0.40, 0.30
M1, M2 = 0.60, 0.35
LC1, LC2 = L1 / 2.0, L2 / 2.0
I1, I2 = M1 * L1 * L1 / 12.0, M2 * L2 * L2 / 12.0
G = 9.81
TAU = np.array([5.0, 1.8])
QD_CAP = 4.0
Q_REF = [0.60, -0.90]
E67_V, E67_A = 3.0461, 66.1436


def _chk(name, cond, detail=""):
    return {"name": name, "pass": bool(cond), "detail": str(detail)}


# ---------------- 独立实现：2R 动力学 ----------------
def M(q):
    c2 = math.cos(q[1])
    m11 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2 * L1 * LC2 * c2) + I1 + I2
    m12 = M2 * (LC2 * LC2 + L1 * LC2 * c2) + I2
    m22 = M2 * LC2 * LC2 + I2
    return np.array([[m11, m12], [m12, m22]])


def Cv(q, qd):
    h = -M2 * L1 * LC2 * math.sin(q[1])
    return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2])


def gv(q):
    g1 = (M1 * LC1 + M2 * L1) * G * math.cos(q[0]) + M2 * LC2 * G * math.cos(q[0] + q[1])
    g2 = M2 * LC2 * G * math.cos(q[0] + q[1])
    return np.array([g1, g2])


def tau_at(q, qd, qdd, mp=0.0):
    qdd = np.zeros(2) if np.isscalar(qdd) else np.asarray(qdd, float)
    return M(q) @ qdd + Cv(q, np.asarray(qd, float)) + gv(q)


def jac(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def max_speed(q, u, qdot_lim=QD_CAP):
    u = np.asarray(u, float)

    def ok(s):
        return bool(np.all(np.abs(tau_at(q, s * u, 0.0)) <= TAU + 1e-12))
    if not ok(1e-9):
        return 0.0
    lo, hi = 1e-9, 20.0
    if ok(hi):
        return min(hi, qdot_lim)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return min(lo, qdot_lim)


def max_accel(q, u):
    u = np.asarray(u, float)

    def ok(s):
        return bool(np.all(np.abs(tau_at(q, [0.0, 0.0], s * u)) <= TAU + 1e-12))
    lo, hi = 1e-9, 1e4
    if not ok(lo):
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def envelope(q, n_dir):
    J = jac(q)
    bv, ba = 0.0, 0.0
    for th in np.linspace(0.0, 2 * math.pi, n_dir, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        bv = max(bv, float(np.linalg.norm(J @ (max_speed(q, u) * u))))
        ba = max(ba, float(np.linalg.norm(J @ (max_accel(q, u) * u))))
    return bv, ba


def wn_nominal():
    kp = [0.9 * TAU[i] / max(abs(Q_REF[i]), 1e-9) for i in range(2)]
    Mm = M(Q_REF)
    return min(math.sqrt(kp[i] / float(Mm[i, i])) for i in range(2))


def _find_src(start=EVAL):
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "planning", "control_strategies.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.dirname(os.path.abspath(start))


def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    A, B, C, D = (rep["A_polar_spectrum"], rep["B_envelope_curve"],
                  rep["C_workspace_map"], rep["D_calibration"])
    checks = []

    # T1 polar 独立重算（逐构型）
    ok1 = True
    for cfg in A["configs"]:
        bv, ba = envelope(cfg["q"], A["configs"][0]["n_dir"])
        if abs(bv - cfg["v_max"]) > 1e-3 or abs(ba - cfg["a_max"]) > 5e-2:
            ok1 = False
    checks.append(_chk("T1_polar_independent", ok1, "polar mismatch"))

    # T2 工作空间网格独立重算（同分辨率 21x21/32）—— argmax 与统计
    n, nd = C["grid"], C["dirs"]
    qs = np.linspace(-1.8, 1.8, n)
    gv_list, ga_list = [], []
    for q1 in qs:
        for q2 in qs:
            bv, ba = envelope([float(q1), float(q2)], nd)
            gv_list.append(bv)
            ga_list.append(ba)
    ok2 = (abs(max(gv_list) - C["v_max"]) < 5e-3 and abs(max(ga_list) - C["a_max"]) < 5e-2
           and abs(min(gv_list) - C["v_min"]) < 5e-3
           and abs(float(np.median(gv_list)) - C["v_median"]) < 5e-3)
    checks.append(_chk("T2_grid_independent", ok2,
                       "v %s/%s a %s/%s" % (round(max(gv_list), 4), C["v_max"],
                                            round(max(ga_list), 4), C["a_max"])))

    # T3 复现 E67（独立 7x7/16）
    rqs = np.linspace(-1.8, 1.8, 7)
    rv, ra = 0.0, 0.0
    for q1 in rqs:
        for q2 in rqs:
            bv, ba = envelope([float(q1), float(q2)], 16)
            rv, ra = max(rv, bv), max(ra, ba)
    ok3 = (abs(rv - E67_V) <= 1e-3 and abs(ra - E67_A) <= 1e-3
           and abs(C["reproduce_e67"]["v_max"] - rv) <= 1e-3
           and abs(C["reproduce_e67"]["a_max"] - ra) <= 1e-3)
    checks.append(_chk("T3_reproduce_e67_independent", ok3,
                       "indep %s/%s | report %s/%s" % (round(rv, 4), round(ra, 4),
                       C["reproduce_e67"]["v_max"], C["reproduce_e67"]["a_max"])))

    # T4 jerk 独立重算
    wn = wn_nominal()
    ok4 = (abs(D["wn_nominal"] - round(wn, 3)) < 1e-2
           and abs(D["jerk_envelope"]["at_e67_settings"] - round(wn * E67_A, 3)) < 5e-1
           and abs(D["jerk_envelope"]["max"] - round(wn * C["a_max"], 3)) < 5e-1)
    checks.append(_chk("T4_jerk_independent", ok4, "wn=%s" % round(wn, 3)))

    # T5 路径曲线抽点独立重算
    mid = B["curve"][len(B["curve"]) // 2]
    bv, ba = envelope(mid["q"], C["dirs"])
    ok5 = (abs(bv - mid["v_max"]) < 1e-2 and abs(ba - mid["a_max"]) < 5e-1)
    checks.append(_chk("T5_path_spot_independent", ok5,
                       "indep %s/%s | report %s/%s" % (round(bv, 4), round(ba, 4), mid["v_max"], mid["a_max"])))

    # T6 黑盒 fuzz 作者模块（坏输入不崩、不撒谎）
    src = _find_src()
    sys.path.insert(0, src)
    sys.path.insert(0, EVAL)
    fuzz_ok, notes = True, []
    try:
        import e73_velocity_spectrum_envelope as E73
        cases = [[[-9.0, 9.0], 16], [[0.0, 0.0], 8], [[math.pi, -math.pi], 12]]
        for q, ndc in cases:
            r = E73.envelope_at([float(q[0]), float(q[1])], n_dir=ndc)
            if not (isinstance(r["v_max"], float) and isinstance(r["a_max"], float)
                    and r["v_max"] >= 0.0 and r["a_max"] >= 0.0
                    and r["binding"] in ("joint_speed_cap", "torque", "mixed")):
                fuzz_ok = False
        # NaN/极值输入：max_accel_capped 不得抛异常
        _ = E73.max_accel_capped([0.7, -1.0], [1.0, 0.0], [1e9, 1e9])
        # 越界构型仍返回合法三段
        r2 = E73.envelope_at([50.0, -50.0], n_dir=8)
        if r2["binding"] not in ("joint_speed_cap", "torque", "mixed"):
            fuzz_ok = False
        # polar 对奇异构型必须标 singular
        if not E73.polar([-1.8, 0.0])["is_singular"]:
            fuzz_ok = False
    except Exception as e:  # noqa: BLE001
        fuzz_ok = False
        notes.append("exception:%s" % e)
    checks.append(_chk("T6_blackbox_fuzz", fuzz_ok, "notes=%s" % notes))

    # T7 内部自洽 + 口径诚实
    bind_ok = (C["binding_cap_only"] + C["binding_torque_only"] + C["binding_mixed"] == C["n_cells"])
    honest = (len(D["caveats"]) >= 5 and 0.0 <= C["a_illposed_frac"] <= 1.0
              and 0.0 < C["a_max_diag"] < C["a_max"] and C["n_singular"] >= 1)
    inv_ok = bool(rep["verdict_pass"] == all(bool(v) for v in rep["criteria"].values()))
    checks.append(_chk("T7_self_consistent_and_honest", bind_ok and honest and inv_ok,
                       "bind=%s honest=%s inv=%s" % (bind_ok, honest, inv_ok)))

    npass = sum(c["pass"] for c in checks)
    out = {"audit": "_audit_e73_replicator_anomaly", "experiment": "e73_velocity_spectrum_envelope",
           "pass": npass, "total": len(checks), "verdict": bool(npass == len(checks)),
           "audit_pass": bool(npass == len(checks)), "n_pass": npass, "n_total": len(checks),
           "checks": checks}
    with open(os.path.join(EVAL, "_audit_e73_replicator_anomaly.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly audit pass = %d/%d  verdict=%s" % (npass, len(checks), out["verdict"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    run()
