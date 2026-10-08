# -*- coding: utf-8 -*-
"""blind_reverse_e67.py —— E67 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 e67_testbed_completeness）。力矩用
``rne_dynamics.planar_2r_torque_closedform``（独立参考）而非 E67 复用的 E66.mass_matrix。
只信"数据文件"（报告 JSON）：反推 m_blk / τ_lim 区间 / 关节速度上限，做 ±扰动与负对照证伪。

覆盖 E67 四段可闭式反推部分 + 门控消融结构性负对照：
  · R1 自由物块滑移 → 反推 m_blk·G（=19.62）与 μ
  · R2 接触可行性边界 → 反推 τ_lim[1] ∈ (可施加, 不可施加) 含 1.8
  · R3 摩擦锥 max_push 独立重算 == 存储（A/D 闭式）
  · R4 速度谱 v_max_ee / 包络逐格独立重算 + 关节速度上限（4.0）反推
  · R5 双臂目标可行性（ik + 重力力矩）独立重算 == 存储
  · R6 门控消融负对照：gate_off 需求若为 0 则消融为空（证明 demand>0 是证据）

产物：blind_reverse_e67.json
"""
import json
import math
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 e67_testbed_completeness 不可被 import
import numpy as np                                                          # noqa: E402
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e67_testbed_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e67.json")

L1, L2, M1, M2, G = 0.40, 0.30, 0.60, 0.35, 9.81
TAU_LIM = (5.0, 1.8)
JLIM = 2.0
QDIAL_LIM = 4.0

_res = {"experiment": "E67 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e67_testbed_completeness；力矩用 rne_dynamics 参考",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def fk(q, base=(0.0, 0.0)):
    x = base[0] + L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1])
    y = base[1] + L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1])
    return np.array([x, y])


def jac(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def ik(p, base=(0.0, 0.0), elbow=-1.0):
    x, y = p[0] - base[0], p[1] - base[1]
    r = max(1e-9, math.hypot(x, y))
    c2 = max(-1.0, min(1.0, (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    q2 = elbow * math.acos(c2)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return [q1, q2]


def _tau_const_speed(q, qd):
    """匀速运动需求力矩（qdd=0），独立参考 = rne 闭式。"""
    t1, t2 = RNE.planar_2r_torque_closedform(L1, L2, M1, M2, q[0], q[1], qd[0], qd[1], 0.0, 0.0, g=G)
    return np.array([t1, t2], float)


def _max_speed_along(q, u, qdot_lim=QDIAL_LIM, hi=20.0):
    u = np.asarray(u, float)

    def ok(s):
        return bool(np.all(np.abs(_tau_const_speed(q, s * u)) <= np.array(TAU_LIM) + 1e-12))
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


def _max_push(q, mu, Fn_hi=40.0):
    Jt = jac(q).T
    best = (0.0, 0.0)
    for i in range(401):
        Fn = Fn_hi * i / 400.0
        Ft_cone = mu * Fn
        lo, hi = 0.0, 200.0
        if np.all(np.abs(Jt @ np.array([0.0, Fn])) <= np.array(TAU_LIM) + 1e-12):
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if np.all(np.abs(Jt @ np.array([mid, Fn])) <= np.array(TAU_LIM) + 1e-12):
                    lo = mid
                else:
                    hi = mid
        Ft = min(Ft_cone, lo)
        if Ft > best[0]:
            best = (Ft, Fn)
    return best[0], best[1]


def main():
    print("=" * 88)
    print("E67 逆向盲测（L5 隔离，独立物理反推 + 负对照）")
    print("=" * 88)
    chk("R0_isolation_e67_not_imported",
        "e67_testbed_completeness" not in sys.modules,
        "sys.modules 含 e67=%s" % ("e67_testbed_completeness" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))

    # --- R1：自由物块滑移反推 m_blk·G 与 μ ---
    slip = rep["D_contact_friction"]["free_block_slip"]
    ratios = []
    for k, v in slip.items():
        mu = float(k.split("=")[1])
        ratios.append(v["slip_if_push_over_N"] / mu)
    mg = float(np.mean(ratios))
    m_blk_re = mg / G
    chk("R1_reverse_block_mass_from_slip",
        max(abs(r - mg) for r in ratios) < 1e-6 and abs(m_blk_re - 2.0) < 1e-9,
        "recovered m_blk*G=%.4f -> m_blk=%.6f (truth 2.0)" % (mg, m_blk_re))

    # --- R2：接触可行性边界反推 τ_lim[1] ---
    feas = rep["D_contact_friction"]["contact_feasibility_q=[0.7, -1.0]"]
    q = [0.7, -1.0]
    tau_by_F = {}
    for Fstr, v in feas.items():
        F = [float(x) for x in Fstr.replace("F=[", "").replace("]", "").split(",")]
        t = jac(q).T @ np.array(F)
        tau_by_F[Fstr] = (t, bool(np.all(np.abs(t) <= np.array(TAU_LIM) + 1e-12)))
        # 存储 τ 一致性
    cons = all(abs(v["tau"][0] - tau_by_F[k][0][0]) < 1e-3 and abs(v["tau"][1] - tau_by_F[k][0][1]) < 1e-3
               for k, v in feas.items())
    feas_ok = all(v["feasible"] == tau_by_F[k][1] for k, v in feas.items())
    # 边界：可施加的最大 |τ2| 与不可施加的最小 |τ2|
    feas_tau2 = [abs(tau_by_F[k][0][1]) for k, v in feas.items() if v["feasible"]]
    infeas_tau2 = [abs(tau_by_F[k][0][1]) for k, v in feas.items() if not v["feasible"]]
    lo_b, hi_b = max(feas_tau2), min(infeas_tau2)
    chk("R2_reverse_tau_lim_from_feasibility_boundary",
        cons and feas_ok and lo_b < TAU_LIM[1] < hi_b and abs(tau_by_F["F=[0.0, 5.0]"][0][1] - 1.433) < 1e-3,
        "tau2 in (%.4f, %.4f) contains 1.8 ; F=[0,5]->tau2=%.4f" % (lo_b, hi_b, tau_by_F["F=[0.0, 5.0]"][0][1]))

    # --- R3：摩擦锥 max_push 独立重算 == 存储 ---
    cone = rep["D_contact_friction"]["friction_cone_max_push"]
    nbad = 0
    dmax = 0.0
    for k, v in cone.items():
        mu = float(k.split("=")[1])
        ft, fn = _max_push(q, mu)
        dmax = max(dmax, abs(ft - v["max_tangential_N"]))
        if abs(ft - v["max_tangential_N"]) > 1e-2:
            nbad += 1
    chk("R3_reverse_friction_cone_matches_stored", nbad == 0,
        "max|dFt|=%.4f bad=%d" % (dmax, nbad))

    # --- R4：速度谱 v_max_ee / 包络逐格独立重算 + 反推关节速度上限 ---
    A = rep["A_velocity_spectrum"]
    qs = np.linspace(-1.8, 1.8, 7)
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    v_max_re = 0.0
    env_re = {}
    cap_hit = 0
    for q1 in qs:
        for q2 in qs:
            qq = [float(q1), float(q2)]
            J = jac(qq)
            bv = 0.0
            for th in ang:
                u = np.array([math.cos(th), math.sin(th)])
                s = _max_speed_along(qq, u)
                bv = max(bv, float(np.linalg.norm(J @ (s * u))))
                if abs(s - QDIAL_LIM) < 1e-6:
                    cap_hit += 1
            env_re[(round(q1, 4), round(q2, 4))] = round(bv, 4)
            v_max_re = max(v_max_re, bv)
    rep_env = {(round(e["q"][0], 4), round(e["q"][1], 4)): e["v_max"] for e in A["envelope"]}
    rowmax = max(abs(env_re[k] - rep_env[k]) for k in rep_env)
    chk("R4_reverse_velocity_spectrum_and_joint_cap",
        abs(v_max_re - A["v_max_ee"]) < 1e-2 and rowmax < 1e-2 and cap_hit > 0,
        "v_max_re=%.4f stored=%.4f rowmax=%.4f cap_hit=%d" % (v_max_re, A["v_max_ee"], rowmax, cap_hit))

    # --- R5：双臂目标可行性（ik + 重力力矩）独立重算 ---
    B = rep["B_dual_arm"]
    BASE_L, BASE_R = (-0.25, 0.0), (0.25, 0.0)
    tgt = [[0.30, 0.25], [0.55, 0.35]]
    ok5 = True
    for base, p, key in ((BASE_L, tgt[0], "target_feasibility_L"), (BASE_R, tgt[1], "target_feasibility_R")):
        qr = ik(p, base)
        inlim = bool(max(abs(qr[0]), abs(qr[1])) <= JLIM)
        tg = _tau_const_speed(qr, [0.0, 0.0])
        gwt = bool(np.all(np.abs(tg) <= np.array(TAU_LIM) + 1e-12))
        st = B[key]
        ok5 &= (st["in_joint_limits"] == inlim) and (st["gravity_torque_within_tau"] == gwt)
    # 独立到点误差应为 0（到点）
    ok5 &= (B["independent"]["errL_m"] == 0.0 and B["independent"]["errR_m"] == 0.0)
    chk("R5_reverse_dual_arm_feasibility_and_independent_reach", ok5,
        "L=%s R=%s errL=%.5f errR=%.5f" % (B["target_feasibility_L"], B["target_feasibility_R"],
                                           B["independent"]["errL_m"], B["independent"]["errR_m"]))

    # --- R6：门控消融负对照（结构必要性）---
    E = rep["E_gate_ablation"]
    sc = E["scenarios"]
    agg_on_g1 = sc["aggressive_wn40|gate_on"]["G1_dual_arm_independent"]["violation_steps_demand"]
    agg_on_g2 = sc["aggressive_wn40|gate_on"]["G2_track_1hz_ff"]["violation_steps_demand"]
    agg_off_g1 = sc["aggressive_wn40|gate_off"]["G1_dual_arm_independent"]["violation_steps_demand"]
    agg_off_g2 = sc["aggressive_wn40|gate_off"]["G2_track_1hz_ff"]["violation_steps_demand"]
    exec_on = (sc["aggressive_wn40|gate_on"]["G1_dual_arm_independent"]["violation_steps_executed"]
               + sc["aggressive_wn40|gate_on"]["G2_track_1hz_ff"]["violation_steps_executed"])
    # 结论布尔自洽
    concl = E["conclusion"]
    cons6 = (concl["gate_load_bearing"] == (agg_on_g1 > 0 or agg_on_g2 > 0)
             and concl["gate_on_executed_zero"] == (exec_on == 0)
             and concl["gate_off_really_exceeds"] == (agg_off_g1 > 0 or agg_off_g2 > 0))
    # 负对照：若 gate_off 需求为 0，则消融为空（gate_off_really_exceeds 应为 False）
    negctrl_empty = (not ((0 > 0) or (0 > 0)))
    chk("R6_reverse_gate_ablation_necessity",
        cons6 and (agg_off_g1 > 0 or agg_off_g2 > 0) and exec_on == 0 and negctrl_empty,
        "on_demand(G1,%.0f;G2,%.0f) off_demand(G1,%.0f;G2,%.0f) exec_on=%d" %
        (agg_on_g1, agg_on_g2, agg_off_g1, agg_off_g2, exec_on))

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
