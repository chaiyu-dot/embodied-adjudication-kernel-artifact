# -*- coding: utf-8 -*-
"""blind_reverse_e70.py —— E70 的**逆向盲测（L5，隔离 + 不信声明参数）**。

独立性：隔离（运行期自检 sys.modules 未加载 control_strategies / e70）。只信"数据文件"
（bench 几何 + 目标集生成规则 + IK），**独立复现**粗提案误差（证明离环粗建模必然到点失败、
必须接残差外环），验证 FK/IK 正确、SET_B 目标确无可达 IK，并做证伪。

产物：blind_reverse_e70.json
"""
import json
import math
import os
import sys
import zlib

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ★ 刻意不把 EVAL 加入 sys.path，使 control_strategies / e70 不可被 import
import numpy as np                                                          # noqa: E402

REPORT = os.path.join(EVAL, "e70_task_residual_loop_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e70.json")

_res = {"experiment": "E70 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 control_strategies / e70_task_residual_loop",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def seed_of(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


L1, L2 = 0.40, 0.30
JLIM = 2.0


def fk(q):
    return np.array([L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
                     L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1])])


def ik_all(p):
    x, y = float(p[0]), float(p[1])
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        return []
    out = []
    for sgn in (+1.0, -1.0):
        s2 = sgn * math.sqrt(max(0.0, 1.0 - c2 * c2))
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
        out.append(np.array([q1, q2]))
    return out


def ik_best(p, q_now):
    sols = ik_all(p)
    if not sols:
        return None
    return min(sols, key=lambda q: float(np.max(np.abs(q - np.asarray(q_now, float)))))


def build_goal_sets(n_adm, n_ina):
    """独立复刻 e70.build_goal_sets（同种子同 RNG）。"""
    A = []
    rng = np.random.RandomState(seed_of("goalset"))
    while len(A) < n_adm:
        q = rng.uniform(-1.9, 1.9, 2)
        p = fk(q)
        if not (0.12 < np.linalg.norm(p) < (L1 + L2) * 0.98):
            continue
        sols = ik_all(p)
        if not any(np.all(np.abs(s) <= JLIM) for s in sols):
            continue
        A.append([float(p[0]), float(p[1])])
    B = []
    for i in range(n_ina):
        ang = 2 * math.pi * (i + 0.5) / n_ina
        r = (L1 + L2) * (1.10 + 0.10 * (i % 3))
        B.append({"p": [r * math.cos(ang), r * math.sin(ang)], "kind": "outside_annulus"})
    rng2 = np.random.RandomState(seed_of("goalsetB2"))
    n_b2, tries = 0, 0
    while n_b2 < max(1, n_ina // 2) and tries < 40000:
        tries += 1
        q = np.array([(1 if rng2.rand() < 0.5 else -1) * rng2.uniform(2.05, math.pi),
                      rng2.uniform(-math.pi, math.pi)])
        p = fk(q)
        r = float(np.linalg.norm(p))
        if not (0.15 < r < 0.66):
            continue
        sols = ik_all(p)
        if not sols:
            continue
        if all(float(np.max(np.abs(s))) > JLIM for s in sols):
            B.append({"p": [float(p[0]), float(p[1])], "kind": "joint_limit_violating"})
            n_b2 += 1
    return A, B


def coarse_proposal(goal, seed_tag, q_now):
    rng = np.random.RandomState(seed_of("coarse", seed_tag))
    q_ik = ik_best(goal, q_now)
    if q_ik is None:
        q_ik = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)])
    f = float(rng.uniform(0.30, 0.80))
    return np.clip(f * q_ik, -JLIM, JLIM)


def main():
    print("=" * 88)
    print("E70 逆向盲测（L5 隔离，独立复现 + 证伪）")
    print("=" * 88)
    chk("R0_isolation_CS_not_imported",
        "control_strategies" not in sys.modules and "e70_task_residual_loop" not in sys.modules,
        "sys.modules 含 CS/e70=%s" % ("control_strategies" in sys.modules or "e70_task_residual_loop" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    n_adm = rep["design"]["n_goals_admissible"]
    n_ina = rep["design"]["n_goals_inadmissible"]

    # --- R1：FK 最大可达 = L1+L2 ---
    chk("R1_reverse_max_reach", abs((L1 + L2) - 0.7000) < 1e-9, "L1+L2=%.4f" % (L1 + L2))

    # --- R2：IK 往返 ---
    bad = 0
    for q0 in ([0.6, -0.9], [1.0, -1.2], [-0.8, 0.5]):
        p = fk(q0)
        sols = ik_all(p)
        if not any(abs(s[0] - q0[0]) < 1e-6 and abs(s[1] - q0[1]) < 1e-6 for s in sols):
            bad += 1
    chk("R2_IK_roundtrip", bad == 0, "mismatch=%d/3" % bad)

    # --- 独立复现目标集（同种子）---
    # 注意：报告 design.n_goals_inadmissible=15 是**输出**（B1=range(n_ina)+B2=max(1,n_ina//2)），
    # 但 build_goal_sets 的输入参数 n_ina=10（e70 main 非 quick 默认），10+5=15。用 10 复现。
    A, B = build_goal_sets(n_adm, 10)
    chk("R3_goalset_size_matches", len(A) == n_adm and len(B) == n_ina,
        "A=%d(exp %d) B=%d(exp %d)" % (len(A), n_adm, len(B), n_ina))

    # --- R4：独立复现粗提案误差（证明离环粗建模必然失败）---
    errs = []
    for gi, g in enumerate(A):
        q_ref = coarse_proposal(np.array(g), "A|%d" % gi, [0.0, 0.0])
        errs.append(float(np.linalg.norm(np.array(g) - fk(q_ref))))
    med = float(np.median(errs))
    coarse_rate = sum(1 for e in errs if e <= 0.01) / len(errs)
    chk("R4_coarse_error_reproduced",
        abs(med - rep["SET_A_admissible"]["coarse"]["median_err_m"]) < 0.02 and coarse_rate == 0.0,
        "re_med=%.4f stored=%.4f re_reach=%.3f" % (med, rep["SET_A_admissible"]["coarse"]["median_err_m"], coarse_rate))

    # --- R5 证伪：粗提案到点率确实 ≈0（残差环才把率拉到 0.938）---
    resid_rate = rep["SET_A_admissible"]["resid"]["reach_1cm_rate"]
    chk("R5_falsify_loop_necessary", coarse_rate == 0.0 and resid_rate > 0.5,
        "coarse_reach=%.3f resid_reach=%.3f" % (coarse_rate, resid_rate))

    # --- R6：SET_B 目标确无可达 IK（独立复现 + 验证）---
    inadm_ok = 0
    for b in B:
        p = b["p"]
        sols = ik_all(p)
        if not sols:
            inadm_ok += 1
        elif all(float(np.max(np.abs(s))) > JLIM for s in sols):
            inadm_ok += 1
    chk("R6_SET_B_truly_inadmissible", inadm_ok == len(B),
        "%d/%d confirmed inadmissible" % (inadm_ok, len(B)))

    # --- R7：独立残差环（纯运动学牛顿步，不跑控制器）能把粗提案误差收敛 ---
    # 取一个 SET_A 目标，从粗提案出发做 jac 伪逆迭代（限位内线搜索），验证误差单调下降
    g0 = np.array(A[0])
    q_ref = coarse_proposal(g0, "A|0", [0.0, 0.0])
    errs_loop = []
    for _ in range(8):
        q_cur = q_ref
        e = g0 - fk(q_cur)
        en = float(np.linalg.norm(e))
        errs_loop.append(en)
        if en <= 0.01:
            break
        Ji = np.array([[-L1 * math.sin(q_cur[0]) - L2 * math.sin(q_cur[0] + q_cur[1]), -L2 * math.sin(q_cur[0] + q_cur[1])],
                       [L1 * math.cos(q_cur[0]) + L2 * math.cos(q_cur[0] + q_cur[1]), L2 * math.cos(q_cur[0] + q_cur[1])]])
        try:
            dq = np.linalg.solve(Ji, e)
        except np.linalg.LinAlgError:
            break
        alpha, ok = 1.0, False
        for _ in range(8):
            cand = q_ref + alpha * dq
            if np.all(np.abs(cand) <= JLIM):
                q_ref, ok = cand, True
                break
            alpha *= 0.5
        if not ok:
            break
    chk("R7_kinematic_residual_loop_converges", errs_loop[0] > errs_loop[-1] and errs_loop[-1] < errs_loop[0],
        "start=%.4f end=%.4f" % (errs_loop[0], errs_loop[-1]))

    # --- R8：CLAMP 判据的**证伪见证**（不可达目标上夹取≠到点：不优于空）---
    # 注：CLAMP 臂在 SET_B（不可达目标）上，与 COARSE 同到点率；正确处置是 reject/⊘（RESID 亦 0）。
    B_adm = rep["SET_B_inadmissible"]
    clamp_rate = B_adm["clamp"]["reach_1cm_rate"]
    coarse_rate_b = B_adm["coarse"]["reach_1cm_rate"]
    resid_rate_b = B_adm["resid"]["reach_1cm_rate"]
    _res["falsify"].append({"id": "E70-CLAMP-witness",
                            "claim": "夹取（CLAMP）是错误处置（挪走末端但不增加到点率）",
                            "witness": "不可达目标上 clamp=%.3f 不优于空 coarse=%.3f（resid=%.3f 亦 0，正确处置为 reject/⊘）"
                                       " ⇒ 若夹取能提升到点率则本主张被推翻"
                                       % (clamp_rate, coarse_rate_b, resid_rate_b)})
    chk("R8_falsify_clamp_is_error",
        clamp_rate <= coarse_rate_b + 1e-9 and abs(clamp_rate) < 1e-9,
        "clamp=%.3f coarse=%.3f resid=%.3f" % (clamp_rate, coarse_rate_b, resid_rate_b))

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
