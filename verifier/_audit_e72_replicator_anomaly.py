# -*- coding: utf-8 -*-
"""_audit_e72_replicator_anomaly.py —— **第三方复刻者视角**的异常排查审核（零 API）。

模拟一个**不信任作者代码**的外部复刻者：只凭论文/台架公开物理参数与 e72 报告里的数字，
自己独立实现整套摩擦锥数学，重算报告里每一个关键数字（T1–T4），再对作者模块做
**异常输入 fuzz**（T5：μ=0 / Fn≤0 / 负法向 / 极端 q / 超大 F / NaN），检查作者代码是否
崩溃或撒谎（诚实降级）。最后做内部自洽（T6），复刻字节一致性由复刻 suite 的 F3 统一覆盖。

与 _audit_e72（作者自查）的本质区别：
  · 本脚本 T1–T4 **不 import planning.friction_cone / e72**，纯第三方独立实现；
  · 本脚本 T5 把作者模块当黑盒，故意喂作者测试没覆盖的坏输入，排查鲁棒性异常；
  · 这正是“别人来复刻我实验数据时排查异常”的口径。

产物：_audit_e72_replicator_anomaly.json（{"audit_pass": bool, "checks": [...]}）
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e72_friction_cone_report.json")


def _find_src(start):
    """向上查找含 planning/control_strategies.py 的目录（与 e72 同口径）。"""
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "planning", "control_strategies.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.dirname(os.path.abspath(start))


SRC = _find_src(EVAL)
sys.path.insert(0, SRC)  # 仅 T5 黑盒测试作者模块需要；T1–T4 为独立实现，不依赖 planning

# ---- 第三方独立实现（公开物理参数，与论文/E67 台架一致，不读作者模块） ----
L1, L2, G = 0.40, 0.30, 9.81
TAU_LIM = [5.0, 1.8]
Q = [0.7, -1.0]
S_SAFE, S_UNSAFE, S_NA = "SAFE", "UNSAFE", "NA"


def jac(q):
    s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def tau_for_wrench(q, F):
    return jac(q).T @ np.asarray(F, float)


def cone_ratio(F, mu):
    Ft, Fn = float(np.asarray(F, float)[0]), float(np.asarray(F, float)[1])
    if mu <= 0.0:
        return float("inf")
    return abs(Ft) / (mu * max(abs(Fn), 1e-9))


def max_tangential_push(q, mu, Fn_hi=40.0):
    best = (0.0, 0.0)
    for i in range(401):
        Fn = Fn_hi * i / 400.0
        Ft_cone = mu * Fn
        lo, hi = 0.0, 200.0
        if np.all(np.abs(tau_for_wrench(q, [0.0, Fn])) <= np.array(TAU_LIM) + 1e-12):
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if np.all(np.abs(tau_for_wrench(q, [mid, Fn])) <= np.array(TAU_LIM) + 1e-12):
                    lo = mid
                else:
                    hi = mid
        Ft = min(Ft_cone, lo)
        if Ft > best[0]:
            best = (Ft, Fn)
    return {"max_tangential_N": round(best[0], 4), "at_normal_N": round(best[1], 4),
            "cone_utilization": round(best[0] / max(mu * best[1], 1e-9), 4)}


def contact_verdict(q, F, mu):
    F = np.asarray(F, float)
    Ft, Fn = float(F[0]), float(F[1])
    tau = tau_for_wrench(q, F)
    feasible = bool(np.all(np.abs(tau) <= np.array(TAU_LIM) + 1e-12))
    rho = cone_ratio(F, mu)
    rho_rep = None if not math.isfinite(rho) else float(rho)
    if Fn <= 1e-9:
        return {"value": S_NA, "rho": rho_rep, "feasible": feasible}
    if not feasible:
        return {"value": S_UNSAFE, "rho": rho_rep, "feasible": False}
    if rho > 1.0 + 1e-9:
        return {"value": S_UNSAFE, "rho": rho_rep, "feasible": True}
    return {"value": S_SAFE, "rho": rho_rep, "feasible": True}


def combine(contact_v, dynamic_v):
    cv = contact_v["value"] if isinstance(contact_v, dict) else contact_v
    dv = dynamic_v["value"] if isinstance(dynamic_v, dict) else dynamic_v
    if cv == S_NA:
        return {"value": dv, "shorted_by": "contact_NA", "dynamic": dv}
    if cv == S_UNSAFE:
        return {"value": S_UNSAFE, "shorted_by": "contact_UNSAFE", "dynamic": dv}
    return {"value": dv, "shorted_by": "none", "dynamic": dv}


def deep_diff(a, b, ignore=("date", "elapsed", "runtime", "wall", "wall_s")):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in (set(a) | set(b)):
            if k in ignore:
                continue
            if k not in b:
                out.append((k, "missing_in_b", a[k]))
            else:
                out += deep_diff(a[k], b[k], ignore)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(("len", a, b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += deep_diff(x, y, ignore)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            out.append(("val", a, b))
    elif a != b:
        out.append(("val", a, b))
    return out


def _chk(name, cond, detail=""):
    return {"name": name, "pass": bool(cond), "detail": str(detail)}


def main():
    rep = json.load(open(REPORT, encoding="utf-8"))
    checks = []

    # ---- T1 独立重算包络（A 段） ----
    env_ok = True
    for mu in (0.2, 0.4, 0.6, 0.8):
        r = max_tangential_push(Q, mu)
        st = rep["A_envelope_replica"]["rows"]["mu=%.1f" % mu]
        if (abs(r["max_tangential_N"] - st["max_tangential_N"]) > 1e-6 or
                abs(r["at_normal_N"] - st["at_normal_N"]) > 1e-6 or
                abs(r["cone_utilization"] - st["cone_utilization"]) > 1e-6):
            env_ok = False
    checks.append(_chk("T1_envelope_independent_recompute", env_ok,
                       "第三方独立实现 vs 报告包络不一致"))

    # ---- T2 独立重算 τ-可行性 + 三值（B 段锚点） ----
    feas = rep["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]
    feas_ok = True
    for fs, row in feas.items():
        F = [float(x) for x in fs[1:-1].split(",")]
        v = contact_verdict(Q, F, 0.8)
        if v["value"] != row["value"] or v["feasible"] != row["feasible"]:
            feas_ok = False
    checks.append(_chk("T2_tau_class_independent_recompute", feas_ok, "τ 分类重算不一致"))

    # ---- T3 独立重算滑移翻转 + ⊘ ----
    sf = rep["B_tau_coupling_three_valued"]["slip_flip_F=[3,5]"]
    lo = contact_verdict(Q, [3.0, 5.0], 0.2)
    hi = contact_verdict(Q, [3.0, 5.0], 0.8)
    na = contact_verdict(Q, [0.0, 0.0], 0.8)
    flip_ok = (lo["value"] == sf["mu=0.2"] == S_UNSAFE and
               hi["value"] == sf["mu=0.8"] == S_SAFE and
               na["value"] == rep["B_tau_coupling_three_valued"]["no_contact_NA"]["value"] == S_NA)
    checks.append(_chk("T3_slip_flip_and_NA_recompute", flip_ok,
                       "lo=%s hi=%s na=%s" % (lo["value"], hi["value"], na["value"])))

    # ---- T4 独立重算组合真值表（C 段） ----
    tbl = rep["C_combination_invariants"]["combination_table"]
    c_NA = contact_verdict(Q, [0.0, 0.0], 0.8)
    c_un = contact_verdict(Q, [8.0, 5.0], 0.8)
    c_sa = contact_verdict(Q, [1.0, 5.0], 0.4)
    expect = {
        "NA_x_SAFE": combine(c_NA, {"value": S_SAFE})["value"],
        "NA_x_UNSAFE": combine(c_NA, {"value": S_UNSAFE})["value"],
        "UNSAFE_x_SAFE": combine(c_un, {"value": S_SAFE})["value"],
        "SAFE_x_UNSAFE": combine(c_sa, {"value": S_UNSAFE})["value"],
        "SAFE_x_SAFE": combine(c_sa, {"value": S_SAFE})["value"],
    }
    combo_ok = all(tbl[k]["value"] == expect[k] for k in expect)
    checks.append(_chk("T4_combination_table_recompute", combo_ok,
                       {k: (tbl[k]["value"], expect[k]) for k in expect}))

    # ---- T5 异常输入 fuzz（黑盒测作者模块：能否崩/撒谎） ----
    sys.path.insert(0, EVAL)
    from planning.friction_cone import (contact_verdict as av,
                                        max_tangential_push as amtp)
    fuzz_ok = True
    fuzz_log = []
    try:
        # μ=0（无摩擦）→ 必 UNSAFE 且 ρ 记为 None（不泄漏 inf / 不撒谎 SAFE）
        v0 = av(Q, [3.0, 5.0], 0.0)
        fuzz_ok &= (v0["value"] == S_UNSAFE and v0["rho"] is None)
        fuzz_log.append("mu=0:%s/rho=%s" % (v0["value"], v0["rho"]))
        # Fn=0（无接触）→ ⊘（不谎称 SAFE）
        vn = av(Q, [0.0, 0.0], 0.5)
        fuzz_ok &= (vn["value"] == S_NA)
        fuzz_log.append("Fn=0:%s" % vn["value"])
        # 负法向（拉离）→ ⊘
        vneg = av(Q, [0.0, -5.0], 0.5)
        fuzz_ok &= (vneg["value"] == S_NA)
        fuzz_log.append("Fn<0:%s" % vneg["value"])
        # 极端 q（奇异位形附近）→ 不崩、返回合法三值
        ve = av([3.14, 3.14], [3.0, 5.0], 0.5)
        fuzz_ok &= (ve["value"] in (S_SAFE, S_UNSAFE, S_NA))
        fuzz_log.append("q=pi:%s" % ve["value"])
        # 超大 F（远超执行器）→ UNSAFE（不可施加）
        vbig = av(Q, [100.0, 100.0], 0.5)
        fuzz_ok &= (vbig["value"] == S_UNSAFE)
        fuzz_log.append("F=100:%s" % vbig["value"])
        # NaN 输入 → 不崩溃（返回合理值或 ⊘，绝不抛异常）
        vnan = av(Q, [float("nan"), 5.0], 0.5)
        fuzz_ok &= (vnan["value"] in (S_SAFE, S_UNSAFE, S_NA))
        fuzz_log.append("NaN:%s" % vnan["value"])
        # 超大 q 下 max_tangential_push 不崩且非负有限
        mt = amtp([10.0, 10.0], 0.5)
        fuzz_ok &= (math.isfinite(mt["max_tangential_N"]) and mt["max_tangential_N"] >= 0)
        fuzz_log.append("mt@q=10:%s" % mt["max_tangential_N"])
    except Exception as ex:  # 复刻者最关心的：作者代码在坏输入下崩了
        fuzz_ok = False
        fuzz_log.append("EXCEPTION:%r" % ex)
    checks.append(_chk("T5_anomaly_input_fuzz_blackbox", fuzz_ok, " | ".join(fuzz_log)))

    # ---- T6 内部自洽（报告判据与 verdict_pass 逻辑一致 + 来源档完备） ----
    c = rep["criteria"]
    C1 = bool(c["C1_envelope_match"] and rep["A_envelope_replica"]["all_match"])
    C2 = bool(c["C2_tau_and_cone_class"] and rep["B_tau_coupling_three_valued"]["all_ok"])
    C3 = bool(c["C3_combination_and_doc"] and rep["C_combination_invariants"]["all_ok"])
    C4 = bool(c["C4_determinism_finite"] and
              rep["C_combination_invariants"]["deterministic"] and
              rep["C_combination_invariants"]["inf_capped"] and
              rep["C_combination_invariants"]["NA_rho_finite"])
    inv = (C1 and C2 and C3 and C4)
    self_ok = (inv == bool(rep["verdict_pass"]))
    # 来源档完备（复刻者从报告可查）：
    #  · B 段锚点每个原子裁决都序列化了非空 source
    #  · C 段 source_doc_ok 布尔证明组合所用原子裁决带 source
    src_ok_b = all(bool(row.get("source")) for row in
                   rep["B_tau_coupling_three_valued"]["tau_feasibility_anchor"].values())
    src_ok_c = bool(rep["C_combination_invariants"].get("source_doc_ok", False))
    src_ok = src_ok_b and src_ok_c
    checks.append(_chk("T6_invariants_self_consistent_and_source_doc", self_ok and src_ok,
                       "inv=%s rep=%s srcB=%s srcC=%s" % (inv, rep["verdict_pass"], src_ok_b, src_ok_c)))

    # ---- T7 复刻字节一致性：由复刻 suite 的 F3 统一覆盖（所有报告逐字段比对），
    #      本脚本不再重复，以免在“原目录 / 复刻目录”两种运行环境下产生非确定性 detail。

    # ---- T8–T10 篡改用例（第三方复刻者构造反例：改报告 → 对应重算必报） ----
    import copy as _copy

    def _env_ok(r):
        for mu in (0.2, 0.4, 0.6, 0.8):
            rr = max_tangential_push(Q, mu)
            st = r["A_envelope_replica"]["rows"]["mu=%.1f" % mu]
            if (abs(rr["max_tangential_N"] - st["max_tangential_N"]) > 1e-6 or
                    abs(rr["at_normal_N"] - st["at_normal_N"]) > 1e-6 or
                    abs(rr["cone_utilization"] - st["cone_utilization"]) > 1e-6):
                return False
        return True

    def _anchor_ok(r):
        for fs, row in r["B_tau_coupling_three_valued"]["tau_feasibility_anchor"].items():
            F = [float(x) for x in fs[1:-1].split(",")]
            v = contact_verdict(Q, F, 0.8)
            if v["value"] != row["value"] or v["feasible"] != row["feasible"]:
                return False
        return True

    def _combo_ok(r):
        tbl = r["C_combination_invariants"]["combination_table"]
        c_NA = contact_verdict(Q, [0.0, 0.0], 0.8)
        c_un = contact_verdict(Q, [8.0, 5.0], 0.8)
        c_sa = contact_verdict(Q, [1.0, 5.0], 0.4)
        exp = {"NA_x_SAFE": combine(c_NA, {"value": S_SAFE})["value"],
               "NA_x_UNSAFE": combine(c_NA, {"value": S_UNSAFE})["value"],
               "UNSAFE_x_SAFE": combine(c_un, {"value": S_SAFE})["value"],
               "SAFE_x_UNSAFE": combine(c_sa, {"value": S_UNSAFE})["value"],
               "SAFE_x_SAFE": combine(c_sa, {"value": S_SAFE})["value"]}
        return all(tbl[k]["value"] == exp[k] for k in exp)

    bad_env = _copy.deepcopy(rep)
    bad_env["A_envelope_replica"]["rows"]["mu=0.2"]["max_tangential_N"] = 9.99
    checks.append(_chk("T8_tamper_envelope_detected", not _env_ok(bad_env),
                       "改包络数值 → 独立重算必报"))

    bad_anchor = _copy.deepcopy(rep)
    bad_anchor["B_tau_coupling_three_valued"]["tau_feasibility_anchor"]["[8.0, 5.0]"]["value"] = S_SAFE
    checks.append(_chk("T9_tamper_anchor_verdict_detected", not _anchor_ok(bad_anchor),
                       "改 [8,5] 裁决为 SAFE（τ 不可行却称安全）→ 重算必报"))

    bad_combo = _copy.deepcopy(rep)
    bad_combo["C_combination_invariants"]["combination_table"]["UNSAFE_x_SAFE"]["value"] = S_SAFE
    checks.append(_chk("T10_tamper_combination_detected", not _combo_ok(bad_combo),
                       "改组合 UNSAFE_x_SAFE → 独立重算必报"))

    npass = sum(x["pass"] for x in checks)
    out = {"audit": "_audit_e72_replicator_anomaly", "experiment": "e72_friction_cone_verdict",
           "pass": npass, "total": len(checks), "verdict": bool(npass == len(checks)),
           "audit_pass": bool(npass == len(checks)), "checks": checks}
    with open(os.path.join(EVAL, "_audit_e72_replicator_anomaly.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly audit pass = %d/%d  verdict=%s"
          % (npass, len(checks), out["verdict"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    main()
