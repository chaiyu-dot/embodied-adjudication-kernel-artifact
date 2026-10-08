# -*- coding: utf-8 -*-
"""_audit_e74_replicator_anomaly.py —— e74 的**第三方复刻者视角**异常排查审核（零 API）。

模拟一个**不信任作者代码**的外部复刻者：
  T1–T3 只凭数据文件（报告 + 台架规格）**独立重算**关键量，不 import 作者模块；
  T4    把作者模块当**黑盒**，喂异常输入（ωn=0/负/NaN/极大、退化边界行）排查鲁棒性；
  T5    核验报告内部自洽与"能力口径诚实"；
  T6    递归扫描全报告 inf/NaN。

与 _audit_e74_bandwidth.py（作者自查）的区别：本脚本 T1–T3 **不 import e66/e67/e74**。

产物：_audit_e74_replicator_anomaly.json
"""
import json
import math
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e74_tracking_bandwidth_report.json")
SRC = os.path.dirname(EVAL)
_BANNED = ("e66", "e67", "e74", "planning")


def _chk(n, ok, d=""):
    return {"name": n, "pass": bool(ok), "detail": str(d)}


def _guard():
    return sorted({m.split(".")[0] for m in list(sys.modules) if m.split(".")[0] in _BANNED})


def _finite(o, p="$"):
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += _finite(v, "%s.%s" % (p, k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _finite(v, "%s[%d]" % (p, i))
    elif isinstance(o, float) and not math.isfinite(o):
        bad.append(p)
    return bad


def f3db(wn, zeta):
    a = 1.0 - 2 * zeta * zeta
    return wn * math.sqrt(a + math.sqrt(a * a + 1.0)) / (2 * math.pi)


def boundary_indep(rows, tol, unsat, sat_tol):
    ok = []
    for k, v in rows.items():
        if v["rms_err_m"] is None or v["diverged"]:
            continue
        if v["rms_err_m"] <= tol and (not unsat or v["sat_frac_demand"] <= sat_tol):
            ok.append(float(k))
    return round(max(ok), 1) if ok else None


def run():
    rep = json.load(open(REPORT, encoding="utf-8"))
    bench = rep["bench"]
    A, B, C, D = rep["A_freq_domain_bw"], rep["B_task_tolerance_bw"], rep["C_e67c_anchor"], rep["D_calibration"]
    ch = []

    # T1 二阶闭式与口径因子（独立重算）
    zeta = bench["zeta"]
    bad = []
    for name, v in A["by_wn"].items():
        fa = f3db(v["wn_rad_s"], zeta)
        if abs(fa - v["analytic_hz"]) > 2e-3:
            bad.append((name, v["analytic_hz"], round(fa, 3)))
    ch.append(_chk("T1_analytic_bw_independent", not bad, "bad=%s" % bad[:3]))

    # T2 两条边界（独立重算，逐格）
    bbad = []
    for name, row in B["by_wn"].items():
        for tk, pt in row["per_tol"].items():
            tol = float(tk.split("=")[1][:-1])
            for kind, unsat in (("any", False), ("unsaturated", True)):
                rb = boundary_indep(row["rows"], tol, unsat, bench["sat_tol"])
                if rb != pt[kind]["hz"]:
                    bbad.append((name, tk, kind, pt[kind]["hz"], rb))
    ch.append(_chk("T2_boundaries_independent", not bbad, "反例=%d %s" % (len(bbad), bbad[:3])))

    # T3 −3 dB 交越点独立重算（从报告的 points 插值复现 measured_hz）
    cbad = []
    for name, v in A["by_wn"].items():
        pts = {float(k): val["gain_db"] for k, val in v["points"].items()}
        prev = None
        est = None
        for f in sorted(pts):
            g = pts[f]
            if g <= -3.0:
                if prev is None:
                    est = float(f)
                else:
                    f0, g0 = prev
                    t = (g0 + 3.0) / (g0 - g) if abs(g0 - g) > 1e-12 else 0.0
                    est = float(f0 * (f / f0) ** max(0.0, min(1.0, t)))
                break
            prev = (f, g)
        if v["measured_hz"] is not None and (est is None or abs(est - v["measured_hz"]) > 2e-3):
            cbad.append((name, v["measured_hz"], None if est is None else round(est, 3)))
    ch.append(_chk("T3_f3db_crossing_independent", not cbad, "反例=%s" % cbad[:3]))

    # T4 黑盒 fuzz 作者模块（坏输入不崩、不撒谎）
    guard_at_t1_t3 = _guard()          # T1–T3 完成时**尚未**加载任何作者模块（独立性证据）
    sys.path.insert(0, SRC)
    sys.path.insert(0, EVAL)
    fuzz_ok, notes = True, []
    try:
        import e74_tracking_bandwidth_curve as E74
        # ωn=0/负/极大：uniform_gains 必须返回有限数
        for wn in (0.0, -5.0, 1e6):
            kp, kd = E74.uniform_gains(wn)
            if not (all(math.isfinite(x) for x in kp) and all(math.isfinite(x) for x in kd)):
                fuzz_ok = False
                notes.append("wn=%s nonfinite gains" % wn)
        # 二阶闭式单调性：−3dB 频率随 ωn 严格递增
        vals = [E74.f3db_second_order(w) for w in (1.0, 2.0, 4.0, 8.0)]
        if not all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            fuzz_ok = False
            notes.append("f3db not monotone")
        # 退化边界行：空表/全 NaN 行不得崩
        if E74._boundary({}, 0.01, False)["hz"] is not None:
            fuzz_ok = False
            notes.append("empty rows not None")
        deg = {"0.1": {"rms_err_m": None, "diverged": False, "sat_frac_demand": 0.0}}
        if E74._boundary(deg, 0.01, False)["hz"] is not None:
            fuzz_ok = False
            notes.append("None rms not skipped")
        # 越界 zeta（>1）不崩
        if not math.isfinite(E74.f3db_second_order(5.0, zeta=1.5)):
            fuzz_ok = False
    except Exception as e:  # noqa: BLE001
        fuzz_ok = False
        notes.append("exception:%s" % e)
    ch.append(_chk("T4_blackbox_fuzz", fuzz_ok, "notes=%s" % notes))

    # T8 E 段（可识别性探针）：只凭数据文件核验逐关节派生量与"约束已激活"
    Esec = rep.get("E_identification_probes") or {}
    mp_spec = (bench.get("payload") or {}).get("mp_probe_kg")
    e_bad = []
    for nm, pr in (Esec.get("probes") or {}).items():
        n = int(pr["n_steps"])
        if mp_spec is None or abs(pr["mp_kg"] - mp_spec) > 1e-12:
            e_bad.append((nm, "mp≠bench.payload"))
        for j in (0, 1):
            if abs(pr["sat_frac_per_joint"][j] - pr["violation_steps_per_joint"][j] / n) > 5e-5:
                e_bad.append((nm, j, "sat_frac≠viol/n"))
    act = [(pr["demand_peak_per_joint"][1], pr["violation_steps_per_joint"][1])
           for pr in (Esec.get("probes") or {}).values()]
    ch.append(_chk("T8_E_section_consistent_and_activated",
                   bool(Esec) and not e_bad and bool(act)
                   and any(d > bench["TAU_LIM"][1] + 1e-9 and v > 0 for d, v in act)
                   and bool(Esec.get("joint1_limit_exercised")),
                   "反例=%s | 关节1(峰值,越限)=%s" % (e_bad[:3], act)))

    # T9 F 段（振幅口径）：纯数据核验自洽 + amp 字段与 bench 一致
    Fsec = rep.get("F_amplitude_calibration") or {}
    amps_spec = bench.get("amp_sweep")
    f_bad = []
    for nm, v in (Fsec.get("by_wn") or {}).items():
        for ak, pt in v["per_amp"].items():
            if amps_spec is None or abs(pt["amp_m"] - float(ak.split("=")[1][:-1])) > 1e-12:
                f_bad.append((nm, ak, "amp字段≠bench.amp_sweep"))
            for t in (0.02, 0.01, 0.005, 0.002):
                k = "tol=%.3fm" % t
                rb = boundary_indep(pt["rows"], t, False, bench["sat_tol"])
                if rb != pt["boundary_any"][k]:
                    f_bad.append((nm, ak, k, pt["boundary_any"][k], rb))
    f_amp_ok = bool(Fsec) and all(v["boundary_non_increasing_in_amp"] for v in (Fsec.get("by_wn") or {}).values())
    ch.append(_chk("T9_F_amplitude_section_independent", (not f_bad) and f_amp_ok,
                   "反例=%s | 对amp非增=%s" % (f_bad[:3], f_amp_ok)))

    # T10 G 段（控制器族口径）：纯数据核验自洽（7 族、spread 可独立重算、pdg_hz==PD_G）+ 口径诚实
    #   （G 由控制代码算出，非数据文件可独立复现；复刻者视角只验报告内部自洽 + 是否已诚实标注）
    Gsec = rep.get("G_controller_family_calibration") or {}
    g_bad = []
    fams = (Gsec or {}).get("by_family") or {}
    exp = ("P", "PD", "PID", "PD_G", "CT", "FO", "MPC")
    if tuple(fams.keys()) != exp:
        g_bad.append(("族集合不符", tuple(fams.keys())))
    vals = [v.get("bandwidth_hz") for v in fams.values()]
    if any(v is None for v in vals):
        g_bad.append("有族未测得")
    else:
        rec_spread = max(vals) / min(vals)
        if Gsec.get("spread_x") is None or abs(Gsec["spread_x"] - rec_spread) > 1e-3:
            g_bad.append(("spread_x 不一致", Gsec.get("spread_x"), round(rec_spread, 3)))
        if Gsec.get("pdg_hz") is None or abs(Gsec["pdg_hz"] - fams["PD_G"]["bandwidth_hz"]) > 1e-9:
            g_bad.append(("pdg_hz 不一致", Gsec.get("pdg_hz"), fams["PD_G"]["bandwidth_hz"]))
    caveat_ok = any("控制器族口径" in c for c in D.get("caveats", []))
    ch.append(_chk("T10_G_controller_family_independent", (not g_bad) and caveat_ok and bool(fams),
                   "反例=%s | 有caveat=%s" % (g_bad[:3], caveat_ok)))

    # T5 内部自洽 + 口径诚实
    honest = (len(D["caveats"]) >= 4 and D["derate_rel_dev"] <= 0.02
              and all(v["gap"] is None or v["gap"] >= 0 for v in D["boundary_gap_tol1cm"].values())
              and B["sat_tol"] == bench["sat_tol"])
    inv = bool(rep["verdict_pass"] == all(bool(v) for v in rep["criteria"].values()))
    ch.append(_chk("T5_self_consistent_and_honest", honest and inv, "honest=%s inv=%s" % (honest, inv)))

    # T6 全报告 inf/NaN
    badf = _finite(rep)
    ch.append(_chk("T6_no_inf_nan", not badf, "非有限=%s" % badf[:5]))

    # T7 E67-C 锚点独立复核
    rowsC = C["rows"]

    def passes(f, tol):
        v = rowsC.get("%.1f" % f)
        return bool(v and v["rms_err_m"] is not None and v["rms_err_m"] <= tol)
    ch.append(_chk("T7_e67c_anchor_independent",
                   passes(0.2, 0.01) and (not passes(0.5, 0.01)) and passes(0.5, 0.02),
                   "0.2(1cm)=%s 0.5(1cm)=%s 0.5(2cm)=%s" % (passes(0.2, 0.01), passes(0.5, 0.01), passes(0.5, 0.02))))

    npass = sum(c["pass"] for c in ch)
    out = {"audit": "_audit_e74_replicator_anomaly", "experiment": "e74_tracking_bandwidth_curve",
           "pass": npass, "total": len(ch), "verdict": bool(npass == len(ch)),
           "audit_pass": bool(npass == len(ch)), "n_pass": npass, "n_total": len(ch),
           "repo_modules_loaded_at_T1_T3": guard_at_t1_t3,
           "repo_modules_loaded_after_fuzz": _guard(), "checks": ch}
    with open(os.path.join(EVAL, "_audit_e74_replicator_anomaly.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("replicator-anomaly e74: pass=%d/%d verdict=%s" % (npass, len(ch), out["verdict"]))
    for c in ch:
        if not c["pass"]:
            print("  FAIL", c["name"], c["detail"])
    return out


if __name__ == "__main__":
    run()
