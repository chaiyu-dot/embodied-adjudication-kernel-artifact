# -*- coding: utf-8 -*-
"""_audit_e64_strict.py —— E64 严格审核（L3 纯 JSON 自洽 + 物理等价 + 表一致性）。

E64 是 wall-clock 基准（时间不可逐位复现）→ 不能重跑计时。本审核：
  · 从存储 measures 重算派生量（D_*_p50_by_N、单调性 H64-1、向量化 H64-2、speedup、meets）
  · 核验 .tex 表与 measures 一致
  · 独立核验 loop(RNE) 与 vector(闭式) **数值等价**（物理一致性，非计时）
  · 篡改必报
产物：_audit_e64_strict.json
"""
import copy
import json
import math
import os
import re
import sys

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e64_latency_vs_order_report.json")
OUT_TEX = r"D:\Fatima\期刊投稿_TMLR_2026-09-06\manuscript\figures\tab_latency_order.tex"
OUT = os.path.join(EVAL, "_audit_e64_strict.json")
L1, L2, M1, M2 = 0.4, 0.3, 0.6, 0.35

_res = {"experiment": "E64 严格审核(L3 纯JSON自洽+物理等价)", "independence_scope":
        "不 import e64；只读 JSON + 独立核物理等价（rne_dynamics 参考）", "checks": [], "tamper": []}


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _cf_batch(traj):
    q1, q2 = traj["q"][:, 0], traj["q"][:, 1]
    d1, d2 = traj["qd"][:, 0], traj["qd"][:, 1]
    a1, a2 = traj["qdd"][:, 0], traj["qdd"][:, 1]
    lc1, lc2 = L1 / 2.0, L2 / 2.0
    i1, i2 = M1 * L1 * L1 / 12.0, M2 * L2 * L2 / 12.0
    c2, s2 = np.cos(q2), np.sin(q2); c12 = np.cos(q1 + q2)
    M11 = M1 * lc1 ** 2 + M2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + i1 + i2
    M12 = M2 * (lc2 ** 2 + L1 * lc2 * c2) + i2
    M22 = M2 * lc2 ** 2 + i2
    C1 = -M2 * L1 * lc2 * s2 * (2 * d1 * d2 + d2 ** 2); C2 = M2 * L1 * lc2 * s2 * d1 ** 2
    G1 = (M1 * lc1 + M2 * L1) * 9.81 * np.cos(q1) + M2 * lc2 * 9.81 * c12
    G2 = M2 * lc2 * 9.81 * c12
    return M11 * a1 + M12 * a2 + C1 + G1, M12 * a1 + M22 * a2 + C2 + G2


def _forward(rep):
    ch = []
    M = rep["results"]["measures"]
    # S1 measures 键数 + D 派生
    d_loop = [M["D_loop_%d" % n]["p50_ms"] for n in (1, 101, 501)]
    d_vec = [M["D_vector_%d" % n]["p50_ms"] for n in (1, 101, 501)]
    ch.append(_chk("S1_measures_and_Dp50_recompute",
                   len(M) == 8 and d_loop == rep["results"]["D_loop_p50_by_N"]
                   and d_vec == rep["results"]["D_vector_p50_by_N"], "n_keys=%d" % len(M)))
    # S2 H64 重算
    mono = all(d_loop[i] <= d_loop[i + 1] + 1e-9 for i in range(2))
    vf = all(d_vec[i] <= d_loop[i] for i in range(3))
    H = rep["preregistered_verdict"]
    ch.append(_chk("S2_H64_recompute", H["H64-1_D_latency_monotone_in_N"] == mono
                   and H["H64-2_vectorized_le_loop"] == vf and rep["verdict_pass"] == (mono and vf),
                   "mono=%s vf=%s verdict=%s" % (mono, vf, rep["verdict_pass"])))
    # S3 meets + speedup
    ch.append(_chk("S3_meets_and_speedup_recompute",
                   rep["results"]["D_loop_101_meets_30ms"] == (d_loop[1] <= 30.0)
                   and (abs(rep["results"]["speedup_loop_over_vector_101"] - round(d_loop[1] / d_vec[1], 2)) < 0.02
                        if d_vec[1] else True), "meets=%s" % rep["results"]["D_loop_101_meets_30ms"]))
    # S4 self_consistency
    ch.append(_chk("S4_self_consistency", rep["results"]["self_consistency"]["keys"] is True))
    # S5 .tex 表一致
    if os.path.exists(OUT_TEX):
        txt = open(OUT_TEX, encoding="utf-8").read()
        rows = re.findall(r"^([A-Za-z0-9_\\]+)\s*&\s*(\S+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\", txt, re.M)
        bad = 0
        for k, nn, p50, p95 in rows:
            key = k.replace("\\_", "_")
            if key in M and (abs(float(p50) - M[key]["p50_ms"]) > 1e-3 or abs(float(p95) - M[key]["p95_ms"]) > 1e-3):
                bad += 1
        ch.append(_chk("S5_tex_table_matches_measures", len(rows) == 8 and bad == 0,
                       "tex_rows=%d mismatch=%d" % (len(rows), bad)))
    else:
        ch.append(_chk("S5_tex_table_exists", False, "missing %s" % OUT_TEX))
    # S6 物理等价：loop(RNE) vs vector(闭式)
    traj = {"t": np.linspace(0, 0.5, 101)}
    q = np.stack([np.linspace(0, 1.2, 101), np.linspace(0, 0.8, 101)], 1)
    qd = np.stack([np.full(101, 1.0), np.full(101, 0.5)], 1)
    qdd = np.zeros((101, 2))
    traj.update({"q": q, "qd": qd, "qdd": qdd})
    t1, t2 = _cf_batch(traj)
    e1 = e2 = 0.0
    for i in range(101):
        lk = RNE.planar_2r_model(L1, L2, M1, M2, plane="xy", q1=float(q[i, 0]), q2=float(q[i, 1]))
        tau = np.asarray(RNE.rne_inverse_dynamics(lk, list(q[i]), list(qd[i]), list(qdd[i]), g=(0.0, -9.81, 0.0)), float)
        e1 = max(e1, abs(tau[0] - t1[i])); e2 = max(e2, abs(tau[1] - t2[i]))
    ch.append(_chk("S6_loop_vs_vector_physics_equivalent", e1 < 1e-8 and e2 < 1e-8,
                   "max|Δ|=(%.2e,%.2e)" % (e1, e2)))
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
    print("E64 严格审核（L3 纯JSON自洽 + 物理等价，不 import e64）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 D_loop_101 p50 打破单调 → S2 必报",
          lambda r: r["results"]["measures"]["D_loop_101"].__setitem__("p50_ms", 0.0), ["S1", "S2"])
    _tamp("R2 篡改 H64-1 → S2 必报",
          lambda r: r["preregistered_verdict"].__setitem__("H64-1_D_latency_monotone_in_N", False), ["S2"])
    _tamp("R3 篡改 speedup → S3 必报",
          lambda r: r["results"].__setitem__("speedup_loop_over_vector_101", 999.0), ["S3"])
    _tamp("R4 篡改 verdict_pass → S2 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["S2"])

    tp = sum(1 for t in _res["tamper"] if t["caught"])
    allpass = (npass == len(fwd)) and (tp == len(_res["tamper"]))
    _res["n_pass"] = npass; _res["n_total"] = len(fwd)
    _res["n_tamper_pass"] = tp; _res["n_tamper_total"] = len(_res["tamper"])
    _res["strict_pass"] = bool(allpass)
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n严格审核：前提 %d/%d ｜ 篡改 %d/%d ｜ strict_pass = %s" % (npass, len(fwd), tp, len(_res["tamper"]), allpass))
    print("wrote", OUT)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
