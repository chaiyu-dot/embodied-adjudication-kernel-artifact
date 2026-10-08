# -*- coding: utf-8 -*-
"""_audit_e58_strict.py —— E58 的**严格审核（L3，审前提 + 独立物理）**。

不 import 任何 e58 / trackability 模块；审计者**自写** C 阶判据
（trajectory_bandwidth_hz + trackability_margin 的闭式 m_ω/m_τ/m_s 三项取 min）、
quintic 轨迹、cf_peak/cf_static，用与报告**同一套确定性种子**复现每个条件，逐行核对
C_ok 与 c_margin；并做篡改用例。

产物：_audit_e58_strict.json
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
import numpy as np                                                          # noqa: E402

REPORT = os.path.join(EVAL, "e58_trackability_report.json")
OUT = os.path.join(EVAL, "_audit_e58_strict.json")

_res = {"experiment": "E58 严格审核(L3 独立物理)", "independence_scope":
        "独立物理：审计者自写 C 阶闭式判据 + quintic + cf_peak/static，不 import e58/trackability",
        "checks": [], "tamper": []}

# ---- 审计者自写的几何/执行器常数（与 e58 同源数据，但不经其代码）----
L1, L2 = 0.4, 0.3
M1B, M2B = 0.6, 0.35
TAU = [8.0, 4.0]
JLIM = 2.0
Q1_NOM, Q1_OVER = [1.2, 0.8], [2.3, 0.8]
N_JUDGE = 101
T_LIST = [3.0, 1.5, 0.8, 0.5, 0.35, 0.2]
LOAD_LIST = [0.5, 1.0, 2.0]
FAMILY_LIST = ["clean", "near_obstacle", "over_limit"]
N_SEEDS = 5
OMEGA_N, ZETA, T_S = 40.0, 0.7, 0.001
K_OMEGA, K_MARGIN = 2.0, 2.0


# ---- 自写物理 ----
def _quintic(a, b, dur, n):
    T = float(dur); d = b - a
    t = np.linspace(0.0, T, int(n))
    q = a + 10 * d / T ** 3 * t ** 3 - 15 * d / T ** 4 * t ** 4 + 6 * d / T ** 5 * t ** 5
    qd = 30 * d / T ** 3 * t ** 2 - 60 * d / T ** 4 * t ** 3 + 30 * d / T ** 5 * t ** 4
    qdd = 60 * d / T ** 3 * t - 180 * d / T ** 4 * t ** 2 + 120 * d / T ** 5 * t ** 3
    return q, qd, qdd


def _traj(qt, dur, n):
    q1, d1, a1 = _quintic(0.0, qt[0], dur, n)
    q2, d2, a2 = _quintic(0.0, qt[1], dur, n)
    return {"t": np.linspace(0.0, float(dur), int(n)), "q": np.stack([q1, q2], 1),
            "qd": np.stack([d1, d2], 1), "qdd": np.stack([a1, a2], 1)}


def _cf_peak(tr, m1, m2):
    q1, q2 = tr["q"][:, 0], tr["q"][:, 1]
    d1, d2 = tr["qd"][:, 0], tr["qd"][:, 1]
    a1, a2 = tr["qdd"][:, 0], tr["qdd"][:, 1]
    lc1, lc2 = L1 / 2.0, L2 / 2.0
    i1, i2 = m1 * L1 * L1 / 12.0, m2 * L2 * L2 / 12.0
    c2, s2 = np.cos(q2), np.sin(q2); c12 = np.cos(q1 + q2)
    M11 = m1 * lc1 ** 2 + m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + i1 + i2
    M12 = m2 * (lc2 ** 2 + L1 * lc2 * c2) + i2
    M22 = m2 * lc2 ** 2 + i2
    C1 = -m2 * L1 * lc2 * s2 * (2 * d1 * d2 + d2 ** 2); C2 = m2 * L1 * lc2 * s2 * d1 ** 2
    G1 = (m1 * lc1 + m2 * L1) * 9.81 * np.cos(q1) + m2 * lc2 * 9.81 * c12
    G2 = m2 * lc2 * 9.81 * c12
    t1 = M11 * a1 + M12 * a2 + C1 + G1; t2 = M12 * a1 + M22 * a2 + C2 + G2
    return float(np.max(np.abs(t1))), float(np.max(np.abs(t2)))


def _cf_static(qt, m1, m2):
    lc1, lc2 = L1 / 2.0, L2 / 2.0; c12 = np.cos(qt[0] + qt[1])
    G1 = (m1 * lc1 + m2 * L1) * 9.81 * np.cos(qt[0]) + m2 * lc2 * 9.81 * c12
    G2 = m2 * lc2 * 9.81 * c12
    return abs(G1), abs(G2)


def _traj_bw_hz(tr, rel_thresh=0.05):
    t = np.asarray(tr["t"], float); n = len(t)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    best = 0.0
    for j in range(tr["qdd"].shape[1]):
        A = np.abs(np.fft.rfft(tr["qdd"][:, j]))
        if A.size < 2:
            continue
        freqs = np.fft.rfftfreq(n, d=dt)
        thr = rel_thresh * float(np.max(A))
        idx = np.where(A >= thr)[0]
        if idx.size:
            best = max(best, float(freqs[idx[-1]]))
    return best


def _trackability_margin(omega_n, zeta, tr, T_s=T_S, k_omega=K_OMEGA, k_margin=K_MARGIN):
    f = _traj_bw_hz(tr)
    w = 2.0 * math.pi * f
    z2 = min(max(zeta, 0.0), 0.99)
    g = math.sqrt(max(0.0, 1 - 2 * z2 ** 2 + math.sqrt((1 - 2 * z2 ** 2) ** 2 + 1)))
    m_omega = 1.0 - (k_omega * w) / omega_n if omega_n > 0 else -1.0
    t_react = 1.0 / omega_n if omega_n > 0 else float("inf")
    t_budget = (1.0 / (k_margin * f)) if f > 0 else float("inf")
    m_tau = 1.0 - t_react / t_budget if math.isfinite(t_budget) else 1.0
    m_s = 1.0 - 10.0 * T_s * f
    margin = min(m_omega, m_tau, m_s)
    binding = "omega" if margin == m_omega else ("tau" if margin == m_tau else "sampling")
    return margin, {"f_traj_hz": round(f, 3), "binding": binding}


def _seed(*p):
    import zlib
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _regen_rows():
    rows = []
    for T in T_LIST:
        for load in LOAD_LIST:
            for fam in FAMILY_LIST:
                base = Q1_OVER if fam == "over_limit" else Q1_NOM
                for seed in range(N_SEEDS):
                    rng = np.random.RandomState(1000 * seed + int(T * 100))
                    qt = [base[0] + rng.uniform(-0.05, 0.05), base[1] + rng.uniform(-0.05, 0.05)]
                    msc = 1.0 + rng.uniform(-0.03, 0.03)
                    m1, m2 = M1B * load * msc, M2B * load * msc
                    tr = _traj(qt, T, N_JUDGE)
                    exc = max(0.0, max(abs(qt[0]), abs(qt[1])) - JLIM)
                    K_ok = bool(exc <= 0.0)
                    st = _cf_static(qt, m1, m2)
                    sm = float(min((TAU[0] - st[0]) / TAU[0], (TAU[1] - st[1]) / TAU[1]))
                    S_ok = bool(K_ok and sm > 0.0)
                    pk = _cf_peak(tr, m1, m2)
                    dm = float(min((TAU[0] - pk[0]) / TAU[0], (TAU[1] - pk[1]) / TAU[1]))
                    D_ok = bool(S_ok and dm > 0.0)
                    c_m, c_d = _trackability_margin(OMEGA_N, ZETA, tr)
                    C_ok = bool(c_m >= 0.0)
                    rows.append({"T_s": T, "load": load, "family": fam, "seed": seed,
                                 "K_ok": K_ok, "S_ok": S_ok, "D_ok": D_ok, "C_ok": C_ok,
                                 "c_margin": round(float(c_m), 4),
                                 "f_traj_hz": c_d["f_traj_hz"], "binding": c_d["binding"],
                                 "dyn_margin": round(dm, 4)})
    return rows


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:160]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    rows_rep = rep["rows"]
    my_rows = _regen_rows()
    # 行数一致
    ch.append(_chk("S1_row_count_matches", len(my_rows) == len(rows_rep),
                   "mine=%d report=%d" % (len(my_rows), len(rows_rep))))
    # 逐行核对 C_ok 与 c_margin（独立判据复现）
    nC = 0; cmax = 0.0; nbad = 0
    for a, b in zip(my_rows, rows_rep):
        if a["C_ok"] != b["C_ok"]:
            nbad += 1
        d = abs(a["c_margin"] - b["c_margin"])
        cmax = max(cmax, d)
        nC += 1
    ch.append(_chk("S2_C_ok_rowwise_match", nbad == 0, "mismatch_rows=%d" % nbad))
    ch.append(_chk("S3_c_margin_rowwise_match", cmax < 1e-3, "max|Δc_margin|=%.4f" % cmax))
    # D_ok 逐行核对（动力学阶独立复现）
    nd = sum(1 for a, b in zip(my_rows, rows_rep) if a["D_ok"] != b["D_ok"])
    ch.append(_chk("S4_D_ok_rowwise_match", nd == 0, "mismatch_rows=%d" % nd))
    # 盲区计数 = D_ok ∧ ¬C_ok
    blind_D = sum(1 for r in my_rows if r["D_ok"] and not r["C_ok"])
    ch.append(_chk("S5_blind_spot_count_matches", blind_D == rep["results"]["blind_spot_vs_D"],
                   "mine=%d report=%d" % (blind_D, rep["results"]["blind_spot_vs_D"])))
    # 盲区全在 T≤0.5s
    blind_rows = [r for r in my_rows if r["D_ok"] and not r["C_ok"]]
    ch.append(_chk("S6_blind_spot_all_T_le_0p5", all(r["T_s"] <= 0.5 for r in blind_rows) and len(blind_rows) > 0,
                   "max_T_in_blind=%.2f" % (max(r["T_s"] for r in blind_rows) if blind_rows else -1)))
    # 慢档(T≥0.8)误报率 < 0.10
    slow = [r for r in my_rows if r["T_s"] >= 0.8]
    slow_fa = sum(1 for r in slow if not r["C_ok"]) / len(slow) if slow else None
    ch.append(_chk("S7_slow_false_alarm_lt_0p10", slow_fa is not None and slow_fa < 0.10, "rate=%.4f" % slow_fa))
    # verdict 自洽
    H = rep["preregistered_verdict"]
    mine_H = {"H58-1_blind_spot_exists": blind_D > 0,
              "H58-2_blind_spot_all_T_le_0.5s": all(r["T_s"] <= 0.5 for r in blind_rows),
              "H58-3_slow_false_alarm_lt_0.10": slow_fa is not None and slow_fa < 0.10}
    ch.append(_chk("S8_verdict_self_consistent",
                   rep["verdict_pass"] == (H["H58-1_blind_spot_exists"]
                                           and H["H58-2_blind_spot_all_T_le_0.5s"]
                                           and H["H58-3_slow_false_alarm_lt_0.10"])
                   and mine_H == {k: H[k] for k in mine_H}))
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
    print("E58 严格审核（L3 独立物理，不 import e58/trackability）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 H58-1=false → S8 必报",
          lambda r: r["preregistered_verdict"].__setitem__("H58-1_blind_spot_exists", False), ["S8"])
    _tamp("R2 篡改 blind_spot_vs_D=0 → S5 必报",
          lambda r: r["results"].__setitem__("blind_spot_vs_D", 0), ["S5"])
    _tamp("R3 篡改 verdict=false → S8 必报",
          lambda r: r.__setitem__("verdict_pass", False), ["S8"])
    _tamp("R4 翻转某行 C_ok → S2 必报",
          lambda r: r["rows"].__setitem__(0, {**r["rows"][0], "C_ok": not r["rows"][0]["C_ok"]}), ["S2"])

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
