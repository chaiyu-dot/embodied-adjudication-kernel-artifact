# -*- coding: utf-8 -*-
"""blind_reverse_e64.py —— E64 的 L5 逆向盲测（隔离 + 代价模型反推 + 物理等价 + 负控）。

独立性：隔离（自检 sys.modules 不含 e64）。与 L3 正交：
  · 从存储 p50(N) 反推**线性代价模型** p50 ≈ a + b·N，并外推未测点 N=201；
  · 独立核验 loop(RNE) 与 vector(闭式) 在**随机构型**上的物理等价（L3 只测固定轨迹）；
  · 负控：若某 N 上 vector 慢于 loop，H64-2 即被反证。
产物：blind_reverse_e64.json
"""
import json
import math
import os
import sys

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(EVAL)
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from planning import rne_dynamics as RNE                                    # noqa: E402

REPORT = os.path.join(EVAL, "e64_latency_vs_order_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e64.json")
L1, L2, M1, M2 = 0.4, 0.3, 0.6, 0.35

_res = {"experiment": "E64 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e64；代价模型反推 + 随机构型物理等价", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _tau_cf(q, qd, qdd):
    q1, q2 = q; d1, d2 = qd; a1, a2 = qdd
    lc1, lc2 = L1 / 2.0, L2 / 2.0
    i1, i2 = M1 * L1 * L1 / 12.0, M2 * L2 * L2 / 12.0
    c2, s2 = math.cos(q2), math.sin(q2); c12 = math.cos(q1 + q2)
    M11 = M1 * lc1 ** 2 + M2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + i1 + i2
    M12 = M2 * (lc2 ** 2 + L1 * lc2 * c2) + i2
    M22 = M2 * lc2 ** 2 + i2
    C1 = -M2 * L1 * lc2 * s2 * (2 * d1 * d2 + d2 ** 2); C2 = M2 * L1 * lc2 * s2 * d1 ** 2
    G1 = (M1 * lc1 + M2 * L1) * 9.81 * math.cos(q1) + M2 * lc2 * 9.81 * c12
    G2 = M2 * lc2 * 9.81 * c12
    return M11 * a1 + M12 * a2 + C1 + G1, M12 * a1 + M22 * a2 + C2 + G2


def main():
    print("=" * 88)
    print("E64 逆向盲测（L5 隔离，代价模型反推 + 物理等价）")
    print("=" * 88)
    chk("R0_isolation_e64_not_imported", "e64_latency_vs_order" not in sys.modules,
        "sys.modules 含 e64=%s" % ("e64_latency_vs_order" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    M = rep["results"]["measures"]
    Ns = [1, 101, 501]
    dl = [M["D_loop_%d" % n]["p50_ms"] for n in Ns]
    dv = [M["D_vector_%d" % n]["p50_ms"] for n in Ns]

    # R1：线性代价模型反推（用 N=101/501 两点定 b，预测 N=201）
    b = (dl[2] - dl[1]) / (501 - 101)
    a = dl[1] - b * 101
    pred201 = a + b * 201
    chk("R1_recover_linear_cost_model", b > 0 and pred201 > dl[1],
        "p50(N)~%.4f+%.5f*N ; pred(201)=%.3f ms (N=101->%.3f, 501->%.3f)" % (a, b, pred201, dl[1], dl[2]))

    # R2：随机构型物理等价（loop=RNE vs vector=闭式）
    rng = np.random.RandomState(20260920)
    worst = 0.0
    for _ in range(40):
        q = rng.uniform(-2, 2, 2); qd = rng.uniform(-3, 3, 2); qdd = rng.uniform(-8, 8, 2)
        lk = RNE.planar_2r_model(L1, L2, M1, M2, plane="xy", q1=float(q[0]), q2=float(q[1]))
        tau = np.asarray(RNE.rne_inverse_dynamics(lk, list(q), list(qd), list(qdd), g=(0.0, -9.81, 0.0)), float)
        c = np.asarray(_tau_cf(q, qd, qdd), float)
        worst = max(worst, float(np.max(np.abs(tau - c))))
    chk("R2_random_config_physics_equivalence", worst < 1e-8, "max|Δ over 40 rnd cfg|=%.2e" % worst)

    # R3：负控——vector 必须 ≤ loop（否则 H64-2 被反证）
    ok3 = all(dv[i] <= dl[i] for i in range(3))
    chk("R3_negative_control_vector_le_loop", ok3, "loop=%s vector=%s" % (dl, dv))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass; _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    _res["recovered"] = {"cost_slope_ms_per_pt": round(b, 5), "predicted_p50_at_201": round(pred201, 3)}
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
