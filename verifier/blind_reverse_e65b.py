# -*- coding: utf-8 -*-
"""blind_reverse_e65b.py —— E65b 的 L5 逆向盲测（隔离 + 负对照 + CI 反推）。

独立性：隔离（自检 sys.modules 不含 e65 / e65b）。与 L3 正交：
  · R1 从 rows 反推失效比例与 ⊘ 选择性；
  · R2 负对照：把称重原语换成推挤原语（读真值）→ 三值守卫不被骗（漏报→~0），
       证明 '被骗' 源于原语假设失效而非逻辑；
  · R3 由 per_arm 独立重算配对差 CI 下界（H65b-1/5 闸门）。
产物：blind_reverse_e65b.json
"""
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e65b_confirmation_report.json")
PREREG = os.path.join(EVAL, "e65b_confirmation_prereg.json")
OUT = os.path.join(EVAL, "blind_reverse_e65b.json")

_res = {"experiment": "E65b 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e65/e65b；负对照 push 原语 + CI 反推", "checks": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _seed(tag, *p):
    return zlib.crc32("|".join(str(x) for x in (tag,) + p).encode()) % (2 ** 31)


def _wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _objs(B):
    o = []
    for i in range(B["n_objects"]):
        rng = np.random.RandomState(_seed(B["seed_tag"], "obj", i))
        invalid = bool(rng.rand() < B["invalid_fraction"])
        if invalid:
            me = float(rng.uniform(B["m_lim_kg"] + 0.5, B["m_lim_kg"] + 2.0))
            wc = float(rng.uniform(B["m_lim_kg"] - 0.30, B["m_lim_kg"] - 0.10))
        else:
            me = float(rng.uniform(0.2, 2.0 * B["m_lim_kg"])); wc = me
        o.append({"id": i, "invalid": invalid, "me": me, "wc": wc, "over": bool(me > B["m_lim_kg"])})
    return o


def _retry(o, rng, B, kind="weigh"):
    ml, kmax, shrink = B["m_lim_kg"], B["k_max"], B["shrink"]
    for _ in range(1):
        c = (o["wc"] if kind == "weigh" else o["me"]) + rng.normal(0, B["sigma_w"] if kind == "weigh" else B["sigma_p"])
        hw = B["eps_w0"] if kind == "weigh" else B["eps_p0"]
    for k in range(1, kmax + 1):
        v = "safe" if c + hw < ml else ("over" if c - hw > ml else "unknown")
        if v != "unknown":
            return v
        hw *= shrink
        c = (o["wc"] if kind == "weigh" else o["me"]) + rng.normal(0, B["sigma_w"] if kind == "weigh" else B["sigma_p"])
    v = "safe" if c + hw < ml else ("over" if c - hw > ml else "over")
    return v


def _miss_rate(B, kind):
    objs = _objs(B); miss = over = 0
    for o in objs:
        rng = np.random.RandomState(_seed(B["seed_tag"], "run", "G3_RETRY", o["id"]))
        if o["over"]:
            over += 1
            if _retry(o, rng, B, kind) == "safe":
                miss += 1
    return miss, over


def main():
    print("=" * 88)
    print("E65b 逆向盲测（L5 隔离，负对照 + CI 反推）")
    print("=" * 88)
    chk("R0_isolation_e65_e65b_not_imported",
        "e65_inapplicable_guard" not in sys.modules and "e65b_confirmation" not in sys.modules,
        "sys.modules 含 e65/e65b=%s" % ("e65_inapplicable_guard" in sys.modules or "e65b_confirmation" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    B = rep["design"]
    rows = rep["rows"]

    # R1：失效比例 + ⊘ 选择性
    inv = sum(1 for r in rows if r["arm"] == "G4_OURS" and r["invalid"])
    tot = sum(1 for r in rows if r["arm"] == "G4_OURS")
    frac = inv / tot
    on_inv = rep["results"]["per_arm"]["G4_OURS"]["inapplicable_on_invalid"]
    on_val = rep["results"]["per_arm"]["G4_OURS"]["inapplicable_on_valid"]
    chk("R1_recover_invalid_fraction_and_selectivity",
        abs(frac - rep["design"]["invalid_fraction"]) < 0.04 and on_inv >= 0.85 and on_val <= 0.10,
        "frac=%.4f (cfg %.2f) ; on_inv=%.3f on_val=%.3f" % (frac, rep["design"]["invalid_fraction"], on_inv, on_val))

    # R2：负对照——push 原语（读真值）→ 三列不被骗
    mr_w, over = _miss_rate(B, "weigh")
    mr_p, _ = _miss_rate(B, "push")
    chk("R2_negative_control_push_not_fooled",
        (mr_p / max(over, 1)) < 0.05 and (mr_w / max(over, 1)) >= 0.30,
        "miss(weigh)=%d/%d=%.3f miss(push)=%d=%.3f" % (mr_w, over, mr_w / max(over, 1), mr_p, mr_p / max(over, 1)))

    # R3：CI 下界反推（方向与量级）
    K, N = rep["results"]["per_arm"]["G3_SHRINK"]["miss_count"], rep["results"]["per_arm"]["G3_SHRINK"]["n_over_limit"]
    k2, n2 = rep["results"]["per_arm"]["G4_OURS"]["miss_count"], rep["results"]["per_arm"]["G4_OURS"]["n_over_limit"]
    l1, u1 = _wilson(K, N); l2, u2 = _wilson(k2, n2)
    p1, p2 = K / N, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    chk("R3_recover_ci_lower_bound", lo > 0.20 and abs(lo - rep["results"]["paired_diff_miss_G3SHRINK_minus_G4"][0]) < 1e-3,
        "CI_lower_re=%.4f store=%.4f" % (lo, rep["results"]["paired_diff_miss_G3SHRINK_minus_G4"][0]))

    npass = sum(1 for c in _res["checks"] if c["pass"])
    _res["n_pass"] = npass; _res["n_total"] = len(_res["checks"])
    _res["reverse_pass"] = bool(npass == len(_res["checks"]))
    json.dump(_res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n逆向盲测：%d/%d pass = %s" % (npass, len(_res["checks"]), _res["reverse_pass"]))
    print("wrote", OUT)
    return 0 if _res["reverse_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
