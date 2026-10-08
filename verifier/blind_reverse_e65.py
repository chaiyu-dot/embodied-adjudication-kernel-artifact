# -*- coding: utf-8 -*-
"""blind_reverse_e65.py —— E65 的**逆向盲测（L5，隔离 + 不信声明）**。

独立性：隔离（运行期自检 sys.modules 未加载 e65_inapplicable_guard）。按声明规则独立重写
对象生成与四臂（含**对照臂**：把称重原语换成推挤原语——推挤测惯性质量、不受支撑条件破坏）。
核心证伪：**若原语假设成立（push 读真值），三值守卫不被骗（漏报→0）** ⇒ 证明漏报源于
"原语假设被打破"（bench 构造有效），而非三值逻辑本身有 bug；且 ⊘ 的选择性（失效子集≫合法子集）
是"检测到原语不一致"的signature。

产物：blind_reverse_e65.json
"""
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e65_inapplicable_guard_report.json")
OUT = os.path.join(EVAL, "blind_reverse_e65.json")

N_OBJECTS = 600
INVALID_FRAC = 0.30
M_LIM = 1.5
SEEDTAG = ""
SIGMA_W, SIGMA_P = 0.02, 0.05
EPS_W0, EPS_P0 = 0.30, 0.45
SHRINK = 0.5
K_MAX = 6
FLOOR = 0.01

_res = {"experiment": "E65 逆向盲测(L5 隔离)", "independence_scope":
        "隔离：运行期 sys.modules 不含 e65_inapplicable_guard；按规则独立重写对象/四臂",
        "checks": [], "falsify": []}


def chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _seed(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _objs(invalid_frac=INVALID_FRAC, m_lim=M_LIM):
    o = []
    for i in range(N_OBJECTS):
        rng = np.random.RandomState(_seed(SEEDTAG, "obj", i))
        invalid = bool(rng.rand() < invalid_frac)
        if invalid:
            m_eff = float(rng.uniform(m_lim + 0.5, m_lim + 2.0))
            w_center = float(rng.uniform(m_lim - 0.30, m_lim - 0.10))
        else:
            m_eff = float(rng.uniform(0.2, 2.0 * m_lim))
            w_center = m_eff
        o.append({"id": i, "invalid": invalid, "m_eff": m_eff, "w_center": w_center,
                  "over": bool(m_eff > m_lim)})
    return o


def _probe(obj, kind, rng, m_lim=M_LIM):
    if kind == "weigh":
        return float(obj["w_center"] + rng.normal(0, SIGMA_W)), EPS_W0
    return float(obj["m_eff"] + rng.normal(0, SIGMA_P)), EPS_P0


def _cls(c, hw, m_lim=M_LIM):
    lo, hi = c - hw, c + hw
    return "safe" if hi < m_lim else ("over" if lo > m_lim else "unknown")


def _retry(obj, rng, kind="weigh", m_lim=M_LIM):
    c, hw = _probe(obj, kind, rng, m_lim)
    for k in range(1, K_MAX + 1):
        v = _cls(c, hw, m_lim)
        if v != "unknown":
            return v, "decided", k
        hw *= SHRINK
        c, _h = _probe(obj, kind, rng, m_lim)
    v = _cls(c, hw, m_lim)
    return (v if v != "unknown" else "over"), "decided", K_MAX


def _ours(obj, rng, kind=None, m_lim=M_LIM):
    cw, hw = _probe(obj, "weigh", rng, m_lim)
    for k in range(1, K_MAX + 1):
        v = _cls(cw, hw, m_lim)
        if v != "unknown":
            return v, "decided", k
        cp, hp = _probe(obj, "push", rng, m_lim)
        if abs(cw - cp) > (hw + hp):
            return "inapplicable", "inapplicable", k
        hw *= SHRINK
        cw, _h = _probe(obj, "weigh", rng, m_lim)
    v = _cls(cw, hw, m_lim)
    if v == "unknown":
        cp, hp = _probe(obj, "push", rng, m_lim)
        return ("inapplicable", "inapplicable", K_MAX) if abs(cw - cp) > (hw + hp) else ("over", "decided", K_MAX)
    return v, "decided", K_MAX


def _miss_rate(fn, arm_name, kind=None, invalid_frac=INVALID_FRAC, m_lim=M_LIM):
    """arm_name 用实验同款 rng key（"run", arm_name, id）以复现；算法本身独立重写。"""
    objs = _objs(invalid_frac, m_lim)
    miss = 0
    over = 0
    for o in objs:
        rng = np.random.RandomState(_seed(SEEDTAG, "run", arm_name, o["id"]))
        v = fn(o, rng, kind, m_lim)[0]
        if o["over"]:
            over += 1
            if v == "safe":
                miss += 1
    return miss / max(over, 1), over


def main():
    print("=" * 88)
    print("E65 逆向盲测（L5 隔离，独立重写 + 负对照证伪）")
    print("=" * 88)
    chk("R0_isolation_e65_not_imported",
        "e65_inapplicable_guard" not in sys.modules,
        "sys.modules 含 e65=%s" % ("e65_inapplicable_guard" in sys.modules))

    rep = json.load(open(REPORT, encoding="utf-8"))
    rows = rep["rows"]

    # --- R1：从数据反推 invalid_fraction + ⊘ 选择性 ---
    inv_frac = sum(1 for r in rows if r["arm"] == "G4_OURS" and r["invalid"]) / \
        sum(1 for r in rows if r["arm"] == "G4_OURS")
    on_inv = rep["results"]["per_arm"]["G4_OURS"]["inapplicable_on_invalid"]
    on_val = rep["results"]["per_arm"]["G4_OURS"]["inapplicable_on_valid"]
    n_inv_store = rep["results"]["self_consistency"]["n_invalid"]
    chk("R1_reverse_invalid_fraction_and_selectivity",
        abs(inv_frac - n_inv_store / 600.0) < 1e-9 and 0.25 < inv_frac < 0.35
        and on_inv >= 0.90 and on_val <= 0.10,
        "invalid_frac=%.4f (store %.4f) ; ⊘|invalid=%.3f ⊘|valid=%.3f" % (inv_frac, n_inv_store / 600.0, on_inv, on_val))

    # --- R2：负对照——把原语换成推挤（读真值），三值守卫不被骗 ---
    mr_push, n_over = _miss_rate(_retry, "G3_RETRY", kind="push")
    mr_weigh, _ = _miss_rate(_retry, "G3_RETRY", kind="weigh")
    chk("R2_negative_control_push_primitive_not_fooled",
        mr_push < 0.05 and mr_weigh >= 0.40,
        "miss(push)=%.4f miss(weigh)=%.4f (over=%d)" % (mr_push, mr_weigh, n_over))

    # --- R3：⊘ 漏报显著低于三值（独立重算与存储一致）---
    mr_ours, _ = _miss_rate(_ours, "G4_OURS")
    store_ours = rep["results"]["per_arm"]["G4_OURS"]["miss_rate"]
    store_retry = rep["results"]["per_arm"]["G3_RETRY"]["miss_rate"]
    chk("R3_ours_miss_below_three_value_and_matches_store",
        mr_ours < mr_weigh * 0.25 and abs(mr_ours - store_ours) < 1e-4 and abs(mr_weigh - store_retry) < 1e-4,
        "ours=%.4f(retry=%.4f) store ours=%.4f retry=%.4f" % (mr_ours, mr_weigh, store_ours, store_retry))

    # --- R4：±扰动失效比例，序关系稳健（⊘ 仍优于三值）---
    ok4 = True
    det = []
    for frac in (0.20, 0.40):
        mw, _ = _miss_rate(_retry, "G3_RETRY", kind="weigh", invalid_frac=frac)
        mo, _ = _miss_rate(_ours, "G4_OURS", invalid_frac=frac)
        ok4 &= (mw > mo and mo <= 0.06)
        det.append("frac=%.2f weigh=%.3f ours=%.4f" % (frac, mw, mo))
    chk("R4_robustness_to_invalid_fraction", ok4, " ｜ ".join(det))

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
