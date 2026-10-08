# -*- coding: utf-8 -*-
"""_audit_e65_strict.py —— E65 严格审核（L3，独立重推 + 审前提）。

不 import e65_inapplicable_guard；审计者**按 docstring 声明的规则**独立重写：
  · 对象生成（invalid 30%；失效对象 m_eff>M_LIM、秤读 w_center<M_LIM；合法对象 w_center=m_eff）
  · 四臂（G3_RETRY/G3_ABSTAIN/G3_SHRINK/G4_OURS）与 σ/ε/shrink/K_max/floor
  · 逐臂统计（漏报率/Wilson95/⊘选择性）与判据 H65-1..6、verdict
并与存储报告逐项核对；篡改必报。

产物：_audit_e65_strict.json
"""
import copy
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np                                                           # noqa: E402

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e65_inapplicable_guard_report.json")
PREREG = os.path.join(EVAL, "e65_inapplicable_guard_prereg.json")
OUT = os.path.join(EVAL, "_audit_e65_strict.json")

# —— 从 docstring/预注册声明的规格独立重写（非 import 实验模块）——
N_OBJECTS = 600
INVALID_FRAC = 0.30
M_LIM = 1.5
SEEDTAG = ""
SIGMA_W, SIGMA_P = 0.02, 0.05
EPS_W0, EPS_P0 = 0.30, 0.45
SHRINK = 0.5
K_MAX = 6
FLOOR = 0.01

_res = {"experiment": "E65 严格审核(L3 独立重推)", "independence_scope":
        "独立重推：按声明规则自写对象生成 + 四臂 + 统计，不 import e65", "checks": [], "tamper": []}


def _seed(*p):
    return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)


def _wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _make_objects():
    objs = []
    for i in range(N_OBJECTS):
        rng = np.random.RandomState(_seed(SEEDTAG, "obj", i))
        invalid = bool(rng.rand() < INVALID_FRAC)
        if invalid:
            m_eff = float(rng.uniform(M_LIM + 0.5, M_LIM + 2.0))
            w_center = float(rng.uniform(M_LIM - 0.30, M_LIM - 0.10))
        else:
            m_eff = float(rng.uniform(0.2, 2.0 * M_LIM))
            w_center = m_eff
        objs.append({"id": i, "invalid": invalid, "m_eff": round(m_eff, 4),
                     "w_center": round(w_center, 4), "over_limit_gt": bool(m_eff > M_LIM)})
    return objs


def _probe(obj, kind, rng):
    if kind == "weigh":
        return float(obj["w_center"] + rng.normal(0, SIGMA_W)), EPS_W0
    return float(obj["m_eff"] + rng.normal(0, SIGMA_P)), EPS_P0


def _classify(c, hw):
    lo, hi = c - hw, c + hw
    if hi < M_LIM:
        return "safe"
    if lo > M_LIM:
        return "over"
    return "unknown"


def _arm_retry(o, rng):
    c, hw = _probe(o, "weigh", rng)
    for k in range(1, K_MAX + 1):
        v = _classify(c, hw)
        if v != "unknown":
            return v, "decided", k
        hw *= SHRINK
        c, _h = _probe(o, "weigh", rng)
    v = _classify(c, hw)
    return (v if v != "unknown" else "over"), "decided", K_MAX


def _arm_abstain(o, rng):
    c, hw = _probe(o, "weigh", rng)
    v = _classify(c, hw)
    if v == "unknown":
        return "abstain", "abstain", 0
    return v, "decided", 0


def _arm_shrink(o, rng):
    c, hw = _probe(o, "weigh", rng)
    for k in range(1, K_MAX + 1):
        v = _classify(c, hw)
        if v != "unknown":
            return v, "decided", k
        hw *= SHRINK
        c, _h = _probe(o, "weigh", rng)
    if hw > FLOOR or _classify(c, hw) == "unknown":
        return "abstain", "abstain", K_MAX
    return _classify(c, hw), "decided", K_MAX


def _arm_ours(o, rng):
    cw, hw = _probe(o, "weigh", rng)
    for k in range(1, K_MAX + 1):
        v = _classify(cw, hw)
        if v != "unknown":
            return v, "decided", k
        cp, hp = _probe(o, "push", rng)
        if abs(cw - cp) > (hw + hp):
            return "inapplicable", "inapplicable", k
        hw *= SHRINK
        cw, _h = _probe(o, "weigh", rng)
    v = _classify(cw, hw)
    if v == "unknown":
        cp, hp = _probe(o, "push", rng)
        return ("inapplicable", "inapplicable", K_MAX) if abs(cw - cp) > (hw + hp) else ("over", "decided", K_MAX)
    return v, "decided", K_MAX


_ARMS = {"G3_RETRY": _arm_retry, "G3_ABSTAIN": _arm_abstain, "G3_SHRINK": _arm_shrink, "G4_OURS": _arm_ours}


def _rerun():
    objs = _make_objects()
    rows = []
    for a, fn in _ARMS.items():
        for o in objs:
            rng = np.random.RandomState(_seed(SEEDTAG, "run", a, o["id"]))
            v, status, k = fn(o, rng)
            rows.append({"arm": a, "id": o["id"], "invalid": o["invalid"],
                         "over_limit_gt": o["over_limit_gt"], "verdict": v, "status": status,
                         "rounds": k, "exec": (v == "safe")})
    return objs, rows


def _stat(rows, a):
    sub = [r for r in rows if r["arm"] == a]
    over = [r for r in sub if r["over_limit_gt"]]
    miss = [r for r in over if r["exec"]]
    inv = [r for r in sub if r["invalid"]]
    val = [r for r in sub if not r["invalid"]]
    return {"n": len(sub), "n_over_limit": len(over), "miss_count": len(miss),
            "miss_rate": round(len(miss) / len(over), 4) if over else None,
            "miss_wilson95": _wilson(len(miss), len(over)),
            "decided_rate": round(sum(1 for r in sub if r["status"] == "decided") / len(sub), 4),
            "abstain_rate": round(sum(1 for r in sub if r["status"] == "abstain") / len(sub), 4),
            "inapplicable_rate": round(sum(1 for r in sub if r["verdict"] == "inapplicable") / len(sub), 4),
            "inapplicable_on_invalid": round(sum(1 for r in inv if r["verdict"] == "inapplicable") / len(inv), 4) if inv else None,
            "inapplicable_on_valid": round(sum(1 for r in val if r["verdict"] == "inapplicable") / len(val), 4) if val else None}


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    objs, rows = _rerun()
    S = rep["results"]["per_arm"]

    # S1 对象统计一致
    n_inv = sum(1 for o in objs if o["invalid"])
    n_over = sum(1 for o in objs if o["over_limit_gt"])
    sc = rep["results"]["self_consistency"]
    ch.append(_chk("S1_objects_match", n_inv == sc["n_invalid"] and n_over == sc["n_over_limit"] and len(rows) == len(rep["rows"]),
                   "n_invalid=%d(store %d) n_over=%d(store %d)" % (n_inv, sc["n_invalid"], n_over, sc["n_over_limit"])))

    # S2 逐臂统计一致
    ok = True
    detail = ""
    for a in _ARMS:
        got = _stat(rows, a)
        for k in ("n", "n_over_limit", "miss_count", "miss_rate", "decided_rate",
                  "abstain_rate", "inapplicable_rate", "inapplicable_on_invalid", "inapplicable_on_valid"):
            gv, sv = got[k], S[a][k]
            same = (gv == sv) if not isinstance(gv, float) else (abs(gv - sv) < 1e-9)
            if not same:
                ok = False
                detail = "%s.%s re=%s store=%s" % (a, k, gv, sv)
                break
        if not ok:
            break
    ch.append(_chk("S2_per_arm_stats_match_independent_rerun", ok, detail or "4 arms all match"))

    # S3 Wilson 区间一致
    ok = all(_wilson(S[a]["miss_count"], S[a]["n_over_limit"]) == S[a]["miss_wilson95"] for a in _ARMS)
    ch.append(_chk("S3_wilson95_match", ok))

    # S4 判据 H65-1..6 自洽 + verdict
    H = {
        "H65-1_three_value_retry_misses_unsafe": bool(S["G3_RETRY"]["miss_rate"] >= 0.40),
        "H65-2_four_value_misses_le_0.02": bool(S["G4_OURS"]["miss_rate"] <= 0.02),
        "H65-3_shrink_check_is_fooled": bool(S["G3_SHRINK"]["miss_rate"] >= 0.40),
        "H65-4_selective_inapplicable": bool(S["G4_OURS"]["inapplicable_on_valid"] <= 0.10
                                             and S["G4_OURS"]["inapplicable_on_invalid"] >= 0.90),
        "H65-5_abstain_cost": bool(S["G3_ABSTAIN"]["decided_rate"] <= 0.60),
        "H65-6_FALSIFIER_inapplicable_necessary": bool(not (S["G3_RETRY"]["miss_rate"] >= 0.40
                                                            or S["G3_SHRINK"]["miss_rate"] >= 0.40)),
    }
    ch.append(_chk("S4_preregistered_verdict_selfconsistent",
                   all(H[k] == rep["preregistered_verdict"][k] for k in H)
                   and all(H.values()) == rep["verdict_pass"],
                   "verdict=%s" % rep["verdict_pass"]))

    # S5 预注册锚
    sha = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    ch.append(_chk("S5_prereg_sha256_anchor", sha == rep["prereg_sha256"], sha[:16]))
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
    print("E65 严格审核（L3 独立重推，不 import e65）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 G3_RETRY miss_count → S2 必报",
          lambda r: r["results"]["per_arm"]["G3_RETRY"].__setitem__("miss_count", 0), ["S2"])
    _tamp("R2 篡改 G4_OURS miss_rate=0.99 → S2 必报",
          lambda r: r["results"]["per_arm"]["G4_OURS"].__setitem__("miss_rate", 0.99), ["S2"])
    _tamp("R3 篡改 self_consistency.n_invalid → S1 必报",
          lambda r: r["results"]["self_consistency"].__setitem__("n_invalid", 0), ["S1"])
    _tamp("R4 篡改 verdict_pass=True（原为 False）→ S4 必报",
          lambda r: r.__setitem__("verdict_pass", True), ["S4"])
    _tamp("R5 篡改 prereg_sha256 → S5 必报",
          lambda r: r.__setitem__("prereg_sha256", "deadbeef" * 8), ["S5"])

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
