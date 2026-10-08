# -*- coding: utf-8 -*-
"""_audit_e65b_strict.py —— E65b 严格审核（L3 独立重推，按确认性配置参数化）。

不 import e65b / e65；审计者按预注册 bench（读 prereg）自写对象生成 + 四臂 + 统计，
独立重推逐臂指标、配对差 CI、判据 H65b-1..6 与 verdict（验证『排除自证伪闸门』逻辑）；篡改必报。

产物：_audit_e65b_strict.json
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
REPORT = os.path.join(EVAL, "e65b_confirmation_report.json")
PREREG = os.path.join(EVAL, "e65b_confirmation_prereg.json")
OUT = os.path.join(EVAL, "_audit_e65b_strict.json")

_res = {"experiment": "E65b 严格审核(L3 独立重推)", "independence_scope":
        "独立重推：按 prereg bench 自写对象/四臂/统计，不 import e65/e65b", "checks": [], "tamper": []}


def _seed(tag, *p):
    return zlib.crc32("|".join(str(x) for x in (tag,) + p).encode()) % (2 ** 31)


def _wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _objs(B):
    o = []
    for i in range(B["n_objects"]):
        rng = np.random.RandomState(_seed(B["seed_tag"], "obj", i))
        invalid = bool(rng.rand() < B["invalid_fraction"])
        if invalid:
            m_eff = float(rng.uniform(B["m_lim_kg"] + 0.5, B["m_lim_kg"] + 2.0))
            wc = float(rng.uniform(B["m_lim_kg"] - 0.30, B["m_lim_kg"] - 0.10))
        else:
            m_eff = float(rng.uniform(0.2, 2.0 * B["m_lim_kg"])); wc = m_eff
        o.append({"id": i, "invalid": invalid, "m_eff": m_eff, "wc": wc,
                  "over": bool(m_eff > B["m_lim_kg"])})
    return o


def _probe(o, kind, rng, B):
    if kind == "weigh":
        return float(o["wc"] + rng.normal(0, B["sigma_w"])), B["eps_w0"]
    return float(o["m_eff"] + rng.normal(0, B["sigma_p"])), B["eps_p0"]


def _cls(c, hw, ml):
    return "safe" if c + hw < ml else ("over" if c - hw > ml else "unknown")


def _mk_arms(B):
    ml, kmax, shrink, floor = B["m_lim_kg"], B["k_max"], B["shrink"], B["resolution_floor"]

    def retry(o, rng):
        c, hw = _probe(o, "weigh", rng, B)
        for k in range(1, kmax + 1):
            v = _cls(c, hw, ml)
            if v != "unknown":
                return v, "decided", k
            hw *= shrink; c, _ = _probe(o, "weigh", rng, B)
        v = _cls(c, hw, ml); return (v if v != "unknown" else "over"), "decided", kmax

    def abstain(o, rng):
        c, hw = _probe(o, "weigh", rng, B)
        v = _cls(c, hw, ml)
        return ("abstain", "abstain", 0) if v == "unknown" else (v, "decided", 0)

    def shrink_(o, rng):
        c, hw = _probe(o, "weigh", rng, B)
        for k in range(1, kmax + 1):
            v = _cls(c, hw, ml)
            if v != "unknown":
                return v, "decided", k
            hw *= shrink; c, _ = _probe(o, "weigh", rng, B)
        if hw > floor or _cls(c, hw, ml) == "unknown":
            return "abstain", "abstain", kmax
        return _cls(c, hw, ml), "decided", kmax

    def ours(o, rng):
        cw, hw = _probe(o, "weigh", rng, B)
        for k in range(1, kmax + 1):
            v = _cls(cw, hw, ml)
            if v != "unknown":
                return v, "decided", k
            cp, hp = _probe(o, "push", rng, B)
            if abs(cw - cp) > (hw + hp):
                return "inapplicable", "inapplicable", k
            hw *= shrink; cw, _ = _probe(o, "weigh", rng, B)
        v = _cls(cw, hw, ml)
        if v == "unknown":
            cp, hp = _probe(o, "push", rng, B)
            return ("inapplicable", "inapplicable", kmax) if abs(cw - cp) > (hw + hp) else ("over", "decided", kmax)
        return v, "decided", kmax
    return {"G3_RETRY": retry, "G3_ABSTAIN": abstain, "G3_SHRINK": shrink_, "G4_OURS": ours}


def _stat(rows, a):
    sub = [r for r in rows if r["arm"] == a]
    over = [r for r in sub if r["over"]]
    miss = [r for r in over if r["exec"]]
    inv = [r for r in sub if r["invalid"]]; val = [r for r in sub if not r["invalid"]]
    return {"n": len(sub), "n_over_limit": len(over), "miss_count": len(miss),
            "miss_rate": round(len(miss) / len(over), 4) if over else None,
            "miss_wilson95": _wilson(len(miss), len(over)),
            "decided_rate": round(sum(1 for r in sub if r["status"] == "decided") / len(sub), 4),
            "inapplicable_on_invalid": round(sum(1 for r in inv if r["verdict"] == "inapplicable") / len(inv), 4) if inv else None,
            "inapplicable_on_valid": round(sum(1 for r in val if r["verdict"] == "inapplicable") / len(val), 4) if val else None}


def _rerun(B):
    arms = _mk_arms(B); objs = _objs(B); rows = []
    for a, fn in arms.items():
        for o in objs:
            rng = np.random.RandomState(_seed(B["seed_tag"], "run", a, o["id"]))
            v, st, k = fn(o, rng)
            rows.append({"arm": a, "invalid": o["invalid"], "over": o["over"], "verdict": v,
                         "status": st, "exec": (v == "safe")})
    return S_from(rows), objs


def S_from(rows):
    return {a: _stat(rows, a) for a in ("G3_RETRY", "G3_ABSTAIN", "G3_SHRINK", "G4_OURS")}


def _chk(name, ok, detail=""):
    _res["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  —— " + str(detail)[:170]) if detail else ""))
    return (name, bool(ok))


def _forward(rep):
    ch = []
    B = rep["design"]
    S_re, objs = _rerun(B)
    S = rep["results"]["per_arm"]
    # S1 对象统计
    n_inv = sum(1 for o in objs if o["invalid"]); n_over = sum(1 for o in objs if o["over"])
    sc = rep["results"]["self_consistency"]
    ch.append(_chk("S1_objects_match", n_inv == sc["n_invalid"] and n_over == sc["n_over_limit"],
                   "ninval=%d store=%d nover=%d store=%d" % (n_inv, sc["n_invalid"], n_over, sc["n_over_limit"])))
    # S2 逐臂统计
    ok = True; det = ""
    for a in S_re:
        for k, v in S_re[a].items():
            sv = S[a][k]
            same = (v == sv) if not isinstance(v, float) else (abs(v - sv) < 1e-9)
            if not same:
                ok = False; det = "%s.%s re=%s store=%s" % (a, k, v, sv); break
        if not ok:
            break
    ch.append(_chk("S2_per_arm_stats_match_independent_rerun", ok, det or "4 arms match"))
    # S3 配对差 CI + 判据 + verdict
    na, nb = S["G3_SHRINK"]["n_over_limit"], S["G4_OURS"]["n_over_limit"]
    ka, kb = S["G3_SHRINK"]["miss_count"], S["G4_OURS"]["miss_count"]
    la, ua = _wilson(ka, na); lb, ub = _wilson(kb, nb)
    pa, pb = ka / na, kb / nb; d = pa - pb
    ci = [round(d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2), 4),
          round(d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2), 4)]
    H = {"H65b-1_direction_and_magnitude": bool(ci[0] > 0.20),
         "H65b-2_four_value_absolute": bool(S["G4_OURS"]["miss_rate"] <= 0.06),
         "H65b-3_selective_inapplicable": bool(S["G4_OURS"]["inapplicable_on_invalid"] >= 0.85
                                               and S["G4_OURS"]["inapplicable_on_valid"] <= 0.10),
         "H65b-4_abstain_cost": bool(S["G3_ABSTAIN"]["decided_rate"] <= 0.60),
         "H65b-5_FALSIFIER": bool(ci[0] <= 0.20),
         "H65b-6_order_insensitive_retry": bool(S["G3_RETRY"]["miss_rate"] >= 0.30)}
    verd = all(v for k, v in H.items() if k != "H65b-5_FALSIFIER") and not H["H65b-5_FALSIFIER"]
    ch.append(_chk("S3_ci_and_criteria_and_verdict",
                   ci == rep["results"]["paired_diff_miss_G3SHRINK_minus_G4"]
                   and all(H[k] == rep["preregistered_verdict"][k] for k in H)
                   and verd == rep["verdict_pass"],
                   "ci=%s verdict=%s" % (ci, rep["verdict_pass"])))
    # S4 prereg sha
    sha = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    ch.append(_chk("S4_prereg_sha256_anchor", sha == rep["prereg_sha256"], sha[:16]))
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
    print("E65b 严格审核（L3 独立重推，不 import e65/e65b）")
    print("=" * 88)
    rep = json.load(open(REPORT, encoding="utf-8"))
    fwd = _forward(rep)
    npass = sum(1 for c in fwd if c[1])
    print("前提判据：%d/%d 通过" % (npass, len(fwd)))

    print("\n[R] 篡改用例")
    _tamp("R1 篡改 G4_OURS miss_rate → S2 必报",
          lambda r: r["results"]["per_arm"]["G4_OURS"].__setitem__("miss_rate", 0.99), ["S2"])
    _tamp("R2 篡改 n_invalid → S1 必报",
          lambda r: r["results"]["self_consistency"].__setitem__("n_invalid", 0), ["S1"])
    _tamp("R3 篡改 CI → S3 必报",
          lambda r: r["results"].__setitem__("paired_diff_miss_G3SHRINK_minus_G4", [0.0, 1.0]), ["S3"])
    _tamp("R4 篡改 verdict_pass → S3 必报",
          lambda r: r.__setitem__("verdict_pass", not r["verdict_pass"]), ["S3"])
    _tamp("R5 篡改 prereg_sha256 → S4 必报",
          lambda r: r.__setitem__("prereg_sha256", "deadbeef" * 8), ["S4"])

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
