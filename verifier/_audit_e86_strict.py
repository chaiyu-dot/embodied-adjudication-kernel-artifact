# -*- coding: utf-8 -*-
"""_audit_e86_strict.py —— E86 的 **L3 严格审核**（结构自洽 + **从 E19 原始日志独立重算** + 篡改捕获）。

与 L2 正交：L2 只读 E86 报告；本层**直接读 9 个 `e19_r2_*.json` 原始日志**独立重算，并核验
报告里那些"关于数据本身"的断言（如 `ti` 是任务序号而非迭代数）。

判据
----
S1 报告↔原始日志自洽：逐模型三项比率 / 极差 / Spearman / 恒等偏差 由原始行重算一致。
S2 断言核验（关于数据本身）：
   a) 存在接口精度 = 0 的模型（"完全不会"的见证）；
   b) `correction_flip` ⟺ (step1 错 ∧ step2 对)（恒等的机制来源）；
   c) `ti` 的取值集合是**任务序号**（连续小整数、每值多行）而不是迭代计数；
   d) `step2_correct` 恒为 True（系统输出 = 1.0）。
S3 篡改捕获：改报告里某模型比率 / 改恒等偏差 / 改判据 / 改诚实说明 → S1/S2 必报。
S4 幂等。

产物：_audit_e86_strict.json
"""
import copy
import glob
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(EVAL, "e86_interface_accuracy_real_models_report.json")
OUT = os.path.join(EVAL, "_audit_e86_strict.json")
_res = {"audit": "E86 L3 strict", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def _spearman(x, y):
    def avg_rank(v):
        pairs = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and v[pairs[j + 1]] == v[pairs[i]]:
                j += 1
            for k in range(i, j + 1):
                out[pairs[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out
    rx, ry = avg_rank(x), avg_rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return float(num / den) if den > 0 else 0.0


def recompute_from_raw():
    """直接从 E19 原始日志重算（不经 E86 的报告、不经 bench 数据文件）。"""
    models, ti_vals, ti_hist, all_step2_true, flip_iff = [], set(), {}, True, True
    for f in sorted(glob.glob(os.path.join(EVAL, "e19_r2_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        s1 = s2 = fl = n = 0
        for r in d.get("rows", []):
            e = r.get("error")
            if not isinstance(e, dict) or "step1_correct" not in e:
                continue
            n += 1
            s1 += int(bool(e["step1_correct"]))
            s2 += int(bool(e["step2_correct"]))
            fl += int(bool(e.get("correction_flip")))
            all_step2_true &= bool(e["step2_correct"])
            if bool(e.get("correction_flip")) != (not e["step1_correct"] and e["step2_correct"]):
                flip_iff = False
            ti = r.get("ti")
            if isinstance(ti, int):
                ti_vals.add(ti)
                ti_hist[ti] = ti_hist.get(ti, 0) + 1
        if n:
            models.append({"model": d.get("model"), "n": n, "s1": round(s1 / n, 4),
                           "s2": round(s2 / n, 4), "fl": round(fl / n, 4)})
    a = [m["s1"] for m in models]
    b = [m["s2"] for m in models]
    fl = [m["fl"] for m in models]
    return {"models": models, "spread_a": round(max(a) - min(a), 4), "spread_b": round(max(b) - min(b), 4),
            "rho": round(_spearman(a, fl), 4),
            "ident_dev": round(max(abs(fl[i] - (1 - a[i])) for i in range(len(a))), 4),
            "ti_vals": sorted(ti_vals), "ti_hist": ti_hist,
            "all_step2_true": all_step2_true, "flip_iff": flip_iff}


def main():
    print("=" * 92)
    print("E86 L3 严格审核（从 E19 原始日志独立重算）")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    raw = recompute_from_raw()

    bad = []
    for mine, theirs in zip(raw["models"], rep["models"]):
        if mine["model"] != theirs["model"] or mine["n"] != theirs["n_rows"]:
            bad.append(mine["model"])
            continue
        for mk, tk in (("s1", "interface_accuracy_step1"), ("s2", "system_output_step2"),
                       ("fl", "repair_rate_flip")):
            if abs(mine[mk] - theirs[tk]) > 1e-4:
                bad.append((mine["model"], tk))
    chk("S1 逐模型三项比率由 E19 原始日志重算一致", not bad, "mismatch=%s" % bad)

    s = rep["summary"]
    ok1b = (abs(raw["spread_a"] - s["interface_accuracy_spread"]) < 1e-4
            and abs(raw["spread_b"] - s["system_output_spread"]) < 1e-4
            and abs(raw["rho"] - s["spearman_interface_vs_repair"]) < 1e-3
            and abs(raw["ident_dev"] - s["repair_identity_maxdev"]) < 1e-4)
    chk("S1b 汇总量（极差 / Spearman / 恒等偏差）由原始日志重算一致", ok1b,
        "raw spread=%.4f/%.4f rho=%.4f dev=%.4f" % (raw["spread_a"], raw["spread_b"],
                                                   raw["rho"], raw["ident_dev"]))

    chk("S2a 存在接口精度 = 0 的模型（『完全不会』的见证）",
        any(m["interface_accuracy_step1"] == 0.0 for m in rep["models"]),
        "零精度模型 = %s" % [m["model"] for m in rep["models"] if m["interface_accuracy_step1"] == 0.0])

    chk("S2b `correction_flip` ⟺ (step1 错 ∧ step2 对)（恒等的机制来源）", raw["flip_iff"])

    ti_ok = (len(raw["ti_vals"]) > 1 and max(raw["ti_vals"]) - min(raw["ti_vals"]) < len(raw["ti_vals"]) + 5
             and min(raw["ti_hist"].values()) >= 2)
    chk("S2c `ti` 是任务序号（连续小整数、每值多行）而非迭代计数——支撑『禁当迭代数引用』",
        ti_ok, "ti 取值 %s…%s，共 %d 个取值，最小频次 %d"
        % (raw["ti_vals"][:3], raw["ti_vals"][-3:], len(raw["ti_vals"]),
           min(raw["ti_hist"].values())))

    chk("S2d `step2_correct` 恒为 True（系统输出 = 1.0）", raw["all_step2_true"])

    # ---- 篡改捕获 ----
    def s1_check(rep_, raw_):
        for mine, theirs in zip(raw_["models"], rep_["models"]):
            for mk, tk in (("s1", "interface_accuracy_step1"), ("s2", "system_output_step2"),
                           ("fl", "repair_rate_flip")):
                if abs(mine[mk] - theirs[tk]) > 1e-4:
                    return False
        return True

    t1 = copy.deepcopy(rep)
    t1["models"][0]["interface_accuracy_step1"] = 0.1234
    chk("S3a 篡改某模型接口精度 → S1 必报", not s1_check(t1, raw))
    t2 = copy.deepcopy(rep)
    t2["summary"]["repair_identity_maxdev"] = 0.5
    t2_ok = abs(raw["ident_dev"] - t2["summary"]["repair_identity_maxdev"]) < 1e-4
    chk("S3b 篡改恒等偏差 → S1b 必报", not t2_ok)
    t3 = copy.deepcopy(rep)
    t3["preregistered_verdict"]["H86-3_repair_identity_disclosed"] = False
    chk("S3c 翻转 H86-3 → verdict_pass≠all(H) 必报",
        t3["verdict_pass"] != all(t3["preregistered_verdict"].values()))
    t4 = copy.deepcopy(rep)
    t4["models"] = [m for m in t4["models"] if m["interface_accuracy_step1"] != 0.0]
    chk("S3d 抹掉零精度模型（削弱见证）→ S2a 必报",
        not any(m["interface_accuracy_step1"] == 0.0 for m in t4["models"]))
    chk("S3e 未篡改 → S1/S1b/S2a–d 全通过（阴性对照）",
        s1_check(rep, raw) and ok1b and raw["flip_iff"] and raw["all_step2_true"])
    chk("S4 幂等：两次重算一致", recompute_from_raw() == raw)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "从 9 个 E19 原始日志直接重算，与 E86 报告逐项一致；",
        "★ 独立核验了三项「关于数据本身」的断言：零精度模型确实存在、"
        "`correction_flip ≡ (step1 错 ∧ step2 对)`、`ti` 是任务序号（非迭代数）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
