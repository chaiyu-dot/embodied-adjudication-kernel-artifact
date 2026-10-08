# -*- coding: utf-8 -*-
"""_audit_extended_coverage.py —— 协议外落盘 report 的 L6（只读 JSON 反向）扩审。

背景：run_audit_protocol.py 只注册 31 个报告（五阶控制链）；落盘 *_report.json 共 106 个，
另有 75 个**零审核层**。本脚本对**协议外**的报告做最小审核（L6）：
  1. JSON 可解析
  2. verdict 字段存在（兼容多种命名）或显式声明无
  3. criteria ↔ verdict 自洽（若有 criteria dict）
  4. 无绝对路径（C2）
  5. 顶层空洞字段（value 全 None）

独立脚本，**不改** run_audit_protocol.py。产物 `_audit_extended_coverage.json`。
运行：在 agent/src/eval 下 `python _audit_extended_coverage.py`。
"""
import glob
import json
import os
import re
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))

VERDICT_KEYS = ("verdict", "verdict_pass", "preregistered_verdict", "posthoc_verdict",
                "posthoc_verdict_effective", "verdict_corrected", "verdicts",
                "audit_pass", "strict_pass", "replicate_pass", "reverse_pass",
                "replicator_pass", "l1_pass", "pass")

WIN = re.compile(r"[A-Za-z]:[\\/]")
POSIX = re.compile(r"/(?:home|Users|mnt|root|var|tmp)/")


def registered_reports():
    try:
        import run_audit_protocol as R
        return {e["report"] for e in getattr(R, "CHAIN", []) if e.get("report")}
    except Exception:  # noqa: BLE001
        return set()


def _walk_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


def audit_one(path):
    name = os.path.basename(path)
    r = {"report": name, "checks": {}, "problems": []}
    try:
        raw = open(path, encoding="utf-8").read()
        d = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        r["checks"]["parse"] = False
        r["problems"].append("JSON 不可解析: %s" % e)
        return r
    r["checks"]["parse"] = True
    if not isinstance(d, dict):
        r["problems"].append("顶层非 dict: %s" % type(d).__name__)
        return r

    vkeys = [k for k in VERDICT_KEYS if k in d]
    r["checks"]["verdict_keys"] = vkeys
    exempt = (d.get("verdict") == "DECLARED_EXEMPT") or (d.get("verdict_exempt") is True)
    if not vkeys and not exempt:
        r["problems"].append("no-verdict-key（无任何 verdict 命名；需文档声明）")

    crit = d.get("criteria")
    if isinstance(crit, dict) and len(crit) > 0 and "verdict_pass" in d:
        if bool(d.get("verdict_pass")) != all(bool(v) for v in crit.values()):
            r["problems"].append("criteria↔verdict_pass 不自洽")

    # 绝对路径：排除 URL（http(s)/ws/ftp 等含 "://" 的字符串，避免 "http:/" 被误判为 "x:/"）
    hits = [s for s in _walk_strings(d)
            if "://" not in s and (WIN.search(s) or POSIX.search(s))]
    r["checks"]["abs_path_hits"] = len(hits)
    r["checks"]["abs_path_samples"] = [s[:120] for s in hits[:3]]
    if hits:
        r["problems"].append("含绝对路径 %d 处（C2）例:%s" % (len(hits), hits[0][:80]))

    empty = [k for k, v in d.items() if v is None]
    r["checks"]["null_top_keys"] = empty
    if empty:
        r["problems"].append("顶层空洞字段: %s" % empty)
    return r


def main():
    reg = registered_reports()
    all_rep = sorted(os.path.basename(p) for p in glob.glob(os.path.join(EVAL, "*_report.json")))
    outside = [f for f in all_rep if f not in reg]
    rows = [audit_one(os.path.join(EVAL, f)) for f in outside]
    bad = [r for r in rows if r["problems"]]
    out = {"scope": "协议外 *_report.json（L6 只读反向）",
           "n_registered": len(reg), "n_all": len(all_rep), "n_outside": len(outside),
           "n_clean": len(rows) - len(bad), "n_with_problems": len(bad),
           "rows": rows, "pass": len(bad) == 0}
    with open(os.path.join(EVAL, "_audit_extended_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("扩审（协议外 L6）：报告 %d 个 ｜ 干净 %d ｜ 有问题 %d"
          % (len(rows), out["n_clean"], len(bad)))
    for r in bad:
        print("  [!] %-52s %s" % (r["report"], "; ".join(r["problems"])))
    print("pass = %s -> _audit_extended_coverage.json" % out["pass"])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
