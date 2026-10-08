# -*- coding: utf-8 -*-
"""_audit_e56c.py — E56c 诚实失败披露的 L1 自洽校验。

仅校验同目录 _audit_e56c.json：checks/tamper 全通过、verdict_pass=False、
audit_integrity_pass=True、n_pass/n_total 自洽。不引绝对路径，可在 replcheck
隔离目录运行。rc=0 表示审计层自洽（失败真实、未伪造、已如实声明）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "_audit_e56c.json")


def fail(msg):
    print("E56c L1 校验 FAIL:", msg)
    sys.exit(1)


if not os.path.exists(AUDIT):
    fail("审计产物 %s 缺失" % AUDIT)
d = json.load(open(AUDIT, encoding="utf-8"))

if not all(c.get("pass") for c in d.get("checks", [])):
    fail("checks 非全通过")
if not all(t.get("pass") for t in d.get("tamper", [])):
    fail("tamper 非全通过")
if d.get("verdict_pass") is not False:
    fail("verdict_pass 应为 False（诚实失败，不得翻成通过）")
if d.get("audit_integrity_pass") is not True:
    fail("audit_integrity_pass 应为 True")
np, nt = d.get("n_pass"), d.get("n_total")
if np is None or nt is None or np != nt:
    fail("n_pass/n_total 不自洽")
print("E56c L1 自洽校验 PASS (verdict_pass=False, audit_integrity_pass=True, n_pass=%s/%s)" % (np, nt))
sys.exit(0)
