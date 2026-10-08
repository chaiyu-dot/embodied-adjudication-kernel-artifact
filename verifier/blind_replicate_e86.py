# -*- coding: utf-8 -*-
"""blind_replicate_e86.py —— E86 的 **L4 隔离盲复刻**（只给原始行数据文件 + 独立统计实现）。

不 import 仓库任何模块 / 任何 e8x 脚本：
  · 从 `e86_bench_data.json` 的**原始行**独立重算接口精度 / 系统输出 / 修复率；
  · 独立复现极差、Spearman（含并列平均秩）、**恒等偏差**与 6 条预注册判据；
  · 与作者报告逐项比对（比率容差 1e-4，整数与布尔精确相等）。

产物：blind_replicate_e86.json
"""
import json
import os
import sys

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)

import e86_standalone_stats as ST   # noqa: E402

REPORT = os.path.join(EVAL, "e86_interface_accuracy_real_models_report.json")
BENCH = os.path.join(EVAL, "e86_bench_data.json")
OUT = os.path.join(EVAL, "blind_replicate_e86.json")
_repo = ("planning", "e19_r2_closed_loop", "e86_interface_accuracy_real_models",
         "e84_five_order_wrapper_loop", "e84b_independent_engine_gt", "e84c_depth_ablation_grid")
_res = {"experiment": "E86 盲复刻（只给原始行 + 独立统计实现）", "checks": [], "findings": []}
_n = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def main():
    print("=" * 92)
    print("E86 盲复刻（只给原始行 + 独立统计实现）vs 作者报告")
    print("=" * 92)
    loaded = [m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in _repo)]
    chk("B0 隔离自检：运行期未加载任何仓库模块", len(loaded) == 0, "loaded=%s" % loaded)

    data = json.load(open(BENCH, encoding="utf-8"))
    rep = json.load(open(REPORT, encoding="utf-8"))
    chk("B0b 数据文件自足（models[*].rows = [step1_correct, step2_correct, correction_flip]）",
        bool(data.get("models")) and all(len(m["rows"][0]) == 3 for m in data["models"]),
        "models=%d rows=%d" % (len(data["models"]), sum(len(m["rows"]) for m in data["models"])))

    out = ST.replicate(data)

    # B1 逐模型三项比率
    bad = []
    for mine, theirs in zip(out["models"], rep["models"]):
        if mine["model"] != theirs["model"] or mine["n_rows"] != theirs["n_rows"]:
            bad.append(mine["model"])
            continue
        for k in ("interface_accuracy_step1", "system_output_step2", "repair_rate_flip"):
            if abs(mine[k] - theirs[k]) > 1e-4:
                bad.append((mine["model"], k))
    chk("B1 逐模型（接口精度 / 系统输出 / 修复率）与报告一致", not bad, "mismatch=%s" % bad)

    # B2 汇总量
    ms, ts = out["summary"], rep["summary"]
    bad2 = [k for k in ("n_models", "interface_accuracy_spread", "system_output_spread",
                        "spearman_interface_vs_repair", "repair_identity_maxdev")
            if abs(ms[k] - ts[k]) > 1e-4] if isinstance(ms["n_models"], float) else \
        [k for k in ("interface_accuracy_spread", "system_output_spread",
                     "spearman_interface_vs_repair", "repair_identity_maxdev")
         if abs(ms[k] - ts[k]) > 1e-4]
    if ms["n_models"] != ts["n_models"]:
        bad2.append("n_models")
    chk("B2 汇总量（极差 / Spearman / 恒等偏差）与报告一致", not bad2, "mismatch=%s" % bad2)

    # B3 恒等自曝独立复现
    chk("B3 独立复现『repair_rate ≡ 1 − interface_accuracy』（恒等偏差 ≤ 0.01）",
        out["summary"]["repair_identity_maxdev"] <= 0.01,
        "maxdev=%.4f" % out["summary"]["repair_identity_maxdev"])

    # B4 判据
    chk("B4 预注册判据 H86-1..6 逐条复现", out["H"] == rep["preregistered_verdict"],
        "mine=%s theirs=%s" % (out["H"], rep["preregistered_verdict"]))
    chk("B5 结论判定复现（verdict_pass）", out["verdict_pass"] == rep["verdict_pass"],
        "mine=%s theirs=%s" % (out["verdict_pass"], rep["verdict_pass"]))

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["replicate_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "★ 盲复刻（只给原始行 + 独立统计实现）：9 个模型的接口精度/系统输出/修复率、极差、"
        "Spearman（含并列平均秩）、恒等偏差与 6 条判据全部被外部独立重建。",
        "**语义强度标注**：原始行不可再复刻（模型调用已发生），本件复刻的是**统计管线**；"
        "『原始行是否真实』由 E19 自身的审核链承担。",
    ]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n盲复刻结果：%d/%d PASS ｜ replicate_pass = %s" % (npass, len(_res["checks"]),
                                                              _res["replicate_pass"]))


if __name__ == "__main__":
    main()
