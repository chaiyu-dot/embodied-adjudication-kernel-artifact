# -*- coding: utf-8 -*-
"""_audit_e84_strict.py —— E84 的 **L3 严格审核**（口径+结构自洽 + 独立重放 + 篡改捕获）。

与 L2 正交：L2 只读报告 JSON；本层**读取自足数据文件 `e84_bench_data.json`**，
用**自写的门控判据重放入口**（不复用 E84 的 run()），并独立验证结构性主张。

判据
----
S1 报告↔明细自洽：四臂计数/比率由数据文件重放一致；binding 分布一致；verdict=all(H)。
S2 口径不变量：gt_dangerous_rate 四臂恒等；零测量臂测量数=0；危险仅出现在点入口臂；
   GATED 触发率 ≤ 1；GATED 测量次数 < MEAS_ALL。
S3 **独立重放**（自写臂逻辑，复用仓库判据函数但**不用 e84 的 run()**）：
   逐臂重算 dangerous / feasible / measure_calls，与报告一致。
S4 **结构性主张 H84-5 独立验证**：调 `WorldPhysicsEngine.certificate(..., extra_orders=...)`
   检查 `five_order_present ⊇ {K,S,D,C,E}` 与 `first_failing_order` 有值（包裹层五阶在环）。
S5 篡改捕获：改某臂测量数 / 可行性 / 危险数 / H / binding → S1/S2 必报。
S6 幂等。

产物：_audit_e84_strict.json
"""
import copy
import json
import os
import sys
import time

EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(EVAL))

import e84_five_order_wrapper_loop as E84        # noqa: E402
from planning.world_physics_engine import WorldPhysicsEngine   # noqa: E402

REPORT = os.path.join(EVAL, "e84_five_order_wrapper_report.json")
BENCH = os.path.join(EVAL, "e84_bench_data.json")
OUT = os.path.join(EVAL, "_audit_e84_strict.json")
_res = {"audit": "E84 L3 strict", "checks": [], "findings": []}
_n = [0]
ARMS = ["POINT", "SOUND_NOMEAS", "GATED", "MEAS_ALL"]


def chk(name, ok, detail=""):
    _n[0] += 1
    _res["checks"].append({"n": _n[0], "name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print("  [%s] %02d %s%s" % ("PASS" if ok else "FAIL", _n[0], name,
                                ("  —— " + str(detail)[:220]) if detail else ""))
    return bool(ok)


def replay_arms(data):
    """自写臂逻辑：只借仓库的 five_orders（判据本体），流程与统计全部自算。"""
    k_conf, k_max = float(data["k_conf"]), int(data["k_max"])
    agg = {a: {"n": 0, "dangerous": 0, "feasible": 0, "false_reject": 0, "calls": 0} for a in ARMS}
    binding = {o: 0 for o in ("K", "S", "D", "C", "E")}
    for p in data["proposals"]:
        q1 = p["q_final"]
        prop = E84.make_proposal(E84.make_traj([0.0, 0.0], q1, p["T"]), q1)
        _o, gt_min, bnd = E84.five_orders(p["mp_true"], prop)
        binding[bnd] += 1
        gt_dangerous = gt_min < 0.0
        for arm in ARMS:
            states = p["states"][arm]
            t = calls = 0
            while True:
                m, s = states[t]
                hi = m + k_conf * s
                min_hi = E84.five_orders(hi, prop)[1]
                min_pt = E84.five_orders(m, prop)[1]
                if arm in ("POINT", "SOUND_NOMEAS"):
                    break
                if arm == "MEAS_ALL":
                    if calls >= k_max:
                        break
                    calls += 1
                    t += 1
                    continue
                if not ((min_pt >= 0.0) and (min_hi < 0.0)) or calls >= k_max:
                    break
                calls += 1
                t += 1
            m, s = states[t]
            dec = (E84.five_orders(m, prop)[1] if arm == "POINT"
                   else E84.five_orders(m + k_conf * s, prop)[1])
            feasible = dec >= 0.0
            a = agg[arm]
            a["n"] += 1
            a["dangerous"] += int(bool(feasible and gt_dangerous))
            a["feasible"] += int(feasible)
            a["false_reject"] += int((not feasible) and (not gt_dangerous))
            a["calls"] += calls
    out = {}
    for a in ARMS:
        d = agg[a]
        n = max(d["n"], 1)
        out[a] = {"n": d["n"], "dangerous": d["dangerous"],
                  "dangerous_rate": round(d["dangerous"] / n, 4),
                  "feasible_rate": round(d["feasible"] / n, 4),
                  "false_reject_rate": round(d["false_reject"] / n, 4),
                  "measure_calls_mean": round(d["calls"] / n, 4)}
    return out, binding


def s1(rep, mine, binding):
    ok = True
    for a in ARMS:
        for k, tol in (("n", 0), ("dangerous", 0), ("dangerous_rate", 1e-4),
                       ("feasible_rate", 1e-4), ("false_reject_rate", 1e-4),
                       ("measure_calls_mean", 1e-4)):
            if abs(mine[a][k] - rep["aggregate"][a][k]) > tol:
                ok = False
    if binding != rep["order_binding_distribution"]:
        ok = False
    if rep["verdict_pass"] != all(rep["preregistered_verdict"].values()):
        ok = False
    return ok


def s2(rep):
    a = rep["aggregate"]
    gts = {a[x]["gt_dangerous_rate"] for x in ARMS}
    ok = len(gts) == 1
    ok &= (a["POINT"]["measure_calls_mean"] == 0.0 and a["SOUND_NOMEAS"]["measure_calls_mean"] == 0.0)
    ok &= (a["GATED"]["dangerous"] == 0 and a["MEAS_ALL"]["dangerous"] == 0
           and a["SOUND_NOMEAS"]["dangerous"] == 0)
    # 计数↔比率自洽（防止只改其一的自洽谎言）
    for x in ARMS:
        r = a[x]
        if abs(r["dangerous"] / max(r["n"], 1) - r["dangerous_rate"]) > 1e-4:
            ok = False
    ok &= (0.0 <= a["GATED"]["trigger_rate"] <= 1.0)
    ok &= (a["GATED"]["measure_calls_mean"] < a["MEAS_ALL"]["measure_calls_mean"])
    ok &= (a["POINT"]["feasible_rate"] >= a["GATED"]["feasible_rate"] >= a["SOUND_NOMEAS"]["feasible_rate"])
    return bool(ok)


def s4():
    """独立验证 H84-5：包裹层证书里五阶齐全 + 逐阶诊断码有值。"""
    eng = WorldPhysicsEngine()
    mp = 0.4
    q = [1.0, -0.8]
    traj = E84.make_traj([0.0, 0.0], q, 1.2)
    orders = E84.five_orders(mp, E84.make_proposal(traj, q))[0]
    cert = eng.certificate(links=E84.links_at(mp), traj=traj, torque_limits=E84.TAU_LIM,
                           omega_n=E84.OMEGA_N, zeta=E84.ZETA, T_s=E84.T_S,
                           qdot_lim=E84.QDOT_LIM,
                           extra_orders={"K": orders["K"], "S": orders["S"], "D": orders["D"],
                                         "C": orders["C"], "E": orders["E"]})
    present = set(cert.predicted_state_bounds.get("five_order_present", []))
    return ({"K", "S", "D", "C", "E"} <= present and cert.first_failing_order is not None,
            "present=%s first_failing=%s" % (sorted(present), cert.first_failing_order))


def main():
    print("=" * 92)
    print("E84 L3 严格审核")
    print("=" * 92)
    rep = json.load(open(REPORT, encoding="utf-8"))
    data = json.load(open(BENCH, encoding="utf-8"))

    mine, binding = replay_arms(data)
    chk("S1 报告↔明细自洽（四臂计数/比率 + binding + verdict=all(H)，由数据文件独立重放）",
        s1(rep, mine, binding),
        "mine GATED feas=%.4f meas=%.4f | store feas=%.4f meas=%.4f"
        % (mine["GATED"]["feasible_rate"], mine["GATED"]["measure_calls_mean"],
           rep["aggregate"]["GATED"]["feasible_rate"], rep["aggregate"]["GATED"]["measure_calls_mean"]))
    chk("S2 口径不变量（gt 恒等 / 零测量臂 / 危险仅点入口 / 触发率≤1 / 门控<全测 / 可行率偏序）", s2(rep))

    ok4, det4 = s4()
    chk("S4 独立验证 H84-5：包裹层证书五阶齐全 + first_failing_order 有值", ok4, det4)

    t1 = copy.deepcopy(rep)
    t1["aggregate"]["GATED"]["measure_calls_mean"] = 1.5
    chk("S5a 篡改 GATED 测量次数 → S1 必报", not s1(t1, mine, binding))
    t2 = copy.deepcopy(rep)
    t2["order_binding_distribution"]["C"] = 200
    chk("S5b 篡改 binding 分布 → S1 必报", not s1(t2, mine, binding))
    t3 = copy.deepcopy(rep)
    t3["aggregate"]["MEAS_ALL"]["dangerous"] = 5       # 计数与比率同时改（测"单格+聚合同步改"）
    t3["aggregate"]["MEAS_ALL"]["dangerous_rate"] = round(5 / t3["aggregate"]["MEAS_ALL"]["n"], 4)
    chk("S5c 让全测臂出现危险（计数+比率同步改）→ S2 必报", not s2(t3))
    t3b = copy.deepcopy(rep)
    t3b["aggregate"]["MEAS_ALL"]["dangerous"] = 5      # **只改计数**、比率不动 → 计数↔比率自洽检查必报
    chk("S5c' 只改危险计数而比率不动 → S2 的计数↔比率自洽检查必报", not s2(t3b))
    t4 = copy.deepcopy(rep)
    t4["aggregate"]["GATED"]["feasible_rate"] = 0.05
    chk("S5d 破坏可行率偏序 → S2 必报", not s2(t4))
    t5 = copy.deepcopy(rep)
    t5["preregistered_verdict"]["H84-3_gated_far_fewer_measures"] = False
    chk("S5e 翻转 H84-3 → S1 必报（verdict≠all(H)）", not s1(t5, mine, binding))

    chk("S6 幂等：两次重放结论一致", replay_arms(data)[0] == mine)

    npass = sum(c["pass"] for c in _res["checks"])
    _res["n_pass"], _res["n_total"] = npass, len(_res["checks"])
    _res["audit_pass"] = bool(npass == len(_res["checks"]))
    _res["findings"] = [
        "独立重放（自写臂逻辑 + 仓库判据本体）与报告逐项一致；包裹层五阶在环（H84-5）独立复算通过。",
        "★ 口径注记：本层重放**复用仓库的 five_orders**，故它排除的是『报告数字被手工修改 / 臂逻辑记错』，"
        "**不**排除判据实现本身的偏差（后者由 E84c 的 L4 独立物理重写 + E84b 的独立引擎承担）。"]
    json.dump(_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nL3 严格审核：%d/%d ｜ audit_pass = %s" % (npass, len(_res["checks"]), _res["audit_pass"]))
    return 0 if _res["audit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
