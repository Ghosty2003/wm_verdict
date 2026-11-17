"""UVA K 族（作者 $COLLAB_UVA 11-13 批次 k-family-20261113-v1，T1–T4 × K1–K9，每格 50 单元 × 2 遍，H=400）逐格核对 → uva_k_cells.json。
只读作者 NAS 目录，不改源文件。核对：
  1. 作者 <cell>/closed/probe-complete.json status == passed，rows = 100（2 遍 × 50）
  2. repeat0/result.json 恰 50 个，与 probe-complete 中 repeat_id=0 的行逐单元 native_success / first_native_success_step 相同
  3. K 参数：result.json 的 family=K、trigger_step / distance_m 与 plan.json 该档一致；horizon=400、diagnostic_only=false
  4. 两遍结论一致率（repeat0 vs repeat1 的 native_success）
  5. 该任务 K0 门槛日志含 K_PHYSICAL_GATE_PASSED
主表用第一遍（repeat0），与其他 UVA 格同口径。
2026-11-16（v6）：加 T0——作者 11-14 正式批 t0-k-ef-20261115/K-formal-v3（9 档 × 50 单元 × 2 遍 = 900 条轨迹）。
  目录结构不同：<D_T0>/output/<K>/closed/task5_anchor*_seed*/repeat{0,1}/result.json；K 参数取 formal-contract.json 的 cells；
  K0 门槛 = t0-k-ef-20261115/K0-health-v1/health-*.out 含 K_PHYSICAL_GATE_PASSED；另查 result.json 的 task_id = 5。核对项 1–4 与 T1–T4 相同。
  python3 uva_k_scan.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os

D = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate/k-family-20261113-v1")
D_T0 = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate/t0-k-ef-20261115/K-formal-v3")      # 2026-11-16 v6
T0_GATE_GLOB = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate/t0-k-ef-20261115/K0-health-v1/health-*.out")
BATCH = {"T0": "t0_formal_v3_20261114", "T1": "k_family_20261113_v1", "T2": "k_family_20261113_v1", "T3": "k_family_20261113_v1", "T4": "k_family_20261113_v1"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_k_cells.json")


def main():
    plan = json.load(open(f"{D}/plan.json"))
    S = {s["key"]: s for s in plan["settings"]}
    contract = json.load(open(f"{D_T0}/formal-contract.json"))
    S0 = {s["key"]: s for s in contract["cells"]}
    assert contract["task"] == "T0" and contract["task_id"] == 5 and contract["horizon"] == 400, contract["task"]
    cells, issues = [], []
    for T in ("T0", "T1", "T2", "T3", "T4"):
        if T == "T0":
            if not os.path.isdir(f"{D_T0}/output"):
                issues.append("T0: K-formal-v3 output 不存在"); continue
            gate = any("K_PHYSICAL_GATE_PASSED" in open(f).read() for f in glob.glob(T0_GATE_GLOB))
            SS, celldir = S0, (lambda K: f"{D_T0}/output/{K}/closed")
        else:
            if not os.path.isdir(f"{D}/{T}"):
                issues.append(f"{T}: 作者批次里没有这个任务"); continue
            gate = any("K_PHYSICAL_GATE_PASSED" in open(f).read() for f in glob.glob(f"{D}/{T}-K0-gate*.out"))
            SS, celldir = S, (lambda K, T=T: f"{D}/{T}/{K}/closed")
        if not gate:
            issues.append(f"{T}: K0 门槛日志没有 K_PHYSICAL_GATE_PASSED")
        for K in [f"K{i}" for i in range(1, 10)]:
            S_ = SS
            cd = celldir(K)
            pc = json.load(open(f"{cd}/probe-complete.json"))
            rows = {(r["repeat_id"], r["anchor_id"], r["seed"]): r for r in pc["rows"]}
            res0 = {}
            for f in sorted(glob.glob(f"{cd}/*/repeat0/result.json")):
                r = json.load(open(f)); res0[(r["anchor_id"], r["seed"])] = r
            res1 = {}
            for f in sorted(glob.glob(f"{cd}/*/repeat1/result.json")):
                r = json.load(open(f)); res1[(r["anchor_id"], r["seed"])] = r
            prob = []
            if pc.get("status") != "passed" or len(pc["rows"]) != 100:
                prob.append(f"probe-complete status={pc.get('status')} rows={len(pc['rows'])}")
            if len(res0) != 50 or len(res1) != 50:
                prob.append(f"result.json repeat0={len(res0)} repeat1={len(res1)}")
            for key, r in res0.items():
                p = rows.get((0,) + key)
                if p is None or bool(p["native_success"]) != bool(r["native_success"]) or p.get("first_native_success_step") != r.get("first_native_success_step"):
                    prob.append(f"repeat0 {key} 与 probe-complete 不符"); break
                if r.get("family") != "K" or r.get("trigger_step") != S_[K]["trigger_step"] or abs(r.get("distance_m", -1) - S_[K]["distance_m"]) > 1e-9 \
                        or r.get("horizon") != 400 or r.get("diagnostic_only") or (T == "T0" and r.get("task_id") != 5):
                    prob.append(f"repeat0 {key} 参数不符：family={r.get('family')} t={r.get('trigger_step')} d={r.get('distance_m')} H={r.get('horizon')}"); break
            same = sum(bool(res0[k]["native_success"]) == bool(res1[k]["native_success"]) for k in res0 if k in res1)
            k0 = sum(bool(r["native_success"]) for r in res0.values())
            steps = sorted(int(r["first_native_success_step"]) for r in res0.values() if r["native_success"] and r.get("first_native_success_step") is not None)
            cells.append(dict(task=T, level=K, closed_dir=cd, batch=BATCH[T], n=len(res0), k=k0, steps=steps,
                              k_repeat1=sum(bool(r["native_success"]) for r in res1.values()), repeat_agree=same,
                              trigger_step=S_[K]["trigger_step"], distance_m=S_[K]["distance_m"], gate_passed=gate,
                              probe_status=pc.get("status"), problems=prob))
            if prob:
                issues.append(f"{T} {K}: " + "; ".join(prob))
    json.dump(dict(source=D, source_T0=D_T0, plan_status=plan.get("status"), formal_results=plan.get("formal_results"),
                   t0_contract_status=contract.get("status"), t0_total_formal_trajectories=contract.get("total_formal_trajectories"),
                   cells=cells, issues=issues),
              open(OUT, "w"), ensure_ascii=False, indent=1)
    for c in cells:
        print(f"{c['task']} {c['level']}  n={c['n']} k={c['k']} (repeat1 k={c['k_repeat1']}, agree {c['repeat_agree']}/50)  {'OK' if not c['problems'] else c['problems']}")
    print("ISSUES:", issues or "none")


if __name__ == "__main__":
    main()
