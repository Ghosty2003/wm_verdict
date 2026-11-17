"""Conclusion-level reruns of closed-loop units whose two original runs were not bit-identical and whose second run was
not stored (recorded before the 2026-10-24 runner hot fix). Ruling 2026-10-24: rerun each unit twice with the fixed
runner; if every run agrees with the stored first run on the success verdict (revision 2026-10-25: no step
tolerance; step differences recorded as noise) -> keep in the closed-loop column, annotate "bit-level mismatch, conclusion consistent, exemption #N"
(libero/exemptions.md); otherwise the cell is reported as an interval (with / without these units) and we STOP.

Run only when the driver is not writing the records (after job 203).

  python ow_cl_rerun.py --run famA
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_wmvp.bootstrap()
import ow_runner as W

EXEMPTIONS = f"{W.ROOT}/libero/exemptions.md"


def next_exemption_id():
    if not os.path.exists(EXEMPTIONS):
        return 1
    ids = [int(l.split("#")[1].split()[0]) for l in open(EXEMPTIONS) if l.startswith("| #")]
    return max(ids, default=0) + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a_ = ap.parse_args()
    W._setup()
    R = W.RUNS[a_.run]; out = R["out"]; os.makedirs(f"{out}/cl_reruns", exist_ok=True)
    recs = {}
    for t in (5, 3):
        p = f"{out}/records_t{t}.jsonl"
        if os.path.exists(p):
            recs[t] = [json.loads(l) for l in open(p)]
    todo = [(t, i) for t, L in recs.items() for i, r in enumerate(L)
            if not r["closed_loop"]["deterministic_rerun_identical"] and "rerun" not in r["closed_loop"] and "exemption" not in r["closed_loop"]]
    print("units to rerun:", [recs[t][i]["unit_id"] for t, i in todo], flush=True)
    if not todo:
        return
    eng, pre, arch, t_load, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_sim_{os.getpid()}")
    from libero_lang import LANG
    levels = sorted({recs[t][i]["setting_id"] for t, i in todo})
    anc_of = {}
    for anc, s0, L, seeds in W.anchors_for(a_.run, levels):
        for k in range(W.K):
            anc_of[f"openwam_{s0}__k{k}" + (f"__{L}" if L else "")] = (anc, int(seeds[k]))
    results, bad = {}, []
    for t, i in todo:
        r = recs[t][i]; uid = r["unit_id"]; anc, seed = anc_of[uid]
        assert seed == r["imagination_seed"]
        ca = np.load(f"{W.ROOT}/{r['action_file']}")["closed_loop"]; z = np.load(f"{W.ROOT}/{r['trajectory_file']}")
        cs, cS = z["closed_loop_success"], z["closed_loop_states"]
        runs = []
        for j in range(2):
            ca2, cs2, cS2 = W.closed_loop(eng, sim, anc, LANG[t], seed)
            div, f2, concl = W.cl_compare(ca, cs, cS, ca2, cs2, cS2)
            runs.append(dict(success=bool(cs2.any()), first_success_step=f2, first_divergence_vs_run1=div,
                             bit_identical_to_run1=div is None, conclusion_consistent_with_run1=concl))
            np.savez_compressed(f"{out}/cl_reruns/{uid}__rerun{j + 1}.npz", actions=ca2, success=cs2, states=cS2)
            print(uid, runs[-1], flush=True)
        ok = all(x["conclusion_consistent_with_run1"] for x in runs)
        results[uid] = (t, i, runs, ok)
        if not ok:
            bad.append(uid)
    sim.close()
    nid = next_exemption_id(); rows = []
    for uid, (t, i, runs, ok) in results.items():
        cl = recs[t][i]["closed_loop"]
        ex = dict(conclusion_consistent=ok, reruns=runs, ruling="2026-10-24", step_tolerance=W.CL_STEP_TOL, machine=os.uname().nodename,
                  note="bit-level mismatch between the two original runs (second run not stored); conclusion-level reruns x2")
        if ok:
            ex["id"] = f"#{nid}"; rows.append(f"| #{nid} | {uid} | closed-loop reference column | bit-level mismatch, success verdict consistent "
                                              f"(run 1 first success {cl['first_success_step']}; reruns "
                                              f"{[x['first_success_step'] for x in runs]}) | ruling 2026-10-24 |")
            nid += 1
        cl["exemption"] = ex
    for t, L in recs.items():                                    # rewrite atomically
        p = f"{out}/records_t{t}.jsonl"
        with open(p + ".tmp", "w") as f:
            for r in L:
                f.write(json.dumps(r) + "\n")
        os.replace(p + ".tmp", p)
    if rows:
        new = not os.path.exists(EXEMPTIONS)
        with open(EXEMPTIONS, "a") as f:
            if new:
                f.write("# 豁免登记（LIBERO）\n\n逐位确定性只是配对轨（想象、真值重放）的硬停点。闭环参照列要求结论级一致"
                        "（成功结论相同、首次成功步 ±10 步），逐位不一致照实登记于此（裁决 2026-10-24，规则澄清）。\n\n"
                        "| 编号 | unit | 范围 | 内容 | 依据 |\n|---|---|---|---|---|\n")
            f.write("\n".join(rows) + "\n")
    print("exemptions", rows, "conclusion-level mismatch", bad, flush=True)
    if bad:
        raise SystemExit(f"closed-loop conclusion differs on rerun: {bad} -- report the cell as an interval and STOP")


if __name__ == "__main__":
    main()
