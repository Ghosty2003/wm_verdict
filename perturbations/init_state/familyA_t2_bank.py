"""T2 family A (object displacement) native anchors: moka_pot_1 along +x (away from the robot, the stove at x ~ -0.2
and the frypan at x ~ -0.05; candidate -y runs toward the frypan at y ~ -0.25, so +x per the registry principle
"away from other objects, collision-free placement" - the probes below verify, and a suspect direction is flagged).
Levels A0 sham / A1 boundary / A2-A4 = 2 / 5 / 10 cm beyond; checks identical to familyA_bank.py. Master 20261105.

  python familyA_t2_bank.py   -> out/bank/familyA_t2/{familyA_t2_anchors.json, familyA_t2_verify.json, bddl/}
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import hashlib
import numpy as np

_wmvp.bootstrap()
import familyA_bank as FA
from familyA_bank import target_point
from libero.libero import benchmark, get_libero_path
from native_env import bddl_region_rewrite, region_ranges, BDDL_HOOK, sha256_text

ROOT = FA.ROOT
OUT = f"{ROOT}/libero/out/bank/familyA_t2"
MASTER, N, TID = 20261105, 5, 2
CFG = dict(obj="moka_pot_1", region="moka_pot_init_region", direction="+x", stage_kind="multi-stage",
           levels={"A0": None, "A1": 0.0, "A2": 0.02, "A3": 0.05, "A4": 0.10})


def main():
    os.makedirs(f"{OUT}/bddl", exist_ok=True)
    suite = benchmark.get_benchmark_dict()[FA.SUITE]()
    rs = np.random.RandomState(MASTER)
    task = suite.get_task(TID)
    src_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    src = open(src_path).read(); rng = region_ranges(src, CFG["region"])
    seeds = rs.randint(0, 2 ** 31 - 1, size=N)
    anchors, bddls = [], {}
    for L, d in CFG["levels"].items():
        if d is None:
            txt = bddl_region_rewrite(src, CFG["region"], rng); tgt = None; new_rng = rng
        else:
            tx, ty = target_point(rng, CFG["direction"], d); e = 0.001
            new_rng = (tx - e, ty - e, tx + e, ty + e); txt = bddl_region_rewrite(src, CFG["region"], new_rng); tgt = (tx, ty)
        rel = f"libero/out/bank/familyA_t2/bddl/t{TID}_{L}.bddl"
        open(f"{ROOT}/{rel}", "w").write(txt)
        bddls[L] = dict(path=rel, sha256=sha256_text(txt), identical_to_original=(txt == src))
        for j in range(N):
            anchors.append(dict(anchor_id=f"libero10_t{TID}_fa{j}_{L}", base_id=f"libero10_t{TID}_fa{j}", level=L,
                                family="A_obj", suite=FA.SUITE, task_id=TID, task=task.name, stage_kind=CFG["stage_kind"],
                                seed=int(seeds[j]), settle_steps=FA.SETTLE, resolution=FA.RES, native=True, init_index=None,
                                bddl_file=rel, bddl_sha256=bddls[L]["sha256"], hook=BDDL_HOOK,
                                displacement=dict(object=CFG["obj"], region=CFG["region"], direction=CFG["direction"],
                                                  beyond_boundary_m=d, native_ranges=list(rng), rewritten_ranges=list(new_rng),
                                                  target_region_xy=None if tgt is None else list(tgt), sham=(L == "A0"))))
    bddls["original"] = dict(path=os.path.relpath(src_path, ROOT), sha256=sha256_text(src))
    ars = np.random.RandomState(MASTER + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(FA.R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(FA.R_STEPS) % 60 < 30, -1.0, 1.0)
    report, prev = [], None
    for a in anchors:
        SA = FA.run(suite, a, actions); SB = FA.run(suite, a, actions)
        dAB = FA.maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values()); dD = None
        if prev is not None:
            _, envP = FA.run(suite, prev, actions, keep_open=True)
            SD = FA.run(suite, a, actions); envP.close()
            dD = FA.maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        row = dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                   init_success=bool(SA[0]["succ"][0]),
                   trajectory_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                             for s in SA for k in sorted(s))).hexdigest())
        if a["level"] == "A0":
            dS = FA.maxdiff(SA, FA.run(suite, a, actions, bddl_abs=os.path.join(ROOT, bddls["original"]["path"])))
            row["sham_vs_untouched_native"] = dS; row["sham_pass"] = all(v == 0.0 for v in dS.values())
        row["placement"] = FA.place_and_reach(suite, a, CFG["obj"])
        report.append(row)
        print(a["anchor_id"], "PASS" if ok else "FAIL", row.get("sham_pass", ""),
              {k: row["placement"][k] for k in ("drift_cm", "planar_dist_to_base_m", "other_contacts", "reach_min_dist_cm")}, flush=True)
        prev = a
    suspect = [r["anchor_id"] for r in report
               if r["placement"]["other_contacts"] or r["placement"]["reach_min_dist_cm"]["pre_grasp_6cm"] > 8.0
               or r["placement"]["drift_cm"] > 2.0]
    bank = dict(family="A_obj", suite=FA.SUITE, master_seed=MASTER, anchors_per_task=N, settle_steps=FA.SETTLE,
                resolution=FA.RES, hook=BDDL_HOOK, levels=CFG["levels"], direction=CFG["direction"],
                direction_reason="away from the robot, the stove (x ~ -0.2) and the frypan (x ~ -0.05, y ~ -0.25); -y would approach the frypan",
                bddl={f"t{TID}_{k}": v for k, v in bddls.items()}, anchors=anchors)
    p = f"{OUT}/familyA_t2_anchors.json"; json.dump(bank, open(p, "w"), indent=1)
    allp = all(r["pass_"] for r in report) and all(r.get("sham_pass", True) for r in report) and not any(r["init_success"] for r in report)
    json.dump(dict(all_pass=allp, direction_suspect=suspect, tolerance=0.0, replay_steps=FA.R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/familyA_t2_verify.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", "DIRECTION_SUSPECT " + str(suspect) if suspect else "direction ok",
          hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
