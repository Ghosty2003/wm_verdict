"""T2 family P (robot initial pose) native anchors: the familyP_bank.py recipe for task 2 (original BDDL +
initialization_noise; P0 0 / 0.02 / 0.06 / 0.12 / 0.24 rad; same checks: bit-exact A/B/C, P0 self-sham, coverage,
|dq| <= magnitude, reset contacts). Master 20261106. Depends on familyA_t2_bank.py only for the original-BDDL record.

  python familyP_t2_bank.py   -> out/bank/familyP_t2/{familyP_t2_anchors.json, familyP_t2_verify.json}
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
import familyP_bank as FP
from libero.libero import benchmark, get_libero_path
from anchor_bank import snapshot, maxdiff  # noqa: F401  (FP.run uses them via its module)
from native_env import NativeLiberoEnv, POSE_HOOK

ROOT = FP.ROOT
OUT = f"{ROOT}/libero/out/bank/familyP_t2"
MASTER, N, TID = 20261106, 5, 2


def orig_bddl(suite):
    t = suite.get_task(TID)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)


def main():
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[FP.SUITE]()
    src = orig_bddl(suite)
    # route FP.make's familyA original-BDDL lookup to the T2 file
    FP.make = lambda s, a, noise="anchor": NativeLiberoEnv(
        task_suite=s, task_id=a["task_id"], task_suite_name=FP.SUITE, observation_width=FP.RES,
        observation_height=FP.RES, n_envs=1, num_steps_wait=FP.SETTLE, hard_reset=True, bddl_file=src,
        initialization_noise=a["initialization_noise"] if noise == "anchor" else noise)
    rs = np.random.RandomState(MASTER)
    seeds = rs.randint(0, 2 ** 31 - 1, size=N)
    anchors = [dict(anchor_id=f"libero10_t{TID}_fp{j}_{L}", base_id=f"libero10_t{TID}_fp{j}", level=L, family="P_pose",
                    suite=FP.SUITE, task_id=TID, task=suite.get_task(TID).name, stage_kind="multi-stage",
                    seed=int(seeds[j]), settle_steps=FP.SETTLE, resolution=FP.RES, native=True, init_index=None,
                    bddl_file=None, initialization_noise={"type": "uniform", "magnitude": mag}, hook=POSE_HOOK,
                    coverage=FP.COVER[L])
               for L, mag in FP.LEVELS.items() for j in range(N)]
    ars = np.random.RandomState(MASTER + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(FP.R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(FP.R_STEPS) % 60 < 30, -1.0, 1.0)
    report, prev, stop = [], None, []
    for a in anchors:
        SA = FP.run(suite, a, actions); SB = FP.run(suite, a, actions)
        dAB = FP.maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values()); dD = None
        if prev is not None:
            _, envP = FP.run(suite, prev, actions, keep_open=True); SD = FP.run(suite, a, actions); envP.close()
            dD = FP.maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        lay, viol, dq, rc = FP.sampled_layout(suite, a)
        row = dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                   init_success=bool(SA[0]["succ"][0]),
                   trajectory_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                             for s in SA for k in sorted(s))).hexdigest(),
                   coverage_all_inside=not viol, coverage_violations=viol,
                   robot_abs_dq_max=float(dq.max()),
                   robot_dq_within_magnitude=bool(dq.max() <= a["initialization_noise"]["magnitude"] + 1e-9),
                   robot_object_contacts_at_reset=rc)
        if a["level"] == "P0":
            row["sham_selfconsistent"] = all(v == 0.0 for v in dAB.values())
            lay_native, _, _, _ = FP.sampled_layout(suite, a, noise=None)
            row["artifact_P0_vs_native_layout_maxdiff_m"] = max(
                float(np.max(np.abs(np.array(lay[k]["xy"]) - np.array(lay_native[k]["xy"])))) for k in lay)
        if viol:
            stop.append(a["anchor_id"])
        report.append(row)
        print(a["anchor_id"], "PASS" if ok else "FAIL", "inside" if not viol else f"OUTSIDE {viol}",
              f"|dq|max {dq.max():.3f}", rc, flush=True)
        prev = a
    bank = dict(family="P_pose", suite=FP.SUITE, master_seed=MASTER, anchors_per_task=N, settle_steps=FP.SETTLE,
                resolution=FP.RES, hook=POSE_HOOK, levels=FP.LEVELS, coverage=FP.COVER,
                bddl={f"t{TID}_original": dict(path=os.path.relpath(src, ROOT),
                                               sha256=hashlib.sha256(open(src, "rb").read()).hexdigest())},
                anchors=anchors)
    p = f"{OUT}/familyP_t2_anchors.json"; json.dump(bank, open(p, "w"), indent=1)
    allp = (all(r["pass_"] for r in report) and all(r.get("sham_selfconsistent", True) for r in report)
            and not stop and not any(r["init_success"] for r in report))
    json.dump(dict(all_pass=allp, coverage_stop=stop, tolerance=0.0, replay_steps=FP.R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/familyP_t2_verify.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
