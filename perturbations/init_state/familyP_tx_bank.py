"""Generic family-P bank for a new LIBERO-10 task (T3/T4 onboarding): familyP_t2_bank.py parameterized.

  python familyP_tx_bank.py <tid> <master>  -> out/bank/familyP_t<tid>/...
Masters (recorded here): t4 = 20261109, t6 = 20261110.
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
from native_env import NativeLiberoEnv, POSE_HOOK

ROOT = FP.ROOT


def main(tid, master):
    OUT = f"{ROOT}/libero/out/bank/familyP_t{tid}"; os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[FP.SUITE]()
    t = suite.get_task(tid); src = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    FP.make = lambda s, a, noise="anchor": NativeLiberoEnv(
        task_suite=s, task_id=a["task_id"], task_suite_name=FP.SUITE, observation_width=FP.RES,
        observation_height=FP.RES, n_envs=1, num_steps_wait=FP.SETTLE, hard_reset=True, bddl_file=src,
        initialization_noise=a["initialization_noise"] if noise == "anchor" else noise)
    rs = np.random.RandomState(master); seeds = rs.randint(0, 2 ** 31 - 1, size=5)
    anchors = [dict(anchor_id=f"libero10_t{tid}_fp{j}_{L}", base_id=f"libero10_t{tid}_fp{j}", level=L, family="P_pose",
                    suite=FP.SUITE, task_id=tid, task=t.name, stage_kind="multi-stage", seed=int(seeds[j]),
                    settle_steps=FP.SETTLE, resolution=FP.RES, native=True, init_index=None, bddl_file=None,
                    initialization_noise={"type": "uniform", "magnitude": mag}, hook=POSE_HOOK, coverage=FP.COVER[L])
               for L, mag in FP.LEVELS.items() for j in range(5)]
    ars = np.random.RandomState(master + 1)
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
                   init_success=bool(SA[0]["succ"][0]), coverage_all_inside=not viol, coverage_violations=viol,
                   robot_abs_dq_max=float(dq.max()),
                   robot_dq_within_magnitude=bool(dq.max() <= a["initialization_noise"]["magnitude"] + 1e-9),
                   robot_object_contacts_at_reset=rc,
                   trajectory_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                             for s in SA for k in sorted(s))).hexdigest())
        if a["level"] == "P0":
            row["sham_selfconsistent"] = all(v == 0.0 for v in dAB.values())
        if viol:
            stop.append(a["anchor_id"])
        report.append(row)
        print(a["anchor_id"], "PASS" if ok else "FAIL", "inside" if not viol else f"OUTSIDE {viol}", flush=True)
        prev = a
    bank = dict(family="P_pose", suite=FP.SUITE, master_seed=master, anchors_per_task=5, settle_steps=FP.SETTLE,
                resolution=FP.RES, hook=POSE_HOOK, levels=FP.LEVELS, coverage=FP.COVER,
                bddl={f"t{tid}_original": dict(path=os.path.relpath(src, ROOT),
                                               sha256=hashlib.sha256(open(src, "rb").read()).hexdigest())},
                anchors=anchors)
    p = f"{OUT}/familyP_t{tid}_anchors.json"; json.dump(bank, open(p, "w"), indent=1)
    allp = (all(r["pass_"] for r in report) and all(r.get("sham_selfconsistent", True) for r in report)
            and not stop and not any(r["init_success"] for r in report))
    json.dump(dict(all_pass=allp, coverage_stop=stop, tolerance=0.0, replay_steps=FP.R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/familyP_t{tid}_verify.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", f"famP t{tid}")
    return allp


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]), int(sys.argv[2])) else 1)
