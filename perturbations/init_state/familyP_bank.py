"""LIBERO family P (robot initial pose, batch 2): freeze per-level NATIVE anchors and verify them.

Ruling of 2026-10-15: levels P0 (uniform noise, magnitude 0; the family's own baseline), P1 0.02 rad (known),
P2 0.06 (known boundary), P3 0.12, P4 0.24 (out of fine-tuning). Hook = robosuite `initialization_noise`
{"type": "uniform", "magnitude": m} passed to OffScreenRenderEnv (native_env.py); original BDDL; init_states=False.
Seeds: master 20261032, 5 per task, shared by all levels (anchor id libero10_t<task>_fp<j>_P<n>).

Checks:
  verify    per anchor: fresh vs fresh, and after another anchor's episode; R steps of a fixed action sequence (tol 0)
  sham      (P family, redefined) P0 generated twice independently with the same seed -> bit-identical (tol 0)
  artifact  P0 vs the untouched native path (initialization_noise None): object placement difference recorded (known:
            switching the noise type changes the order of global-RNG draws before placement; not physical)
  coverage  every object / fixture of every anchor lies inside its native BDDL region (sampler reference_pos + ranges),
            on the sampler's own output (before the settle steps); any violation -> STOP
  robot     per-joint |q - init_qpos| <= magnitude; contacts between the robot and any object at reset (listed)

  python familyP_bank.py
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
from libero.libero import benchmark
from anchor_bank import snapshot, maxdiff, SUITE, SETTLE, RES, R_STEPS
from native_env import NativeLiberoEnv, POSE_HOOK

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/bank/familyP"
MASTER = 20261032
N = 5
LEVELS = {"P0": 0.0, "P1": 0.02, "P2": 0.06, "P3": 0.12, "P4": 0.24}
COVER = {"P0": "known (family baseline)", "P1": "known", "P2": "known (boundary)",
         "P3": "out-of-fine-tuning (pre-training coverage not excluded)",
         "P4": "out-of-fine-tuning (pre-training coverage not excluded)"}
STAGE = {5: "single-stage", 3: "multi-stage"}


def make(suite, a, noise="anchor"):
    bank_A = json.load(open(f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"))
    orig = os.path.join(ROOT, bank_A["bddl"][f"t{a['task_id']}_original"]["path"])
    return NativeLiberoEnv(task_suite=suite, task_id=a["task_id"], task_suite_name=SUITE, observation_width=RES,
                           observation_height=RES, n_envs=1, num_steps_wait=SETTLE, hard_reset=True, bddl_file=orig,
                           initialization_noise=a["initialization_noise"] if noise == "anchor" else noise)


def run(suite, a, actions, keep_open=False, noise="anchor"):
    env = make(suite, a, noise); obs, _ = env.reset(seed=a["seed"])
    snaps = [snapshot(env, obs)]
    for x in actions:
        obs, *_ = env.step(x); snaps.append(snapshot(env, obs))
    if keep_open:
        return snaps, env
    env.close(); return snaps


def sampled_layout(suite, a, noise="anchor"):
    """the sampler's own output: positions right after the native reset, before the settle steps; + region bounds."""
    env = make(suite, a, noise); env._ensure_env(); u = env._env
    u.seed(a["seed"]); u.reset()
    m, d = u.sim.model, u.sim.data
    out, viol = {}, []
    for name, s in u.env.placement_initializer.samplers.items():
        for mo in s.mujoco_objects:
            jn = f"{mo.name}_joint0"
            try:
                adr = m.get_joint_qpos_addr(jn)[0]; xy = d.qpos[adr:adr + 2].copy()
            except Exception:
                xy = d.body_xpos[m.body_name2id(mo.root_body)][:2].copy()
            ref = np.asarray(s.reference_pos[:2])
            boxes = [(ref[0] + xr[0], ref[1] + yr[0], ref[0] + xr[1], ref[1] + yr[1]) for xr, yr in zip(s.x_ranges, s.y_ranges)]
            inside = any(b[0] - 1e-6 <= xy[0] <= b[2] + 1e-6 and b[1] - 1e-6 <= xy[1] <= b[3] + 1e-6 for b in boxes)
            out[mo.name] = dict(xy=np.round(xy, 5).tolist(), inside=bool(inside), boxes=[list(np.round(b, 4)) for b in boxes])
            if not inside:
                viol.append(mo.name)
    rob = u.robots[0]
    dq = (d.qpos[np.array(rob._ref_joint_pos_indexes)] - np.array(rob.init_qpos))
    rc = set()
    for i in range(d.ncon):
        b1 = m.body_id2name(m.geom_bodyid[d.contact[i].geom1]); b2 = m.body_id2name(m.geom_bodyid[d.contact[i].geom2])
        if (b1.startswith("robot0") or b1.startswith("gripper0")) != (b2.startswith("robot0") or b2.startswith("gripper0")):
            other = b2 if b1.startswith(("robot0", "gripper0")) else b1
            if "table" not in other and "mount" not in other and "base" not in other:
                rc.add(other)
    env.close()
    return out, viol, np.abs(dq), sorted(rc)


def main():
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    rs = np.random.RandomState(MASTER)
    anchors = []
    for tid in (5, 3):
        seeds = rs.randint(0, 2 ** 31 - 1, size=N)
        for L, mag in LEVELS.items():
            for j in range(N):
                anchors.append(dict(anchor_id=f"libero10_t{tid}_fp{j}_{L}", base_id=f"libero10_t{tid}_fp{j}", level=L,
                                    family="P_pose", suite=SUITE, task_id=tid, task=suite.get_task(tid).name,
                                    stage_kind=STAGE[tid], seed=int(seeds[j]), settle_steps=SETTLE, resolution=RES,
                                    native=True, init_index=None, bddl_file=None,
                                    initialization_noise={"type": "uniform", "magnitude": mag}, hook=POSE_HOOK,
                                    coverage=COVER[L]))
    ars = np.random.RandomState(MASTER + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(R_STEPS) % 60 < 30, -1.0, 1.0)
    report, prev, stop = [], None, []
    for a in anchors:
        a["bddl_file"] = None
        SA = run(suite, a, actions); SB = run(suite, a, actions)
        dAB = maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values()); dD = None
        if prev is not None:
            _, envP = run(suite, prev, actions, keep_open=True); SD = run(suite, a, actions); envP.close()
            dD = maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        lay, viol, dq, rc = sampled_layout(suite, a)
        row = dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                   trajectory_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                            for s in SA for k in sorted(s))).hexdigest(),
                   coverage_all_inside=not viol, coverage_violations=viol, layout=lay,
                   robot_abs_dq_max=float(dq.max()), robot_dq_within_magnitude=bool(dq.max() <= a["initialization_noise"]["magnitude"] + 1e-9),
                   robot_object_contacts_at_reset=rc)
        if a["level"] == "P0":
            row["sham_selfconsistent"] = all(v == 0.0 for v in dAB.values())      # P0 twice, same seed
            lay_native, _, _, _ = sampled_layout(suite, a, noise=None)
            row["artifact_P0_vs_native_layout_maxdiff_m"] = max(
                float(np.max(np.abs(np.array(lay[k]["xy"]) - np.array(lay_native[k]["xy"])))) for k in lay)
        if viol:
            stop.append(a["anchor_id"])
        report.append(row)
        print(a["anchor_id"], "PASS" if ok else "FAIL", "inside" if not viol else f"OUTSIDE {viol}",
              f"|dq|max {dq.max():.3f}", "contacts", rc, row.get("artifact_P0_vs_native_layout_maxdiff_m", ""), flush=True)
        prev = a
    bank = dict(family="P_pose", suite=SUITE, master_seed=MASTER, anchors_per_task=N, settle_steps=SETTLE, resolution=RES,
                restore_recipe="native_env.NativeLiberoEnv(init_states=False, original BDDL, initialization_noise="
                               "{type: uniform, magnitude: m}, hard_reset=True): new env per restore -> reset(seed) -> settle",
                hook=POSE_HOOK, levels=LEVELS, coverage=COVER,
                bddl={f"t{t}_original": json.load(open(f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"))["bddl"][f"t{t}_original"]
                      for t in (5, 3)},
                anchors=anchors)
    p = f"{OUT}/familyP_anchors.json"; json.dump(bank, open(p, "w"), indent=1)
    for tid in (5, 3):
        ids = sorted({a["base_id"] for a in anchors if a["task_id"] == tid}, key=lambda s: int(s.split("fp")[1]))
        json.dump(dict(note="family P base anchor ids (s0_id); the level is the setting", source=os.path.relpath(p, ROOT),
                       source_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                       entries=[dict(s0_id=i) for i in ids]), open(f"{OUT}/familyP_entries_t{tid}.json", "w"), indent=1)
    allp = all(r["pass_"] for r in report) and all(r.get("sham_selfconsistent", True) for r in report)
    json.dump(dict(all_pass=allp, coverage_stop=stop, tolerance=0.0, replay_steps=R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()), open(f"{OUT}/familyP_verify.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", "COVERAGE STOP " + str(stop) if stop else "coverage all inside",
          hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
