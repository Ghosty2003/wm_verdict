"""LIBERO family A (object displacement, batch 1): freeze per-level NATIVE anchors and verify them.

Levels (ruling of 2026-10-15): A0 sham (0 cm: the region rewritten with its own values), A1 on the native region
boundary, A2 / A3 / A4 = 2 / 5 / 10 cm beyond it (t3 A4 = 7 cm, downgraded: 10 cm is at the Panda reach limit).
Direction (eval_design section 19/20): t5 book +y (away from caddy and mug), t3 bowl +x (away from the robot).
A1..A4: the object's init region becomes a 2 mm square centred on the target point; yaw keeps its native range.
Hook = BDDL region rewrite + native sampler (native_env.py); anchors use init_states=False. Seeds: master 20261030,
5 per task, shared by all levels (anchor id = libero10_t<task>_fa<j>_<level>, base id = libero10_t<task>_fa<j>).

Checks (tolerance 0 where marked):
  verify   per level anchor: fresh vs fresh and after another anchor's episode, R steps of a fixed action sequence,
           full physics / body poses / warm start / gripper / controller goal / success / both images  (0)
  sham     A0 anchor (hook path, identity rewrite) vs the untouched native path (original BDDL), same seed  (0)
  place    object settled where intended (drift), no contact with anything but the table, distance to the robot base,
           other objects' initial positions vs A0 (same seed)
  reach    scripted OSC reach (gripper open, orientation unchanged) to 6 cm above the object and to 2 cm above it;
           min distance achieved (evidence for "physically graspable")

  python familyA_bank.py
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
from libero.libero import benchmark, get_libero_path
from anchor_bank import snapshot, maxdiff, SUITE, SETTLE, RES, R_STEPS
from native_env import NativeLiberoEnv, bddl_region_rewrite, region_ranges, BDDL_HOOK, sha256_text

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/bank/familyA"
MASTER = 20261030
N = 5
CFG = {5: dict(obj="black_book_1", region="black_book_init_region", direction="+y", stage_kind="single-stage",
               levels={"A0": None, "A1": 0.0, "A2": 0.02, "A3": 0.05, "A4": 0.10}),
       3: dict(obj="akita_black_bowl_1", region="akita_black_bowl_init_region", direction="+x", stage_kind="multi-stage",
               levels={"A0": None, "A1": 0.0, "A2": 0.02, "A3": 0.05, "A4": 0.07})}


def target_point(rng, direction, d):
    x0, y0, x1, y1 = rng; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"+x": (x1 + d, cy), "-x": (x0 - d, cy), "+y": (cx, y1 + d), "-y": (cx, y0 - d)}[direction]


def make(suite, a, bddl_abs=None):
    return NativeLiberoEnv(task_suite=suite, task_id=a["task_id"], task_suite_name=SUITE, observation_width=RES,
                           observation_height=RES, n_envs=1, num_steps_wait=SETTLE, hard_reset=True,
                           bddl_file=bddl_abs if bddl_abs is not None else os.path.join(ROOT, a["bddl_file"]))


def run(suite, a, actions, bddl_abs=None, keep_open=False):
    env = make(suite, a, bddl_abs)
    obs, _ = env.reset(seed=a["seed"])
    snaps = [snapshot(env, obs)]
    for x in actions:
        obs, *_ = env.step(x)
        snaps.append(snapshot(env, obs))
    if keep_open:
        return snaps, env
    env.close()
    return snaps


def free_joint_xy(env):
    sim = env._env.sim; m = sim.model
    out = {}
    for i in range(m.njnt):
        n = m.joint_id2name(i)
        if n and n.endswith("_joint0") and m.jnt_type[i] == 0:
            adr = m.jnt_qposadr[i]; out[n[:-7]] = sim.data.qpos[adr:adr + 3].copy()
    return out


def place_and_reach(suite, a, obj):
    env = make(suite, a); obs, _ = env.reset(seed=a["seed"]); u = env._env
    q0 = free_joint_xy(env)
    for _ in range(20):
        u.step([0, 0, 0, 0, 0, 0, -1])
    q1 = free_joint_xy(env)
    m, d = u.sim.model, u.sim.data
    oth = set()
    for i in range(d.ncon):
        b1 = m.body_id2name(m.geom_bodyid[d.contact[i].geom1]); b2 = m.body_id2name(m.geom_bodyid[d.contact[i].geom2])
        if obj in b1 or obj in b2:
            other = b2 if obj in b1 else b1
            if "table" not in other:
                oth.add(other)
    base = d.body_xpos[m.body_name2id("robot0_base")]
    reach = {}
    for name, dz in (("pre_grasp_6cm", 0.06), ("grasp_2cm", 0.02)):
        env.close(); env = make(suite, a); env.reset(seed=a["seed"]); u = env._env
        tgt = free_joint_xy(env)[obj] + np.array([0, 0, dz])
        best = 9.0
        for _ in range(250):
            eef = u.robots[0].controller.ee_pos
            err = tgt - eef; best = min(best, float(np.linalg.norm(err)))
            u.step(np.concatenate([np.clip(err * 20.0, -1, 1), [0, 0, 0, -1]]))
        reach[name] = round(best * 100, 2)          # cm
    env.close()
    return dict(placed_xyz=np.round(q0[obj], 4).tolist(), settled_xyz=np.round(q1[obj], 4).tolist(),
                drift_cm=round(float(np.linalg.norm(q1[obj][:2] - q0[obj][:2]) * 100), 2),
                planar_dist_to_base_m=round(float(np.linalg.norm(q1[obj][:2] - base[:2])), 3),
                other_contacts=sorted(oth), reach_min_dist_cm=reach,
                others_initial={k: np.round(v, 5).tolist() for k, v in q0.items() if k != obj})


def main():
    os.makedirs(f"{OUT}/bddl", exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    rs = np.random.RandomState(MASTER)
    anchors, bddls = [], {}
    for tid in (5, 3):
        c = CFG[tid]; task = suite.get_task(tid)
        src_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        src = open(src_path).read(); rng = region_ranges(src, c["region"])
        seeds = rs.randint(0, 2 ** 31 - 1, size=N)
        for L, d in c["levels"].items():
            if d is None:
                txt = bddl_region_rewrite(src, c["region"], rng); tgt = None; new_rng = rng
            else:
                tx, ty = target_point(rng, c["direction"], d); e = 0.001
                new_rng = (tx - e, ty - e, tx + e, ty + e); txt = bddl_region_rewrite(src, c["region"], new_rng); tgt = (tx, ty)
            rel = f"libero/out/bank/familyA/bddl/t{tid}_{L}.bddl"
            open(f"{ROOT}/{rel}", "w").write(txt)
            bddls[(tid, L)] = dict(path=rel, sha256=sha256_text(txt), identical_to_original=(txt == src))
            for j in range(N):
                anchors.append(dict(
                    anchor_id=f"libero10_t{tid}_fa{j}_{L}", base_id=f"libero10_t{tid}_fa{j}", level=L, family="A_obj",
                    suite=SUITE, task_id=tid, task=task.name, stage_kind=c["stage_kind"], seed=int(seeds[j]),
                    settle_steps=SETTLE, resolution=RES, native=True, init_index=None,
                    bddl_file=rel, bddl_sha256=bddls[(tid, L)]["sha256"], hook=BDDL_HOOK,
                    displacement=dict(object=c["obj"], region=c["region"], direction=c["direction"],
                                      beyond_boundary_m=d, native_ranges=list(rng), rewritten_ranges=list(new_rng),
                                      target_region_xy=None if tgt is None else list(tgt),
                                      sham=(L == "A0"))))
        bddls[(tid, "original")] = dict(path=os.path.relpath(src_path, ROOT), sha256=sha256_text(src))
    ars = np.random.RandomState(MASTER + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(R_STEPS) % 60 < 30, -1.0, 1.0)
    report, prev = [], None
    for a in anchors:
        SA = run(suite, a, actions); SB = run(suite, a, actions)
        dAB = maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values()); dD = None
        if prev is not None:
            _, envP = run(suite, prev, actions, keep_open=True)
            SD = run(suite, a, actions); envP.close()
            dD = maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        row = dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                   trajectory_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                            for s in SA for k in sorted(s))).hexdigest(),
                   init_success=bool(SA[0]["succ"][0]))
        if a["level"] == "A0":         # sham: hook path vs untouched native path (original BDDL), same seed
            orig = os.path.join(ROOT, bddls[(a["task_id"], "original")]["path"])
            dS = maxdiff(SA, run(suite, a, actions, bddl_abs=orig))
            row["sham_vs_untouched_native"] = dS
            row["sham_pass"] = all(v == 0.0 for v in dS.values())
        row["placement"] = place_and_reach(suite, a, a["displacement"]["object"])
        report.append(row)
        print(a["anchor_id"], "PASS" if ok else "FAIL", "sham" if a["level"] == "A0" else "",
              row.get("sham_pass", ""), {k: row["placement"][k] for k in ("settled_xyz", "drift_cm", "planar_dist_to_base_m",
                                                                        "other_contacts", "reach_min_dist_cm")}, flush=True)
        prev = a
    # other objects identical across levels for the same seed?
    by = {}
    for a, r in zip(anchors, report):
        by.setdefault(a["base_id"], {})[a["level"]] = r["placement"]["others_initial"]
    cross = {b: {L: max(float(np.max(np.abs(np.array(v[k]) - np.array(lv["A0"][k])))) for k in v)
                 for L, v in lv.items() if L != "A0"} for b, lv in by.items()}
    bank = dict(family="A_obj", suite=SUITE, master_seed=MASTER, anchors_per_task=N, settle_steps=SETTLE, resolution=RES,
                restore_recipe="native_env.NativeLiberoEnv(init_states=False, bddl_file=<level BDDL>, hard_reset=True): "
                               "new env per restore -> reset(seed) [native placement sampler] -> settle_steps x dummy action",
                hook=BDDL_HOOK, levels={str(t): CFG[t]["levels"] for t in CFG},
                directions={str(t): CFG[t]["direction"] for t in CFG},
                bddl={f"t{t}_{L}": v for (t, L), v in bddls.items()}, anchors=anchors)
    p = f"{OUT}/familyA_anchors.json"
    json.dump(bank, open(p, "w"), indent=1)
    for tid in (5, 3):
        ids = sorted({a["base_id"] for a in anchors if a["task_id"] == tid}, key=lambda s: int(s.split("fa")[1]))
        json.dump(dict(note="family A base anchor ids (s0_id); the level is the setting", source=os.path.relpath(p, ROOT),
                       source_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                       entries=[dict(s0_id=i) for i in ids]), open(f"{OUT}/familyA_entries_t{tid}.json", "w"), indent=1)
    allp = all(r["pass_"] for r in report) and all(r.get("sham_pass", True) for r in report)
    json.dump(dict(all_pass=allp, tolerance=0.0, replay_steps=R_STEPS, results=report, other_objects_vs_A0_maxdiff=cross,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/familyA_verify.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", hashlib.sha256(open(p, "rb").read()).hexdigest())
    print("other objects vs A0 (max abs diff per base anchor/level):", json.dumps(cross))


if __name__ == "__main__":
    main()
