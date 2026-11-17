"""Truth (a) for the non-input families (E, G, N): open-loop replay of the main-bank baseline's stored imagined env
actions (first 400) under the perturbation; no new imagination (ruling 2026-10-26). Per unit and level: two replays
in fresh envs, bit-identical; sham level must equal the stored baseline replay bit for bit (check-8 analogue).
Friction family: contact readout checks that every object-pad / object-table contact's sliding friction equals
factor x the unperturbed combined value.

  python phys_truth.py probe            stability probe: zero-action holds at Em max / Ef min on all 10 anchors
  python phys_truth.py run [fam,...]    -> out/runs/<model>_phys/truth_t<k>.jsonl + trajectories/
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from phys_perturb import FAMILIES, TRUTH_FAMILIES, settings, spec_of, apply_physics, ActionStream

BANK = f"{ROOT}/libero/out/bank/libero_anchors.json"
H = 400
MODELS = {"lingbot": "lingbot_base", "openwam": "openwam_base"}


def _state(u):
    import mujoco
    m, d = u._env.sim.model._model, u._env.sim.data._data
    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return st


def _fric_ref(u, spec):
    """unperturbed combined sliding friction per (object geom, partner geom) for the two registry pairs."""
    import mujoco
    from phys_perturb import _subtree
    p = u._env.env; m = u._env.sim.model._model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    obj = {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(_subtree(m, p.obj_body_id[spec["object"]]))}
    from phys_perturb import support_geoms
    part = {g for g in range(m.ngeom) if "pad_collision" in names[g] and names[g].startswith("gripper0_")} | set(support_geoms(u, obj))
    return obj, part, m.geom_friction[:, 0].copy()


def replay(anchor, acts, spec):
    from libero_env_adapter import _make_env
    env = _make_env(anchor, H=len(acts)); u = env.envs[0]
    fr = None
    if spec and spec["kind"] == "friction":
        fr = _fric_ref(u, spec)
    info = apply_physics(u, spec) if spec else {}
    stream = ActionStream(spec, H) if spec else None
    S, succ, ex, ferr, nfc = [_state(u)], [], [], 0.0, 0
    for t, a in enumerate(acts):
        e = stream(t, a) if stream else np.asarray(a, np.float32)
        ex.append(e); u.step(e); succ.append(bool(u._env.check_success())); S.append(_state(u))
        if fr is not None:
            d = u._env.sim.data._data; obj, part, f0 = fr
            for i in range(d.ncon):
                c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in obj and g2 in part) or (g2 in obj and g1 in part):
                    want = spec["value"] * max(f0[g1], f0[g2]); ferr = max(ferr, abs(float(c.friction[0]) - want)); nfc += 1
    env.close()
    return np.stack(S), np.array(succ), np.stack(ex), info, (ferr, nfc)


def _unit(args):
    model, t, r, setting, anchor = args
    spec = spec_of(setting, int(t[1:]), r["unit_id"])
    acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
    S1, s1, ex, info, (ferr, nfc) = replay(anchor, acts, spec)
    S2, s2, _, _, _ = replay(anchor, acts, spec)
    same = bool(np.array_equal(S1, S2) and np.array_equal(s1, s2))
    sham_ok = None
    if spec["sham"]:
        ref = np.load(f"{ROOT}/{r['trajectory_file']}")["replay1"][: H + 1]
        sham_ok = bool(np.array_equal(S1, ref))
    uid = f"{r['unit_id']}__{setting}"
    out = f"{ROOT}/libero/out/runs/{model}_phys/trajectories/{uid}.npz"
    np.savez_compressed(out, states=S1, success=s1, executed=ex)
    first = int(np.argmax(s1)) + 1 if s1.any() else None
    return dict(unit_id=uid, base_unit=r["unit_id"], model=model, task=t, setting_id=setting, perturbation=spec,
                physics_change=info, real_verdict=bool(s1.any()), real_first_success_step=first, real_verdict_final=bool(s1[-1]),
                replay_check_identical=same, sham_bit_identical_to_baseline=sham_ok,
                friction_contact_max_abs_err=ferr if spec["kind"] == "friction" else None, friction_contacts_checked=nfc,
                executed_differs_from_commanded=int((np.abs(ex - acts).sum(1) > 0).sum()),
                trajectory_file=os.path.relpath(out, ROOT), action_file=r["action_file"], H_env=H)


def anchors():
    return {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}


def run(fams):
    import multiprocessing as mp
    A = anchors(); jobs = []
    for model, run_ in MODELS.items():
        os.makedirs(f"{ROOT}/libero/out/runs/{model}_phys/trajectories", exist_ok=True)
        for t in ("t5", "t3"):
            path = f"{ROOT}/libero/out/runs/{model}_phys/truth_{t}.jsonl"
            done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
            for l in open(f"{ROOT}/libero/out/runs/{run_}/records_{t}.jsonl"):
                r = json.loads(l)
                for fam in fams:
                    for s in settings(fam):
                        if f"{r['unit_id']}__{s}" not in done:
                            jobs.append((model, t, r, s, A[r["s0_id"]]))
    print("jobs", len(jobs), flush=True); t0 = time.time(); bad = []
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 10)), maxtasksperchild=20) as pool:
        for i, rec in enumerate(pool.imap_unordered(_unit, jobs, chunksize=1)):
            with open(f"{ROOT}/libero/out/runs/{rec['model']}_phys/truth_{rec['task']}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if not rec["replay_check_identical"] or rec["sham_bit_identical_to_baseline"] is False or \
               (rec["friction_contact_max_abs_err"] or 0) > 1e-9:
                bad.append(rec["unit_id"]); print("BAD", json.dumps(rec)[:400], flush=True)
            if i % 200 == 0:
                print(i, "done", round(time.time() - t0), "s", flush=True)
    print("DONE bad", len(bad), bad[:10])
    if bad:
        sys.exit("truth replay check failed -- STOP")


def probe():
    """zero-action hold, 400 steps, at the heaviest / slipperiest level, all 10 anchors: object drift and NaN."""
    from libero_env_adapter import _make_env
    res = {}
    for aid, a in anchors().items():
        tid = a["task_id"]
        for setting in ("Em0", "Em4", "Ef0", "Ef4"):
            env = _make_env(a, H=H); u = env.envs[0]; spec = spec_of(setting, tid)
            apply_physics(u, spec); p = u._env.env
            bid = p.obj_body_id[spec["object"]]; x0 = u._env.sim.data.body_xpos[bid].copy()
            for _ in range(H):
                u.step(np.zeros(7, np.float32))
            x1 = u._env.sim.data.body_xpos[bid].copy(); q = u._env.sim.data.qpos
            res[f"{aid}/{setting}"] = dict(drift_mm=round(float(np.linalg.norm(x1 - x0)) * 1000, 3), nan=bool(np.isnan(q).any()))
            env.close()
    ok = all(v["drift_mm"] < 5 and not v["nan"] for v in res.values())
    json.dump(dict(rule="zero-action hold 400 steps: object drift < 5 mm and no NaN", pass_=ok, results=res),
              open(f"{ROOT}/libero/out/runs/phys_probe.json", "w"), indent=1)
    print(json.dumps(res, indent=0)); print("PROBE PASS" if ok else "PROBE FAIL")


def pushcheck():
    """K family: right after `push_after_step` steps the target object's position jumps by push_m along push_dir."""
    from libero_env_adapter import _make_env
    from phys_perturb import maybe_push
    A = anchors(); ok = True
    for run_, t in (("lingbot_base", "t5"), ("lingbot_base", "t3")):
        r = json.loads(open(f"{ROOT}/libero/out/runs/{run_}/records_{t}.jsonl").readline())
        acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
        for s in ("K1", "K6", "K9"):
            spec = spec_of(s, int(t[1:])); env = _make_env(A[r["s0_id"]], H=H); u = env.envs[0]
            import mujoco
            m, d = u._env.sim.model._model, u._env.sim.data._data
            bid = u._env.env.obj_body_id[spec["object"]]; jumped = None
            qa = m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
            for i, a in enumerate(acts):
                # measured on the free joint's qpos: body_xpos after env.step is the kinematics of the last substep's
                # start (stale by one substep while the object moves), so it is not a valid "before" reading
                before = d.qpos[qa:qa + 3].copy()
                if maybe_push(u, spec, i):
                    jumped = d.qpos[qa:qa + 3] - before
                    print("  xpos - qpos after push", np.abs(u._env.sim.data.body_xpos[bid] - d.qpos[qa:qa + 3]).max(), flush=True)
                u.step(np.asarray(a, np.float32))
            env.close()
            good = jumped is not None and np.allclose(jumped, spec["push_m"] * np.asarray(spec["push_dir"]), atol=1e-9)
            ok &= bool(good); print("pushcheck", t, s, spec["push_after_step"], spec["push_m"], np.round(jumped, 6) if jumped is not None else None, good, flush=True)
    print("PUSHCHECK", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "pushcheck":
        pushcheck()
    elif sys.argv[1] == "probe":
        probe()
    else:
        run(sys.argv[2].split(",") if len(sys.argv) > 2 else TRUTH_FAMILIES)
