"""Physics / action-stream perturbation families that do NOT enter the model input (perturbation registry E, G, N1, NR,
NL; ruling 2026-10-26: open-loop imagination reused from the main-bank baseline, truth replay and closed loop rerun).

Every family has a sham level (x1 / no drop / no delay) that must reproduce the unperturbed path bit for bit.

  Em  target-object mass x 1 / 1.5 / 2 / 4 / 8          body_mass and body_inertia of every body in the object's subtree,
                                                         body_subtreemass updated by the same delta (no mj_setConst)
  Ef  contact friction x 1 / 0.8 / 0.6 / 0.4 / 0.2     MuJoCo combines a contact's friction as the MAX of the two geoms
                                                         (object 0.95, Panda finger pads 2.0, table 1.0), so scaling the
                                                         object alone changes nothing. To scale the two registry contact
                                                         pairs (object-gripper, object-support) by the same factor, the
                                                         friction of the object geoms, the finger-pad geoms and the table
                                                         geoms is scaled together (side effect: pads / table against other
                                                         objects scale too; recorded). Verified by contact readout.
  G   arm controller gain x 1 / 0.9 / 0.75 / 0.55 / 0.4  OSC kp x f, kd x sqrt(f) (damping ratio kept)
  N1  zero action at env step 50 (1-based), then resume
  NR  each step independently replaced by the zero action with p = 0 / 5 / 10 / 20 / 30 / 40 %; mask from a recorded
      seed (crc32 of unit id and level), the same mask for both executions of a trial
  NL  execute the command issued d = 0..5 env steps earlier; zero action while the history is short
  NLw same, d = 0 / 4 / 8 / 16 / 32 / 64 env steps (0.2 - 3.2 s; ruling 2026-10-28)
Zero action = 7 zeros: OSC delta pose 0 (hold) and gripper sign 0 (robosuite PandaGripper keeps its current target).
"""
import zlib
import numpy as np

FAMILIES = {
    "Em": dict(kind="mass", levels=[1.0, 1.5, 2.0, 4.0, 8.0]),
    "Ef": dict(kind="friction", levels=[1.0, 0.8, 0.6, 0.4, 0.2]),
    "G": dict(kind="gain", levels=[1.0, 0.9, 0.75, 0.55, 0.4]),
    "N1": dict(kind="drop1", levels=[None, 50]),
    "NR": dict(kind="dropr", levels=[0.0, 0.05, 0.10, 0.20, 0.30, 0.40]),
    "NL": dict(kind="delay", levels=[0, 1, 2, 3, 4, 5]),
    # NLw (ruling 2026-10-28; paused and resumed 2026-10-29): delay in ENV steps (same physical lag for every model:
    # 1 step = 0.05 s at 20 Hz), wide levels spanning both models' decision intervals (LingBot-VA re-plans every 16
    # steps, OpenWAM-alpha every 32); NL (1..5) kept as the short-delay evidence. Read in closed loop only: in the
    # open-loop replay a delay is a pure time shift of a fixed action sequence, reported as a reference, not counted.
    "NLw": dict(kind="delay", levels=[0, 4, 8, 16, 32, 64]),
    # K (closed loop only; ruling 2026-10-26): push the target object by d cm along family A's direction right after
    # env step t, t in {60, 160, 260} (absolute, both tasks), d = 2 / 5 / 10 cm; level 0 = sham (no push)
    "K": dict(kind="push", levels=[None] + [(t, d) for t in (60, 160, 260) for d in (0.02, 0.05, 0.10)], closed_loop_only=True),
}
TRUTH_FAMILIES = [f for f, v in FAMILIES.items() if not v.get("closed_loop_only")]
PUSH_DIR = {5: np.array([0.0, 1.0, 0.0]), 3: np.array([1.0, 0.0, 0.0]),     # = family A directions (t5 book +y, t3 bowl +x)
            2: np.array([1.0, 0.0, 0.0]),                                    # t2 moka pot +x (familyA_t2_bank, 2026-11-02)
            4: np.array([1.0, 0.0, 0.0]), 6: np.array([0.0, -1.0, 0.0])}     # t4 mug +x / t6 mug -y (= family-A probed directions, 2026-11-03)
OBJECT = {5: "black_book_1", 3: "akita_black_bowl_1", 2: "moka_pot_1", 4: "porcelain_mug_1", 6: "porcelain_mug_1"}


def settings(fam):
    return [f"{fam}{i}" if fam != "N1" else f"N1_{i}" for i in range(len(FAMILIES[fam]["levels"]))]


def spec_of(setting, task_id, unit_id=""):
    fam = setting.split("_")[0] if setting.startswith("N1") else setting.rstrip("0123456789")
    i = int(setting.split("_")[1]) if setting.startswith("N1") else int(setting[len(fam):])
    F = FAMILIES[fam]; v = F["levels"][i]
    s = dict(family=fam, kind=F["kind"], level=i, value=v, sham=(i == 0), object=OBJECT[task_id])
    if F["kind"] == "push" and v is not None:
        s.update(push_after_step=v[0], push_m=v[1], push_dir=PUSH_DIR[task_id].tolist())
    if F["kind"] == "dropr":
        s["mask_seed"] = zlib.crc32(f"{unit_id}|{setting}".encode())
    return s


def _subtree(m, root):
    out, stack = [], [root]
    while stack:
        b = stack.pop(); out.append(b)
        stack += [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
    return out


def support_geoms(u, obj_geoms):
    """non-object geoms in contact with the object right after reset (the supporting surface)."""
    d = u._env.sim.data._data; obj = set(obj_geoms); out = set()
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if g1 in obj and g2 not in obj:
            out.add(g2)
        elif g2 in obj and g1 not in obj:
            out.add(g1)
    return sorted(out)


def apply_physics(u, spec):
    """call right after env reset (LeRobot LiberoEnv u = env.envs[0]); returns a dict describing what was changed."""
    import mujoco
    k, f = spec["kind"], spec["value"]
    p = u._env.env; m = u._env.sim.model._model
    if k == "mass":
        bid = p.obj_body_id[spec["object"]]; bodies = _subtree(m, bid)
        delta = 0.0
        for b in bodies:
            delta += (f - 1.0) * m.body_mass[b]
            m.body_mass[b] *= f; m.body_inertia[b] *= f
        for b in bodies:                                   # subtree masses inside the object
            m.body_subtreemass[b] *= f
        a = m.body_parentid[bid]
        while True:                                        # ancestors gain the same delta
            m.body_subtreemass[a] += delta
            if a == 0:
                break
            a = m.body_parentid[a]
        return dict(bodies=len(bodies), mass_delta_kg=delta)
    if k == "friction":
        bid = p.obj_body_id[spec["object"]]; bodies = set(_subtree(m, bid))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        obj = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
        pads = [g for g in range(m.ngeom) if "pad_collision" in names[g] and names[g].startswith("gripper0_")]
        table = support_geoms(u, obj)                      # whatever the object rests on at reset (t5 / t3 scenes differ)
        assert pads and table, (pads, table)
        for g in obj + pads + table:
            m.geom_friction[g, 0] *= f                     # sliding friction only
        return dict(object_geoms=len(obj), pad_geoms=len(pads),
                    support_geoms=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}" for g in table])
    if k == "gain":
        c = p.robots[0].controller
        c.kp = c.kp * f; c.kd = c.kd * np.sqrt(f)
        return dict(kp=float(c.kp[0]), kd=float(c.kd[0]))
    return {}


def maybe_push(u, spec, steps_done):
    """K family: right after env step `steps_done` == spec['push_after_step'], move the target object's free joint by
    push_m along push_dir, zero its velocity, mj_forward. Returns True when the push happened."""
    if spec.get("kind") != "push" or spec.get("value") is None or steps_done != spec["push_after_step"]:
        return False
    import mujoco
    p = u._env.env; sim = u._env.sim; m, d = sim.model._model, sim.data._data
    bid = p.obj_body_id[spec["object"]]
    j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(j) == 1, j
    qa, va = m.jnt_qposadr[j[0]], m.jnt_dofadr[j[0]]
    d.qpos[qa:qa + 3] += spec["push_m"] * np.asarray(spec["push_dir"]); d.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(m, d)
    return True


class ActionStream:
    """maps the commanded action at env step t (0-based) to the executed action."""

    def __init__(self, spec, H=400):
        self.spec, self.k = spec, spec["kind"]; self.hist = []
        self.mask = (np.random.RandomState(spec["mask_seed"]).rand(H + 200) < spec["value"]) if self.k == "dropr" else None

    def __call__(self, t, a):
        a = np.asarray(a, np.float32); self.hist.append(a)
        z = np.zeros(7, np.float32)
        if self.k == "drop1":
            return z if (self.spec["value"] is not None and t == self.spec["value"] - 1) else a
        if self.k == "dropr":
            return z if self.mask[t] else a
        if self.k == "delay":
            d = self.spec["value"]
            return self.hist[t - d] if t >= d else z
        return a
