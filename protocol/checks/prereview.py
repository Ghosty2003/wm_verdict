"""Judgeability pre-review for the three new tasks (ruling 2026-11-01; deliverable BEFORE any bank step is released):

  t2 (LIBERO-10 #2)  stove on/off: 5 frozen T2 anchors, each rendered with the knob at its native (off) qpos and
                     forced to 1.3 (on); LIBERO Turnon flag recorded; changed-pixel box located.
  t4 (LIBERO-10 #4)  two mugs on two plates: 5 init states, reset frame (not in place) vs the same state with both
                     mugs teleported onto their plates and settled; both On predicates must flip to True.
  t6 (LIBERO-10 #6)  mug on plate + pudding in the fixed table region plate_right_region: same construction.

Per task: 10 contrast frames (5 pairs) at the LingBot view resolution 128 (shown 4x), a contact sheet, zoom crops of
the changed region, and report.json (predicate flags, changed-pixel stats). Every env lives in its own subprocess
(a second EGL context in one process crashes at teardown).

  python prereview.py   (lerobot_libero interpreter, GPU/EGL)  -> out/task_prereview/<task>/
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
HERE = f"{ROOT}/libero/out/task_prereview"
ZERO = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
SETTLE = 15


def _free_joint(m, bid):
    import mujoco
    j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(j) == 1, (bid, j)
    return m.jnt_qposadr[j[0]], m.jnt_dofadr[j[0]]


def pair_job(args):
    """one subprocess = one env: returns (tag, frame_before, frame_after, flags_before, flags_after)."""
    import mujoco
    from libero_env_adapter import _make_env
    task, anchor = args
    env = _make_env(anchor, H=40); u = env.envs[0]; p = u._env.env
    sim = u._env.sim; m, d = sim.model._model, sim.data._data

    def frame():
        o = p._get_observations(force_update=True)
        return np.asarray(o["agentview_image"]).copy()

    def flags():
        return {" ".join(map(str, g)): bool(p._eval_predicate(g)) for g in p.parsed_problem["goal_state"]}

    f0, g0 = frame(), flags()
    if task == "t2":
        for j in p.get_object("flat_stove_1").joints:
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = 1.3
        mujoco.mj_forward(m, d)
    else:
        place = {"t4": [("porcelain_mug_1", ("body", "plate_1")), ("white_yellow_mug_1", ("body", "plate_2"))],
                 "t6": [("porcelain_mug_1", ("body", "plate_1")), ("chocolate_pudding_1", ("site", "living_room_table_plate_right_region"))]}[task]
        for obj, (kind, target) in place:
            qa, va = _free_joint(m, p.obj_body_id[obj])
            if kind == "body":
                t = d.xpos[p.obj_body_id[target]].copy(); t[2] += 0.06
            else:
                t = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, target)].copy(); t[2] += 0.04
            d.qpos[qa:qa + 3] = t; d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE):
            u.step(ZERO)
    f1, g1 = frame(), flags()
    env.close()
    return anchor["anchor_id"], f0, f1, g0, g1


def main():
    import multiprocessing as mp
    from PIL import Image, ImageDraw
    bank2 = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors_t2.json"))["anchors"]
    jobs = [("t2", a) for a in bank2]
    for task, tid in (("t4", 4), ("t6", 6)):
        jobs += [(task, dict(anchor_id=f"libero10_{task}_pre{i}", suite="libero_10", task_id=tid, init_index=i,
                             seed=20261101, settle_steps=10, resolution=128)) for i in range(5)]
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        results = list(pool.imap(pair_job, jobs))
    rep = {}
    for (task, _), (aid, f0, f1, g0, g1) in zip(jobs, results):
        od = f"{HERE}/{task}"; os.makedirs(od, exist_ok=True)
        diff = np.abs(f1.astype(int) - f0.astype(int)).sum(-1)
        ys, xs = np.where(diff > 30)
        bb = [int(x) for x in (xs.min(), ys.min(), xs.max(), ys.max())] if len(ys) else None
        rep.setdefault(task, {})[aid] = dict(flags_before=g0, flags_after=g1, changed_pixels=int(len(ys)), bbox=bb)
        tiles = []
        for f, lab in ((f0, "before"), (f1, "after")):
            im = Image.fromarray(f).resize((512, 512), Image.NEAREST).convert("RGB")
            dr = ImageDraw.Draw(im)
            if bb:
                dr.rectangle([4 * bb[0] - 5, 4 * bb[1] - 5, 4 * bb[2] + 5, 4 * bb[3] + 5], outline=(255, 0, 0), width=2)
            dr.text((6, 6), f"{aid} {lab}", fill=(255, 255, 0))
            im.save(f"{od}/{aid}_{lab}.png"); tiles.append(im)
        if bb:
            cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2; r = max(14, (bb[2] - bb[0]) // 2 + 8)
            for f, lab in ((f0, "before"), (f1, "after")):
                c = Image.fromarray(f).crop((max(0, cx - r), max(0, cy - r), cx + r, cy + r))
                c.resize((c.width * 6, c.height * 6), Image.NEAREST).save(f"{od}/{aid}_zoom_{lab}.png")
    for task in rep:
        od = f"{HERE}/{task}"
        ims = [Image.open(f"{od}/{aid}_{lab}.png") for aid in rep[task] for lab in ("before", "after")]
        W, Hh = ims[0].size; sheet = Image.new("RGB", (W * 2, Hh * len(rep[task])), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * W, (i // 2) * Hh))
        sheet.save(f"{od}/sheet.png")
    json.dump(rep, open(f"{HERE}/report.json", "w"), indent=1)
    for task, R in rep.items():
        ok = all((not any(r["flags_before"].values())) and all(r["flags_after"].values()) for r in R.values())
        px = [r["changed_pixels"] for r in R.values()]
        print(f"PREREVIEW {task}: predicate_flips_all={ok} changed_pixels min/median/max = {min(px)}/{int(np.median(px))}/{max(px)}")
    print("PREREVIEW_DONE")


if __name__ == "__main__":
    main()
