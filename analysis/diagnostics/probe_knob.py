"""T2 (LIBERO-10 #2, stove + moka pot) prerequisite probe (2026-11-01): is "stove on" visible at 128 px?

Renders the same reset with the stove knob joint forced to OFF (native qpos) and ON (every candidate angle inside the
turnon range), at 128 and 256, agentview and wrist; checks that LIBERO's Turnon predicate flips; saves full frames,
knob-centred crops and a side-by-side sheet, plus report.json (joint names, qpos, predicate readings, pixel diff
statistics between off/on frames).

  python probe_knob.py        (lerobot_libero interpreter, GPU/EGL)   -> out/task2/probe_knob/
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
OUT = f"{ROOT}/libero/out/task2/probe_knob"
ANGLES = [0.0, 0.5, 1.0, 1.5, 2.1]                               # turnon range [0.5, 2.1]; 0.0 = off


def frames(res):
    import mujoco
    from libero_env_adapter import _make_env
    from ow_view import raw_obs
    a = dict(suite="libero_10", task_id=2, init_index=0, seed=20261101, settle_steps=10, resolution=res)
    env = _make_env(a, H=10); u = env.envs[0]; p = u._env.env
    sim = u._env.sim; m, d = sim.model._model, sim.data._data
    stove = p.get_object("flat_stove_1")
    joints = list(stove.joints)
    info = dict(res=res, joints=joints, qpos0={}, shots={})
    knob = [j for j in joints if m.jnt_type[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] == mujoco.mjtJoint.mjJNT_HINGE] or joints
    shots = {}
    for ang in ANGLES:
        for j in knob:
            qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            if ang == ANGLES[0]:
                info["qpos0"][j] = float(d.qpos[qa])
            d.qpos[qa] = ang
        mujoco.mj_forward(m, d)
        o = raw_obs(u)
        on = bool(p.object_states_dict["flat_stove_1"].turn_on())
        shots[ang] = (o["agentview_image"], o["robot0_eye_in_hand_image"])
        info["shots"][str(ang)] = dict(turn_on=on)
    env.close()
    return info, shots, knob


def main():
    from PIL import Image, ImageDraw
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for res in (128, 256):
        info, shots, knob = frames(res)
        base = shots[ANGLES[0]][0].astype(int)
        for ang, (av, wr) in shots.items():
            Image.fromarray(av).save(f"{OUT}/agent_{res}_q{ang}.png")
            diff = np.abs(av.astype(int) - base)
            ys, xs = np.where(diff.sum(-1) > 30)
            info["shots"][str(ang)]["diff_pixels"] = int(len(ys))
            info["shots"][str(ang)]["diff_bbox"] = [int(x) for x in (xs.min(), ys.min(), xs.max(), ys.max())] if len(ys) else None
        # side-by-side sheet (agentview, 4x) with the changed-pixel box drawn on the ON frames
        tiles = []
        for ang in ANGLES:
            im = Image.fromarray(shots[ang][0]).resize((res * 4, res * 4), Image.NEAREST).convert("RGB")
            dr = ImageDraw.Draw(im); bb = info["shots"][str(ang)]["diff_bbox"]
            if bb:
                dr.rectangle([4 * bb[0] - 6, 4 * bb[1] - 6, 4 * bb[2] + 6, 4 * bb[3] + 6], outline=(255, 0, 0), width=3)
            dr.text((8, 8), f"q = {ang}  turn_on = {info['shots'][str(ang)]['turn_on']}", fill=(255, 255, 0))
            tiles.append(im)
        W = sum(t.width for t in tiles); sheet = Image.new("RGB", (W, tiles[0].height), "white"); x = 0
        for t in tiles:
            sheet.paste(t, (x, 0)); x += t.width
        sheet.save(f"{OUT}/sheet_{res}.png")
        # knob-centred crop pair at the largest angle
        bb = info["shots"][str(ANGLES[-1])]["diff_bbox"]
        if bb:
            cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2; r = max(16, (bb[2] - bb[0]) // 2 + 10)
            for ang in (ANGLES[0], ANGLES[-1]):
                crop = Image.fromarray(shots[ang][0]).crop((max(0, cx - r), max(0, cy - r), cx + r, cy + r))
                crop.resize((crop.width * 6, crop.height * 6), Image.NEAREST).save(f"{OUT}/knob_{res}_q{ang}.png")
        report[res] = info
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    ok = report[128]["shots"]["0.0"]["turn_on"] is False and report[128]["shots"]["2.1"]["turn_on"] is True
    print("PROBE_DONE predicate_flips", ok, json.dumps({a: report[128]["shots"][a] for a in report[128]["shots"]}))


if __name__ == "__main__":
    main()
