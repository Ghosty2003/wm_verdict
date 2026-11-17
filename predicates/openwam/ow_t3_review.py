"""OpenWAM-alpha t3: check 7 layer-1 content attribution (stratified) + truth-side sub-goal diagnosis for check 6.
Rulings 2026-10-25 (1: stratified attribution; 2: check-6 interpretation).

Attribution sample: 13 cells (base none, A0..A4, P0..P4), 8 false-safe units per cell (all if fewer), seed 20261041.
Sheet per unit = the established t3 triptych (review_criteria.md): (1) imagined end (step 400), (2) imagined first
trigger of the frozen predicate (k = 1), (3) REAL end (step 400, simulator: fail); each panel = agentview | wrist of the
OpenWAM canvas. Plus two imagined film strips from the stored canvases: coarse (every 20 steps, sub-label
bowl_entered_visibly / bowl_vanished as for LingBot) and dense around the trigger (every imagined frame = 4 steps, 48
steps before to 12 after; OpenWAM-specific observation: placing action drawn completely vs jumps).

Diagnosis (check 6): every t3 unit of famA / famP replayed (truth a, bit-exact replay of the stored env actions; success
trace asserted): per step the BDDL sub-goals In(bowl, bottom region) and Close(bottom drawer), bowl height above its
start, gripper-bowl distance.

  python ow_t3_review.py sample            (any python)
  python ow_t3_review.py replay            (lerobot_libero, GPU render) -> review_t3/units/*.npz, diag/*.npz
  python ow_t3_review.py sheets            (any python)  -> review_t3/sheets/*.png, cards/*.png
  python ow_t3_review.py diag              (any python)  -> review_t3/check6_diag.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from ow_view import canvas_from_raw, raw_obs

OUT = f"{ROOT}/libero/out/content_review/openwam_t3"
RUNS = {"base": ("openwam_base", f"{ROOT}/libero/out/bank/libero_anchors.json"),
        "famA": ("openwam_famA", f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"),
        "famP": ("openwam_famP", f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json")}
SEED = 20261041
PER_CELL = 8
H = 400


def records(fam):
    run, _ = RUNS[fam]
    return [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/{run}/records_t3.jsonl")]


def sample():
    os.makedirs(OUT, exist_ok=True)
    rs = np.random.RandomState(SEED); units, cells = [], {}
    for fam in ("base", "famA", "famP"):
        R = records(fam)
        for L in sorted({r["setting_id"] for r in R}):
            fs = sorted(r["unit_id"] for r in R if r["setting_id"] == L and r["model_verdict"] and not r["real_verdict"])
            pick = sorted(rs.choice(fs, min(PER_CELL, len(fs)), replace=False).tolist())
            cells[f"{fam}/{L}"] = dict(false_safe=len(fs), sampled=len(pick))
            units += [dict(unit=u, family=fam, cell=f"{fam}/{L}") for u in pick]
    json.dump(dict(seed=SEED, per_cell=PER_CELL, cells=cells, units=units), open(f"{OUT}/sample.json", "w"), indent=1)
    print(json.dumps(cells), len(units))


def _replay(args):
    """truth (a) replay of a unit's stored env actions: per-step sub-goals, bowl height, gripper-bowl distance; OpenWAM
    canvases at the requested steps."""
    from libero_env_adapter import _make_env, _anchor
    fam, r, steps = args
    a = dict(_anchor(RUNS[fam][1], r["anchor_id"]), resolution=256)
    acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
    env = _make_env(a, H=len(acts)); u = env.envs[0]; prob = u._env.env; sim = u._env.sim
    goals = prob.parsed_problem["goal_state"]
    bowl = [k for k in prob.obj_body_id if "bowl" in k]; assert len(bowl) == 1, bowl
    bid = prob.obj_body_id[bowl[0]]
    def bowl_pos():
        return sim.data.body_xpos[bid].copy()
    z0 = bowl_pos()[2]
    eid = prob.robots[0].eef_site_id; eid = eid["right"] if isinstance(eid, dict) else eid      # robosuite 1.4 / 1.5
    sub, succ, bz, gd, canv = [], [], [], [], {}
    for t in range(len(acts) + 1):
        if t:
            u.step(np.asarray(acts[t - 1], np.float32)); succ.append(bool(u._env.check_success()))
        o = raw_obs(u) if t in steps else None
        if o is not None:
            canv[t] = canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])
        sub.append([bool(prob._eval_predicate(g)) for g in goals]); p = bowl_pos(); bz.append(p[2] - z0)
        gd.append(float(np.linalg.norm(sim.data.site_xpos[eid] - p)))
    env.close()
    same = bool(np.array_equal(np.array(succ), np.load(f"{ROOT}/{r['trajectory_file']}")["real_success"]))
    return r["unit_id"], dict(sub=np.array(sub), goal_names=np.array([" ".join(g) for g in goals]), bowl_dz=np.array(bz),
                              grip_dist=np.array(gd), success_match=same,
                              **{f"canvas_{t}": c for t, c in canv.items()})


def replay():
    import multiprocessing as mp
    os.makedirs(f"{OUT}/units", exist_ok=True); os.makedirs(f"{OUT}/diag", exist_ok=True)
    smp = {u["unit"] for u in json.load(open(f"{OUT}/sample.json"))["units"]}
    jobs = []
    for fam in ("base", "famA", "famP"):
        for r in records(fam):
            if r["unit_id"] not in smp and fam == "base":
                continue
            steps = {H, int(r["model_first_success_step"])} if r["unit_id"] in smp else set()
            if not os.path.exists(f"{OUT}/diag/{r['unit_id']}.npz"):
                jobs.append((fam, r, steps))
    bad = []
    with mp.get_context("spawn").Pool(2, maxtasksperchild=20) as pool:
        for uid, d in pool.imap_unordered(_replay, jobs, chunksize=1):
            if not d["success_match"]:
                bad.append(uid)
            np.savez_compressed(f"{OUT}/diag/{uid}.npz", **{k: v for k, v in d.items() if not k.startswith("canvas_")})
            cv = {k: v for k, v in d.items() if k.startswith("canvas_")}
            if cv:
                np.savez_compressed(f"{OUT}/units/{uid}.npz", **cv)
    print("replayed", len(jobs), "success trace mismatches", bad)
    if bad:
        raise SystemExit("replay does not reproduce the recorded success trace -- stop")


# ------------------------------------------------------------------------------------------------ sheets
def _panel(canvas, text, w=480):
    from PIL import Image, ImageDraw
    c = np.asarray(canvas); top = Image.fromarray(np.ascontiguousarray(c[:256])); wr = Image.fromarray(np.ascontiguousarray(c[256:, :160]))
    im = Image.new("RGB", (320 + 160 + 4, 256 + 16), "white"); im.paste(top, (0, 16)); im.paste(wr, (324, 16))
    ImageDraw.Draw(im).text((3, 2), text, fill="black")
    return im if w == im.width else im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)


def _strip(frames, labels, per_row, tw):
    from PIL import Image, ImageDraw
    th = int(tw * 256 / 320); rows = (len(frames) + per_row - 1) // per_row
    im = Image.new("RGB", (per_row * (tw + 2), rows * (th + 12)), "white"); d = ImageDraw.Draw(im)
    for i, (f, lb) in enumerate(zip(frames, labels)):
        x, y = (i % per_row) * (tw + 2), (i // per_row) * (th + 12)
        im.paste(Image.fromarray(np.ascontiguousarray(np.asarray(f)[:256])).resize((tw, th), Image.LANCZOS), (x, y + 12)); d.text((x + 2, y), lb, fill="black")
    return im


def sheets():
    from PIL import Image, ImageDraw
    sm = json.load(open(f"{OUT}/sample.json"))
    recs = {r["unit_id"]: r for fam in RUNS for r in records(fam)}
    for d_ in ("sheets", "filmstrips", "cards"):
        os.makedirs(f"{OUT}/{d_}", exist_ok=True)
    cards = []
    for u in sm["units"]:
        r = recs[u["unit"]]; tr = np.load(f"{ROOT}/{r['trajectory_file']}"); IC = tr["imagined_canvas"]   # frame g = step 4g
        rc = np.load(f"{OUT}/units/{u['unit']}.npz"); trig = int(r["model_first_success_step"]); g = trig // 4
        P = [_panel(IC[100], "imagined end (step 400)"), _panel(IC[g], f"imagined first trigger, k=1 (step {trig})"),
             _panel(rc[f"canvas_{H}"], "REAL end (step 400), simulator: fail")]
        sh = Image.new("RGB", (sum(p.width for p in P), P[0].height + 18), "white")
        ImageDraw.Draw(sh).text((3, 2), f"{u['unit']}   [{u['cell']}]", fill="black"); x = 0
        for p in P:
            sh.paste(p, (x, 18)); x += p.width
        sh.save(f"{OUT}/sheets/{u['unit']}.png")
        coarse = _strip([IC[i] for i in range(0, 101, 5)], [f"s{4 * i}" for i in range(0, 101, 5)], 11, 128)
        dense_idx = list(range(max(0, g - 12), min(100, g + 3) + 1))
        dense = _strip([IC[i] for i in dense_idx], [f"s{4 * i}" + (" T" if i == g else "") for i in dense_idx], 8, 160)
        coarse.save(f"{OUT}/filmstrips/{u['unit']}_coarse.png"); dense.save(f"{OUT}/filmstrips/{u['unit']}_dense.png")
        W = 1300; parts = [sh.resize((W, int(sh.height * W / sh.width)), Image.LANCZOS),
                           dense.resize((W, int(dense.height * W / dense.width)), Image.LANCZOS),
                           coarse.resize((W, int(coarse.height * W / coarse.width)), Image.LANCZOS)]
        card = Image.new("RGB", (W, sum(p.height for p in parts) + 8), "white"); y = 0
        for p in parts:
            card.paste(p, (0, y)); y += p.height + 4
        card.save(f"{OUT}/cards/{u['unit']}.png")
    print("sheets", len(sm["units"]))


# ------------------------------------------------------------------------------------------------ check-6 diagnosis
def diag():
    out = {}
    for fam in ("famA", "famP"):
        R = records(fam); by = collections.defaultdict(collections.Counter); rows = collections.defaultdict(list)
        for r in R:
            z = np.load(f"{OUT}/diag/{r['unit_id']}.npz"); names = [str(x) for x in z["goal_names"]]
            iI = [i for i, n in enumerate(names) if n.lower().startswith("in")][0]; iC = [i for i, n in enumerate(names) if n.lower().startswith("close")][0]
            In, Cl = z["sub"][1:, iI], z["sub"][1:, iC]; lifted = bool(z["bowl_dz"].max() > 0.03)
            if r["real_verdict"]:
                m = "success"
            elif not lifted:
                m = "bowl_never_lifted"
            elif not In.any():
                m = "lifted_never_in_drawer"
            elif In[-1] and not Cl[-1]:
                m = "in_drawer_not_closed"
            elif In.any() and not In[-1]:
                m = "in_then_lost"
            else:
                m = "other"
            by[r["setting_id"]][m] += 1
            rows[r["setting_id"]].append(dict(min_grip_dist=float(z["grip_dist"].min()), max_bowl_dz=float(z["bowl_dz"].max())))
        out[fam] = {L: dict(modes=dict(c), min_grip_dist_median=float(np.median([x["min_grip_dist"] for x in rows[L]])),
                            lifted_share=float(np.mean([x["max_bowl_dz"] > 0.03 for x in rows[L]])))
                    for L, c in sorted(by.items())}
    json.dump(out, open(f"{OUT}/check6_diag.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"sample": sample, "replay": replay, "sheets": sheets, "diag": diag}[sys.argv[1]]()
