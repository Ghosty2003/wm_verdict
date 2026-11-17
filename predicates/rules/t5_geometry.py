"""t5 success geometry made visible (2026-10-21, follow-up to ruling 2).

LIBERO's t5 goal In(black_book_1, desk_caddy_1_back_contain_region) is a POINT test: the book body origin must lie
strictly inside the region's axis-aligned box (site centre +- |R @ size|, lower z bound relaxed by 1 cm;
site_object.py in_box; check_contact of a site is always True). The book's tilt and how far it sticks out do not matter.

This script reads the simulator state of real replays and projects into the agentview image (128 x 128, the left half
of the LingBot-VA model view):
  green  = top face of the back-compartment box (z = upper bound)
  red    = "ceiling": the same rectangle raised by the upright book's half height (centre-to-top). An UPRIGHT book is
           in the box iff its top edge is below the red line (and it is inside the green rectangle's footprint).
  dot    = book origin (real frames only; green if inside the box)

  python t5_geometry.py verify      # orientation check on a stored v1 rollout -> out/content_review/geometry/verify.png
  python t5_geometry.py real        # real replays (famA/famP t5): state statistics + visual-rule agreement
  python t5_geometry.py overlay     # draw the box / ceiling on every t5 false-safe imagined endpoint
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from libero_env_adapter import _make_env, _anchor

OUT = f"{ROOT}/libero/out/content_review/geometry"
BOOK, REGION = "black_book_1", "desk_caddy_1_back_contain_region"
RES = 128
FLIP = None          # image orientation of the stored model-view frames relative to the pinhole projection; set by `verify`


class Geo:
    def __init__(self, u):
        self.u = u; self.p = u._env.env; sim = u._env.sim
        self.m, self.d = sim.model._model, sim.data._data
        self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        self.bid = self.p.obj_body_id[BOOK]
        self.book_geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.bid
                           and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        assert self.book_geoms, "book has no mesh geoms"

    def state(self):
        d, sim = self.d, self.u._env.sim
        site = self.p.object_sites_dict[REGION]
        c = sim.data.get_site_xpos(REGION).copy(); R = sim.data.get_site_xmat(REGION).copy()
        half = np.abs(R @ site.size); lb = c - half; ub = c + half; lb[2] -= 0.01
        centre = d.xpos[self.bid].copy()
        zs = []
        for g in self.book_geoms:
            mid = self.m.geom_dataid[g]; a = self.m.mesh_vertadr[mid]; n = self.m.mesh_vertnum[mid]
            v = self.m.mesh_vert[a:a + n]
            zs.append((d.geom_xpos[g][None] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].max())
        inside = bool(np.all(centre > lb) and np.all(centre < ub))
        return dict(lb=lb, ub=ub, centre=centre, top_z=float(max(zs)), inside=inside)

    def project(self, P):
        d, m = self.d, self.m
        cp = d.cam_xpos[self.cam]; R = d.cam_xmat[self.cam].reshape(3, 3)
        f = (RES / 2) / np.tan(np.deg2rad(m.cam_fovy[self.cam]) / 2)
        rel = (np.atleast_2d(P) - cp) @ R
        u = RES / 2 + f * rel[:, 0] / -rel[:, 2]; v = RES / 2 - f * rel[:, 1] / -rel[:, 2]
        if FLIP in ("h", "hv"):
            u = RES - u
        if FLIP in ("v", "hv"):
            v = RES - v
        return np.stack([u, v], 1)


def rect(lb, ub, z):
    return np.array([[lb[0], lb[1], z], [ub[0], lb[1], z], [ub[0], ub[1], z], [lb[0], ub[1], z]])


def draw(img, geo, st, half_up, centre=True, scale=3):
    """img: model-view frame (128 x 256); draws on the agentview half, returns the upscaled agentview crop."""
    im = Image.fromarray(img[:, :RES]).resize((RES * scale, RES * scale), Image.NEAREST); dr = ImageDraw.Draw(im)
    for z, col in ((st["ub"][2], (0, 220, 0)), (st["ub"][2] + half_up, (230, 30, 30))):
        q = geo.project(rect(st["lb"], st["ub"], z)) * scale
        dr.polygon([tuple(x) for x in q], outline=col)
    if centre:
        c = geo.project(st["centre"])[0] * scale
        col = (0, 220, 0) if st["inside"] else (230, 30, 30)
        dr.ellipse([c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4], fill=col)
    return im


def v1_anchor(z):
    bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))
    tpl = [a for a in bank["anchors"] if a["task_id"] == 5][0]
    return dict(tpl, init_index=int(z["init_index"]), seed=int(z["seed"]))


def replay_states(a, acts, steps):
    env = _make_env(a, H=len(acts)); u = env.envs[0]; geo = Geo(u)
    out = {0: (geo.state(), None)} if 0 in steps else {}
    half_up = None
    for t, x in enumerate(acts, 1):
        u.step(np.asarray(x, np.float32))
        if t in steps:
            out[t] = geo.state()
    return env, u, geo, out


def verify():
    global FLIP
    os.makedirs(OUT, exist_ok=True)
    f = sorted(os.listdir(f"{ROOT}/libero/out/predicate/t5/rollouts"))[0]
    z = np.load(f"{ROOT}/libero/out/predicate/t5/rollouts/{f}")
    a = v1_anchor(z); acts = z["actions"]; lab = z["labels"]; fs = int(np.argmax(lab))
    steps = [0, fs - 12, fs, fs + 24]
    env = _make_env(a, H=len(acts)); u = env.envs[0]; geo = Geo(u)
    S = {0: geo.state()}
    for t, x in enumerate(acts, 1):
        u.step(np.asarray(x, np.float32))
        if t in steps:
            S[t] = geo.state()
    half_up = S[0]["top_z"] - S[0]["centre"][2]
    rows = []
    for flip in (None, "h", "v", "hv"):
        FLIP = flip
        rows.append([draw(z["frames"][t], geo, S[t], half_up) for t in steps])
    env.close()
    W, H = rows[0][0].size
    page = Image.new("RGB", (W * len(steps), (H + 14) * 4), "white"); dr = ImageDraw.Draw(page)
    for r, (flip, row) in enumerate(zip((None, "h", "v", "hv"), rows)):
        for c, (t, im) in enumerate(zip(steps, row)):
            page.paste(im, (c * W, r * (H + 14) + 14))
            dr.text((c * W + 3, r * (H + 14)), f"flip={flip} t={t} sim={'S' if lab[t] else 'f'} inside={S[t]['inside']}", fill="black")
    page.save(f"{OUT}/verify.png")
    print(json.dumps({t: dict(inside=S[t]["inside"], sim=bool(lab[t]), top_z=round(S[t]["top_z"], 4),
                              centre_z=round(float(S[t]["centre"][2]), 4), ub_z=round(float(S[t]["ub"][2]), 4)) for t in steps}),
          "half_up", round(half_up, 4))


def _units():
    for run, bank in (("lingbot_famA", "familyA"), ("lingbot_famP", "familyP")):
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t5.jsonl"):
            r = json.loads(l)
            yield r, _anchor(f"{ROOT}/libero/out/bank/{bank}/{bank}_anchors.json", r["anchor_id"])


def real():
    """visual-rule agreement on real replays: rule = book top below the ceiling (box top + upright half height) and the
    book origin's xy inside the box footprint, vs the simulator's point test. 100 units drawn with a fixed seed."""
    U = list(_units()); rs = np.random.RandomState(20261021); pick = sorted(rs.choice(len(U), 100, replace=False))
    tab = np.zeros((2, 2), int); tab_top_only = np.zeros((2, 2), int); half_ups = []
    for i in pick:
        r, a = U[i]; acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
        env = _make_env(a, H=len(acts)); u = env.envs[0]; geo = Geo(u); s0 = geo.state()
        half_up = s0["top_z"] - s0["centre"][2]; half_ups.append(half_up)
        for x in acts:
            u.step(np.asarray(x, np.float32)); s = geo.state()
            xy_in = bool(np.all(s["centre"][:2] > s["lb"][:2]) and np.all(s["centre"][:2] < s["ub"][:2]))
            top_ok = s["top_z"] < s["ub"][2] + half_up
            tab[int(s["inside"]), int(top_ok and xy_in)] += 1; tab_top_only[int(s["inside"]), int(top_ok)] += 1
        env.close()
    res = dict(units=len(pick), steps=int(tab.sum()), half_up_m=dict(min=float(min(half_ups)), max=float(max(half_ups))),
               rule="top_z < box_top_z + upright half height AND origin xy inside the box footprint",
               table_sim_x_rule=tab.tolist(),
               tpr=float(tab[1, 1] / tab[1].sum()), tnr=float(tab[0, 0] / tab[0].sum()),
               top_only=dict(table=tab_top_only.tolist(), tpr=float(tab_top_only[1, 1] / tab_top_only[1].sum()),
                             tnr=float(tab_top_only[0, 0] / tab_top_only[0].sum())))
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/real_rule_agreement.json", "w"), indent=1); print(json.dumps(res))


def overlay():
    """every t5 false-safe unit: imagined endpoint (f520) and real endpoint, both with the box top (green) and the
    ceiling (red); the real panel also carries the book origin (green dot = inside). Box from the unit's own state at
    step 0 (imagined panel: the imagination starts from that state) and at step 520 (real panel)."""
    os.makedirs(f"{OUT}/units", exist_ok=True)
    rev = {u["unit_id"]: u for u in json.load(open(f"{ROOT}/libero/out/content_review/initial_review.json"))["units"]}
    for r, a in _units():
        uid = r["unit_id"]
        if uid not in rev:
            continue
        acts = np.load(f"{ROOT}/{r['action_file']}")["env"]; tr = np.load(f"{ROOT}/{r['trajectory_file']}")
        env = _make_env(a, H=len(acts)); u = env.envs[0]; geo = Geo(u); s0 = geo.state()
        half_up = s0["top_z"] - s0["centre"][2]
        for x in acts:
            u.step(np.asarray(x, np.float32))
        s_end = geo.state()
        idx = list(tr["decoded_frames_idx"]); im_end = tr["decoded_frames_sampled"][idx.index(520)]
        real_end = np.load(f"{ROOT}/libero/out/content_review/units/{uid}.npz")["real_end_frame"]
        A = draw(im_end, geo, s0, half_up, centre=False); B = draw(real_end, geo, s_end, half_up, centre=True)
        env.close()
        page = Image.new("RGB", (A.width * 2 + 6, A.height + 16), "white"); dr = ImageDraw.Draw(page)
        page.paste(A, (0, 16)); page.paste(B, (A.width + 6, 16))
        dr.text((3, 2), f"{uid}  IMAGINED end f520 (agentview)", fill="black")
        dr.text((A.width + 9, 2), f"REAL end (sim: fail; origin inside={s_end['inside']})", fill="black")
        page.save(f"{OUT}/units/{uid}.png")
        print(uid, flush=True)


if __name__ == "__main__":
    {"verify": verify, "real": real, "overlay": overlay}[sys.argv[1]]()
