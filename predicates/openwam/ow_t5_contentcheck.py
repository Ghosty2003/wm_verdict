"""t5 OpenWAM-view predicate: content check of the generated-row false positives (user ruling 2026-10-24, ruling 1).

The generated-row gate failed (timing-rule TNR 0.728 < 0.85). Before ruling pending / not pending, check what the false
positive generated frames actually DEPICT: 80 FP frames, stratified 40 from source episodes that never succeed and 40
from starts before the episode's success whose chunk never succeeds (categories of the 2026-10-24 diagnostic), each as a
triptych (generated frame | real frame at the same step, the model's own actions executed in the simulator | real
success reference), with the T0 geometry reference lines (t5_geometry: green = top face of the back-compartment box,
red = ceiling = box top + upright book half height; dot = book origin on real frames). Label: does the picture show the
book in the compartment (T0 standard). First pass by the agent, Jasmine re-checks 20.

FPs are those of the GATE itself: the out-of-fold predictions of ow_predicate.train (5 folds, same seeds), recomputed
here and required to reproduce the stored generated rows exactly; they are saved for the later re-judge.

  oof      (lerobot_libero, GPU)  -> predicate/t5/contentcheck/gen_oof.npz   (stops if the stored rows do not reproduce)
  sample                          -> predicate/t5/contentcheck/sample.json   (seed 20261040)
  render   (lerobot_libero)       -> predicate/t5/contentcheck/units/*.png, pages/*.png, jasmine/
  fixflip  (any python, no GPU)   repairs the overlays of the first render (2026-10-24, job 185), which used the LingBot
                                  orientation (FLIP=None). The OpenWAM canvas is the raw frame rotated 180 deg, i.e.
                                  mirrored left-right vs the LingBot view, so the correct lines are the drawn ones
                                  mirrored (u -> 128 - u). Lines are taken from the burned panels and mirrored onto
                                  clean frames: generated = gen_out.npz, reference = the reference episode's stored frame,
                                  real same-step = recovered from the NEAREST x3 panel (each source pixel is a 3x3 block;
                                  blocks fully covered by the old dot are filled from neighbours). Old files -> units_flipNone/.
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
import ow_predicate as P
from ow_view import canvas_from_raw, pred_view, raw_obs

D = f"{P.OUT}/t5"
CC = f"{D}/contentcheck"
SAMPLE_SEED = 20261040
N_PER = 40
N_JASMINE = 20
STRATA = ("episode_never_succeeds", "start_before_success_chunk_never_succeeds")


def _first_succ():
    out = {}
    for e in P._registry(5):
        lab = np.load(f"{D}/episodes/{e['id']}.npz")["labels"]
        out[e["id"]] = int(np.argmax(lab)) if lab.any() else None
    return out


def oof():
    import torch
    from recal_v3 import _fit, _row
    from recal_predicate import fail_terminal_mask
    from lingbot_predicate import Net, predict
    os.makedirs(CC, exist_ok=True)
    E = P._registry(5); fold_of = {e["id"]: e["fold"] for e in E}
    X, Y, M, EP = [], [], [], []
    for e in E:
        z = np.load(f"{D}/episodes/{e['id']}.npz"); X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        EP += [e["id"]] * len(z["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(bool); M = np.concatenate(M); FO = np.array([fold_of[x] for x in EP])
    gn = dict(np.load(f"{D}/gen_all.npz")); fo_gn = np.array([fold_of[x] for x in gn["episode"]])
    oof_gn = np.zeros(len(gn["success"]))
    for k in range(5):                                       # = ow_predicate.train, t5 (V = all frames)
        net, loss = _fit(X[FO != k], Y[FO != k], M[FO != k], seed=k)
        oof_gn[fo_gn == k] = predict(net, gn["frames"][fo_gn == k]); print("fold", k, "loss", loss, flush=True)
        del net; torch.cuda.empty_cache()
    cal = json.load(open(f"{D}/calibration_success.json")); thr = cal["threshold"]
    first = _first_succ(); failed = np.array([first[x] is None for x in gn["episode"]])
    chunk_ref = {}
    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"]):
        if y and ((ep, t0) not in chunk_ref or s < chunk_ref[(ep, t0)]):
            chunk_ref[(ep, t0)] = int(s)
    win = np.array([(not y) and (chunk_ref.get((ep, t0), first[ep]) is not None)
                    and abs(int(s) - chunk_ref.get((ep, t0), first[ep])) <= P.WINDOW
                    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"])])
    y = gn["success"].astype(bool)
    rows = dict(generated_raw=_row(oof_gn, y, failed, thr), generated_timing_rule=_row(oof_gn[~win], y[~win], failed[~win], thr))
    same = all(json.dumps(rows[r], sort_keys=True) == json.dumps(cal[r], sort_keys=True) for r in rows)
    print("reproduced:", same, json.dumps(rows), flush=True)
    m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}")); m.eval()
    np.savez_compressed(f"{CC}/gen_oof.npz", oof=oof_gn, deployed=predict(m, gn["frames"]), window=win, success=y,
                        episode=gn["episode"], t0=gn["t0"], step=gn["step"], threshold=thr, reproduced=same)
    if not same:
        raise SystemExit("out-of-fold predictions do not reproduce the stored generated rows -- stop")


def category(i, g, first, chunk_succ):
    e = str(g["episode"][i]); fs = first[e]
    if chunk_succ[(e, int(g["t0"][i]))]:
        return "chunk_reaches_success"
    if fs is None:
        return STRATA[0]
    return STRATA[1] if g["t0"][i] < fs else "start_after_success_chunk_lost_it"


def sample():
    g = dict(np.load(f"{CC}/gen_oof.npz")); assert bool(g["reproduced"])
    thr = float(g["threshold"]); first = _first_succ()
    chunk_succ = collections.defaultdict(bool)
    for e, t0, y in zip(g["episode"], g["t0"], g["success"]):
        chunk_succ[(str(e), int(t0))] |= bool(y)
    fp = (g["oof"] >= thr) & ~g["success"]
    cat = np.array([category(i, g, first, chunk_succ) if not g["success"][i] else "positive" for i in range(len(fp))])
    counts = {c: dict(negatives=int(((cat == c)).sum()), fp=int((fp & (cat == c)).sum()),
                      fp_in_gate_row=int((fp & (cat == c) & ~g["window"]).sum()), fp_in_window=int((fp & (cat == c) & g["window"]).sum()))
              for c in sorted(set(cat) - {"positive"})}
    rs = np.random.RandomState(SAMPLE_SEED); units = []
    for c in STRATA:
        pool = np.where(fp & (cat == c))[0]
        pick = sorted(rs.choice(pool, min(N_PER, len(pool)), replace=False).tolist())
        for i in pick:
            units.append(dict(id=f"cc{len(units):02d}", index=int(i), stratum=c, episode=str(g["episode"][i]), t0=int(g["t0"][i]),
                              step=int(g["step"][i]), j=int((g["step"][i] - g["t0"][i]) // 4), p_oof=float(g["oof"][i]),
                              p_deployed=float(g["deployed"][i]), in_gate_row=bool(not g["window"][i]),
                              episode_first_success=first[str(g["episode"][i])]))
    jas = sorted(rs.choice(len(units), N_JASMINE, replace=False).tolist())       # Jasmine: 20 of the 80
    out = dict(seed=SAMPLE_SEED, threshold=thr, fp_definition="out-of-fold p >= threshold on a negative generated frame (the gate's own predictions)",
               category_counts=counts, units=units, jasmine=[units[i]["id"] for i in jas])
    json.dump(out, open(f"{CC}/sample.json", "w"), indent=1)
    print(json.dumps(counts, indent=1)); print("sampled", collections.Counter(u["stratum"] for u in units))


def _ref_episode(eid, first, E):
    """success reference: the source episode itself if it ever succeeds, else the first successful registry episode
    with the same anchor (same init state), else the first successful episode."""
    if first[eid] is not None:
        return eid
    anc = json.dumps(E[eid]["anc"], sort_keys=True)
    same = [k for k in sorted(E) if first[k] is not None and json.dumps(E[k]["anc"], sort_keys=True) == anc and not E[k].get("disp")]
    return (same or [k for k in sorted(E) if first[k] is not None])[0]


FLIP = "h"     # OpenWAM canvas = raw frame rotated 180 deg -> mirrored left-right vs the LingBot view (t5_geometry FLIP=None);
               # confirmed by `verify` (contentcheck/verify_flip.png: book origin of a real success frame on the back compartment)


def verify():
    """4 orientations on real success frames of two reference episodes (as t5_geometry.verify did for the LingBot view)."""
    import t5_geometry as T
    from PIL import Image, ImageDraw
    E = {e["id"]: e for e in P._registry(5)}; rows = []
    for rid in ("v1_r000", "v1_r001"):
        z = np.load(f"{D}/episodes/{rid}.npz"); acts, lab = z["actions"], z["labels"]; fs = int(np.argmax(lab))
        env, u = P._restore(E[rid]["anc"], E[rid].get("disp"), len(acts)); geo = T.Geo(u); s0 = geo.state(); hu = s0["top_z"] - s0["centre"][2]
        for t, a in enumerate(acts, 1):
            u.step(np.asarray(a, np.float32))
            if t == fs:
                break
        st = geo.state(); o = raw_obs(u); img = pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
        row = []
        for f in (None, "h", "v", "hv"):
            T.FLIP = f; row.append(T.draw(img, geo, st, hu, centre=True, scale=3))
        env.close(); rows.append((rid, fs, row))
    W = rows[0][2][0].width; page = Image.new("RGB", (W * 4, (W + 14) * len(rows)), "white"); dr = ImageDraw.Draw(page)
    for r, (rid, fs, row) in enumerate(rows):
        for c, (f, im) in enumerate(zip((None, "h", "v", "hv"), row)):
            page.paste(im, (c * W, r * (W + 14) + 14)); dr.text((c * W + 3, r * (W + 14)), f"{rid} t={fs} (first success) flip={f}", fill="black")
    os.makedirs(CC, exist_ok=True); page.save(f"{CC}/verify_flip.png")


def _panels(img, geo, st, half_up, centre):
    """agentview half with the geometry lines (drawn while the env is alive: Geo reads live MuJoCo data) + wrist half."""
    from PIL import Image
    import t5_geometry
    from t5_geometry import draw
    t5_geometry.FLIP = FLIP
    out = {}
    for sc in (2, 3):
        a = draw(img, geo, st, half_up, centre=centre, scale=sc)
        w = Image.fromarray(np.ascontiguousarray(img[:, 128:])).resize((64 * sc, 64 * sc), Image.BILINEAR)
        out[sc] = (a, w)
    return out


def _render_unit(args):
    from t5_geometry import Geo
    u_, ref_id = args
    E = {e["id"]: e for e in P._registry(5)}
    gen = np.load(f"{D}/gen_out.npz"); G_ep, G_t0 = gen["episode"], gen["t0"]
    ch = [i for i, (e, t) in enumerate(zip(G_ep, G_t0)) if e == u_["episode"] and t == u_["t0"]]
    assert len(ch) == 1
    gen_actions = gen["actions"][ch[0]]; gen_frame = gen["frames"][ch[0]][u_["j"] - 1]
    acts = np.load(f"{D}/episodes/{u_['episode']}.npz")["actions"]; e = E[u_["episode"]]
    env, u = P._restore(e["anc"], e.get("disp"), len(acts) + P.N_GEN); geo = Geo(u)
    s0 = geo.state(); half_up = s0["top_z"] - s0["centre"][2]
    for a in acts[:u_["t0"]]:
        u.step(np.asarray(a, np.float32))
    for a in gen_actions[:u_["step"] - u_["t0"]]:
        u.step(np.asarray(a, np.float32))
    sim_ok = bool(u._env.check_success()); st = geo.state()
    o = raw_obs(u); real = pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
    P_gen = _panels(gen_frame, geo, st, half_up, False); P_real = _panels(real, geo, st, half_up, True)
    env.close()
    # success reference: replay the reference episode to its first success
    r = E[ref_id]; z = np.load(f"{D}/episodes/{ref_id}.npz"); racts, rlab = z["actions"], z["labels"]; fs = int(np.argmax(rlab))
    env, u = P._restore(r["anc"], r.get("disp"), len(racts)); rgeo = Geo(u)
    r0 = rgeo.state(); r_half = r0["top_z"] - r0["centre"][2]
    for a in racts[:fs]:
        u.step(np.asarray(a, np.float32))
    assert bool(u._env.check_success()), f"reference {ref_id} not successful at its first-success step"
    rst = rgeo.state()
    o = raw_obs(u); ref = pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
    P_ref = _panels(ref, rgeo, rst, r_half, True)
    env.close()
    return dict(panels=(P_gen, P_real, P_ref), sim_ok=sim_ok, inside=bool(st["inside"]), ref_step=fs)


def _make_page(u_, R, sc):
    from PIL import Image, ImageDraw
    titles = ("generated (OpenWAM)", f"real, same step {u_['step']} (model's own actions)",
              f"real success ref ({u_['ref_episode']} t={R['ref_step']})")
    S = 128 * sc; W = 64 * sc; hdr = 34
    page = Image.new("RGB", (3 * S + 20, hdr + S + W + 18), "white"); dr = ImageDraw.Draw(page)
    dr.text((4, 2), f"{u_['id']}  {u_['stratum']}  src={u_['episode']} t0={u_['t0']} step={u_['step']} (frame j={u_['j']})  "
                    f"p_oof={u_['p_oof']:.3f}  in_gate_row={u_['in_gate_row']}", fill="black")
    dr.text((4, 16), f"sim success: {R['sim_ok']}   real book origin in box: {R['inside']}   "
                     "green = box top, red = ceiling (upright book top below red = in)", fill="black")
    for c, (title, Pn) in enumerate(zip(titles, R["panels"])):
        a, w = Pn[sc]
        x = c * (S + 10); page.paste(a, (x, hdr)); page.paste(w, (x + (S - W) // 2, hdr + S + 4))
        dr.text((x + 3, hdr + S + W + 5), title, fill="black")
    return np.asarray(page)


def _render_and_draw(args):
    R = _render_unit(args)
    return _make_page(args[0], R, 3), _make_page(args[0], R, 2), R["sim_ok"], R["inside"]


def render():
    import multiprocessing as mp
    from PIL import Image
    s = json.load(open(f"{CC}/sample.json")); E = {e["id"]: e for e in P._registry(5)}; first = _first_succ()
    for u_ in s["units"]:
        u_["ref_episode"] = _ref_episode(u_["episode"], first, E)
    for d_ in ("units", "pages", "jasmine/units"):
        os.makedirs(f"{CC}/{d_}", exist_ok=True)
    # geometry objects hold MuJoCo handles: draw inside the worker, return images
    with mp.get_context("spawn").Pool(2, maxtasksperchild=10) as pool:
        res = pool.map(_render_and_draw, [(u_, u_["ref_episode"]) for u_ in s["units"]], chunksize=1)
    for u_, (big, small, sim_ok, inside) in zip(s["units"], res):
        assert not sim_ok, f"{u_['id']}: simulator says success at a negative frame"
        u_["real_book_origin_inside"] = inside
        Image.fromarray(big).save(f"{CC}/units/{u_['id']}.png")
        if u_["id"] in s["jasmine"]:
            Image.fromarray(big).save(f"{CC}/jasmine/units/{u_['id']}.png")
        u_["_small"] = small
    for p in range(0, len(s["units"]), 5):                     # review pages, 5 triptychs each
        ims = [Image.fromarray(u_.pop("_small")) for u_ in s["units"][p:p + 5]]
        pg = Image.new("RGB", (ims[0].width, sum(i.height for i in ims)), "white"); y = 0
        for i in ims:
            pg.paste(i, (0, y)); y += i.height
        pg.save(f"{CC}/pages/page_{p // 5:02d}.png")
    json.dump(s, open(f"{CC}/sample.json", "w"), indent=1)
    print("rendered", len(s["units"]))


LINE = ((0, 220, 0), (230, 30, 30))


def fixflip():
    import shutil
    from PIL import Image
    sm = json.load(open(f"{CC}/sample.json")); sc, S, W, hdr = 3, 384, 192, 34
    gen = np.load(f"{D}/gen_out.npz"); G_fr, G_ep, G_t0 = gen["frames"], gen["episode"], gen["t0"]
    old = f"{CC}/units_flipNone"
    if not os.path.isdir(old):
        shutil.copytree(f"{CC}/units", old)
    filled = {}; REAL, GEN, REF = [], [], []
    for u_ in sm["units"]:
        im = np.asarray(Image.open(f"{old}/{u_['id']}.png").convert("RGB")).copy()
        ch = [i for i, (e, t) in enumerate(zip(G_ep, G_t0)) if e == u_["episode"] and t == u_["t0"]][0]
        z = np.load(f"{D}/episodes/{u_['ref_episode']}.npz")
        clean = {0: G_fr[ch][u_["j"] - 1][:, :128], 2: z["frames"][int(np.argmax(z["labels"]))][:, :128]}
        for c in range(3):
            x = c * (S + 10); pan = im[hdr:hdr + S, x:x + S]
            mask = np.zeros(pan.shape[:2], bool); col = np.zeros(pan.shape, np.uint8)
            for L in LINE:
                m = np.all(pan == L, -1); mask |= m; col[m] = L
            if c == 1:                                           # recover the 128 x 128 source from the 3 x 3 blocks
                src = np.zeros((128, 128, 3), np.uint8); hole = np.zeros((128, 128), bool)
                for a in range(128):
                    for b in range(128):
                        blk = pan[a * 3:a * 3 + 3, b * 3:b * 3 + 3].reshape(-1, 3); ok = ~mask[a * 3:a * 3 + 3, b * 3:b * 3 + 3].reshape(-1)
                        if ok.any():
                            v = blk[ok]; assert (v == v[0]).all(), f"{u_['id']}: panel is not a clean NEAREST x3 upscale"
                            src[a, b] = v[0]
                        else:
                            hole[a, b] = True
                for a, b in zip(*np.where(hole)):
                    nb = [src[i, j] for i, j in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)) if 0 <= i < 128 and 0 <= j < 128 and not hole[i, j]]
                    src[a, b] = np.mean(nb, 0) if nb else 0
                filled[u_["id"]] = int(hole.sum()); clean[1] = src
            else:                                                # the clean source must match the panel off the lines
                up = np.asarray(Image.fromarray(np.ascontiguousarray(clean[c])).resize((S, S), Image.NEAREST))
                assert (up[~mask] == pan[~mask]).all(), f"{u_['id']} panel {c}: clean source does not match the rendered panel"
            new = np.asarray(Image.fromarray(np.ascontiguousarray(clean[c])).resize((S, S), Image.NEAREST)).copy()
            mm = mask[:, ::-1]; new[mm] = col[:, ::-1][mm]          # mirror the lines (u -> 128 - u)
            im[hdr:hdr + S, x:x + S] = new
        Image.fromarray(im).save(f"{CC}/units/{u_['id']}.png")
        REAL.append(clean[1]); GEN.append(G_fr[ch][u_["j"] - 1]); REF.append(z["frames"][int(np.argmax(z["labels"]))])
        if u_["id"] in sm["jasmine"]:
            Image.fromarray(im).save(f"{CC}/jasmine/units/{u_['id']}.png")
    for p in range(0, len(sm["units"]), 5):
        ims = [Image.open(f"{CC}/units/{u_['id']}.png") for u_ in sm["units"][p:p + 5]]
        ims = [i.resize((i.width * 2 // 3, i.height * 2 // 3), Image.LANCZOS) for i in ims]
        pg = Image.new("RGB", (ims[0].width, sum(i.height for i in ims)), "white"); y = 0
        for i in ims:
            pg.paste(i, (0, y)); y += i.height
        pg.save(f"{CC}/pages/page_{p // 5:02d}.png")
    np.savez_compressed(f"{CC}/frames.npz", ids=np.array([u_["id"] for u_ in sm["units"]]), gen=np.stack(GEN),
                        real_agent=np.stack(REAL), ref=np.stack(REF))    # real: agentview half only (wrist panel is bilinear)
    sm["overlay_fix"] = dict(note="overlays mirrored to FLIP='h' by fixflip (first render used FLIP=None)", source_blocks_filled=filled)
    json.dump(sm, open(f"{CC}/sample.json", "w"), indent=1)
    print("fixed", len(sm["units"]), "blocks filled from neighbours:", {k: v for k, v in filled.items() if v})


def realscore():
    """companion numbers for the first pass (CPU): the deployed t5 detector on
      gen       the generated frame (exact; = what the gate scored, up to OOF vs deployed weights)
      real      the real frame at the same step: agentview half exact (recovered from the panel), wrist half resampled
                back from the panel's 192 px bilinear copy (approximate)
      realA_genW real agentview half + generated wrist half (which half drives the generated frame's score)
    and the mean absolute pixel difference generated vs real agentview in the caddy region."""
    import torch
    from PIL import Image
    from lingbot_predicate import Net, to_t
    sm = json.load(open(f"{CC}/sample.json")); f = np.load(f"{CC}/frames.npz")
    cal = json.load(open(f"{D}/calibration_success.json"))
    m = Net(); m.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}", map_location="cpu")); m.eval()
    S, hdr = 384, 34; RW = []
    for u_ in sm["units"]:
        im = Image.open(f"{CC}/units/{u_['id']}.png").convert("RGB"); x = (S + 10) + (S - 192) // 2
        RW.append(np.asarray(im.crop((x, hdr + S + 4, x + 192, hdr + S + 4 + 192)).resize((128, 128), Image.BILINEAR)))
    gen, ra = f["gen"], f["real_agent"]
    with torch.no_grad():
        P_ = lambda X: torch.sigmoid(m(to_t(X))).numpy().ravel()
        p_gen = P_(gen); p_real = P_(np.concatenate([ra, np.stack(RW)], 2)); p_mix = P_(np.concatenate([ra, gen[:, :, 128:]], 2))
    mad = np.abs(gen[:, 4:92, 40:124].astype(float) - ra[:, 4:92, 40:124].astype(float)).mean((1, 2, 3))
    for u_, a_, b_, c_, d_ in zip(sm["units"], p_gen, p_real, p_mix, mad):
        u_.update(p_deployed=dict(gen=float(a_), real=float(b_), realA_genW=float(c_)), caddy_mad_gen_vs_real=float(d_))
    json.dump(sm, open(f"{CC}/sample.json", "w"), indent=1)
    print("threshold", cal["threshold"])
    for u_ in sm["units"]:
        q = u_["p_deployed"]
        print(u_["id"], u_["stratum"][:5], "j", u_["j"], "gen %.2f real %.2f realA+genW %.2f mad %.1f" % (q["gen"], q["real"], q["realA_genW"], u_["caddy_mad_gen_vs_real"]))


if __name__ == "__main__":
    {"oof": oof, "sample": sample, "render": render, "verify": verify, "fixflip": fixflip, "realscore": realscore}[sys.argv[1]]()
