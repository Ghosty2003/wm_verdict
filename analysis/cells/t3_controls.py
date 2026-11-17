"""OpenWAM-alpha t3 attribution review: 10 control cards + one uniform blind deck of 30 (rulings 2026-10-29: add 10
controls; agent-judge pass rule = disagreement <= 15 % AND kappa >= 0.6, stratified blind subsets of 30).

Controls have a known answer: REAL trajectories (truth replay of stored actions) shown where the other cards show the
imagination - 5 real failures (3 "bowl in drawer, drawer not closed", 2 "bowl never in the drawer") and 5 real
successes, from famA / famP units outside every attribution sample. Their canvases (every 4 env steps, as the
imagined canvases) pass through OpenWAM's own Wan VAE (encode -> decode), so they look like generated frames.

Deck: all 30 cards (20 sampled false-safe imaginations + 10 controls) in one format, shuffled, anonymised (card_01..30):
(1) end (step 400), (2) reference frame (originals: the predicate's first trigger; real successes: the first success
step; real failures: a step drawn from the originals' trigger steps), (3) dense strip around the reference frame,
(4) coarse strip every 20 steps. No "real end" panel (it would give the controls away). The key is written to
jasmine/.agent_firstpass/deck_key.json only.

  python t3_controls.py select                      (any python)
  python t3_controls.py render                      (lerobot_libero, GPU render)  -> ctrl/units/<uid>.npz
  python t3_controls.py roundtrip                   (openwam, GPU)                -> ctrl/rt/<uid>.npz
  python t3_controls.py deck                        (any python)                  -> deck/card_XX.png, jasmine/audit_sheet.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
SEED = 20261029
H = 400
RUNS = {"famA": ("openwam_famA", f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"),
        "famP": ("openwam_famP", f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json")}
C = f"{HERE}/ctrl"


def recs(fam):
    return [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/{RUNS[fam][0]}/records_t3.jsonl")]


def select():
    used = {u["unit"] for u in json.load(open(f"{HERE}/sample.json"))["units"]}
    pools = {"in_drawer_not_closed": [], "never_in_drawer": [], "success": []}
    for fam in RUNS:
        for r in recs(fam):
            if r["unit_id"] in used or not os.path.exists(f"{HERE}/diag/{r['unit_id']}.npz"):
                continue
            z = np.load(f"{HERE}/diag/{r['unit_id']}.npz"); close, inn = z["sub"][:, 0], z["sub"][:, 1]
            if r["real_verdict"]:
                pools["success"].append((fam, r["unit_id"]))
            elif inn[-1] and not close[-1]:
                pools["in_drawer_not_closed"].append((fam, r["unit_id"]))
            elif not inn.any():
                pools["never_in_drawer"].append((fam, r["unit_id"]))
    rs = np.random.RandomState(SEED); pick = []
    for k, n in (("in_drawer_not_closed", 3), ("never_in_drawer", 2), ("success", 5)):
        P = sorted(pools[k]); idx = rs.choice(len(P), n, replace=False)
        pick += [dict(fam=P[i][0], unit=P[i][1], kind=k, answer="looks_success" if k == "success" else "not_success_but_triggered") for i in idx]
    os.makedirs(C, exist_ok=True)
    json.dump(dict(seed=SEED, pool_sizes={k: len(v) for k, v in pools.items()}, controls=pick), open(f"{C}/controls.json", "w"), indent=1)
    print({k: len(v) for k, v in pools.items()}, [p["unit"] for p in pick])


def _render(args):
    from libero_env_adapter import _make_env, _anchor
    from ow_view import canvas_from_raw, raw_obs
    fam, uid = args
    r = next(x for x in recs(fam) if x["unit_id"] == uid)
    acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
    env = _make_env(dict(_anchor(RUNS[fam][1], r["anchor_id"]), resolution=256), H=len(acts)); u = env.envs[0]
    cv, succ = [], []
    for t in range(H + 1):
        if t:
            u.step(np.asarray(acts[t - 1], np.float32)); succ.append(bool(u._env.check_success()))
        if t % 4 == 0:
            o = raw_obs(u); cv.append(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
    env.close()
    same = bool(np.array_equal(np.array(succ), np.load(f"{ROOT}/{r['trajectory_file']}")["real_success"]))
    return uid, np.stack(cv), np.array(succ), same


def render():
    import multiprocessing as mp
    ctl = json.load(open(f"{C}/controls.json"))["controls"]; os.makedirs(f"{C}/units", exist_ok=True)
    with mp.get_context("spawn").Pool(2) as pool:
        for uid, cv, succ, same in pool.imap_unordered(_render, [(c["fam"], c["unit"]) for c in ctl]):
            assert same, f"{uid}: replay does not reproduce the recorded success trace"
            np.savez_compressed(f"{C}/units/{uid}.npz", canvas=cv, success=succ); print("rendered", uid, flush=True)


def roundtrip():
    sys.path[:0] = [_wmvp.P("$OPENWAM_REPO"), _wmvp.P("$OPENWAM_REPO/third_party")]
    os.environ.setdefault("WAM_ATTENTION_IMPL", "sdpa")
    import torch
    from PIL import Image
    from openwam_unit import _engine
    from openwam.model.video_backbone.wan import encode as wan_encode
    eng, pre, arch, _, _ = _engine(); vb = arch.video_backbone
    os.makedirs(f"{C}/rt", exist_ok=True)
    for c in json.load(open(f"{C}/controls.json"))["controls"]:
        cv = np.load(f"{C}/units/{c['unit']}.npz")["canvas"]; out = []
        with torch.no_grad():
            for f in cv:
                t = wan_encode.preprocess_video([Image.fromarray(f)], encoder=vb.video_encoder, dtype=vb.dtype, device=vb.device)
                lat = wan_encode.encode_video(t.to(vb.device), vae=vb.vae, encoder=vb.video_encoder)
                out.append(np.asarray(vb.decode_video(lat, tiled=True)[0]))
        np.savez_compressed(f"{C}/rt/{c['unit']}.npz", canvas=np.stack(out)); print("roundtrip", c["unit"], np.stack(out).shape, flush=True)


def deck():
    from PIL import Image, ImageDraw
    from ow_t3_review import _panel, _strip
    sheet = json.load(open(f"{HERE}/jasmine/.agent_firstpass/audit_sheet_with_agent_labels.json"))
    R = {r["unit_id"]: r for fam in ("famA", "famP") for r in recs(fam)}
    R.update({json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{ROOT}/libero/out/runs/openwam_base/records_t3.jsonl")})
    rs = np.random.RandomState(SEED + 1); items = []
    trig = [int(R[x["unit"]]["model_first_success_step"]) for x in sheet]
    for x in sheet:
        r = R[x["unit"]]; IC = np.load(f"{ROOT}/{r['trajectory_file']}")["imagined_canvas"]
        items.append(dict(source="imagination", unit=x["unit"], ref=int(r["model_first_success_step"]), canvas=IC,
                          agent_firstpass=dict(label=x["agent_label"], sublabel=x["agent_sublabel"], motion=x["agent_motion"])))
    for c in json.load(open(f"{C}/controls.json"))["controls"]:
        cv = np.load(f"{C}/rt/{c['unit']}.npz")["canvas"]; succ = np.load(f"{C}/units/{c['unit']}.npz")["success"]
        ref = (int(np.argmax(succ)) + 1) if succ.any() else int(rs.choice(trig))
        items.append(dict(source="control_real_vae_roundtrip", unit=c["unit"], kind=c["kind"], answer=c["answer"], ref=4 * round(ref / 4), canvas=cv))
    order = rs.permutation(len(items)); os.makedirs(f"{HERE}/deck", exist_ok=True); key, audit = [], []
    for n, i in enumerate(order, 1):
        it = items[i]; IC = it["canvas"]; g = min(100, it["ref"] // 4); cid = f"card_{n:02d}"
        P = [_panel(IC[100], "end (step 400)"), _panel(IC[g], f"reference frame (step {4 * g})")]
        sh = Image.new("RGB", (sum(p.width for p in P), P[0].height + 18), "white"); ImageDraw.Draw(sh).text((3, 2), cid, fill="black"); x = 0
        for p in P:
            sh.paste(p, (x, 18)); x += p.width
        coarse = _strip([IC[k] for k in range(0, 101, 5)], [f"s{4 * k}" for k in range(0, 101, 5)], 11, 128)
        di = list(range(max(0, g - 12), min(100, g + 3) + 1))
        dense = _strip([IC[k] for k in di], [f"s{4 * k}" + (" R" if k == g else "") for k in di], 8, 160)
        W = 1300; parts = [sh.resize((W, int(sh.height * W / sh.width)), Image.LANCZOS), dense.resize((W, int(dense.height * W / dense.width)), Image.LANCZOS),
                           coarse.resize((W, int(coarse.height * W / coarse.width)), Image.LANCZOS)]
        card = Image.new("RGB", (W, sum(p.height for p in parts) + 8), "white"); y = 0
        for p in parts:
            card.paste(p, (0, y)); y += p.height + 4
        card.save(f"{HERE}/deck/{cid}.png")
        key.append({k: v for k, v in it.items() if k != "canvas"} | dict(card=cid))
        audit.append(dict(card=cid, image=f"deck/{cid}.png", jasmine_label=None, jasmine_sublabel=None, jasmine_motion=None, jasmine_note=None))
    json.dump(key, open(f"{HERE}/jasmine/.agent_firstpass/deck_key.json", "w"), indent=1)
    json.dump(audit, open(f"{HERE}/jasmine/audit_sheet.json", "w"), indent=1)
    print("deck", len(audit))


if __name__ == "__main__":
    dict(select=select, render=render, roundtrip=roundtrip, deck=deck)[sys.argv[1]]()
