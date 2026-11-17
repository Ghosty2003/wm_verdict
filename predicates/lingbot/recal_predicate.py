"""LIBERO batch, LingBot-VA: predicate RE-CALIBRATION with failure-terminal negatives (ruling 2, 2026-10-15).

Why: the v1 predicate (lingbot_predicate.py) only saw negatives from the pre-success part of SUCCESSFUL closed-loop
rollouts; on real failure end states (book hovering over / held sideways above the caddy) it outputs p ~ 1
(lingbot_base check 7 layer 2: t5 real-endpoint TNR 0/8). The perturbation runs will produce such states in bulk.

Data = v1 data (rollouts, roundtrip, gen_chunk, drift; reused unchanged from out/predicate/t<task>/) PLUS failure
episodes (out/predicate_v2/t<task>/fail_eps/), every frame labelled by the simulator's check_success:
  fs_replay     the false-safe units of runs.lingbot_base: open-loop replay of the recorded env actions from the anchor
  pert_gain     trial perturbation (family-B-like, NOT a protocol setting): LingBot closed loop with the translation and
                rotation action components scaled by g in {0.4, 0.6} before env.step
  pert_disp     trial perturbation (family-A-like, NOT a protocol setting): the task object (obj_of_interest[0]) moved
                6-10 cm in a random horizontal direction after reset, 10 no-op settle steps, then LingBot closed loop
  rand_inject   LingBot closed loop with a random-action segment (start 60-220, length 40-90 steps) replacing the policy
  pure_random   smoothed random actions for H steps (no policy)
Failure-terminal subset = frames in the second half (t >= T/2) of episodes in which the simulator never reports success.
Splits are by episode (per source 60/20/20). Seeds: master 20261028.

Rows (each: TPR / TNR overall + TNR on the failure-terminal subset; gate 0.85 on all of them):
  holdout_real            v1 test rollouts + test failure episodes
  holdout_vae_roundtrip   v1 roundtrip + roundtrip of test failure-terminal frames (every 3rd)
  model_generated_1chunk  v1 gen_chunk + one-chunk imaginations started from test failure episodes' second half
                          (unperturbed-dynamics sources only: fs_replay, rand_inject, pure_random), labelled by simulating
Monitor (no gate): v1 full-rollout drift.

  python recal_predicate.py data <task_id>
  python recal_predicate.py train <task_id>
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
import torch
import torch.nn.functional as F

_wmvp.bootstrap()
from lingbot_predicate import Net, predict, metrics, model_view, batched, OUT as V1_OUT, BANK

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/predicate_v2"
BASE = f"{ROOT}/libero/out/runs/lingbot_base"
MASTER = 20261028
H = 520
POST = 60
N_PER_SOURCE = 8
SOURCES = ("fs_replay", "pert_gain", "pert_disp", "rand_inject", "pure_random")
UNPERTURBED_DYN = ("fs_replay", "rand_inject", "pure_random")


# --------------------------------------------------------------------------------------------- episodes
def _snap_frame(rig, ob):
    return model_view(rig, rig.obs_to_batch(ob))


def _displace(u, obj, rs):
    sim = u._env.sim
    j = sim.model.joint_name2id(f"{obj}_joint0")
    adr = sim.model.jnt_qposadr[j]
    ang = rs.uniform(0, 2 * np.pi); r = rs.uniform(0.06, 0.10)
    off = np.array([r * np.cos(ang), r * np.sin(ang)])
    sim.data.qpos[adr:adr + 2] += off
    sim.forward()
    return dict(object=obj, offset_xy_m=off.round(4).tolist())


def closed_episode(rig, anc, pseed, rs, gain=None, disp=False, inject=None):
    """LingBot closed loop with an optional trial perturbation. Returns frames, labels, EXECUTED actions, spec."""
    env, obs = rig.make_env(anc); u = env.envs[0]
    spec, ob = {}, obs
    if disp:
        spec["displacement"] = _displace(u, u._env.obj_of_interest[0], rs)
        for _ in range(10):
            o, *_ = u.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
        ob = batched(o)
    if gain is not None:
        spec["gain_xyz_rpy"] = gain
    if inject is not None:
        spec["random_segment"] = dict(start=inject[0], length=inject[1])
    rig.reset_policy(pseed)
    frames, labels, acts, first = [], [], [], None
    irs = np.random.RandomState(pseed % (2 ** 31 - 1))
    with torch.inference_mode():
        for t in range(H):
            batch = rig.obs_to_batch(ob)
            frames.append(model_view(rig, batch)); labels.append(bool(u._env.check_success()))
            a = rig.to_env_action(rig.policy.select_action(batch))[0].astype(np.float32)
            if inject is not None and inject[0] <= t < inject[0] + inject[1]:
                a = np.concatenate([irs.uniform(-1, 1, 6), [irs.choice([-1.0, 1.0])]]).astype(np.float32)
            if gain is not None:
                a = a.copy(); a[:6] *= gain
            acts.append(a)
            o, r, term, trunc, info = u.step(a)
            ob = batched(o)
            if info.get("is_success") and first is None:
                first = t + 1
            if first is not None and t + 1 >= first + POST:
                break
        frames.append(_snap_frame(rig, ob)); labels.append(bool(u._env.check_success()))
    env.close()
    return np.stack(frames), np.array(labels), np.stack(acts), spec


def replay_episode(rig, anc, acts):
    env, obs = rig.make_env(anc); u = env.envs[0]
    frames, labels = [_snap_frame(rig, obs)], [bool(u._env.check_success())]
    for a in acts:
        o, *_ = u.step(np.asarray(a, np.float32))
        frames.append(_snap_frame(rig, batched(o))); labels.append(bool(u._env.check_success()))
    env.close()
    return np.stack(frames), np.array(labels)


def random_episode(rig, anc, rs):
    env, obs = rig.make_env(anc); u = env.envs[0]
    frames, labels, acts = [_snap_frame(rig, obs)], [bool(u._env.check_success())], []
    a6, g = np.zeros(6), -1.0
    for t in range(H):
        a6 = 0.8 * a6 + 0.2 * rs.uniform(-1, 1, 6)
        if t % 50 == 0:
            g = rs.choice([-1.0, 1.0])
        a = np.concatenate([a6, [g]]).astype(np.float32); acts.append(a)
        o, *_ = u.step(a)
        frames.append(_snap_frame(rig, batched(o))); labels.append(bool(u._env.check_success()))
    env.close()
    return np.stack(frames), np.array(labels), np.stack(acts)


def fail_terminal_mask(labels):
    T = len(labels)
    m = np.zeros(T, bool)
    if not labels.any():
        m[T // 2:] = True
    return m


def gen_data(task_id):
    from lingbot_unit import LingBotRig
    rig = LingBotRig(task_id)
    bank = json.load(open(BANK))
    anchors = {a["anchor_id"]: a for a in bank["anchors"]}
    used = {a["init_index"] for a in bank["anchors"] if a["task_id"] == task_id}
    v1_used = set()
    for f in os.listdir(f"{V1_OUT}/t{task_id}/rollouts"):
        v1_used.add(int(np.load(f"{V1_OUT}/t{task_id}/rollouts/{f}")["init_index"]))
    pool = [i for i in range(50) if i not in used]
    template = [a for a in bank["anchors"] if a["task_id"] == task_id][0]
    d = f"{OUT}/t{task_id}"; os.makedirs(f"{d}/fail_eps", exist_ok=True)
    rs = np.random.RandomState(MASTER + task_id)

    def save(name, frames, labels, acts, meta):
        np.savez_compressed(f"{d}/fail_eps/{name}.npz", frames=frames, labels=labels, actions=acts,
                            meta=json.dumps(meta))
        print(f"t{task_id} {name}: T={len(acts)} success={bool(labels.any())} fail_terminal_frames="
              f"{int(fail_terminal_mask(labels).sum())} spec={meta.get('spec')}", flush=True)

    # fs_replay: the false-safe units of runs.lingbot_base
    fs = [json.loads(l) for l in open(f"{BASE}/records_t{task_id}.jsonl")]
    fs = [r for r in fs if r["model_verdict"] and not r["real_verdict"]]
    for i, r in enumerate(fs):
        name = f"fs_replay_{i:02d}"
        if os.path.exists(f"{d}/fail_eps/{name}.npz"):
            continue
        acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
        a = anchors[r["s0_id"]]
        fr, lab = replay_episode(rig, a, acts)
        save(name, fr, lab, acts, dict(source="fs_replay", unit_id=r["unit_id"], anchor=r["s0_id"],
                                       init_index=a["init_index"], seed=a["seed"], spec={}))
    # closed-loop trial perturbations and random-action episodes, from non-anchor inits
    for src in ("pert_gain", "pert_disp", "rand_inject", "pure_random"):
        for i in range(N_PER_SOURCE):
            init = int(pool[rs.randint(len(pool))]); seed = int(rs.randint(2 ** 31 - 1)); pseed = int(rs.randint(2 ** 31 - 1))
            gain = [0.4, 0.6][i % 2]; inj = (int(rs.randint(60, 221)), int(rs.randint(40, 91)))
            ers = np.random.RandomState(pseed)
            name = f"{src}_{i:02d}"
            if os.path.exists(f"{d}/fail_eps/{name}.npz"):
                continue
            anc = dict(template, init_index=init, seed=seed)
            if src == "pure_random":
                fr, lab, acts = random_episode(rig, anc, ers); spec = dict(random="smoothed uniform, gripper flips /50")
            else:
                fr, lab, acts, spec = closed_episode(rig, anc, pseed, ers, gain=gain if src == "pert_gain" else None,
                                                     disp=(src == "pert_disp"), inject=inj if src == "rand_inject" else None)
            save(name, fr, lab, acts, dict(source=src, init_index=init, seed=seed, policy_seed=pseed, spec=spec,
                                           init_in_v1_rollouts=init in v1_used))
    # splits by episode, per source
    eps = sorted(f[:-4] for f in os.listdir(f"{d}/fail_eps") if f.endswith(".npz"))
    split = {"train": [], "val": [], "test": []}
    for src in SOURCES:
        S = [e for e in eps if e.startswith(src)]
        n = len(S); nt = max(1, round(.6 * n)); nv = max(1, round(.2 * n)) if n >= 3 else 0
        split["train"] += S[:nt]; split["val"] += S[nt:nt + nv]; split["test"] += S[nt + nv:]
    json.dump(split, open(f"{d}/split.json", "w"), indent=1)
    # counts vs positives
    nft = sum(int(fail_terminal_mask(np.load(f"{d}/fail_eps/{e}.npz")["labels"]).sum()) for e in eps)
    npos = sum(int(np.load(f"{V1_OUT}/t{task_id}/rollouts/{f}")["labels"].sum()) for f in os.listdir(f"{V1_OUT}/t{task_id}/rollouts"))
    npos += sum(int(np.load(f"{d}/fail_eps/{e}.npz")["labels"].sum()) for e in eps)
    print(f"t{task_id} failure-terminal frames {nft} vs positive frames {npos}", flush=True)
    # roundtrip of test failure-terminal frames
    if not os.path.exists(f"{d}/roundtrip_fail.npz"):
        p = rig.policy; p._ensure_frozen_modules()
        RT, RL, RF = [], [], []
        _im = torch.inference_mode(); _im.__enter__()
        for e in split["test"]:
            z = np.load(f"{d}/fail_eps/{e}.npz"); m = fail_terminal_mask(z["labels"])
            for i in range(0, len(z["frames"]), 3):
                fr = torch.from_numpy(z["frames"][i]).float().div(255)
                halves = fr.permute(2, 0, 1).split(fr.shape[1] // 2, dim=-1)
                vids = torch.stack([h * 2 - 1 for h in halves])[:, :, None].to(p.config.device).to(p.dtype)
                p._frozen["streaming_vae"].clear_cache()
                enc = p._streaming_vae.encode_chunk(vids.to(next(p._vae.parameters()).device))
                lat = torch.cat(p._normalize_vae_latent(enc).split(1, dim=0), dim=-1)
                RT.append(rig.decode(lat)[0].numpy()); RL.append(bool(z["labels"][i])); RF.append(bool(m[i]))
        p._frozen["streaming_vae"].clear_cache(); _im.__exit__(None, None, None)
        np.savez_compressed(f"{d}/roundtrip_fail.npz", frames=np.stack(RT), labels=np.array(RL), fail_terminal=np.array(RF))
        print("roundtrip_fail", len(RT), "fail_terminal", int(np.sum(RF)), flush=True)
    # one-chunk imaginations from test failure episodes (unperturbed dynamics only)
    if not os.path.exists(f"{d}/gen_chunk_fail.npz"):
        GF, GL, GM = [], [], []
        grs = np.random.RandomState(MASTER + 100 + task_id)
        for e in split["test"]:
            if not e.startswith(UNPERTURBED_DYN):
                continue
            z = np.load(f"{d}/fail_eps/{e}.npz"); meta = json.loads(str(z["meta"]))
            n = len(z["actions"]); failed = not z["labels"].any()
            anc = anchors[meta["anchor"]] if meta["source"] == "fs_replay" else dict(template, init_index=meta["init_index"], seed=meta["seed"])
            for t0 in np.linspace(n // 2, n - 13, 4).astype(int):
                env, obs = rig.make_env(anc); u = env.envs[0]
                o = None
                for a in z["actions"][:t0]:
                    o, *_ = u.step(np.asarray(a, np.float32))
                batch = rig.obs_to_batch(batched(o) if o is not None else obs)
                rig.reset_policy(int(grs.randint(2 ** 31 - 1)))
                with torch.inference_mode():
                    p = rig.policy; p._ensure_frozen_modules(); p._maybe_init_prompt(batch)
                    acts = p.predict_action_chunk(batch); lat = p.last_predicted_latents
                frames = rig.decode(lat).numpy()
                ea = np.concatenate([rig.to_env_action(acts[:, k]) for k in range(acts.shape[1])], 0)
                labs = []
                for a in ea:
                    u.step(a.astype(np.float32)); labs.append(bool(u._env.check_success()))
                env.close()
                for k in range(1, min(len(frames), len(labs) + 1)):
                    GF.append(frames[k]); GL.append(labs[k - 1]); GM.append(failed)
        np.savez_compressed(f"{d}/gen_chunk_fail.npz", frames=np.stack(GF), labels=np.array(GL), fail_terminal=np.array(GM))
        print("gen_chunk_fail", len(GF), "positives", int(np.sum(GL)), "from failed episodes", int(np.sum(GM)), flush=True)


# --------------------------------------------------------------------------------------------- training
def _load_v1(task_id, names):
    X, Y = [], []
    for f in names:
        z = np.load(f"{V1_OUT}/t{task_id}/rollouts/{f}"); X.append(z["frames"]); Y.append(z["labels"])
    return np.concatenate(X), np.concatenate(Y).astype(bool)


def _load_fail(d, names, with_src=False):
    X, Y, M, S = [], [], [], []
    for e in names:
        z = np.load(f"{d}/fail_eps/{e}.npz"); X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        S += [e.rsplit("_", 1)[0]] * len(z["labels"])
    out = (np.concatenate(X), np.concatenate(Y).astype(bool), np.concatenate(M))
    return out + (np.array(S),) if with_src else out


def row(p, y, ft, thr):
    out = metrics(p, y, thr)
    out["fail_terminal"] = metrics(p[ft], y[ft], thr) if ft.any() else None
    return out


def train(task_id):
    d = f"{OUT}/t{task_id}"
    v1 = json.load(open(f"{V1_OUT}/t{task_id}/split.json")); sp = json.load(open(f"{d}/split.json"))
    parts = {}
    for k in ("train", "val", "test"):
        Xa, Ya = _load_v1(task_id, v1[k]); Xb, Yb, Mb = _load_fail(d, sp[k])
        parts[k] = (np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb]), np.concatenate([np.zeros(len(Ya), bool), Mb]))
    Xtr, Ytr, Mtr = parts["train"]; Xva, Yva, Mva = parts["val"]; Xte, Yte, Mte = parts["test"]
    log = []
    for attempt in range(3):
        torch.manual_seed(attempt)
        net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        pos = np.where(Ytr)[0]; negf = np.where(~Ytr & Mtr)[0]; nego = np.where(~Ytr & ~Mtr)[0]
        rs = np.random.RandomState(attempt)
        for ep in range(25):
            net.train()
            idx = np.concatenate([rs.choice(pos, 3000), rs.choice(negf, 1500), rs.choice(nego, 1500)]); rs.shuffle(idx)
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                loss = F.binary_cross_entropy_with_logits(net(torch.from_numpy(Xtr[b]).permute(0, 3, 1, 2).float().div(255).cuda()),
                                                          torch.from_numpy(Ytr[b].astype(np.float32)).cuda())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if ep % 5 == 4:
                net.eval(); pv = predict(net, Xva)
                log.append(dict(attempt=attempt, epoch=ep + 1, loss=float(loss), val_prob_std=float(pv.std()),
                                val_bal_acc_05=0.5 * (float(np.mean(pv[Yva] >= .5)) + float(np.mean(pv[~Yva] < .5))),
                                val_fail_tnr_05=float(np.mean(pv[Mva & ~Yva] < .5))))
                print(log[-1], flush=True)
        if log[-1]["val_prob_std"] > 0.05 and log[-1]["val_bal_acc_05"] > 0.8:
            break
    else:
        raise SystemExit("predicate training collapsed in 3 attempts -- stop and report")
    net.eval(); pv = predict(net, Xva)
    cands = np.unique(np.quantile(pv, np.linspace(0, 1, 400)))
    thr = float(cands[int(np.argmax([0.5 * (np.mean(pv[Yva] >= c) + np.mean(pv[~Yva] < c)) for c in cands]))])
    res = dict(task_id=task_id, version="v2 (failure-terminal negatives)", threshold=thr, train_log=log,
               n_train=int(len(Ytr)), n_train_pos=int(Ytr.sum()), n_train_fail_terminal=int((Mtr & ~Ytr).sum()),
               val=row(pv, Yva, Mva, thr), holdout_real=row(predict(net, Xte), Yte, Mte, thr))
    z1 = np.load(f"{V1_OUT}/t{task_id}/roundtrip.npz"); z2 = np.load(f"{d}/roundtrip_fail.npz")
    X = np.concatenate([z1["frames"], z2["frames"]]); Y = np.concatenate([z1["labels"], z2["labels"]]).astype(bool)
    M = np.concatenate([np.zeros(len(z1["labels"]), bool), z2["fail_terminal"]])
    res["holdout_vae_roundtrip"] = row(predict(net, X), Y, M, thr)
    g1 = np.load(f"{V1_OUT}/t{task_id}/gen_chunk.npz"); g2 = np.load(f"{d}/gen_chunk_fail.npz")
    X = np.concatenate([g1["frames"], g2["frames"]]); Y = np.concatenate([g1["labels"], g2["labels"]]).astype(bool)
    M = np.concatenate([np.zeros(len(g1["labels"]), bool), g2["fail_terminal"]])
    res["model_generated_1chunk"] = row(predict(net, X), Y, M & ~Y, thr)
    dz = np.load(f"{V1_OUT}/t{task_id}/drift.npz")
    res["model_generated_full_rollout"] = metrics(predict(net, dz["frames"]), dz["labels"].astype(bool), thr)
    # per-source TNR on the failure-terminal subset of the real test episodes
    Xs, Ys, Ms, Ss = _load_fail(d, sp["test"], with_src=True)
    ps = predict(net, Xs)
    res["holdout_real"]["fail_terminal_by_source"] = {
        src: metrics(ps[(Ss == src) & Ms], Ys[(Ss == src) & Ms], thr) for src in SOURCES if ((Ss == src) & Ms).any()}
    res["composition"] = dict(
        sources={s: [e for e in sorted(os.listdir(f"{d}/fail_eps")) if e.startswith(s)] for s in SOURCES},
        split=sp, v1_split=v1,
        fail_terminal_frames={k: int((parts[k][2] & ~parts[k][1]).sum()) for k in parts},
        positive_frames={k: int(parts[k][1].sum()) for k in parts})
    path = f"{d}/predicate.pt"; torch.save(net.state_dict(), path)
    res["artifact"] = dict(path=os.path.relpath(path, ROOT), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
    gate = {}
    for k in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk"):
        r = res[k]; ft = r["fail_terminal"]
        gate[k] = bool(r["tpr"] >= .85 and r["tnr"] >= .85)
        gate[k + "/fail_terminal_tnr"] = bool(ft is not None and ft["tnr"] is not None and ft["tnr"] >= .85)
    res["gate_85"] = gate; res["gate_pass"] = all(gate.values())
    json.dump(res, open(f"{d}/calibration.json", "w"), indent=1)
    print("RESULT", json.dumps({k: res[k] for k in ("threshold", "holdout_real", "holdout_vae_roundtrip",
                                                    "model_generated_1chunk", "model_generated_full_rollout", "gate_85", "gate_pass")}))


if __name__ == "__main__":
    (gen_data if sys.argv[1] == "data" else train)(int(sys.argv[2]))
