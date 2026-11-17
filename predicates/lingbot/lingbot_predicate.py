"""LIBERO batch, LingBot-VA: success predicate on DECODED frames (one predicate per task).

Frame format = the model's own view: per camera (obs_cam_keys order) the processed observation image, horizontally
flipped back (config.image_hflip), resized to 128x128, concatenated on width -> 128x256 uint8. Decoded imagined
frames come out in exactly this layout.

Data (init states NOT used as anchors; seeds from master 20261027; split by rollout 60/20/20):
  real       LingBot-VA closed-loop rollouts, continued 60 steps past the first success; label = check_success
  roundtrip  test real frames -> streaming VAE encode (single frame) -> decode
  gen_chunk  from a real mid-episode state (restored by replaying the recorded action prefix in a fresh env), imagine
             ONE chunk (12 actions) and decode; label = check_success after executing the same actions in the simulator
  drift      full 520-step imaginations from fresh starts, decoded; label = open-loop replay of the same actions
             (monitor only, mixes predicate error with model drift)
Classifier: small CNN, BCE, balanced sampling; threshold by balanced accuracy on val; collapse guard.
Gate (85% TPR and TNR): real, roundtrip, gen_chunk.

  python lingbot_predicate.py data <task_id>     # generate data -> out/predicate/<task>/
  python lingbot_predicate.py train <task_id>    # train + evaluate -> out/predicate/<task>/calibration.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = _wmvp.P("$WMV_ROOT/libero/out/predicate")
BANK = _wmvp.P("$WMV_ROOT/libero/out/bank/libero_anchors.json")
MASTER = 20261027
N_ROLL = 60
POST = 60
N_DRIFT = 10


def model_view(rig, batch):
    cfg = rig.policy.config
    outs = []
    for k in cfg.obs_cam_keys:
        img = batch[k][:1].float()
        if cfg.image_hflip:
            img = torch.flip(img, dims=[-1])
        img = F.interpolate(img, size=(cfg.height, cfg.width), mode="bilinear", align_corners=False)
        outs.append(img)
    x = torch.cat(outs, dim=-1)[0].clamp(0, 1)                       # [C, H, W*ncam]
    return (x.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()


def batched(obs):
    return {k: (batched(v) if isinstance(v, dict) else np.asarray(v)[None]) for k, v in obs.items()}


def gen_data(task_id):
    from lingbot_unit import LingBotRig, H
    rig = LingBotRig(task_id)
    bank = json.load(open(BANK))
    used = {a["init_index"] for a in bank["anchors"] if a["task_id"] == task_id}
    pool = [i for i in range(50) if i not in used]
    rs = np.random.RandomState(MASTER + task_id)
    d = f"{OUT}/t{task_id}"; os.makedirs(f"{d}/rollouts", exist_ok=True)
    template = [a for a in bank["anchors"] if a["task_id"] == task_id][0]
    # ---------------- real closed-loop rollouts (continued past success, no auto-reset)
    for r in range(N_ROLL):
        path = f"{d}/rollouts/r{r:03d}.npz"
        if os.path.exists(path):
            continue
        anc = dict(template, init_index=int(pool[rs.randint(len(pool))]), seed=int(rs.randint(2 ** 31 - 1)))
        pseed = int(rs.randint(2 ** 31 - 1))
        env, obs = rig.make_env(anc)
        u = env.envs[0]
        rig.reset_policy(pseed)
        frames, labels, acts, first = [], [], [], None
        with torch.inference_mode():
            ob = obs
            for t in range(H):
                batch = rig.obs_to_batch(ob)
                frames.append(model_view(rig, batch))
                labels.append(bool(u._env.check_success()))
                a = rig.to_env_action(rig.policy.select_action(batch))[0]
                acts.append(a)
                o, rew, term, trunc, info = u.step(a.astype(np.float32))
                ob = batched(o)
                if info.get("is_success") and first is None:
                    first = t + 1
                if first is not None and t + 1 >= first + POST:
                    break
            batch = rig.obs_to_batch(ob); frames.append(model_view(rig, batch)); labels.append(bool(u._env.check_success()))
        env.close()
        np.savez_compressed(path, frames=np.stack(frames), labels=np.array(labels), actions=np.stack(acts),
                            init_index=anc["init_index"], seed=anc["seed"], policy_seed=pseed,
                            first_success=-1 if first is None else first)
        print(f"t{task_id} roll {r}: steps {len(acts)} first_success {first}", flush=True)
    # ---------------- splits
    R = sorted(os.listdir(f"{d}/rollouts"))
    split = {"train": R[:int(.6 * len(R))], "val": R[int(.6 * len(R)):int(.8 * len(R))], "test": R[int(.8 * len(R)):]}
    json.dump(split, open(f"{d}/split.json", "w"), indent=1)
    # ---------------- roundtrip of test frames (single-frame streaming VAE encode -> decode)
    if not os.path.exists(f"{d}/roundtrip.npz"):
        RT, RL = [], []
        p = rig.policy
        p._ensure_frozen_modules()
        _im = torch.inference_mode(); _im.__enter__()          # streaming-VAE cache holds inference tensors
        for f in split["test"]:
            z = np.load(f"{d}/rollouts/{f}")
            idx = np.arange(0, len(z["frames"]), 3)
            for i in idx:
                fr = torch.from_numpy(z["frames"][i]).float().div(255)                      # [H, W*2, 3] model view
                halves = fr.permute(2, 0, 1).split(fr.shape[1] // 2, dim=-1)              # per camera [3,H,W]
                vids = torch.stack([h * 2 - 1 for h in halves])[:, :, None].to(p.config.device).to(p.dtype)  # [cam,3,1,H,W]
                p._frozen["streaming_vae"].clear_cache()
                enc = p._streaming_vae.encode_chunk(vids.to(next(p._vae.parameters()).device))
                lat = torch.cat(p._normalize_vae_latent(enc).split(1, dim=0), dim=-1)
                RT.append(rig.decode(lat)[0].numpy()); RL.append(bool(z["labels"][i]))
        p._frozen["streaming_vae"].clear_cache()
        _im.__exit__(None, None, None)
        np.savez_compressed(f"{d}/roundtrip.npz", frames=np.stack(RT), labels=np.array(RL))
        print("roundtrip", len(RT), flush=True)
    # ---------------- generated one-chunk-ahead frames from real mid-episode states
    if not os.path.exists(f"{d}/gen_chunk.npz"):
        GF, GL = [], []
        grs = np.random.RandomState(MASTER + 100 + task_id)
        for f in split["test"]:
            z = np.load(f"{d}/rollouts/{f}")
            fs = int(z["first_success"]); n = len(z["actions"])
            centre = fs if fs > 0 else n
            starts = sorted({int(np.clip(centre + o, 1, n - 1)) for o in range(-48, 13, 6)})
            for t0 in starts:
                anc = dict(template, init_index=int(z["init_index"]), seed=int(z["seed"]))
                env, obs = rig.make_env(anc); u = env.envs[0]
                for a in z["actions"][:t0]:
                    o, *_ = u.step(a.astype(np.float32))
                ob = batched(o) if t0 > 0 else obs
                batch = rig.obs_to_batch(ob)
                rig.reset_policy(int(grs.randint(2 ** 31 - 1)))
                with torch.inference_mode():
                    p = rig.policy; p._ensure_frozen_modules(); p._maybe_init_prompt(batch)
                    acts = p.predict_action_chunk(batch)                      # [1, 12, n_used]
                    lat = p.last_predicted_latents
                frames = rig.decode(lat).numpy()                              # 13 frames: 0 = conditioning
                ea = np.concatenate([rig.to_env_action(acts[:, k]) for k in range(acts.shape[1])], 0)
                labs = []
                for a in ea:
                    o, r, term, trunc, info = u.step(a.astype(np.float32)); labs.append(bool(u._env.check_success()))
                env.close()
                for k in range(1, min(len(frames), len(labs) + 1)):
                    GF.append(frames[k]); GL.append(labs[k - 1])
        np.savez_compressed(f"{d}/gen_chunk.npz", frames=np.stack(GF), labels=np.array(GL))
        print("gen_chunk", len(GF), "positives", int(np.sum(GL)), flush=True)
    # ---------------- drift: full imaginations from fresh starts, labelled by open-loop replay
    if not os.path.exists(f"{d}/drift.npz"):
        DF, DL = [], []
        drs = np.random.RandomState(MASTER + 200 + task_id)
        for j in range(N_DRIFT):
            anc = dict(template, init_index=int(pool[drs.randint(len(pool))]), seed=int(drs.randint(2 ** 31 - 1)))
            I = rig.imagine(anc, seed=int(drs.randint(2 ** 31 - 1)))
            fr = rig.decode(I["latents"]).numpy()
            R1 = rig.replay(anc, I["env_actions"])
            lab = np.concatenate([[False], R1["succ"]])
            m = min(len(fr), len(lab))
            for k in range(1, m, 2):
                DF.append(fr[k]); DL.append(bool(lab[k]))
        np.savez_compressed(f"{d}/drift.npz", frames=np.stack(DF), labels=np.array(DL))
        print("drift", len(DF), "positives", int(np.sum(DL)), flush=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 32, 64, 96, 128, 160]
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1) for i in range(5)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(160 * 4 * 8, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):                    # [B, 3, 128, 256] in [0,1] -> logit of success
        for c in self.convs:
            x = F.relu(c(x))
        return self.head(x).squeeze(-1)


def to_t(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)


@torch.no_grad()
def predict(net, X, bs=256):
    dev = next(net.parameters()).device                    # release: follow the net's device (GPU as before; CPU allowed)
    return np.concatenate([torch.sigmoid(net(to_t(X[i:i + bs]).to(dev))).cpu().numpy() for i in range(0, len(X), bs)])


def metrics(p, y, thr):
    yh = p >= thr
    return dict(n=int(len(y)), positives=int(y.sum()), agreement=float(np.mean(yh == y)),
                tpr=float(np.mean(yh[y])) if y.any() else None, tnr=float(np.mean(~yh[~y])) if (~y).any() else None)


def load_split(d, names):
    X, Y = [], []
    for f in names:
        z = np.load(f"{d}/rollouts/{f}")
        X.append(z["frames"]); Y.append(z["labels"])
    return np.concatenate(X), np.concatenate(Y).astype(bool)


def train(task_id):
    d = f"{OUT}/t{task_id}"
    split = json.load(open(f"{d}/split.json"))
    Xtr, Ytr = load_split(d, split["train"]); Xva, Yva = load_split(d, split["val"]); Xte, Yte = load_split(d, split["test"])
    log = []
    for attempt in range(3):
        torch.manual_seed(attempt)
        net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        pos, neg = np.where(Ytr)[0], np.where(~Ytr)[0]
        rs = np.random.RandomState(attempt)
        for ep in range(25):
            net.train()
            idx = np.concatenate([rs.choice(pos, 3000), rs.choice(neg, 3000)]); rs.shuffle(idx)   # balanced 50/50
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                loss = F.binary_cross_entropy_with_logits(net(to_t(Xtr[b]).cuda()), torch.from_numpy(Ytr[b].astype(np.float32)).cuda())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if ep % 5 == 4:
                net.eval(); pv = predict(net, Xva)
                log.append(dict(attempt=attempt, epoch=ep + 1, loss=float(loss), val_prob_std=float(pv.std()),
                                val_bal_acc_05=0.5 * (float(np.mean(pv[Yva] >= .5)) + float(np.mean(pv[~Yva] < .5)))))
                print(log[-1], flush=True)
        if log[-1]["val_prob_std"] > 0.05 and log[-1]["val_bal_acc_05"] > 0.8:
            break
    else:
        raise SystemExit("predicate training collapsed in 3 attempts -- stop and report")
    net.eval(); pv = predict(net, Xva)
    cands = np.unique(np.quantile(pv, np.linspace(0, 1, 400)))
    thr = float(cands[int(np.argmax([0.5 * (np.mean(pv[Yva] >= c) + np.mean(pv[~Yva] < c)) for c in cands]))])
    res = dict(task_id=task_id, model="CNN success classifier on the model-view frame (128x256, both cameras)",
               threshold=thr, train_log=log, n_train=int(len(Ytr)), n_train_pos=int(Ytr.sum()),
               val=metrics(pv, Yva, thr), holdout_real=metrics(predict(net, Xte), Yte, thr))
    for name in ("roundtrip", "gen_chunk", "drift"):
        z = np.load(f"{d}/{name}.npz")
        res[{"roundtrip": "holdout_vae_roundtrip", "gen_chunk": "model_generated_1chunk", "drift": "model_generated_full_rollout"}[name]] = \
            metrics(predict(net, z["frames"]), z["labels"].astype(bool), thr)
    path = f"{d}/predicate.pt"; torch.save(net.state_dict(), path)
    res["artifact"] = dict(path=os.path.relpath(path, _wmvp.P("$WMV_ROOT")),
                           sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
    gate = {k: (res[k]["tpr"] is not None and res[k]["tnr"] is not None and res[k]["tpr"] >= .85 and res[k]["tnr"] >= .85)
            for k in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk")}
    res["gate_85"] = gate
    json.dump(res, open(f"{d}/calibration.json", "w"), indent=1)
    print("RESULT", json.dumps({k: res[k] for k in ("threshold", "holdout_real", "holdout_vae_roundtrip",
                                                    "model_generated_1chunk", "model_generated_full_rollout", "gate_85")}))


if __name__ == "__main__":
    _wmvp.bootstrap()
    (gen_data if sys.argv[1] == "data" else train)(int(sys.argv[2]))
