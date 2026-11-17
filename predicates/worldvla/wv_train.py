"""WorldVLA-view predicate (appendix only; WorldVLA = exclusion E4): k-fold training + gate, same recipe as v3.

Inputs: out/predicate_wv/t<task>/frames/*.npz (128x128 WorldVLA view of every v3 episode, recal_wv.py render),
roundtrip_all.npz (VQGAN round trip, wv_roundtrip.py), and for the generated row the WorldVLA baseline units
(out/worldvla/<anchor>/s1/imagined.npz frames, labelled by the paired truth (a) replay at the same step).
Folds = recal_v3 folds (by episode, stratified by source). Threshold rule = v3 (OOF real TPR, TNR, failure-terminal TNR
>= 0.85; logit midpoint of the largest feasible run). Generated row: timing window = one WorldVLA action chunk (10
steps) around the replay's first success step (ruling 1 rule, chunk length of this model); raw and rule both reported.
Generated frames never enter training; they are scored with the model refit on all episodes.

  python wv_train.py train <task>     # k-fold + refit + real / round-trip rows
  python wv_train.py gen <task>       # generated row from the WorldVLA baseline units (needs the units)
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_wmvp.bootstrap()
from recal_v3 import registry, GATE, K
from lingbot_predicate import metrics

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/predicate_wv"
WINDOW = 10


class NetWV(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 32, 64, 96, 128, 160]
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1) for i in range(5)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(160 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        for cv in self.convs:
            x = F.relu(cv(x))
        return self.head(x).squeeze(-1)


def to_t(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)


@torch.no_grad()
def predict(net, X, bs=512):
    return np.concatenate([torch.sigmoid(net(to_t(X[i:i + bs]).cuda())).cpu().numpy() for i in range(0, len(X), bs)])


def fail_terminal_mask(labels):
    m = np.zeros(len(labels), bool)
    if not labels.any():
        m[len(labels) // 2:] = True
    return m


def fit(X, Y, M, seed):
    torch.manual_seed(seed)
    net = NetWV().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    pos = np.where(Y)[0]; negf = np.where(~Y & M)[0]; nego = np.where(~Y & ~M)[0]
    rs = np.random.RandomState(seed)
    for ep in range(25):
        net.train()
        idx = np.concatenate([rs.choice(pos, 3000), rs.choice(negf, 1500), rs.choice(nego, 1500)]); rs.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(net(to_t(X[b]).cuda()), torch.from_numpy(Y[b].astype(np.float32)).cuda())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net


def row(p, y, ft, thr):
    out = metrics(p, y, thr)
    out["fail_terminal"] = metrics(p[ft & ~y], y[ft & ~y], thr) if (ft & ~y).any() else None
    return out


def train(tid):
    d = f"{OUT}/t{tid}"
    E = registry(tid); fold = {e["id"]: e["fold"] for e in E}
    X, Y, M, EP = [], [], [], []
    for e in E:
        z = np.load(f"{d}/frames/{e['id']}.npz"); X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        EP += [e["id"]] * len(z["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(bool); M = np.concatenate(M); EP = np.array(EP)
    FO = np.array([fold[x] for x in EP])
    rt = np.load(f"{d}/roundtrip_all.npz"); fo_rt = np.array([fold[x] for x in rt["episode"]])
    oof = np.zeros(len(Y)); oof_rt = np.zeros(len(rt["labels"]))
    for k in range(K):
        tr = FO != k
        net = fit(X[tr], Y[tr], M[tr], seed=k)
        oof[~tr] = predict(net, X[~tr]); oof_rt[fo_rt == k] = predict(net, rt["frames"][fo_rt == k])
        print(f"t{tid} fold {k} done (oof std {oof[~tr].std():.3f})", flush=True)
        del net; torch.cuda.empty_cache()
    zs = np.arange(-12, 12.0001, 0.05); ps = 1 / (1 + np.exp(-zs)); fneg = M & ~Y
    feas = np.array([bool((oof[Y] >= p_).mean() >= GATE and (oof[~Y] < p_).mean() >= GATE and (oof[fneg] < p_).mean() >= GATE) for p_ in ps])
    runs, cur = [], None
    for i, f in enumerate(feas):
        if f and cur is None:
            cur = i
        if (not f or i == len(feas) - 1) and cur is not None:
            runs.append((cur, i if f else i - 1)); cur = None
    res = dict(task_id=tid, version="WorldVLA-view (appendix; E4)", view="agentview 256 rotated 180 -> 128x128, no wrist",
               feasible_runs=[[float(ps[a]), float(ps[b])] for a, b in runs])
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0]); thr = float(1 / (1 + np.exp(-(zs[a] + zs[b]) / 2)))
    else:
        thr = float(ps[int(np.argmax([0.5 * ((oof[Y] >= p_).mean() + (oof[~Y] < p_).mean()) for p_ in ps]))])
        res["stop"] = "no feasible threshold (predicate failure)"
    res["threshold"] = thr if runs else None; res["diagnostic_threshold"] = thr
    res["holdout_real"] = row(oof, Y, M, thr)
    res["holdout_vqgan_roundtrip"] = row(oof_rt, rt["labels"].astype(bool), rt["fail_terminal"], thr)
    net = fit(X, Y, M, seed=100); path = f"{d}/predicate.pt"; torch.save(net.state_dict(), path)
    res["artifact"] = dict(path=os.path.relpath(path, ROOT), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
    json.dump(res, open(f"{d}/calibration.json", "w"), indent=1)
    print("RESULT", json.dumps({k: res[k] for k in ("threshold", "feasible_runs", "holdout_real", "holdout_vqgan_roundtrip")})[:900])


def gen(tid):
    d = f"{OUT}/t{tid}"; cal = json.load(open(f"{d}/calibration.json")); thr = cal["threshold"] or cal["diagnostic_threshold"]
    net = NetWV().cuda(); net.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}")); net.eval()
    P, Y, FT, WIN, UNIT = [], [], [], [], []
    for j in range(5):
        u = f"{ROOT}/libero/out/worldvla/libero10_t{tid}_a{j}/s1"
        if not os.path.exists(f"{u}/replay.npz"):
            continue
        fr = np.load(f"{u}/imagined.npz")["frames"][1:]; succ = np.load(f"{u}/replay.npz")["succ"][:len(fr)]
        fr = fr[:len(succ)]
        X = np.stack([np.asarray(Image.fromarray(f).resize((128, 128), Image.BILINEAR)) for f in fr])
        p = predict(net, X); s_ref = int(np.argmax(succ)) + 1 if succ.any() else None
        steps = np.arange(1, len(succ) + 1)
        win = (~succ) & (np.abs(steps - s_ref) <= WINDOW) if s_ref else np.zeros(len(succ), bool)
        P.append(p); Y.append(succ); FT.append(fail_terminal_mask(succ)); WIN.append(win); UNIT += [f"t{tid}_a{j}"] * len(succ)
    P = np.concatenate(P); Y = np.concatenate(Y).astype(bool); FT = np.concatenate(FT); WIN = np.concatenate(WIN)
    raw = row(P, Y, FT, thr); keep = ~WIN; rule = row(P[keep], Y[keep], FT[keep], thr)
    cal["model_generated_full_rollout"] = dict(raw=raw, timing_window_rule=rule, window_steps=WINDOW, units=sorted(set(UNIT)),
                                               labels="paired truth (a) replay success at the same step")
    json.dump(cal, open(f"{d}/calibration.json", "w"), indent=1)
    print("GEN", json.dumps(dict(raw=raw, rule=rule))[:700])


if __name__ == "__main__":
    {"train": train, "gen": gen}[sys.argv[1]](int(sys.argv[2]))
