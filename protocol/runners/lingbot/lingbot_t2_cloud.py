"""LingBot-VA T2 (LIBERO-10 #2) production units on the cloud node $NODE_CLOUD (ruling 2026-11-01: baseline + A/P, O in
parallel; imagination to H = 400; closed loops run later on the cluster 5090s as the reference column, machine noted).

Per unit: imagination (26-27 chunks truncated to 400) + open-loop truth replay x2 (bit-identical required; a mismatch
is recorded, the unit is kept, and the run continues). Records + actions + latents land under $WMV_RESULTS so the
NAS auto-pull picks them up. Resumable (done set = union of records files of the family).

Seeds: imagination seed = RandomState(20261104).randint(0, 2**31-1, (n_anchors, 10))[anchor_index, k]; levels of a
family share the base anchor's seeds (t5/t3 convention).

  WMV_ROOT=... WMV_RESULTS=... python lingbot_t2_cloud.py <base|famP|famA|famO> <shard> <nshards>
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

ROOT = os.environ.get("WMV_ROOT", _wmvp.P("$WMV_ROOT"))
RES = os.environ.get("WMV_RESULTS", f"{ROOT}/libero/out/runs/lingbot_t2")
_wmvp.bootstrap()
MASTER, K, H2 = 20261104, 10, 400
BANKS = {"base": "libero/out/bank/libero_anchors_t2.json",
         "famO": "libero/out/bank/libero_anchors_t2.json",
         "famP": "libero/out/bank/familyP_t2/familyP_t2_anchors.json",
         "famA": "libero/out/bank/familyA_t2/familyA_t2_anchors.json"}
RATIOS = {"O0": None, "O1": 1 / 16, "O2": 1 / 8, "O3": 3 / 16, "O4": 1 / 4, "O5": 5 / 16}


def occlude(img, ratio):
    img = np.array(img, copy=True); h = img.shape[-3]; s = int(round(ratio * h)); o = (h - s) // 2
    img[..., o:o + s, o:o + s, :] = 0
    return img


def units_of(fam, bank):
    A = bank["anchors"]
    if fam == "base":
        return [(a, a["anchor_id"], None, i) for i, a in enumerate(A)]
    if fam == "famO":
        return [(a, a["anchor_id"], L, i) for i, a in enumerate(A) for L in ("O1", "O2", "O3", "O4", "O5")]
    bases = sorted({a["base_id"] for a in A})
    return [(a, a["base_id"], a["level"], bases.index(a["base_id"])) for a in A]


def main():
    fam, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    import torch
    import lingbot_unit as LU
    LU.H = H2                                                   # ruling 2026-11-01: T2 imagination stops at 400
    bank = json.load(open(f"{ROOT}/{BANKS[fam]}"))
    for a in bank["anchors"]:
        if a.get("bddl_file"):                                  # native_env.make_native joins a hard-coded root; an
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])  # absolute path passes through os.path.join unchanged
    rig = LU.LingBotRig(2)
    out = f"{RES}/{fam}"; os.makedirs(f"{out}/actions", exist_ok=True); os.makedirs(f"{out}/latents", exist_ok=True)
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(16, K))
    path = f"{out}/records.{shard}.jsonl"
    done = set()
    for f in os.listdir(out):
        if f.startswith("records."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{out}/{f}")}
    jobs = [(a, s0, L, ai, k) for a, s0, L, ai in units_of(fam, bank) for k in range(K)]
    jobs = [j for n, j in enumerate(jobs) if n % nsh == shard]
    for a, s0, L, ai, k in jobs:
        uid = f"lingbot_{s0}__k{k}" + (f"__{L}" if L else "")
        if uid in done:
            continue
        t0 = time.time(); seed = int(seeds[ai, k])
        if fam == "famO":                                       # occlude only the first real frame the model reads
            env, obs = rig.make_env(a)
            obs = dict(obs); obs["pixels"] = dict(obs["pixels"])
            obs["pixels"]["image"] = occlude(obs["pixels"]["image"], RATIOS[L])
            batch = rig.obs_to_batch(obs); env.close()
            rig.reset_policy(seed)
            img = imagine_with_batch(rig, batch)
        else:
            img = rig.imagine(a, seed)
        split = bool(os.environ.get("WMV_SKIP_REPLAY"))        # truth replays done by t2_replay.py on the CPU pool
        if not split:
            r1 = rig.replay(a, img["env_actions"]); r2 = rig.replay(a, img["env_actions"])
            same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["succ"], r2["succ"]))
            succ = r1["succ"]
        np.savez_compressed(f"{out}/actions/{uid}.npz", env=img["env_actions"])
        torch.save(img["latents"], f"{out}/latents/{uid}.pt")
        rec = dict(unit_id=uid, model="lingbot_va_libero_long", task_id=2, s0_id=s0, anchor_id=a["anchor_id"],
                   seed_index=k, imagination_seed=seed, seed_master=MASTER, setting_id=L or "none",
                   family=fam, H_env=H2, criterion="reach_anytime", n_chunks=img["n_chunks"],
                   real_verdict=None if split else bool(succ.any()),
                   real_first_success_step=None if split or not succ.any() else int(np.argmax(succ)) + 1,
                   replay_check_identical=None if split else same, truth_pending=split, closed_loop=None,
                   closed_loop_note="runs later on the cluster RTX 5090 (reference column; ruling 2026-11-01)",
                   action_sha256=hashlib.sha256(np.ascontiguousarray(img["env_actions"]).tobytes()).hexdigest(),
                   seconds=round(time.time() - t0, 1), seconds_imagine=round(img["seconds_imagine"], 1),
                   machine=os.uname().nodename, gpu=torch.cuda.get_device_name(0))
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, rec["real_verdict"], rec["real_first_success_step"], "same", rec["replay_check_identical"], rec["seconds"], flush=True)
    print("SHARD_DONE", fam, shard, "/", nsh)


def imagine_with_batch(rig, batch):
    """LU.imagine but starting from a pre-built (occluded) first batch."""
    import torch
    import lingbot_unit as LU
    p = rig.policy
    p._ensure_frozen_modules(); p._maybe_init_prompt(batch)
    t0 = time.time()
    acts_norm = [p.predict_action_chunk(batch)]
    lat = [p.last_predicted_latents]; c = 1
    while sum(a.shape[1] for a in acts_norm) < LU.H:
        fsid = 4 * c
        p._frame_st_id = fsid
        actions, latents = p._infer(None, frame_st_id=fsid)
        p._executed_actions = actions
        used = p.config.used_action_channel_ids
        a = actions[:, used].squeeze(-1).flatten(2).transpose(1, 2).contiguous().to(torch.float32)
        acts_norm.append(a); lat.append(latents.detach().to("cpu")); c += 1
    A = torch.cat(acts_norm, dim=1)[:, :LU.H]
    env_actions = np.concatenate([rig.to_env_action(A[:, t]) for t in range(LU.H)], 0)
    return dict(env_actions=env_actions, latents=torch.cat(lat, dim=2), n_chunks=c,
                seconds_imagine=time.time() - t0)


if __name__ == "__main__":
    main()
