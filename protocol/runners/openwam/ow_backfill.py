"""Closed-loop imagination backfill for OpenWAM-alpha (store-all-data ruling 2026-11-08; full backfill ordered
2026-11-09). Every recorded OpenWAM closed-loop unit is re-run x2 with exactly the loop that produced it, now keeping
for every planning call (13 per run): the frames the model imagined (out["video"]) and the real agentview frame it read
(after occlusion / corruption, i.e. what the model saw). Original records are never touched.

Per unit -> <run_dir>/closed_frames/<uid>.npz
            <run_dir>/closed_frames_check.<shard>.jsonl  (rerun success / first step vs the recorded closed loop)
Units whose frames already exist (patched runs: closed/<uid>.npz with `imagined`, famB/closed_imagined/<uid>.npz)
are skipped.

  python ow_backfill.py <shard> <nshards>   (openwam interpreter, GPU)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys
import time
import zlib

import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
_wmvp.bootstrap()

PROMPT_TX = {2: "turn on the stove and put the moka pot on it",
             4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
             6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}
RATIO_LIST = [None, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16]            # phys/lingbot_famO.RATIOS (t3/t5 famO)
RATIO_TX = {"O1": 1 / 16, "O2": 1 / 8, "O3": 3 / 16, "O4": 1 / 4, "O5": 5 / 16}   # out/task2/ow_tx_run.RATIOS


def rows(pattern):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pattern}")):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def bank(path):
    b = json.load(open(path))["anchors"]
    for a in b:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
    return {a["anchor_id"]: a for a in b}


def jobs():
    """(run_dir, uid, kind, anchor, prompt, seed, extra, recorded_closed_loop)."""
    from libero_lang import LANG
    from phys_perturb import spec_of
    J = []
    main = bank(f"{ROOT}/libero/out/bank/libero_anchors.json")
    # ---- t3 / t5 (main bank, default render resolution) ----
    base35 = {r["unit_id"]: r for r in rows("openwam_base/records_t*.jsonl")}
    for r in base35.values():
        J.append(("openwam_base", r["unit_id"], "v1", main[r["s0_id"]], LANG[r["task_id"]], r["imagination_seed"], None, r["closed_loop"]))
    for fam, bp in (("famA", "familyA/familyA_anchors.json"), ("famP", "familyP/familyP_anchors.json")):
        B = bank(f"{ROOT}/libero/out/bank/{bp}")
        for r in rows(f"openwam_{fam}/records_t*.jsonl"):
            J.append((f"openwam_{fam}", r["unit_id"], "v1", B[r["anchor_id"]], LANG[r["task_id"]], r["imagination_seed"], None, r["closed_loop"]))
    for r in rows("openwam_famO/records_t*.jsonl"):
        i = int(r["setting_id"][1:])
        J.append(("openwam_famO", r["unit_id"], "v1", main[r["s0_id"]], LANG[r["task_id"]], r["imagination_seed"], RATIO_LIST[i], r["closed_loop"]))
    for r in rows("openwam_phys/closed_t*.jsonl"):
        b = base35[r["base_unit"]]; tid = b["task_id"]
        J.append(("openwam_phys", r["unit_id"], "v2", main[b["s0_id"]], LANG[tid], b["imagination_seed"],
                  spec_of(r["setting_id"], tid, r["base_unit"]), r["closed_loop"]))
    for r in rows("openwam_famB/records_t*.jsonl"):
        J.append(("openwam_famB", r["unit_id"], "v3", main[r["s0_id"]], LANG[r["task_id"]], r["imagination_seed"], r["setting_id"], r["closed_loop"]))
    # ---- t2 / t4 / t6 (per-task banks, 256 render; famB 128) ----
    for tid in (2, 4, 6):
        run = f"openwam_t{tid}"
        B0 = bank(f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json")
        banks = {"base": B0, "famO": B0,
                 "famP": bank(f"{ROOT}/libero/out/bank/familyP_t{tid}/familyP_t{tid}_anchors.json"),
                 "famA": bank(f"{ROOT}/libero/out/bank/familyA_t{tid}/familyA_t{tid}_anchors.json")}
        for fam in ("base", "famO", "famP", "famA"):
            for r in rows(f"{run}/records_{fam}.jsonl"):
                ratio = RATIO_TX.get(r["setting_id"]) if fam == "famO" else None
                J.append((run, r["unit_id"], "v1", dict(banks[fam][r["anchor_id"]], resolution=256), PROMPT_TX[tid],
                          r["imagination_seed"], ratio, r["closed_loop"]))
        if tid == 2:      # extra T2 base closed batch (same units / seeds / loop as records_base -> same uid, deduped)
            for r in rows(f"{run}/closed.jsonl"):
                J.append((run, r["unit_id"], "v1", dict(B0[r["anchor_id"]], resolution=256), PROMPT_TX[tid], r["imagination_seed"], None, r["closed_loop"]))
        baseT = {r["unit_id"]: r for r in rows(f"{run}/records_base.jsonl")}
        for r in rows(f"{run}/phys/closed.*.jsonl"):
            b = baseT[r["base_unit"]]
            J.append((run, r["unit_id"], "v2", dict(B0[b["anchor_id"]], resolution=256), PROMPT_TX[tid], b["imagination_seed"],
                      spec_of(r["setting_id"], tid, r["base_unit"]), r["closed_loop"]))
        for r in rows(f"{run}/famB/records*.jsonl"):
            kind = "v4" if r["setting_id"] == "B0" else "v3"
            J.append((f"{run}/famB", r["unit_id"], kind, B0[r["anchor_id"]], PROMPT_TX[tid], r["imagination_seed"], r["setting_id"], r["closed_loop"]))
    seen, out = set(), []
    for j in J:
        if (j[0], j[1]) not in seen:
            seen.add((j[0], j[1])); out.append(j)
    return out


def has_frames(run, uid):
    if os.path.exists(f"{RUNS}/{run}/closed_frames/{uid}.npz"):
        return True
    p = f"{RUNS}/{run}/closed/{uid}.npz"                       # patched owpc runs (2026-11-08 onward)
    if run.startswith("openwam_t") and "/" not in run and os.path.exists(p):
        try:
            return "imagined" in np.load(p).files
        except Exception:
            return False
    if run.endswith("/famB") and os.path.exists(f"{RUNS}/{run}/closed_imagined/{uid}.npz"):
        return True
    return False


def run_loop(W, eng, sim, corr, kind, anchor, prompt, seed, extra, uid):
    """One closed loop, identical to the producing runner's; returns actions, success, imagined, read frames."""
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    from lingbot_famO import occlude
    if kind == "v2":
        f = f"{sim.scratch}/obs.npz"; sim.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, perturb=extra); o = dict(np.load(f))
    elif kind == "v3":
        from ow_famB import reset128
        o = reset128(sim, anchor)
    elif kind == "v4":
        o = sim.reset(dict(anchor, resolution=128))
    else:
        o = sim.reset(anchor)
    acts, succ, imag, read = [], [], [], []
    for c in range(W.N_CALLS):
        if kind == "v3":
            from ow_famB import corrupt
            oc = corrupt(o, corr, extra, uid, len(acts), None)
            av, wr = oc["agentview_image"], oc["robot0_eye_in_hand_image"]
        else:
            av = o["agentview_image"]
            if kind == "v1" and c == 0 and extra is not None:
                av = occlude(av, extra)
            wr = o["robot0_eye_in_hand_image"]
        read.append(np.asarray(av, np.uint8))
        img = Image.fromarray(W.canvas_from_raw(av, wr))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        imag.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(acts)]
        r = sim.step(A); acts += A; succ += r["success"].tolist(); o = r
    return np.stack(acts), np.array(succ, bool), imag, read


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import ow_runner as W
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_backfill_{os.getpid()}")
    corr = None
    allj = jobs()
    todo = [j for j in allj if zlib.crc32(f"{j[0]}/{j[1]}".encode()) % nsh == shard and not has_frames(j[0], j[1])]
    print("OW_BACKFILL", "all", len(allj), "this shard todo", len(todo), flush=True)
    for run, uid, kind, anchor, prompt, seed, extra, rec in todo:
        if kind == "v3" and corr is None:
            from bcorrupt import Corruptor
            corr = Corruptor()
        t0 = time.time()
        res = [run_loop(W, eng, sim, corr, kind, anchor, prompt, seed, extra, uid) for _ in range(2)]
        os.makedirs(f"{RUNS}/{run}/closed_frames", exist_ok=True)
        d = {}
        for i, (acts, succ, imag, read) in enumerate(res, 1):
            sfx = "" if i == 1 else "2"
            d.update({f"imagined{sfx}": np.concatenate(imag), f"imagined{sfx}_call_lens": np.array([len(x) for x in imag]),
                      f"read{sfx}": np.stack(read), f"actions{sfx}": acts, f"success{sfx}": succ})
        tmp = f"{RUNS}/{run}/closed_frames/{uid}.tmp.npz"
        np.savez_compressed(tmp, **d); os.replace(tmp, f"{RUNS}/{run}/closed_frames/{uid}.npz")
        fs = [int(np.argmax(s)) + 1 if s.any() else None for _, s, _, _ in res]
        row = dict(unit_id=uid, run=run, kind=kind, success=[bool(s.any()) for _, s, _, _ in res], first_success_step=fs,
                   recorded_success=rec.get("success"), recorded_first_success_step=rec.get("first_success_step"),
                   matches_recorded=bool(res[0][1].any()) == bool(rec.get("success")),
                   frames_file=f"libero/out/runs/{run}/closed_frames/{uid}.npz", checkpoint=ckname,
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        with open(f"{RUNS}/{run}/closed_frames_check.{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(uid, row["success"], "rec", row["recorded_success"], row["seconds"], flush=True)
    sim.close()
    print("OW_BACKFILL_DONE shard", shard)


if __name__ == "__main__":
    main()
