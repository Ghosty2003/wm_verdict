"""Closed loop (reference column) of OpenWAM-alpha under the non-input families E, G, N (ruling 2026-10-26). Same loop as
ow_runner.closed_loop (per block of 32: real canvas + real proprio -> generate -> the simulator executes the block; to
H = 400), with the perturbation passed to sim_server's reset (physics right after reset, action stream on every executed
step). Same anchor and seed as the baseline unit. Two runs per unit; hard requirement = same success verdict.
Sham levels are not rerun.

  python phys_closed_openwam.py [fam,...]   (openwam interpreter)   -> out/runs/openwam_phys/closed_t<k>.jsonl
  python phys_closed_openwam.py <shard> <nshards> [fam,...]           -> out/runs/openwam_phys/closed_t<k>.<shard>.jsonl
  (units already in ANY closed_t<k>*.jsonl are skipped, so sharded and unsharded runs never duplicate a unit)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import glob
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
import ow_runner as W
from phys_perturb import FAMILIES, settings, spec_of

OUT = f"{ROOT}/libero/out/runs/openwam_phys"


def closed(eng, sim, anchor, prompt, seed, spec):
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    f = f"{sim.scratch}/obs.npz"; sim.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, perturb=spec); o = dict(np.load(f))
    cmd, ex, succ, imag = [], [], [], []
    for c in range(W.N_CALLS):
        img = Image.fromarray(W.canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        # store-all-data ruling 2026-11-08: keep every imagined canvas frame of every call
        imag.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(cmd)]
        r = sim.step(A); cmd += A; ex += list(r["executed"]); succ += r["success"].tolist(); o = r
    return np.stack(cmd), np.stack(ex), np.array(succ), imag


def main():
    a = sys.argv[1:]
    shard, nsh = (int(a[0]), int(a[1])) if a and a[0].isdigit() else (0, 1)
    a = a[2:] if a and a[0].isdigit() else a
    fams = a[0].split(",") if a else list(FAMILIES)
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_phys_{os.getpid()}")
    from libero_lang import LANG
    A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    os.makedirs(f"{OUT}/closed", exist_ok=True)
    for t in ("t5", "t3"):
        path = f"{OUT}/closed_{t}.jsonl" if nsh == 1 else f"{OUT}/closed_{t}.{shard}.jsonl"
        done = {json.loads(l)["unit_id"] for f in glob.glob(f"{OUT}/closed_{t}*.jsonl") for l in open(f)}
        n_ = -1
        for l in open(f"{ROOT}/libero/out/runs/openwam_base/records_{t}.jsonl"):
            r = json.loads(l)
            for fam in fams:
                for s in settings(fam)[1:]:
                    uid = f"{r['unit_id']}__{s}"; n_ += 1
                    if n_ % nsh != shard or uid in done:
                        continue
                    spec = spec_of(s, int(t[1:]), r["unit_id"]); t0 = time.time(); seed = r["imagination_seed"]
                    c1, e1, s1, i1 = closed(eng, sim, A[r["s0_id"]], LANG[int(t[1:])], seed, spec)
                    c2, e2, s2, i2 = closed(eng, sim, A[r["s0_id"]], LANG[int(t[1:])], seed, spec)
                    n = min(len(e1), len(e2)); diff = np.where(np.any(e1[:n] != e2[:n], axis=1))[0]
                    div = int(diff[0]) + 1 if len(diff) else None
                    f1 = int(np.argmax(s1)) + 1 if s1.any() else None; f2 = int(np.argmax(s2)) + 1 if s2.any() else None
                    np.savez_compressed(f"{OUT}/closed/{uid}.npz", commanded=c1, executed=e1, success=s1, commanded2=c2, executed2=e2, success2=s2,
                                        imagined=np.concatenate(i1), imagined_call_lens=np.array([len(x) for x in i1]),
                                        imagined2=np.concatenate(i2), imagined2_call_lens=np.array([len(x) for x in i2]))
                    rec = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, perturbation=spec, machine=os.uname().nodename, checkpoint=ckname,
                               closed_loop=dict(success=f1 is not None, first_success_step=f1, steps=len(e1),
                                                deterministic_rerun_identical=div is None,
                                                rerun=dict(success=f2 is not None, first_success_step=f2, steps=len(e2)),
                                                first_divergence_step=div, conclusion_consistent=(f1 is None) == (f2 is None)),
                               closed_loop_file=os.path.relpath(f"{OUT}/closed/{uid}.npz", ROOT), seconds=round(time.time() - t0, 1))
                    with open(path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(json.dumps(dict(unit=uid, s1=f1, s2=f2, div=div, sec=rec["seconds"])), flush=True)
    sim.close(); print("DONE")


_CTX = {}


def SHAM_RUN(r, t):
    if not _CTX:
        W._setup(); eng, pre, arch, _, ck = W._engine(); _CTX.update(eng=eng, sim=W.Sim(f"/dev/shm/ow_physsham_{os.getpid()}"))
    from libero_lang import LANG
    A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    c, e, s = closed(_CTX["eng"], _CTX["sim"], A[r["s0_id"]], LANG[int(t[1:])], r["imagination_seed"], spec_of("Em0", int(t[1:])))
    return c


def sham_check(n=5):
    """closed-loop code path check: the sham level (Em0) must reproduce the baseline's stored closed-loop actions
    bit for bit on the first n units of each task."""
    import json as _j
    ok = []
    for t in ("t5", "t3"):
        recs = [_j.loads(l) for l in open(f"{ROOT}/libero/out/runs/openwam_base/records_{t}.jsonl")][:n]
        for r in recs:
            ref = np.load(f"{ROOT}/{r['action_file']}")["closed_loop"]
            got = SHAM_RUN(r, t)
            same = bool(got.shape == ref.shape and np.array_equal(got, ref))
            ok.append(same); print("sham", r["unit_id"], "identical", same, got.shape, ref.shape, flush=True)
    print("SHAM_CHECK", "PASS" if all(ok) else "FAIL", sum(ok), "/", len(ok))
    if not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sham":
        sham_check()
    else:
        main()
