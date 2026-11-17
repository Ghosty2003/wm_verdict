#!/usr/bin/env python3
"""Reproduction check for one cell: OpenWAM-alpha, T1 (LIBERO-10 #3, internal task id 3), no perturbation
(run openwam_base, records_t3.jsonl). Uses only this repository's code; data and predicate weights come from the
data root ($WMV_ROOT, see scripts/make_data_root.py).

  replay   per unit: a fresh env from the frozen anchor (libero_env_adapter._make_env, the runner's recipe: 256 px,
           episode length H + 100), then (a) the stored imagined env actions (truth (a)) and (b) the stored closed-loop
           actions are replayed open loop; per-step success and full-physics states must equal the archive bit for
           bit, and the first success steps must equal the record.
  score    the release predicate weights (OpenWAM view, t3: close AND NOT bowl_out; sha256 checked against the
           predicate manifest) score the stored imagined canvases on CPU; per-frame verdicts, model_verdict and
           model_first_success_step must equal the record; max |p - p_archived| is reported.
  verify   the release verify_run.py on a run (--verify-run, default openwam_base); every check's status must equal the
           archived report (checks added later are listed as new).
  --no-gl  GPU-less nodes have no OpenGL: robosuite's offscreen render context is replaced by a stub that returns
           blank images. Only camera observations are affected; physics, states and success are not.

  python scripts/repro_check_openwam_t1.py replay|score --units 0:50 --out DIR [--workers 8] [--no-gl]
Writes DIR/<part>_<a>_<b>.json (per unit) and prints a summary. Exit code 1 on any mismatch or code-provenance error.
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import sys
import time

import numpy as np

H, TID, RUN = 400, 3, "openwam_base"


def no_gl():
    import types
    import robosuite.environments.base as EB
    import robosuite.utils.binding_utils as BU

    class _Stub:
        def __init__(self, sim, device_id=-1, *a, **k):
            self.sim = sim
            self.vopt = types.SimpleNamespace(geomgroup=np.ones(6, np.uint8), flags=np.zeros(32, np.uint8))
            self.cam = types.SimpleNamespace(); self.scn = types.SimpleNamespace(flags=np.zeros(16, np.uint8))
            sim.add_render_context(self)                    # as robosuite's MjRenderContext.__init__ does

        def render(self, *a, **k):
            pass

        def read_pixels(self, width, height, depth=False, segmentation=False):
            img = np.zeros((height, width, 3), np.uint8)
            return (img, np.zeros((height, width), np.float32)) if depth else img

        def __getattr__(self, name):
            return lambda *a, **k: None

    BU.MjRenderContextOffscreen = _Stub
    EB.MjRenderContextOffscreen = _Stub


def provenance():
    """every imported module of ours must come from this repository; report sys.path and offending modules."""
    forbid = [os.path.realpath(p) for p in os.environ.get("WMV_FORBID_CODE", "").split(":") if p]
    repo = os.path.realpath(_wmvp.REPO)
    bad_mod, ours = [], []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or not os.path.isabs(f):
            continue
        rf = os.path.realpath(f)
        if rf.startswith(repo + os.sep):
            ours.append(name)
        elif any(rf.startswith(p + os.sep) for p in forbid):
            bad_mod.append((name, rf))
    bad_path = [p for p in sys.path if p and any(os.path.realpath(p).startswith(x + os.sep) or os.path.realpath(p) == x for x in forbid)
                and not os.path.realpath(p).startswith(repo)]
    return dict(repo=repo, forbidden_prefixes=forbid, modules_from_repo=sorted(ours), modules_from_forbidden=bad_mod,
                sys_path_into_forbidden=bad_path, ok=not bad_mod and not bad_path, script=os.path.realpath(__file__))


def records():
    return [json.loads(l) for l in open(_wmvp.P(f"$WMV_ROOT/libero/out/runs/{RUN}/records_t{TID}.jsonl"))]


def _first(succ):
    return int(np.argmax(succ)) + 1 if np.any(succ) else None


def replay_unit(args):
    i, use_stub = args
    if use_stub:
        no_gl()
    from libero_env_adapter import _anchor, _make_env
    from sim_server import _phys                         # the runner's simulator side (same state snapshot)
    r = records()[i]
    anc = _anchor(_wmvp.P("$WMV_ROOT/libero/out/bank/libero_anchors.json"), r["anchor_id"])
    A = np.load(_wmvp.P(f"$WMV_ROOT/{r['action_file']}")); T = np.load(_wmvp.P(f"$WMV_ROOT/{r['trajectory_file']}"))

    def run(acts):
        env = _make_env(dict(anc, resolution=256), H=H + 100); u = env.envs[0]
        S, succ = [_phys(u)], []
        for a in acts:
            u.step(np.asarray(a, np.float32)); succ.append(bool(u._env.check_success())); S.append(_phys(u))
        env.close()
        return np.array(succ), np.stack(S)
    t0 = time.time()
    s_a, S_a = run(A["env"]); s_c, S_c = run(A["closed_loop"])
    cS = T["closed_loop_states"]; cS_cmp = S_c if len(cS) == len(S_c) else S_c[1:]
    out = dict(unit_id=r["unit_id"],
               truth_success_identical=bool(np.array_equal(s_a, T["real_success"])),
               truth_states_identical=bool(np.array_equal(S_a, T["replay1"])),
               truth_first_success=_first(s_a), truth_first_success_record=r["real_first_success_step"],
               closed_success_identical=bool(np.array_equal(s_c, T["closed_loop_success"])),
               closed_states_identical=bool(np.array_equal(cS_cmp, cS)),
               closed_first_success=_first(s_c), closed_first_success_record=r["closed_loop"]["first_success_step"],
               n_truth_steps=int(len(s_a)), n_closed_steps=int(len(s_c)), seconds=round(time.time() - t0, 1))
    pv = provenance(); out["worker_provenance_ok"] = pv["ok"]; out["worker_modules_from_forbidden"] = pv["modules_from_forbidden"]
    out["worker_repo_modules"] = [m for m in pv["modules_from_repo"] if m in ("libero_env_adapter", "sim_server", "ow_view", "wmv_paths")]
    out["ok"] = bool(pv["ok"] and out["truth_success_identical"] and out["truth_states_identical"] and out["closed_success_identical"]
                     and out["closed_states_identical"] and out["truth_first_success"] == out["truth_first_success_record"]
                     and out["closed_first_success"] == out["closed_first_success_record"])
    return out


def score(idx):
    import torch
    import yaml
    from lingbot_predicate import Net, predict
    from ow_score import onset
    from ow_view import pred_view
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    pr = yaml.safe_load(open(_wmvp.P("$WMV_ROOT/protocol.yaml")))["libero_openwam"]["tasks"][f"t{TID}"]["predicate"]
    man = json.load(open(_wmvp.P(f"$WMV_ROOT/{pr['path']}")))
    nets, wsha = {}, {}
    for n, d in man["detectors"].items():
        p = _wmvp.P(f"$WMV_ROOT/{d['path']}")
        wsha[n] = dict(file=os.path.realpath(p), sha256_ok=hashlib.sha256(open(p, "rb").read()).hexdigest() == d["sha256"])
        m = Net(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); nets[n] = m
    thr = pr["threshold_m"]; kf = pr["trajectory_rule"]["k_imagined_frames"]
    R = records(); res = []
    for i in idx:
        r = R[i]; T = np.load(_wmvp.P(f"$WMV_ROOT/{r['trajectory_file']}"))
        F = np.stack([pred_view(c) for c in T["imagined_canvas"]])
        with torch.no_grad():
            pc, pb = predict(nets["close"], F), predict(nets["bowl_out"], F)
        ok = (pc >= thr["close"]) & (pb < thr["bowl_out"])
        on = onset(ok[1:], kf)
        Z = np.load(_wmvp.P(f"$WMV_ROOT/libero/out/runs/{RUN}/trajectories/{r['unit_id']}_scores.npz"))
        ok_arch = (Z["imag_p_close"] >= thr["close"]) & (Z["imag_p_bowl"] < thr["bowl_out"])
        mv, mfs = on is not None, (4 * (on + 1) if on is not None else None)
        d = dict(unit_id=r["unit_id"], model_verdict=mv, model_verdict_record=r["model_verdict"],
                 model_first_success_step=mfs, model_first_success_step_record=r["model_first_success_step"],
                 frame_verdicts_identical_to_archive=bool(np.array_equal(ok, ok_arch)),
                 max_abs_dp_close=float(np.abs(pc - Z["imag_p_close"]).max()), max_abs_dp_bowl=float(np.abs(pb - Z["imag_p_bowl"]).max()),
                 n_frames=int(len(F)))
        d["ok"] = bool(mv == r["model_verdict"] and mfs == r["model_first_success_step"] and d["frame_verdicts_identical_to_archive"])
        res.append(d)
    return res, wsha


def verify(out, use_stub, run=RUN):
    """run the release verify_run.py on a run (all checks) and compare each check's status with the archived report;
    checks added after the archived report was written are listed as new (status_archived = None) and not compared."""
    import runpy
    if use_stub:
        no_gl()
    rep = os.path.join(out, f"verify_report_{run}.json")
    sys.argv = ["verify_run.py", "--run-dir", _wmvp.P("$WMV_ROOT"), "--protocol", _wmvp.P("$WMV_ROOT/protocol.yaml"), "--run", run,
                "--report", rep]
    try:
        runpy.run_path(_wmvp.code("verify_run.py"), run_name="__main__")
    except SystemExit:
        pass
    import glob
    olds = glob.glob(_wmvp.P(f"$WMV_ROOT/libero/out/runs/*/verify_report_{run}.json"))
    new = json.load(open(rep)); old = json.load(open(olds[0]))
    so = {c["id"]: c["status"] for c in old["checks"]}; sn = {c["id"]: c["status"] for c in new["checks"]}
    res = [dict(unit_id=f"check{k}", status=sn.get(k), status_archived=so.get(k), new_check=k not in so, ok=(k not in so) or sn.get(k) == so.get(k))
           for k in sorted(set(so) | set(sn))]
    c1 = next((c for c in new["checks"] if c["id"] == 1), {})
    return res, dict(verify_run_name=run, overall=new.get("overall"), overall_archived=old.get("overall"),
                     check1_release_copy_hashes_used=(c1.get("values") or {}).get("release_copy_hashes_used", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["replay", "score", "verify"]); ap.add_argument("--units", default="0:50")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--no-gl", action="store_true")
    ap.add_argument("--verify-run", default=RUN, help="verify part: protocol run name (e.g. openwam_famA_obj)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.units.split(":")); idx = list(range(lo, min(hi, len(records()))))
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    if a.part == "replay":
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
            res = pool.map(replay_unit, [(i, a.no_gl) for i in idx], chunksize=1)
        extra = {}
        import libero_env_adapter, sim_server  # noqa: F401  (load them here too so provenance covers this process)
    elif a.part == "score":
        res, wsha = score(idx); extra = dict(weights=wsha)
    else:
        res, extra = verify(a.out, a.no_gl, a.verify_run)
    prov = provenance()
    summary = dict(part=a.part, run=RUN, task_id=TID, units=len(res), ok=sum(r["ok"] for r in res), host=socket.gethostname(),
                   no_gl=a.no_gl, seconds=round(time.time() - t0, 1), provenance=prov, **extra, units_detail=res)
    path = os.path.join(a.out, f"{a.part}_{lo}_{hi}.json" if a.part != "verify" else f"verify_{a.verify_run}.json")
    json.dump(summary, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("units_detail", "provenance")}),
          "provenance_ok", prov["ok"], "->", path, flush=True)
    sys.exit(0 if summary["ok"] == len(res) and prov["ok"] else 1)


if __name__ == "__main__":
    main()
