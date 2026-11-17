"""Score OpenWAM-alpha records with the frozen OpenWAM-view predicate (protocol libero_openwam): fills the model-side
fields of every record that has model_verdict = null, plus the predicate readouts on the REAL frames (truth (a) replay
re-rendered in the OpenWAM view) that verify_run's check 7 needs. Per-frame scores are stored next to the trajectory.

Model verdict: success frames among imagined frames 1..100 (frame g = after 4g actions, steps 4..400); trajectory rule
= k_imagined_frames consecutive success frames (k steps -> ceil(k/4) frames). Real side: frames 1..400 with the same
detector(s) and k steps. Onset convention (eval_design section 28): the completion step is the START of the qualifying run.

  python ow_score.py --run base|famA|famP|famO|famB   (famO / famB: --tasks 3 only; t5 stays pending)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from ow_view import canvas_from_raw, pred_view, raw_obs

RUN_DIR = {"base": "openwam_base", "famA": "openwam_famA", "famP": "openwam_famP", "famO": "openwam_famO", "famB": "openwam_famB"}
BANK = {"base": f"{ROOT}/libero/out/bank/libero_anchors.json", "famA": f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json",
        "famP": f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json",
        "famO": f"{ROOT}/libero/out/bank/libero_anchors.json", "famB": f"{ROOT}/libero/out/bank/libero_anchors.json"}                    # family O: main-bank anchors, truth replay unoccluded
H = 400


def _render_real(args):
    """truth (a) replay of the unit's recorded actions in the OpenWAM view (worker process; see ow_predicate._label_start)."""
    from libero_env_adapter import _make_env, _anchor
    bank, anchor_id, action_file, trajectory_file = args
    a = dict(_anchor(bank, anchor_id), resolution=256)
    acts = np.load(f"{ROOT}/{action_file}")["env"][:H]
    env = _make_env(a, H=len(acts)); u = env.envs[0]
    o = raw_obs(u); F = [pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))]; S = []
    for x in acts:
        u.step(np.asarray(x, np.float32)); S.append(bool(u._env.check_success()))
        o = raw_obs(u); F.append(pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])))
    env.close()
    same = bool(np.array_equal(np.array(S), np.load(f"{ROOT}/{trajectory_file}")["real_success"]))
    return np.stack(F), same


def onset(ok, k):
    run = 0
    for i, x in enumerate(ok):
        run = run + 1 if x else 0
        if run >= k:
            return i - k + 1
    return None


def main():
    import torch, yaml
    from lingbot_predicate import Net, predict
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--tasks", default="5,3"); a_ = ap.parse_args()
    sec = yaml.safe_load(open(f"{ROOT}/protocol.yaml"))["libero_openwam"]["tasks"]
    out_dir = f"{ROOT}/libero/out/runs/{RUN_DIR[a_.run]}"
    for tid in [int(x) for x in a_.tasks.split(",")]:
        pr = sec[f"t{tid}"]["predicate"]; k = pr["trajectory_rule"]["k_steps"]; kf = pr["trajectory_rule"]["k_imagined_frames"]
        nets = {}
        if tid == 5:
            m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{json.load(open(f'{ROOT}/' + pr['calibration_path']))['artifact']['path']}")); m.eval(); nets["success"] = m
        else:
            man = json.load(open(f"{ROOT}/{pr['path']}"))
            for n, dd in man["detectors"].items():
                m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{dd['path']}")); m.eval(); nets[n] = m
        thr = pr["threshold_m"]

        def readout(frames):
            """per frame: success flag and feat_dist (margin readout; success iff feat_dist <= feat_dist_threshold)."""
            if tid == 5:
                p = predict(nets["success"], frames); return p >= thr, 1 - p, dict(p=p)
            pc, pb = predict(nets["close"], frames), predict(nets["bowl_out"], frames)
            ok = (pc >= thr["close"]) & (pb < thr["bowl_out"])
            return ok, -np.minimum(pc - thr["close"], thr["bowl_out"] - pb), dict(p_close=pc, p_bowl=pb)
        path = f"{out_dir}/records_t{tid}.jsonl"
        recs = [json.loads(l) for l in open(path)]
        todo = [i for i, r in enumerate(recs) if r.get("model_verdict") is None]
        jobs = [(BANK[a_.run], recs[i]["anchor_id"], recs[i]["action_file"], recs[i]["trajectory_file"]) for i in todo]
        with mp.get_context("spawn").Pool(2, maxtasksperchild=20) as pool:
            reals = pool.map(_render_real, jobs, chunksize=1)
        for i, (RF, same) in zip(todo, reals):
            r = recs[i]
            if not same:
                raise SystemExit(f"{r['unit_id']}: OpenWAM-view truth replay does not reproduce the recorded success trace -- stop")
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            IF = np.stack([pred_view(c) for c in tr["imagined_canvas"]])            # frames 0..100
            ok_i, fd_i, sc_i = readout(IF); ok_r, fd_r, sc_r = readout(RF)
            on_i = onset(ok_i[1:], kf); on_r = onset(ok_r[1:], k)
            np.savez_compressed(f"{out_dir}/trajectories/{r['unit_id']}_scores.npz", **{f"imag_{x}": v for x, v in sc_i.items()},
                                **{f"real_{x}": v for x, v in sc_r.items()})
            r.update(threshold=thr, predicate_sha256=pr["sha256"], criterion="reach_anytime",
                     model_verdict=on_i is not None, model_verdict_final=bool(ok_i[-1]),
                     model_first_success_step=(4 * (on_i + 1) if on_i is not None else None),
                     model_feat_dist=float(fd_i[1:].min()),
                     real_final_feat_dist=float(fd_r[1:].min()), predicate_on_real_final=on_r is not None,
                     predicate_on_real_final_step=bool(ok_r[-1]), trajectory_rule=dict(k_steps=k, k_imagined_frames=kf))
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        R = recs
        print(f"t{tid}", len(todo), "scored; model success", sum(bool(x["model_verdict"]) for x in R), "real", sum(x["real_verdict"] for x in R),
              "false-safe", sum(bool(x["model_verdict"]) and not x["real_verdict"] for x in R), flush=True)


if __name__ == "__main__":
    main()
