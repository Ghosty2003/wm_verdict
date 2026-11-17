"""B family predicate robustness gate, TRAJECTORY level (ruling 2026-11-01, replacing the per-frame reading of
bgate.py: the deployed verdicts are trajectory rules, so the gate is scored the way the predicate is used).

Real episodes: truth replays of the model's own main-bank baseline units (as bgate.py). Every real frame the rule
reads is corrupted per type and severity (seed rule with unit id "gate_<unit>"); truth label = simulator
check_success reach-anytime over env steps 1..400. Per (task x setting): TPR / TNR over episodes, 85 % line.

  LingBot-VA   scores on EVERY env step 1..400 (full decoding is the deployed reading);
               t5 verdict = v3 score >= thr on k = 24 consecutive steps; t3 = v4 conjunction on k = 1 step.
  OpenWAM-a    canvas every 4th env step (the model's imagined-frame stride); t3 verdict = frozen conjunction on
               k = 1 sampled frame among steps 4..400; t5 has no deployed predicate. clean_256 reference row kept.

  python bgate_traj.py lingbot|openwam   (lerobot_libero interpreter, GPU)   -> out/b_family/gate/gate_traj_<model>.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from bcorrupt import SETTINGS, Corruptor

OUT = f"{ROOT}/libero/out/b_family/gate"
H, LINE = 400, 0.85
K_T5, K_T3 = 24, 1                                               # cells_final.py (not imported: scipy-free here)


def fires(h, k):
    run = 0
    for x in h:
        run = run + 1 if x else 0
        if run >= k:
            return True
    return False


def rates(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tpr = float(pred[y].mean()) if y.any() else None; tnr = float((~pred[~y]).mean()) if (~y).any() else None
    return dict(n=int(len(y)), positives=int(y.sum()), tpr=tpr, tnr=tnr,
                pass_85=bool((tpr is None or tpr >= LINE) and (tnr is None or tnr >= LINE)))


def lingbot():
    import torch
    import yaml  # noqa: F401  (environment check)
    from lingbot_unit import LingBotRig
    from lingbot_predicate import Net, predict, model_view, batched
    corr = Corruptor(); A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    res = {}
    for tid in (5, 3):
        rig = LingBotRig(tid)
        if tid == 5:
            cal = json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))
            net = Net().cuda(); net.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}")); net.eval()
            ok_of = lambda F: predict(net, F) >= float(cal["threshold"])
            verdict_of = lambda ok: fires(ok[1:], K_T5)          # steps 1..H, k consecutive
        else:
            V4 = f"{ROOT}/libero/out/predicate_v4_t3"; nets, thr = {}, {}
            for d in ("close", "bowl_out"):
                c = json.load(open(f"{V4}/calibration_{d}.json")); thr[d] = float(c["threshold"])
                nets[d] = Net().cuda(); nets[d].load_state_dict(torch.load(f"{ROOT}/{c['artifact']['path']}")); nets[d].eval()
            ok_of = lambda F: (predict(nets["close"], F) >= thr["close"]) & (predict(nets["bowl_out"], F) < thr["bowl_out"])
            verdict_of = lambda ok: fires(ok[1:], K_T3)
        P = {s: [] for s in SETTINGS}; Y = []
        recs = [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_t{tid}.jsonl")]
        for r in recs:
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
            env, obs = rig.make_env(A[r["s0_id"]]); u = env.envs[0]; F = {s: [] for s in SETTINGS}; y = []
            for t in range(H + 1):
                y.append(bool(u._env.check_success()))
                for s in SETTINGS:
                    o = obs
                    if s is not None:
                        o = dict(obs); o["pixels"] = dict(obs["pixels"])
                        o["pixels"]["image"] = corr(obs["pixels"]["image"][0], s, f"gate_{r['unit_id']}", t)[None]
                    F[s].append(model_view(rig, rig.obs_to_batch(o)))
                if t < H:                                        # unwrapped env: no vector auto-reset at success
                    obs = batched(u.step(acts[t].astype(np.float32))[0])
            env.close()
            for s in SETTINGS:
                P[s].append(verdict_of(ok_of(np.stack(F[s]))))
            Y.append(bool(np.array(y[1:]).any()))
            print(f"t{tid}", r["unit_id"], "real", Y[-1], {("none" if s is None else s): int(v[-1]) for s, v in P.items()}, flush=True)
        res[f"t{tid}"] = {("none" if s is None else s): rates(P[s], Y) for s in SETTINGS}
        del rig; torch.cuda.empty_cache()
    corr.close()
    return res


def openwam():
    import torch  # noqa: F401
    import yaml
    from lingbot_predicate import Net, predict
    from libero_env_adapter import _make_env, _anchor
    from ow_view import canvas_from_raw, pred_view, raw_obs
    corr = Corruptor(); bank = f"{ROOT}/libero/out/bank/libero_anchors.json"
    pr = yaml.safe_load(open(f"{ROOT}/protocol.yaml"))["libero_openwam"]["tasks"]["t3"]["predicate"]
    man = json.load(open(f"{ROOT}/{pr['path']}")); nets = {}
    for n, dd in man["detectors"].items():
        m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{dd['path']}")); m.eval(); nets[n] = m
    thr = pr["threshold_m"]
    ok_of = lambda F: (predict(nets["close"], F) >= thr["close"]) & (predict(nets["bowl_out"], F) < thr["bowl_out"])
    rows = ["clean_256", "B0"] + SETTINGS[1:]; P = {s: [] for s in rows}; Y = []
    for l in open(f"{ROOT}/libero/out/runs/openwam_base/records_t3.jsonl"):
        r = json.loads(l); acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
        envs = {res_: _make_env(dict(_anchor(bank, r["anchor_id"]), resolution=res_), H=len(acts)) for res_ in (128, 256)}
        F = {s: [] for s in rows}; y = []
        for t in range(H + 1):
            if t % 4 == 0:
                u128, u256 = envs[128].envs[0], envs[256].envs[0]
                lab = bool(u128._env.check_success()); assert lab == bool(u256._env.check_success()); y.append(lab)
                o2 = raw_obs(u256); F["clean_256"].append(pred_view(canvas_from_raw(o2["agentview_image"], o2["robot0_eye_in_hand_image"])))
                o1 = raw_obs(u128)
                for s in rows[1:]:
                    av = o1["agentview_image"] if s == "B0" else corr(o1["agentview_image"], s, f"gate_{r['unit_id']}", t)
                    F[s].append(pred_view(canvas_from_raw(av, o1["robot0_eye_in_hand_image"])))
            if t < H:
                for e in envs.values():
                    e.envs[0].step(np.asarray(acts[t], np.float32))
        for e in envs.values():
            e.close()
        for s in rows:
            P[s].append(fires(ok_of(np.stack(F[s]))[1:], 1))     # sampled frames at steps 4..400 (index 0 = step 0)
        Y.append(bool(np.array(y[1:]).any()))
        print("t3", r["unit_id"], "real", Y[-1], {s: int(v[-1]) for s, v in P.items()}, flush=True)
    corr.close()
    return {"t3": {s: rates(P[s], Y) for s in rows}}


def main():
    model = sys.argv[1]; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    res = lingbot() if model == "lingbot" else openwam()
    out = dict(model=model, reading="trajectory (deployed rule; ruling 2026-11-01)", line=LINE, H=H,
               rule=dict(t5=f"v3 k = {K_T5} consecutive env steps", t3="conjunction k = 1",
                         openwam_t3="conjunction, k = 1 sampled frame (stride 4)"),
               cells=res, seconds=round(time.time() - t0),
               failed=[f"{t}/{s}" for t, d in res.items() for s, v in d.items() if not v["pass_85"]], machine=os.uname().nodename)
    json.dump(out, open(f"{OUT}/gate_traj_{model}.json", "w"), indent=1)
    for t, d in res.items():
        for s, v in d.items():
            print(model, t, s, v)
    print("GATE_TRAJ_DONE", model, "failed:", out["failed"])


if __name__ == "__main__":
    main()
