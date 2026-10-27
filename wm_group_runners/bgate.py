"""B family: predicate robustness gate (ruling 3, 2026-10-27). Real frames are corrupted per type and severity and the
DEPLOYED success predicate is scored against the simulator's success flag; 85 % TPR and TNR per (task x type x
severity). A cell that fails keeps its data, but its verdict column is marked "predicate unreliable at this level,
pending".

Real frames: truth replays of the model's own main-bank baseline units (anchors are disjoint from every predicate's
training rollouts), every 4th env step 0..400; label = check_success() of the state shown. Each frame goes through the
same path as the model's real input under B: raw agentview corrupted (bcorrupt, seed rule with unit id "gate_<unit>"),
then the model's own view.
  LingBot-VA   rendered at 128; model view = LingBotRig.obs_to_batch + lingbot_predicate.model_view;
               t5 v3 (threshold from calibration) and t3 v4 conjunction (close >= thr and bowl_out < thr).
  OpenWAM-a    rendered at 128 (B render path); view = pred_view(canvas_from_raw(agentview, wrist)); t3 only (frozen
               libero_openwam conjunction; t5 has no deployed predicate). Extra reference row: clean at 256 (main path).

  python bgate.py lingbot|openwam   (lerobot_libero interpreter, GPU)   -> out/b_family/gate/gate_<model>.json
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/qnap/chenmg93/wm_verdict"
sys.path[:0] = [HERE, ROOT, f"{ROOT}/libero", f"{ROOT}/libero/openwam"]
from bcorrupt import SETTINGS, Corruptor

OUT = f"{ROOT}/libero/out/b_family/gate"
H, STRIDE, LINE = 400, 4, 0.85


def rates(ok, y):
    ok, y = np.asarray(ok, bool), np.asarray(y, bool)
    tpr = float(ok[y].mean()) if y.any() else None; tnr = float((~ok[~y]).mean()) if (~y).any() else None
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
            judge = lambda F: predict(net, F) >= float(cal["threshold"])
        else:
            V4 = f"{ROOT}/libero/out/predicate_v4_t3"; nets, thr = {}, {}
            for d in ("close", "bowl_out"):
                c = json.load(open(f"{V4}/calibration_{d}.json")); thr[d] = float(c["threshold"])
                nets[d] = Net().cuda(); nets[d].load_state_dict(torch.load(f"{ROOT}/{c['artifact']['path']}")); nets[d].eval()
            judge = lambda F: (predict(nets["close"], F) >= thr["close"]) & (predict(nets["bowl_out"], F) < thr["bowl_out"])
        ok = {s: [] for s in SETTINGS}; Y = []
        recs = [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_t{tid}.jsonl")]
        for r in recs:
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
            env, obs = rig.make_env(A[r["s0_id"]]); u = env.envs[0]; F = {s: [] for s in SETTINGS}; y = []
            for t in range(H + 1):
                if t % STRIDE == 0:
                    y.append(bool(u._env.check_success()))
                    for s in SETTINGS:
                        o = obs
                        if s is not None:
                            o = dict(obs); o["pixels"] = dict(obs["pixels"])
                            o["pixels"]["image"] = corr(obs["pixels"]["image"][0], s, f"gate_{r['unit_id']}", t)[None]
                        F[s].append(model_view(rig, rig.obs_to_batch(o)))
                if t < H:                                    # unwrapped env: no vector auto-reset at success
                    obs = batched(u.step(acts[t].astype(np.float32))[0])
            env.close()
            for s in SETTINGS:
                ok[s] += list(judge(np.stack(F[s])))
            Y += y
            print(f"t{tid}", r["unit_id"], "frames", len(y), "pos", sum(y), flush=True)
        res[f"t{tid}"] = {("none" if s is None else s): rates(ok[s], Y) for s in SETTINGS}
        del rig; torch.cuda.empty_cache()
    corr.close()
    return res


def openwam():
    import torch
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
    judge = lambda F: (predict(nets["close"], F) >= thr["close"]) & (predict(nets["bowl_out"], F) < thr["bowl_out"])
    rows = ["clean_256", "B0"] + SETTINGS[1:]; ok = {s: [] for s in rows}; Y = []
    for l in open(f"{ROOT}/libero/out/runs/openwam_base/records_t3.jsonl"):
        r = json.loads(l); acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
        envs = {res_: _make_env(dict(_anchor(bank, r["anchor_id"]), resolution=res_), H=len(acts)) for res_ in (128, 256)}
        F = {s: [] for s in rows}; y = []
        for t in range(H + 1):
            if t % STRIDE == 0:
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
            ok[s] += list(judge(np.stack(F[s])))
        Y += y
        print("t3", r["unit_id"], "frames", len(y), "pos", sum(y), flush=True)
    corr.close()
    return {"t3": {s: rates(ok[s], Y) for s in rows}}


def main():
    model = sys.argv[1]; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    res = lingbot() if model == "lingbot" else openwam()
    out = dict(model=model, line=LINE, stride=STRIDE, H=H, cells=res, seconds=round(time.time() - t0),
               failed=[f"{t}/{s}" for t, d in res.items() for s, v in d.items() if not v["pass_85"]], machine=os.uname().nodename)
    json.dump(out, open(f"{OUT}/gate_{model}.json", "w"), indent=1)
    for t, d in res.items():
        for s, v in d.items():
            print(model, t, s, v)
    print("GATE_DONE", model, "failed:", out["failed"])


if __name__ == "__main__":
    main()
