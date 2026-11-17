"""False-safe attribution material for a LingBot-VA LIBERO run (verify check-7 caliber + endpoint sheets).

For every false-safe unit (model_verdict and not real_verdict, reach_anytime):
  numbers  predicate on REAL frames (reach: any frame 1..H >= thr; final: frame H), predicate on IMAGINED frames
           (reach / final), first frames, closed-loop outcome; check-7 class = predicate_error if the predicate also
           fires on a real frame, else prediction_error.
  sheet    PNG: imagined (top) vs real (bottom) model-view frames at the same steps, labelled with p. Real frames come
           from a fresh open-loop replay of the recorded actions through the lerobot-eval preprocessing chain
           (the exact predicate input); imagined frames are the decoded frames stored with the unit.
    python attrib_sheets.py [--units all]      # default: false-safe units only
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
_wmvp.bootstrap()
import lingbot_unit as LU
from lingbot_predicate import model_view, batched

ROOT = _wmvp.P("$WMV_ROOT")
RUN = f"{ROOT}/libero/out/runs/lingbot_base"
OUT = f"{RUN}/attrib"


class ViewRig(LU.LingBotRig):
    """LingBotRig without the policy weights: env + processors + model_view config only."""
    def __init__(self, task_id, device="cuda"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.envs.factory import make_env_pre_post_processors
        from libero.libero import benchmark
        self.task_id = task_id
        self.suite = benchmark.get_benchmark_dict()["libero_10"]()
        self.env_cfg = LU.LiberoEnvCfg(task="libero_10", task_ids=[task_id], observation_height=128, observation_width=128)
        cfg = PreTrainedConfig.from_pretrained(LU.CKPT); cfg.pretrained_path = LU.CKPT; cfg.device = device
        self.cfg = cfg
        self.policy = type("P", (), {"config": cfg})()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=LU.CKPT,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.task_desc = None


def real_frames(rig, a, acts, steps):
    env, obs = rig.make_env(a); u = env.envs[0]
    want, out = set(steps), {}
    if 0 in want:
        out[0] = model_view(rig, rig.obs_to_batch(obs))
    for t, act in enumerate(acts, 1):
        o, *_ = u.step(act.astype(np.float32))
        if t in want:
            out[t] = model_view(rig, rig.obs_to_batch(batched(o)))
        if t >= max(want):
            break
    env.close()
    return out


def label(img, text):
    im = Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST)
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0)); d.text((3, 1), text, fill=(255, 255, 0))
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units", default="false_safe"); A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    bank = {x["anchor_id"]: x for x in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    rows = []
    for tid in (5, 3):
        R = [json.loads(l) for l in open(f"{RUN}/records_t{tid}.jsonl")]
        sel = [r for r in R if A.units == "all" or (r["model_verdict"] and not r["real_verdict"])]
        if not sel:
            continue
        rig = ViewRig(tid)
        for r in sel:
            thr = r["threshold"]
            z = np.load(f"{ROOT}/{r['trajectory_file']}")
            mp, rp, idx, dec = z["model_pred"], z["real_pred"], list(z["decoded_frames_idx"]), z["decoded_frames_sampled"]
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
            M = r["model_first_success_frame"]
            rmax = int(np.argmax(rp[1:])) + 1
            cols = sorted({M, 200, 300, 400, 520} | ({rmax} if rmax in idx else set()))
            cols = [c for c in cols if c in idx]
            RF = real_frames(rig, bank[r["s0_id"]], acts, cols + [rmax])
            top = [label(dec[idx.index(c)], f"IMAG t={c} p={mp[c]:.3f}") for c in cols]
            bot = [label(RF[c], f"REAL t={c} p={rp[c]:.3f}") for c in cols]
            W, Hh = top[0].width, top[0].height
            sheet = Image.new("RGB", (W * len(cols), Hh * 2 + 18), (40, 40, 40))
            for i, (a_, b_) in enumerate(zip(top, bot)):
                sheet.paste(a_, (i * W, 18)); sheet.paste(b_, (i * W, 18 + Hh))
            ImageDraw.Draw(sheet).text((4, 3), f"{r['unit_id']}  thr={thr:.3g}  model first success={M}  real: never "
                                                f"(sim)  real-pred max p={rp[1:].max():.3f} @t={rmax}  closed-loop={r['closed_loop']['success']}",
                                       fill=(255, 255, 255))
            path = f"{OUT}/{r['unit_id']}.png"; sheet.save(path)
            if rmax not in cols:
                label(RF[rmax], f"REAL argmax-p t={rmax} p={rp[rmax]:.3f}").save(f"{OUT}/{r['unit_id']}__real_argmax.png")
            rows.append(dict(unit_id=r["unit_id"], task_id=tid, threshold=thr,
                             model_first_success_frame=M, imag_final_pred=bool(mp[-1] >= thr), imag_final_p=float(mp[-1]),
                             real_reach_pred=bool(rp[1:].max() >= thr), real_max_p=float(rp[1:].max()), real_max_p_step=rmax,
                             real_final_pred=bool(rp[-1] >= thr), real_final_p=float(rp[-1]),
                             check7_class="predicate_error" if rp[1:].max() >= thr else "prediction_error",
                             closed_loop_success=r["closed_loop"]["success"], sheet=os.path.relpath(path, ROOT)))
            print(json.dumps(rows[-1]), flush=True)
        del rig; torch.cuda.empty_cache()
    json.dump(rows, open(f"{OUT}/attrib_{A.units}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
