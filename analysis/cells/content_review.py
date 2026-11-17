"""Check-7 layer-1 material and the t3 sub-goal diagnosis (2026-10-20 rulings 1 and 3/4).

For every false-safe unit of lingbot_famA / lingbot_famP (model_verdict and not real_verdict; rule v1 = single-frame
reach-anytime): replay the recorded post-processed env actions from the unit's anchor (new env, same recipe as the
truth (a) branch; bit-exactness is asserted against the stored replay1 states and per-step success), and record
  - per step, every BDDL goal sub-predicate (parsed_problem["goal_state"], LIBERO's own _eval_predicate)
  - the REAL endpoint frame (step 520) in the LingBot-VA model view (128 x 256, agentview + wrist)
and write a triptych: imagined endpoint (decoded frame 520) | imagined first-trigger frame (rule v1) | real endpoint.
Imagined frames come from the stored decoded_frames_sampled (no new generation).

  python content_review.py            -> out/content_review/units/<unit>.npz, sheets/<unit>.png, replay_log.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw

_wmvp.bootstrap()
from lingbot_unit import LingBotRig
from lingbot_predicate import model_view, batched

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/content_review"
RUNS = {"famA": ("lingbot_famA", f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"),
        "famP": ("lingbot_famP", f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json")}


def false_safe_units(t):
    out = []
    for fam, (run, bank) in RUNS.items():
        anchors = {a["anchor_id"]: a for a in json.load(open(bank))["anchors"]}
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_{t}.jsonl"):
            r = json.loads(l)
            if r["model_verdict"] and not r["real_verdict"]:
                out.append((fam, r, anchors[r["anchor_id"]]))
    return out


def replay(rig, anchor, acts):
    env, obs = rig.make_env(anchor); u = env.envs[0]
    prob = u._env.env
    goals = prob.parsed_problem["goal_state"]

    def snap():
        m, d = u._env.sim.model._model, u._env.sim.data._data
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS); return st
    S, succ, sub = [snap()], [], [[bool(prob._eval_predicate(g)) for g in goals]]
    for a in acts:
        obs, *_ = u.step(a.astype(np.float32))
        S.append(snap()); succ.append(bool(u._env.check_success()))
        sub.append([bool(prob._eval_predicate(g)) for g in goals])
    with torch.inference_mode():
        end = model_view(rig, rig.obs_to_batch(batched(obs)))
    env.close()
    return np.stack(S), np.array(succ), np.array(sub), [" ".join(g) for g in goals], end


def label(img, text, scale=2):
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    canvas = Image.new("RGB", (im.width, im.height + 16), "white"); canvas.paste(im, (0, 16))
    ImageDraw.Draw(canvas).text((3, 2), text, fill="black"); return canvas


def main():
    os.makedirs(f"{OUT}/units", exist_ok=True); os.makedirs(f"{OUT}/sheets", exist_ok=True)
    log = []
    for t, tid in (("t5", 5), ("t3", 3)):
        rig = LingBotRig(tid)
        for fam, r, a in false_safe_units(t):
            uid = r["unit_id"]
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            S, succ, sub, goal_names, real_end = replay(rig, a, acts)
            exact = bool(S.shape == tr["replay1"].shape and np.array_equal(S, tr["replay1"]))
            succ_ok = bool(np.array_equal(succ, tr["real_success"]))
            idx = list(tr["decoded_frames_idx"]); fr = tr["decoded_frames_sampled"]
            trig = int(r["model_first_success_frame"])
            im_end, im_trig = fr[idx.index(520)], fr[idx.index(trig)]
            np.savez_compressed(f"{OUT}/units/{uid}.npz", subgoals=sub, goal_names=np.array(goal_names),
                                real_end_frame=real_end, imagined_end_frame=im_end, imagined_trigger_frame=im_trig)
            panels = [label(im_end, "imagined end (frame 520)"),
                      label(im_trig, f"imagined first trigger, rule v1 k=1 (frame {trig})"),
                      label(real_end, "REAL end (step 520), simulator: fail")]
            W = sum(p.width for p in panels); sheet = Image.new("RGB", (W, panels[0].height + 18), "white")
            ImageDraw.Draw(sheet).text((3, 2), f"{uid}   [{fam} {r['setting_id']}]", fill="black")
            x = 0
            for p in panels:
                sheet.paste(p, (x, 18)); x += p.width
            sheet.save(f"{OUT}/sheets/{uid}.png")
            log.append(dict(unit=uid, task=t, family=fam, setting=r["setting_id"], replay_bit_exact=exact,
                            success_trace_match=succ_ok, trigger_frame_v1=trig, goal_names=goal_names,
                            ever_true=[bool(x) for x in sub.any(0)], final=[bool(x) for x in sub[-1]]))
            print(uid, "exact" if exact else "NOT-EXACT", "succ-ok" if succ_ok else "SUCC-MISMATCH",
                  dict(zip(goal_names, sub.any(0).tolist())), flush=True)
        del rig; torch.cuda.empty_cache()
    json.dump(dict(n=len(log), all_exact=all(x["replay_bit_exact"] for x in log),
                   all_success_match=all(x["success_trace_match"] for x in log), units=log),
              open(f"{OUT}/replay_log.json", "w"), indent=1)
    print("DONE", len(log), "all exact", all(x["replay_bit_exact"] for x in log))


if __name__ == "__main__":
    main()
