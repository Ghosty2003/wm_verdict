"""Zoomed endpoint comparison per false-safe unit: agentview (top) and wrist (bottom) halves of the model-view frame at
3x: IMAG@first-success, IMAG@520, REAL@argmax-p, REAL@520, and a reference TRUE-SUCCESS real frame of the same task."""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
_wmvp.bootstrap()
from attrib_sheets import ViewRig, real_frames, ROOT, RUN, OUT


def tile(img, text, s=3):
    im = Image.fromarray(img).resize((img.shape[1] * s, img.shape[0] * s), Image.LANCZOS)
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0)); d.text((3, 1), text, fill=(255, 255, 0))
    return im


bank = {x["anchor_id"]: x for x in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
rows = json.load(open(f"{OUT}/attrib_false_safe.json"))
for tid in (5, 3):
    R = [json.loads(l) for l in open(f"{RUN}/records_t{tid}.jsonl")]
    rig = ViewRig(tid)
    ref = next(r for r in R if r["real_verdict"] and r["real_verdict_final"])          # stays successful to the end
    rs = ref["real_first_success_step"]
    RF = real_frames(rig, bank[ref["s0_id"]], np.load(f"{ROOT}/{ref['action_file']}")["env"], [rs + 20])
    ref_img = (RF[rs + 20], f"REF true success {ref['unit_id'][17:30]} t={rs + 20}")
    for u in [x for x in rows if x["task_id"] == tid]:
        r = next(x for x in R if x["unit_id"] == u["unit_id"])
        z = np.load(f"{ROOT}/{r['trajectory_file']}"); idx = list(z["decoded_frames_idx"]); dec = z["decoded_frames_sampled"]
        mp, rp = z["model_pred"], z["real_pred"]; M = u["model_first_success_frame"]; k = u["real_max_p_step"]
        F = real_frames(rig, bank[r["s0_id"]], np.load(f"{ROOT}/{r['action_file']}")["env"], [k, 520])
        panels = [(dec[idx.index(M)], f"IMAG t={M} p={mp[M]:.3f}"), (dec[idx.index(520)], f"IMAG t=520 p={mp[520]:.3f}"),
                  (F[k], f"REAL t={k} p={rp[k]:.3f}"), (F[520], f"REAL t=520 p={rp[520]:.3f}"), ref_img]
        W = 128 * 3
        sheet = Image.new("RGB", (W * len(panels), W * 2 + 16), (40, 40, 40))
        ImageDraw.Draw(sheet).text((4, 2), u["unit_id"] + "   top: agentview  bottom: wrist", fill=(255, 255, 255))
        for i, (img, lab) in enumerate(panels):
            sheet.paste(tile(img[:, :128], lab), (i * W, 16)); sheet.paste(tile(img[:, 128:], lab), (i * W, 16 + W))
        sheet.save(f"{OUT}/{u['unit_id']}__zoom.png"); print("saved", u["unit_id"], flush=True)
    del rig
