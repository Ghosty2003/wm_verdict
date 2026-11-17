"""LingBot-VA family O (occlusion) false-safe units -> content-attribution cards (2026-10-28), from stored data only
(decoded imagined frames every 20 steps, model view agentview | wrist; no GPU). False-safe = final-rule verdict
success (t5 v3 k = 24, t3 v4 k = 1, H = 400) while the open-loop truth replay fails within 400 steps.
Card: top (2x) = imagined frames at step 400 (LAST, judged; first panel), 0 (occluded input), onset (nearest stored frame), 300; bottom = strip every 20.
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json, glob, numpy as np
from PIL import Image, ImageDraw
ROOT = _wmvp.P("$WMV_ROOT"); D = f"{ROOT}/libero/out/content_review/lingbot_famO"
thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
V4 = f"{ROOT}/libero/out/predicate_v4_t3"; thr4 = {d: json.load(open(f"{V4}/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}


def onset(h, k):
    run = 0
    for i, x in enumerate(h):
        run = run + 1 if x else 0
        if run >= k:
            return i - k + 1
    return None


units = []
for t in (5, 3):
    for r in [json.loads(l) for f in sorted(glob.glob(f"{ROOT}/libero/out/runs/lingbot_famO/records_t{t}.*.jsonl")) for l in open(f)]:
        z = np.load(f"{ROOT}/{r['trajectory_file']}")
        ok = (z["model_pred"] >= thr5)[1:401] if t == 5 else ((z["p_close"] >= thr4["close"]) & (z["p_bowl"] < thr4["bowl_out"]))[1:401]
        o = onset(ok, 24 if t == 5 else 1); rs = r["real_first_success_step"]; real = rs is not None and rs <= 400
        if o is None or real:
            continue
        F, idx = z["decoded_frames_sampled"], list(z["decoded_frames_idx"]); on = o + 1
        near = min(range(len(idx)), key=lambda i: abs(idx[i] - on))
        top = np.asarray(Image.fromarray(np.concatenate([F[idx.index(400)], F[0], F[near], F[idx.index(300)]], 1)).resize((2048, 256), Image.LANCZOS))   # 2x; judged frame first
        strip = np.concatenate([np.asarray(Image.fromarray(F[i]).resize((96, 48))) for i in range(len(F))], 1)
        W = max(top.shape[1], strip.shape[1]); card = Image.new("RGB", (W, 18 + 256 + 16 + 48), "white"); dr = ImageDraw.Draw(card)
        dr.text((4, 2), f"{r['unit_id']}   onset step {on}   real: FAIL in 400 steps", fill=(0, 0, 0))
        card.paste(Image.fromarray(top), (0, 18))
        for j, lab in enumerate(["LAST FRAME (step 400) - judge this one", "step 0 (occluded input), not the end", f"~onset (step {idx[near]}), not the end", "step 300, not the end"]):
            dr.text((j * 512 + 4, 18 + 256 + 2), lab, fill=(150, 0, 0))
        card.paste(Image.fromarray(strip), (0, 18 + 256 + 16))
        card.save(f"{D}/cards/{r['unit_id']}.png") if False else None
        units.append(dict(unit=r["unit_id"], task=f"t{t}", level=r["setting_id"], onset=on, card=f"cards/{r['unit_id']}.png"))
        import os; os.makedirs(f"{D}/cards", exist_ok=True); card.save(f"{D}/{units[-1]['card']}")
json.dump(units, open(f"{D}/units.json", "w"), indent=1)
print(len(units), {k: sum(1 for u in units if u["task"] == k) for k in ("t5", "t3")})
