"""重建验证扩展：腐蚀与朝向变换的先后顺序 × 4 种朝向 × 2 种种子串，LingBot T2 famB 每档 3 个单元第 0 步，对存档 real_00。"""
import json, sys, os
import numpy as np
sys.argv = [sys.argv[0], os.path.dirname(os.path.abspath(__file__))]
import sharpness as S
T = {"raw": lambda a: a, "flip_v": lambda a: a[::-1], "flip_h": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1]}
recs = [json.loads(l) for f in sorted(S.glob.glob(f"{S.RUNS}/lingbot_t2/famB/records.*.jsonl")) for l in open(f) if l.strip()]
by = {}
for r in recs:
    by.setdefault((r.get("setting_id") or "").split("__")[-1], []).append(r["unit_id"])
out = {}
for lv in ("Bz2", "Bz5", "Bz9", "Bf2", "Bf5", "Bf9", "Bg5"):
    res = []
    for uid in by.get(lv, [])[:3]:
        clean0 = np.load(f"{S.RUNS}/lingbot_t2/famB/replay_frames/{uid}.npz")["agentview"][0]
        stored = S.to_u8(np.load(f"{S.RUNS}/lingbot_t2/famB/closed_frames/{uid}.npz")["real_00__observation_images_image"])[0]
        cands = []
        for order in ("corrupt_then_T", "T_then_corrupt"):
            for o, f in T.items():
                for fmt in ("level", "sev"):
                    rec = f(S.corrupt(clean0, uid, 0, lv, fmt)) if order == "corrupt_then_T" else S.corrupt(np.ascontiguousarray(f(clean0)), uid, 0, lv, fmt)
                    d = np.abs(np.ascontiguousarray(rec).astype(int) - stored.astype(int))
                    cands.append((float((d == 0).mean()), int(d.max()), order, o, fmt))
        cands.sort(reverse=True); res.append(dict(uid=uid, best=cands[0], second=cands[1]))
    out[lv] = res
json.dump(out, open("verify2.json", "w"), indent=1)
for k, v in out.items(): print(k, [r["best"] for r in v])
