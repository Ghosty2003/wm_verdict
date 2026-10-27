"""supplement to bcheck.py: per-image order of levels 2 / 5 / 9 (PSNR) and SSIM by level for fog and glass_blur."""
import sys, json, numpy as np
sys.path.insert(0, "/mnt/qnap/chenmg93/wm_verdict/libero/out/b_family/healthcheck")
from bcheck import load_functions, frames, call, psnr, SEEDS, W
from skimage.metrics import structural_similarity as ssim
ns, _, _ = load_functions(); F, names = frames(); out = {}
for fn in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("gaussian_blur", "fog", "glass_blur")):
    S = np.zeros((len(F), 10)); P = np.zeros((len(F), 10))
    for li in range(10):
        for j, img in enumerate(F):
            s_, p_ = [], []
            for s in (SEEDS if fn in ("fog", "glass_blur") else (0,)):
                u = call(ns, fn, img, li + 1, s)[1]
                s_.append(ssim(img, u, channel_axis=-1, data_range=255)); p_.append(psnr(img, u))
            S[j, li] = np.mean(s_); P[j, li] = np.mean(p_)
    out[fn] = dict(ssim_mean_by_level=[round(float(x), 4) for x in S.mean(0)],
                   ssim_non_decreasing_steps=[l + 1 for l in range(9) if S.mean(0)[l + 1] >= S.mean(0)[l]],
                   order_2_5_9_psnr_per_image={names[j]: bool(P[j, 1] > P[j, 4] > P[j, 8]) for j in range(len(F))},
                   order_2_5_9_ssim_per_image={names[j]: bool(S[j, 1] > S[j, 4] > S[j, 8]) for j in range(len(F))},
                   psnr_2_5_9_per_image={names[j]: [round(float(P[j, k]), 2) for k in (1, 4, 8)] for j in range(len(F))})
json.dump(out, open(f"{W}/report_pinned_extra" + ("_" + "_".join(out) if len(sys.argv) > 1 else "") + ".json", "w"), indent=1); print(json.dumps(out, indent=1))
