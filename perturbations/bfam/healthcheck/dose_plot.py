import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = _wmvp.P("$WMV_ROOT/libero/out/b_family/healthcheck")
a = json.load(open(f"{W}/report_pinned.json"))["checks"]["3_dose"]; b = json.load(open(f"{W}/report_pinned_extra.json"))
a.update(json.load(open(f"{W}/report_pinned_zoom_blur.json"))["checks"]["3_dose"]); b.update(json.load(open(f"{W}/report_pinned_extra_zoom_blur.json")))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); L = list(range(1, 11))
for fn, c in (("gaussian_blur", "#4269d0"), ("fog", "#3ca951"), ("zoom_blur", "#6c3fc1"), ("glass_blur", "#ff725c")):
    p = [a[fn]["psnr_mean_by_level"][str(l)] for l in L]; s = b[fn]["ssim_mean_by_level"]
    ax[0].plot(L, p, "-o", ms=3, c=c, label=fn + (" (dropped)" if fn == "glass_blur" else "")); ax[1].plot(L, s, "-o", ms=3, c=c, label=fn)
for x in ax:
    for l in (2, 5, 9): x.axvline(l, c="#bbb", lw=0.8, zorder=0)
    x.set_xticks(L); x.set_xlabel("LIBERO-plus severity (grey lines: 2 / 5 / 9)")
ax[0].set_ylabel("PSNR vs original (dB)"); ax[1].set_ylabel("SSIM vs original"); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Dose curves on 6 real LIBERO agentview frames, 128x128 (fog / glass: mean of 5 seeds; glass_blur dropped, zoom_blur replaces it)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{W}/dose_curves.png", dpi=130)
