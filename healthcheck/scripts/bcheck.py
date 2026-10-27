"""B-family function check (2026-10-27): gaussian_blur, fog, glass_blur from LIBERO-plus commit 4976dc3, UNMODIFIED.

The three functions (and plasma_fractal, which fog calls) are taken verbatim from src/env_wrapper.py (sha256 verified)
via the AST and executed with the module's own imports (skimage.filters.gaussian, scipy.ndimage.zoom). Each call follows
the upstream call site (ControlEnv.step / reset): PIL image in, function, then `.astype(np.uint8)`. The only addition is
`np.random.seed(seed)` before a call (fog and glass_blur draw from the global numpy RNG).

Frames: real LIBERO agentview at 128 x 128 (stored real episodes; left half of the 128 x 256 model-view frames, i.e. no
rescaling), t5 and t3, start / middle / end.

  python bcheck.py pinned    (numpy 1.22.4 / skimage 0.21.0, LIBERO-plus requirements)  -> report_pinned.json, sheets
  python bcheck.py current   (numpy 2.2.6, as in our lerobot_libero / openwam envs)    -> report_current.json
  python bcheck.py pinned|current zoom_blur   (same five checks for zoom_blur)            -> report_*_zoom_blur.json
"""
import ast
import sys
import json
import time
import hashlib
import numpy as np
from PIL import Image, ImageDraw

W = "/mnt/qnap/chenmg93/wm_verdict/libero/out/b_family/healthcheck"
OUT = "/mnt/qnap/chenmg93/wm_verdict/libero/out"
EXPECTED = "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c"
FUNCS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("gaussian_blur", "fog", "glass_blur")
TAG = "" if len(sys.argv) <= 2 else "_" + "_".join(FUNCS)     # e.g. zoom_blur (ruling 2026-10-27: replaces glass_blur)
LEVELS = range(1, 11)
SEEDS = (0, 1, 2, 3, 4)


def load_functions():
    src = open(f"{W}/src/env_wrapper.py", "rb").read()
    sha = hashlib.sha256(src).hexdigest(); assert sha == EXPECTED, sha
    text = src.decode(); tree = ast.parse(text)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("gaussian_blur", "fog", "glass_blur", "zoom_blur", "plasma_fractal", "clipped_zoom")]
    from skimage.filters import gaussian
    from scipy.ndimage import zoom as scizoom
    ns = dict(np=np, gaussian=gaussian, scizoom=scizoom)
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"{W}/src/env_wrapper.py", "exec"), ns)
    seg = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in keep}
    return ns, sha, seg


def frames():
    F, names = [], []
    for path, tag in ((f"{OUT}/predicate_v3/t5/nearmiss_eps/nm_base_00.npz", "t5"), (f"{OUT}/predicate_v4_t3/episodes/v1_r005.npz", "t3")):
        fr = np.load(path)["frames"]
        for i in (0, len(fr) // 2, len(fr) - 1):
            F.append(np.ascontiguousarray(fr[i, :, :128])); names.append(f"{tag}_f{i}")
    return F, names


def call(ns, fn, img, sev, seed):
    np.random.seed(seed)
    out = ns[fn](Image.fromarray(img), severity=sev)
    return out, out.astype(np.uint8)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if m == 0 else float(10 * np.log10(255 ** 2 / m))


def main(mode):
    ns, sha, seg = load_functions()
    F, names = frames()
    rep = dict(mode=mode, numpy=np.__version__, source_sha256=sha, function_sha256=seg, frames=names, checks={})
    import skimage, scipy
    rep.update(skimage=skimage.__version__, scipy=scipy.__version__)
    if mode == "current":
        res = {}
        ref = np.load(f"{W}/outputs_pinned{TAG}.npz")
        for fn in FUNCS:
            try:
                same = []
                for sev in (2, 5, 9):
                    for j, img in enumerate(F):
                        _, u = call(ns, fn, img, sev, 0); same.append(bool(np.array_equal(u, ref[f"{fn}_{sev}_{j}"])))
                res[fn] = dict(runs=True, identical_to_pinned=f"{sum(same)}/{len(same)}")
            except Exception as e:
                res[fn] = dict(runs=False, error=f"{type(e).__name__}: {e}")
        rep["checks"]["cross_version"] = res
        json.dump(rep, open(f"{W}/report_current{TAG}.json", "w"), indent=1); print(json.dumps(res, indent=1)); return

    C = rep["checks"]; store = {}
    # 1 determinism ------------------------------------------------------------------------------------------------
    det = {}
    for fn in FUNCS:
        same_seed, diff_seed, rng_untouched = [], [], []
        for sev in LEVELS:
            for j, img in enumerate(F):
                a_f, a = call(ns, fn, img, sev, 0); b_f, b = call(ns, fn, img, sev, 0)
                same_seed.append(bool(np.array_equal(a_f, b_f) and np.array_equal(a, b)))
                _, c = call(ns, fn, img, sev, 1); diff_seed.append(not np.array_equal(a, c))
                np.random.seed(123); s0 = np.random.get_state()[1].copy(); ns[fn](Image.fromarray(img), severity=sev)
                rng_untouched.append(bool(np.array_equal(np.random.get_state()[1], s0)))
                if sev in (2, 5, 9):
                    store[f"{fn}_{sev}_{j}"] = a
        det[fn] = dict(same_seed_identical=f"{sum(same_seed)}/{len(same_seed)}",
                       other_seed_differs=f"{sum(diff_seed)}/{len(diff_seed)}",
                       consumes_global_numpy_rng=not all(rng_untouched))
    C["1_determinism"] = det
    np.savez_compressed(f"{W}/outputs_pinned{TAG}.npz", **store)
    # 2 output validity --------------------------------------------------------------------------------------------
    val = {}
    for fn in FUNCS:
        r = dict(float_dtype=set(), nan=0, below0=0, above255=0, shape_ok=0, n=0, trunc_bias_mean=[], border_vs_interior=[],
                 border_dark_px_added=0)
        for sev in LEVELS:
            for j, img in enumerate(F):
                f_, u = call(ns, fn, img, sev, 0); r["n"] += 1
                r["float_dtype"].add(str(f_.dtype)); r["nan"] += int(np.isnan(f_).sum())
                r["below0"] += int((f_ < 0).sum()); r["above255"] += int((f_ > 255).sum())
                r["shape_ok"] += int(u.shape == img.shape and u.dtype == np.uint8)
                r["trunc_bias_mean"].append(float(np.mean(f_ - u)))
                d = np.abs(u.astype(float) - img.astype(float)).mean(-1); b = 6
                inner = d[b:-b, b:-b].mean(); border = np.concatenate([d[:b].ravel(), d[-b:].ravel(), d[:, :b].ravel(), d[:, -b:].ravel()]).mean()
                r["border_vs_interior"].append((sev, round(float(border / max(inner, 1e-9)), 3)))
                dark_o = (img[:2].max(-1) < 8).sum() + (img[-2:].max(-1) < 8).sum() + (img[:, :2].max(-1) < 8).sum() + (img[:, -2:].max(-1) < 8).sum()
                dark_u = (u[:2].max(-1) < 8).sum() + (u[-2:].max(-1) < 8).sum() + (u[:, :2].max(-1) < 8).sum() + (u[:, -2:].max(-1) < 8).sum()
                r["border_dark_px_added"] += int(max(0, dark_u - dark_o))
        bvi = {}
        for sev, x in r["border_vs_interior"]:
            bvi.setdefault(sev, []).append(x)
        r["border_vs_interior"] = {s: round(float(np.median(v)), 3) for s, v in bvi.items()}
        r["trunc_bias_mean"] = round(float(np.mean(r["trunc_bias_mean"])), 3); r["float_dtype"] = sorted(r["float_dtype"])
        r["shape_ok"] = f"{r['shape_ok']}/{r['n']}"
        val[fn] = r
    C["2_validity"] = val
    # 3 dose monotonicity ------------------------------------------------------------------------------------------
    dose, times = {}, {}
    for fn in FUNCS:
        P = np.zeros((len(F), 10)); T = []
        for li, sev in enumerate(LEVELS):
            for j, img in enumerate(F):
                v = []
                for s in (SEEDS if fn in ("fog", "glass_blur") else (0,)):
                    t0 = time.time(); _, u = call(ns, fn, img, sev, s); T.append(time.time() - t0); v.append(psnr(img, u))
                P[j, li] = np.mean(v)
        mean = P.mean(0)
        per_img_viol = [(names[j], [int(l + 1) for l in range(9) if P[j, l + 1] >= P[j, l]]) for j in range(len(F))]
        dose[fn] = dict(psnr_mean_by_level={int(l): round(float(x), 2) for l, x in zip(LEVELS, mean)},
                        mean_monotone_decreasing=bool(np.all(np.diff(mean) < 0)),
                        non_decreasing_steps_mean=[int(l + 1) for l in range(9) if mean[l + 1] >= mean[l]],
                        non_decreasing_steps_per_image=[x for x in per_img_viol if x[1]],
                        psnr_2_5_9=[round(float(mean[1]), 2), round(float(mean[4]), 2), round(float(mean[8]), 2)],
                        gap_2_5_db=round(float(mean[1] - mean[4]), 2), gap_5_9_db=round(float(mean[4] - mean[8]), 2))
        times[fn] = round(1000 * float(np.median(T)), 1)
    C["3_dose"] = dose; C["seconds_per_call_ms_median"] = times
    # 4 small resolution ---------------------------------------------------------------------------------------------
    sr = {}
    big = [np.asarray(Image.fromarray(x).resize((256, 256), Image.BICUBIC)) for x in F]
    for fn in FUNCS:
        rel = {}
        for sev in (2, 5, 9):
            p128 = np.mean([psnr(x, call(ns, fn, x, sev, 0)[1]) for x in F])
            p256 = np.mean([psnr(x, call(ns, fn, x, sev, 0)[1]) for x in big])
            rel[sev] = dict(psnr_at_128=round(float(p128), 2), psnr_same_frame_at_256=round(float(p256), 2))
        sr[fn] = rel
    from skimage.registration import phase_cross_correlation
    sh = {}
    for sev in (2, 5, 9):
        v = []
        for j, img in enumerate(F):
            for s in SEEDS:
                u = call(ns, "glass_blur", img, sev, s)[1].astype(float).mean(-1)
                g = call(ns, "gaussian_blur", img, 1, 0)[1].astype(float).mean(-1)
                v.append(phase_cross_correlation(g, u, upsample_factor=20)[0])
        v = np.array(v); sh[sev] = dict(mean_shift_rows_cols=[round(float(x), 3) for x in v.mean(0)], sd=[round(float(x), 3) for x in v.std(0)])
    sr["glass_blur_global_shift_vs_gaussian"] = sh
    # the swap statement of glass_blur, evaluated on a coordinate-coded array (diagnostic of numpy semantics only)
    H = 128; x = np.arange(H * H, dtype=np.int64).reshape(H, H, 1).repeat(3, 2); np.random.seed(0); d = 1
    for h in range(H - d, d, -1):
        for w in range(H - d, d, -1):
            dx, dy = np.random.randint(-d, d, size=(2,)); hp, wp = h + dy, w + dx
            x[h, w], x[hp, wp] = x[hp, wp], x[h, w]
    sr["glass_swap_statement"] = dict(unique_pixels_before=H * H, unique_pixels_after_one_pass=int(len(np.unique(x[..., 0]))),
                                      randint_range=f"np.random.randint(-d, d) draws from [-d, d-1]")
    C["4_small_resolution"] = sr
    json.dump(rep, open(f"{W}/report_pinned{TAG}.json", "w"), indent=1)
    # 5 contact sheets ---------------------------------------------------------------------------------------------
    for fn in FUNCS:
        cols = len(F); sheet = Image.new("RGB", (60 + 128 * cols, 11 * 132), "white"); dr = ImageDraw.Draw(sheet)
        for r_, sev in enumerate([0] + list(LEVELS)):
            lab = "orig" if sev == 0 else f"L{sev}" + (" *" if sev in (2, 5, 9) else "")
            dr.text((4, r_ * 132 + 58), lab, fill=(200, 0, 0) if sev in (2, 5, 9) else (0, 0, 0))
            for j, img in enumerate(F):
                im = img if sev == 0 else call(ns, fn, img, sev, 0)[1]
                sheet.paste(Image.fromarray(im), (60 + 128 * j, r_ * 132))
        sheet.save(f"{W}/sheet_{fn}.png")
    print(json.dumps(C, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
