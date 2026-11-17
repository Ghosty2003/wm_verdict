"""WorldVLA appendix: content-attribution sheets for the 10 baseline units (2026-10-22 ruling 3; "content attribution"
column, not a predicate column). Per unit: imagined and real filmstrips (every 40 steps, WorldVLA view = agentview
256 px rotated 180 deg) and the two endpoints side by side; t5 endpoints carry the geometry reference of
t5_geometry.py (green = back-compartment box top, red = ceiling for an upright book, dot = real book origin).

  python wv_content.py    -> out/worldvla/content/<unit>.png
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap(); _wmvp.bootstrap()
from libero_env_adapter import _make_env, _anchor
import t5_geometry as G

OUT = f"{ROOT}/libero/out/worldvla/content"


def tile(img, text, color="black", size=160):
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    c = Image.new("RGB", (size, size + 14), "white"); c.paste(im, (0, 14)); ImageDraw.Draw(c).text((3, 1), text, fill=color)
    return c


def endpoint(img, geo, st, half_up, centre):
    G.RES, G.FLIP = 256, None              # WorldVLA view = LeRobot-processed view (rotated 180 once), same as the verified LingBot view
    im = Image.fromarray(img).resize((512, 512), Image.NEAREST); dr = ImageDraw.Draw(im)
    for z, col in ((st["ub"][2], (0, 220, 0)), (st["ub"][2] + half_up, (230, 30, 30))):
        q = geo.project(G.rect(st["lb"], st["ub"], z)) * 2
        dr.polygon([tuple(x) for x in q], outline=col)
    if centre:
        c = geo.project(st["centre"])[0] * 2
        dr.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], fill=(0, 220, 0) if st["inside"] else (230, 30, 30))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = {u["unit"]: u for u in json.load(open(f"{ROOT}/libero/out/worldvla/e4_analysis.json"))["units"]}
    for tid in (5, 3):
        for j in range(5):
            u = f"libero10_t{tid}_a{j}"; d = f"{ROOT}/libero/out/worldvla/{u}/s1"
            I = np.load(f"{d}/imagined.npz")["frames"]; Rz = np.load(f"{d}/replay.npz"); R = Rz["frames"]
            e = ex[u]
            steps = list(range(0, 521, 40))
            row1 = [tile(I[s], f"imag f{s}") for s in steps]; row2 = [tile(R[s], f"real f{s}" + (" S" if s and Rz["succ"][s - 1] else "")) for s in steps]
            head = (f"{u}  WorldVLA (E4)  VLA malformed {e['vla_malformed']}/{e['vla_calls']}, first at step {e['first_vla_malformed_step']}, "
                    f"free-output share {e['free_output_share_vla']}  |  truth (a): {'SUCCESS' if e['truth_a_success'] else 'fail'}"
                    + ("  |  CLEAN (no intervention)" if e["vla_malformed"] == 0 else "  |  constrained-retry dominated"))
            if tid == 5:
                a = _anchor(f"{ROOT}/libero/out/bank/libero_anchors.json", u)
                acts = np.load(f"{d}/imagined.npz")["env_actions"]
                env = _make_env(a, H=len(acts)); uu = env.envs[0]; geo = G.Geo(uu); s0 = geo.state()
                half_up = s0["top_z"] - s0["centre"][2]
                for x in acts:
                    uu.step(np.asarray(x, np.float32))
                s_end = geo.state(); env.close()
                A = endpoint(I[-1], geo, s0, half_up, False); B = endpoint(R[-1], geo, s_end, half_up, True)
            else:
                A = Image.fromarray(I[-1]).resize((512, 512), Image.NEAREST); B = Image.fromarray(R[-1]).resize((512, 512), Image.NEAREST)
            W = max(len(steps) * 160, 1040)
            page = Image.new("RGB", (W, 20 + 2 * 174 + 20 + 512), "white"); dr = ImageDraw.Draw(page)
            dr.text((4, 4), head, fill="black")
            for k, t in enumerate(row1):
                page.paste(t, (k * 160, 20))
            for k, t in enumerate(row2):
                page.paste(t, (k * 160, 20 + 174))
            y = 20 + 2 * 174
            dr.text((4, y + 4), "IMAGINED end (f520)" + ("  green = back-compartment box top, red = ceiling for an upright book" if tid == 5 else ""), fill="black")
            dr.text((524, y + 4), "REAL end (step 520)" + ("  dot = book origin (green = inside)" if tid == 5 else ""), fill="black")
            page.paste(A, (0, y + 20)); page.paste(B, (520, y + 20))
            page.save(f"{OUT}/{u}.png"); print(u, flush=True)


if __name__ == "__main__":
    main()
