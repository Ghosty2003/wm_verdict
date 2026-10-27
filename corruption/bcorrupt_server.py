"""B-family corruption worker (protocol v1.0 + ruling 2026-10-27). Runs under the LIBERO-plus pinned dependencies
(numpy 1.22.4, scikit-image 0.21.0; ~/bcheck_venvs/venv_pinned on each node), in its own process: the corruption RNG
(numpy global MT19937, reseeded per call) can never touch the simulator's or the policy's random streams.

Functions are taken verbatim (AST) from LIBERO-plus 4976dc3 libero/libero/envs/env_wrapper.py (sha256 checked at start)
and called as the upstream call site does: PIL image -> function(severity) -> .astype(np.uint8).

Wire protocol (binary stdin / stdout): request = one JSON line {"fn", "sev", "seed", "h", "w"} + h*w*3 uint8 bytes;
reply = h*w*3 uint8 bytes. First stdout line: "READY <source sha256> <numpy> <skimage>".
"""
import os
import ast
import sys
import json
import hashlib
import numpy as np
from PIL import Image

SRC = os.environ.get("LIBERO_PLUS_ENV_WRAPPER", "/mnt/qnap/chenmg93/wm_verdict/libero/out/b_family/healthcheck/src/env_wrapper.py")
EXPECTED = "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c"
NAMES = ("gaussian_blur", "fog", "zoom_blur", "glass_blur", "plasma_fractal", "clipped_zoom")   # glass_blur loadable (shared self-test); not in the WM settings


def load():
    src = open(SRC, "rb").read(); sha = hashlib.sha256(src).hexdigest()
    assert sha == EXPECTED, sha
    tree = ast.parse(src.decode())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    from skimage.filters import gaussian
    from scipy.ndimage import zoom as scizoom
    ns = dict(np=np, gaussian=gaussian, scizoom=scizoom)
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    return ns, sha


def main():
    ns, sha = load()
    import skimage
    out = sys.stdout.buffer; inp = sys.stdin.buffer
    out.write(f"READY {sha} {np.__version__} {skimage.__version__}\n".encode()); out.flush()
    while True:
        line = inp.readline()
        if not line:
            return
        q = json.loads(line); n = q["h"] * q["w"] * 3
        img = np.frombuffer(inp.read(n), np.uint8).reshape(q["h"], q["w"], 3)
        np.random.seed(q["seed"])
        y = ns[q["fn"]](Image.fromarray(img), severity=q["sev"]).astype(np.uint8)
        assert y.shape == img.shape
        out.write(np.ascontiguousarray(y).tobytes()); out.flush()


if __name__ == "__main__":
    main()
