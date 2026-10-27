"""B family (image degradation, unified protocol v1.0; rulings 2026-10-27): settings, seed rule and the client of the
corruption worker (bcorrupt_server.py, LIBERO-plus pinned dependencies, separate process).

Settings: three types x severities 2 / 5 / 9 (glass_blur dropped: parameter-table dose defect, see
out/b_family/healthcheck/report.md). Camera agentview only, rendered at 128 x 128; corruption on the raw LIBERO
observation (as LIBERO-plus ControlEnv.step / reset do), before any model-specific preprocessing; wrist untouched.
Window: every real observation from t = 0 (reset) to episode end. Open loop: only the real t = 0 frame (imagined
frames are model state, not camera output - never corrupted); closed loop: every real frame the policy reads.

Seed rule: seed = first 8 hex digits of sha256("<unit_id>|<env step>|agentview|<fn>|<severity>") as an integer
(< 2**32); numpy legacy MT19937 (np.random.seed) in the worker. gaussian_blur and zoom_blur draw no random numbers.
"""
import os
import json
import hashlib
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED_PY = os.environ.get("B_PINNED_PY", os.path.expanduser("~/bcheck_venvs/venv_pinned/bin/python"))   # pinned-deps interpreter
COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
SOURCE_SHA256 = "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c"
TYPES = {"g": ("gaussian_blur", 10), "f": ("fog", 30), "z": ("zoom_blur", 20)}     # noise code = base + severity
SEVERITIES = (2, 5, 9)
SETTINGS = [None] + [f"B{t}{s}" for t in TYPES for s in SEVERITIES]           # Bg2 .. Bz9; None = no corruption


def spec(setting):
    t, s = setting[1], int(setting[2:]); fn, base = TYPES[t]
    return dict(family="B", fn=fn, severity=s, noise_code=base + s, camera="agentview_image", render_resolution=128,
                start_step=0, end="episode end", libero_plus_commit=COMMIT, source_sha256=SOURCE_SHA256,
                rng="numpy 1.22.4 legacy MT19937 via np.random.seed, separate process",
                seed_rule='int(sha256("<unit_id>|<step>|agentview|<fn>|<severity>").hexdigest()[:8], 16)')


def seed_of(unit_id, step, fn, sev):
    return int(hashlib.sha256(f"{unit_id}|{step}|agentview|{fn}|{sev}".encode()).hexdigest()[:8], 16)


class Corruptor:
    def __init__(self):
        self.p = subprocess.Popen([PINNED_PY, "-I", f"{HERE}/bcorrupt_server.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        ready = self.p.stdout.readline().decode().split()
        assert ready[0] == "READY" and ready[1] == SOURCE_SHA256, ready
        self.versions = dict(numpy=ready[2], skimage=ready[3])

    def raw(self, img, fn, sev, seed):
        img = np.ascontiguousarray(img, dtype=np.uint8); h, w, _ = img.shape
        self.p.stdin.write((json.dumps(dict(fn=fn, sev=int(sev), seed=int(seed), h=h, w=w)) + "\n").encode() + img.tobytes())
        self.p.stdin.flush()
        buf = self.p.stdout.read(h * w * 3)
        return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

    def __call__(self, img, setting, unit_id, step):
        """img: raw agentview [128, 128, 3] uint8 at env step `step` (0 = reset observation)."""
        sp = spec(setting)
        return self.raw(img, sp["fn"], sp["severity"], seed_of(unit_id, step, sp["fn"], sp["severity"]))

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
