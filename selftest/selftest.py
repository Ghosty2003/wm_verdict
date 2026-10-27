"""Cross-group equivalence self-test for the B-family corruption worker (protocol v1.0 section 8).

Runs the worker (corruption/bcorrupt_server.py, pinned dependencies) on the 6 bundled 128 x 128 LIBERO frames and
checks:
  1  determinism: same image + severity + seed -> identical bytes, within one worker and across two worker processes;
     fog changes with the seed
  2  equivalence: bit-identical to the WM group's archived outputs (reference_outputs.npz: gaussian_blur, fog,
     zoom_blur, glass_blur at severities 2 / 5 / 9, seed 0) -> both groups run the same algorithm on the same stack
  3  the calling process's numpy global RNG is untouched (the corruption RNG lives in the worker process)
  4  the seed rule is the agreed one (sha256 of "<unit_id>|<step>|agentview|<fn>|<severity>", first 8 hex digits)

  bash corruption/fetch_upstream.sh                       # pinned LIBERO-plus source, sha256-verified
  python3.10 -m venv .venv_pinned && .venv_pinned/bin/pip install -r corruption/requirements_pinned.txt
  B_PINNED_PY=$PWD/.venv_pinned/bin/python python selftest/selftest.py
"""
import os
import sys
import json
import hashlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LIBERO_PLUS_ENV_WRAPPER", f"{REPO}/upstream/env_wrapper.py")
os.environ.setdefault("B_PINNED_PY", f"{REPO}/.venv_pinned/bin/python")
sys.path.insert(0, f"{REPO}/corruption")
from bcorrupt import Corruptor, SETTINGS, seed_of


def main():
    F = np.load(f"{REPO}/selftest/test_frames.npz")["frames"]
    ref = np.load(f"{REPO}/selftest/reference_outputs.npz")
    c1, c2 = Corruptor(), Corruptor(); R = dict(worker_versions=c1.versions); ok = True
    n = same = cross = fog = 0
    for s in SETTINGS[1:]:
        for j, img in enumerate(F):
            a = c1(img, s, f"selftest_{j}", 7); n += 1
            same += np.array_equal(a, c1(img, s, f"selftest_{j}", 7)); cross += np.array_equal(a, c2(img, s, f"selftest_{j}", 7))
            if s[1] == "f":
                fog += not np.array_equal(a, c1(img, s, f"selftest_{j}", 8))
    R["1_determinism"] = dict(same_worker=f"{same}/{n}", two_workers=f"{cross}/{n}", fog_seed_sensitive=f"{fog}/{n // 3}")
    ok &= same == n and cross == n and fog == n // 3
    m = k = 0; bad = []
    for key in ref.files:
        fn, sev, j = key.rsplit("_", 2); k += 1
        eq = np.array_equal(c1.raw(F[int(j)], fn, int(sev), 0), ref[key]); m += eq
        if not eq:
            bad.append(key)
    R["2_matches_wm_group_reference"] = f"{m}/{k}"; R["2_mismatches"] = bad; ok &= m == k
    np.random.seed(123); st = np.random.get_state()[1].copy()
    for s in SETTINGS[1:]:
        c1(F[0], s, "selftest_rng", 3)
    R["3_caller_rng_untouched"] = bool(np.array_equal(np.random.get_state()[1], st)); ok &= R["3_caller_rng_untouched"]
    R["4_seed_rule"] = dict(example='seed_of("unit_x", 12, "fog", 5)', value=seed_of("unit_x", 12, "fog", 5),
                            expected=int(hashlib.sha256(b"unit_x|12|agentview|fog|5").hexdigest()[:8], 16))
    ok &= R["4_seed_rule"]["value"] == R["4_seed_rule"]["expected"]
    c1.close(); c2.close(); R["pass"] = bool(ok)
    print(json.dumps(R, indent=1)); print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
