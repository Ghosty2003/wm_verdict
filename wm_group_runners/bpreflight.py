"""B family pre-run checks, unified protocol v1.0 section 8 (CPU; lerobot_libero interpreter, worker in the pinned venv).

  1  same raw image + severity + seed -> identical output (twice in one worker, and across two worker processes);
     a different seed changes fog
  2  the worker reproduces the function check's archived outputs bit for bit (outputs_pinned*.npz, seed 0): the
     production path runs exactly the code that was checked
  3  baseline leaves the image untouched (no-corruption setting never calls the worker; checked here on the hook of
     the OpenWAM runner, and on GPU by lingbot_famB.py sham for LingBot-VA)
  4  only the agentview changes (the hook replaces one key; the wrist array is the same object)
  5  the client process's numpy global RNG is untouched by corruption calls (separate process)
  6  the seed rule is stable (sha256; no Python hash())
Per-step application from t = 0 on is asserted inside the runners (corrupted frames == steps read) and recorded.

  python bpreflight.py   -> out/b_family/preflight.json; exit 1 on any failure
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/qnap/chenmg93/wm_verdict"
HC = f"{ROOT}/libero/out/b_family/healthcheck"
sys.path[:0] = [HERE, HC]
from bcorrupt import Corruptor, SETTINGS, TYPES, seed_of


def frames():
    out = []
    for path in (f"{ROOT}/libero/out/predicate_v3/t5/nearmiss_eps/nm_base_00.npz", f"{ROOT}/libero/out/predicate_v4_t3/episodes/v1_r005.npz"):
        fr = np.load(path)["frames"]
        for i in (0, len(fr) // 2, len(fr) - 1):
            out.append(np.ascontiguousarray(fr[i, :, :128]))
    return out


def main():
    F = frames(); R = {}; ok = True
    c1, c2 = Corruptor(), Corruptor()
    # 1 determinism
    same, cross, seed_diff = 0, 0, 0; n = 0
    for s in SETTINGS[1:]:
        for j, img in enumerate(F):
            a = c1(img, s, f"pre_{j}", 7); b = c1(img, s, f"pre_{j}", 7); c = c2(img, s, f"pre_{j}", 7); n += 1
            same += np.array_equal(a, b); cross += np.array_equal(a, c)
            if s[1] == "f":
                seed_diff += not np.array_equal(a, c1(img, s, f"pre_{j}", 8))
    R["1_determinism"] = dict(same_worker=f"{same}/{n}", two_workers=f"{cross}/{n}", fog_other_step_seed_differs=f"{seed_diff}/{n // 3}")
    ok &= same == n and cross == n and seed_diff == n // 3
    # 2 identical to the archived function-check outputs
    ref = dict(np.load(f"{HC}/outputs_pinned.npz")); ref.update(np.load(f"{HC}/outputs_pinned_zoom_blur.npz"))
    m, k = 0, 0
    for t, (fn, _) in TYPES.items():
        for sev in (2, 5, 9):
            for j, img in enumerate(F):
                k += 1; m += np.array_equal(c1.raw(img, fn, sev, 0), ref[f"{fn}_{sev}_{j}"])
    R["2_matches_function_check"] = f"{m}/{k}"; ok &= m == k
    # 3 + 4 hook behaviour (OpenWAM runner hook; LingBot hook asserts the same inside lingbot_famB and its sham on GPU)
    sys.path.insert(0, f"{ROOT}/libero/openwam")
    src = open(f"{HERE}/ow_famB.py").read(); ns = {"np": np}
    exec(compile(src[src.index("def corrupt("):src.index("def closed_b(")], "ow_famB.corrupt", "exec"), ns)
    o = dict(agentview_image=F[0].copy(), robot0_eye_in_hand_image=F[1].copy(), robot0_eef_pos=np.zeros(3))
    o0 = ns["corrupt"](o, c1, "B0", "pre_hook", 0)
    o9 = ns["corrupt"](o, c1, "Bg9", "pre_hook", 0)
    R["3_baseline_untouched"] = bool(o0 is o and np.array_equal(o0["agentview_image"], F[0]))
    R["4_only_agentview"] = bool(o9["robot0_eye_in_hand_image"] is o["robot0_eye_in_hand_image"] and o9["robot0_eef_pos"] is o["robot0_eef_pos"]
                                 and not np.array_equal(o9["agentview_image"], F[0]) and np.array_equal(o["agentview_image"], F[0]))
    ok &= R["3_baseline_untouched"] and R["4_only_agentview"]
    # 5 client RNG untouched
    np.random.seed(123); st0 = np.random.get_state()[1].copy()
    for s in SETTINGS[1:]:
        c1(F[0], s, "pre_rng", 3)
    R["5_client_rng_untouched"] = bool(np.array_equal(np.random.get_state()[1], st0)); ok &= R["5_client_rng_untouched"]
    # 6 seed rule
    want = int(hashlib.sha256(b"unit_x|12|agentview|fog|5").hexdigest()[:8], 16)
    R["6_seed_rule_stable"] = bool(seed_of("unit_x", 12, "fog", 5) == want); ok &= R["6_seed_rule_stable"]
    R["worker_versions"] = c1.versions; R["pass"] = bool(ok)
    c1.close(); c2.close()
    os.makedirs(f"{ROOT}/libero/out/b_family", exist_ok=True)
    json.dump(R, open(f"{ROOT}/libero/out/b_family/preflight.json", "w"), indent=1)
    print(json.dumps(R, indent=1)); print("PREFLIGHT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
