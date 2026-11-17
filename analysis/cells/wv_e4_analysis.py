"""WorldVLA = exclusion E4: per-unit malformation onset / share curves, image-quality decay vs step (PSNR of the imagined
frame against the paired truth (a) replay frame at the same step, both in WorldVLA's 256 px view), alignment of the
two, and the appendix verdict under the constrained-retry definition (with the free-output share per unit).

  python wv_e4_analysis.py          -> out/worldvla/e4_analysis.json + printed table
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import json
import numpy as np
from scipy.stats import spearmanr

PSNR_THR = 15.0      # fixed before the remaining units were seen (2026-10-15, eval_design section 21):
MA = 10              #   absolute: first step whose trailing 10-step mean PSNR < 15 dB
REL_DROP = 5.0       #   relative: first step whose trailing 10-step mean PSNR < (the unit's own first-10-step mean - 5 dB)
# n = 10: the Spearman coefficient is reported DESCRIPTIVELY (coefficient + scatter), no p-value, no significance claim

ROOT = _wmvp.P("$WMV_ROOT/libero/out/worldvla")


def psnr(a, b):
    m = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))


def main():
    rows = []
    for tid in (5, 3):
        for j in range(5):
            d = f"{ROOT}/libero10_t{tid}_a{j}/s1"
            if not os.path.exists(f"{d}/replay.json"):
                continue
            I = json.load(open(f"{d}/imagined.json")); R = json.load(open(f"{d}/replay.json"))
            fi = np.load(f"{d}/imagined.npz")["frames"]; fr = np.load(f"{d}/replay.npz")["frames"]
            n = min(len(fi), len(fr))
            ps = np.array([psnr(fi[t], fr[t]) for t in range(1, n)])            # step t = after t actions
            ms = sorted(I.get("vla_malformed_steps", []))
            onset = I.get("first_vla_malformed_step")
            pre = ps[:onset] if onset else ps
            post = ps[onset:] if onset else np.array([])
            # psnr decay summary per 50-step bin
            bins = {f"{b}-{b + 49}": round(float(ps[b:b + 50].mean()), 2) for b in range(0, len(ps), 50) if len(ps[b:b + 50])}
            ma = np.convolve(ps, np.ones(MA) / MA, mode="valid")              # ma[i] = mean of steps i+1 .. i+MA
            below = np.where(ma < PSNR_THR)[0]
            decay_onset = int(below[0] + MA) if len(below) else None
            below_rel = np.where(ma < ps[:MA].mean() - REL_DROP)[0]
            decay_onset_rel = int(below_rel[0] + MA) if len(below_rel) else None
            # step-normalised malformation: steps whose action came from a chunk of a malformed (retried) VLA call
            z = np.load(f"{d}/imagined.npz"); calls = list(z["t_vla_step"].astype(int)) + [I["n_actions"]]
            mal = set(ms); mal_steps = 0
            for c0, c1 in zip(calls[:-1], calls[1:]):
                if c0 in mal:
                    mal_steps += c1 - c0
            rows.append(dict(unit=f"libero10_t{tid}_a{j}", n_actions=I["n_actions"], status=I["status"],
                             vla_calls=I["n_vla_calls"], vla_malformed=I["n_vla_malformed"], first_vla_malformed_step=onset,
                             free_output_share_vla=round((I["n_vla_calls"] - I["n_vla_malformed"]) / max(1, I["n_vla_calls"]), 3),
                             malformed_step_share=round(mal_steps / 520, 3), decay_onset_step=decay_onset,
                             decay_onset_step_relative=decay_onset_rel,
                             wm_calls=I["n_wm_calls"], wm_malformed=I["n_wm_malformed"],
                             wm_first_malformed_step=I.get("first_wm_malformed_step"),
                             psnr_first10=round(float(ps[:10].mean()), 2), psnr_pre_onset=round(float(pre.mean()), 2) if len(pre) else None,
                             psnr_post_onset=round(float(post.mean()), 2) if len(post) else None, psnr_by_50_steps=bins,
                             truth_a_success=R["success"], truth_a_first_success_step=R["first_success_step"],
                             replay_bit_identical=R["replay_bit_identical"], generation_invalid=I.get("generation_invalid")))
    align = {}
    for key, rule in (("decay_onset_step", "absolute: trailing-10 mean PSNR < 15 dB"),
                      ("decay_onset_step_relative", "relative: trailing-10 mean PSNR < own first-10 mean - 5 dB")):
        pairs = [(r["unit"], r[key], r["first_vla_malformed_step"]) for r in rows
                 if r[key] is not None and r["first_vla_malformed_step"] is not None]
        rho = float(spearmanr([a for _, a, _ in pairs], [b for _, _, b in pairs])[0]) if len(pairs) >= 3 else None
        align[key] = dict(rule=rule, n=len(pairs), spearman_rho_descriptive=rho,
                          scatter=[dict(unit=u, decay_onset=a, collapse_onset=b) for u, a, b in pairs],
                          note="descriptive only (n<=10): no p-value, no significance claim")
    json.dump(dict(units=rows, alignment=align, summary=dict(
        malformed_step_share_mean=float(np.mean([r["malformed_step_share"] for r in rows])) if rows else None,
        units_truth_a_success=sum(r["truth_a_success"] for r in rows), n_units=len(rows))),
        open(f"{ROOT}/e4_analysis.json", "w"), indent=1)
    print("ALIGN", json.dumps(align))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
