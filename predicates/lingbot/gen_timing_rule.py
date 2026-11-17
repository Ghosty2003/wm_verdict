"""Ruling 1 of 2026-10-15 (t5 predicate, option 1): content-based timing window for the GENERATED row.

Rule (fixed before computing; window never changed afterwards):
  For a one-chunk imagination started at step t0 (frame k = state after its k-th action, simulated step s = t0 + k),
  the reference real success step s_ref is the first step at which the simulator reports success when the chunk's own
  actions are executed (the paired real continuation of that chunk); if that continuation does not reach success
  within the chunk, s_ref is the source episode's first real success step; if neither exists there is no reference.
  A NEGATIVE generated frame (simulator: not yet / not successful) with |s - s_ref| <= 12 (= one LingBot-VA action
  chunk on either side) is listed separately: it is excluded from the generated row's TNR and reported on its own
  (count, and how many the predicate calls success). TPR is unaffected.
Why: inside that window an imagination that runs a few steps ahead of physics shows a genuinely successful frame
while the simulator-step label still says "not yet"; the predicate's job is to read frame content, so the generator's
time offset must not be charged to it. The excluded frames are themselves evidence (imagination ahead of physics,
same family as false-safe) and are saved as a list (early_success_chunks.json).
Applied to t5 AND t3 with the v3 thresholds and out-of-fold predictions; t3 must still pass or the rule is void.

  python gen_timing_rule.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np

_wmvp.bootstrap()
from recal_v3 import registry, OUT, GATE
from lingbot_predicate import metrics

WINDOW = 12


def starts_of(e):
    """the start steps recal_v3.gen_data used for this episode (same code path)."""
    z = np.load(e["path"]); lab = z["labels"]; n = len(z["actions"])
    if not lab.any():
        starts = np.linspace(n // 2, n - 13, 3).astype(int)
    else:
        fs = int(np.argmax(lab))
        starts = np.clip(np.array([fs - 36, fs - 12, fs + 12]), 1, n - 13)
    return sorted(set(int(x) for x in starts)), (int(np.argmax(lab)) if lab.any() else None)


def main():
    summary = {}
    for tid in (5, 3):
        d = f"{OUT}/t{tid}"
        cal = json.load(open(f"{d}/calibration.json")); thr = cal["threshold"]
        gn = np.load(f"{d}/gen_all.npz"); z = np.load(f"{d}/oof.npz")
        y = gn["labels"].astype(bool); ft = gn["fail_terminal"]; ep = gn["episode"]; p = z["oof_gn"]
        E = [e for e in registry(tid) if e["gen_ok"]]
        # reconstruct (episode, t0, k) for every generated frame; 12 frames per start in gen_data order
        meta, i = [], 0
        for e in E:
            st, s_ep = starts_of(e)
            for t0 in st:
                for k in range(1, 13):
                    meta.append((e["id"], e["source"], t0, k, s_ep)); i += 1
        assert len(meta) == len(y), (len(meta), len(y))
        assert all(m[0] == ep[j] for j, m in enumerate(meta)), "episode order mismatch"
        win = np.zeros(len(y), bool); sref = np.full(len(y), -1)
        for b in range(0, len(y), 12):
            blk = slice(b, b + 12)
            yb = y[blk]; e_id, src, t0, _, s_ep = meta[b]
            s_chunk = t0 + int(np.argmax(yb)) + 1 if yb.any() else None
            ref = s_chunk if s_chunk is not None else s_ep
            if ref is None:
                continue
            for j in range(b, b + 12):
                s = meta[j][2] + meta[j][3]
                sref[j] = ref
                if not y[j] and abs(s - ref) <= WINDOW:
                    win[j] = True
        keep = ~win
        raw = metrics(p, y, thr)
        rule = metrics(p[keep], y[keep], thr)
        ftn = ft & ~y
        rule["fail_terminal"] = metrics(p[keep & ftn], y[keep & ftn], thr) if (keep & ftn).any() else None
        raw["fail_terminal"] = metrics(p[ftn], y[ftn], thr) if ftn.any() else None
        wneg = dict(n=int(win.sum()), predicate_calls_success=int((p[win] >= thr).sum()),
                    share_of_raw_false_positives=float((p[win] >= thr).sum() / max(1, ((~y) & (p >= thr)).sum())))
        # early-success chunks: window negatives the predicate reads as success
        es = {}
        for j in np.where(win & (p >= thr))[0]:
            e_id, src, t0, k, s_ep = meta[j]
            key = f"{e_id}@{t0}"
            es.setdefault(key, dict(episode=e_id, source=src, t0=int(t0), reference_success_step=int(sref[j]),
                                    frames=[]))["frames"].append(dict(k=int(k), step=int(t0 + k), p=float(p[j]),
                                                                     steps_before_reference=int(sref[j] - (t0 + k))))
        es = sorted(es.values(), key=lambda r: (r["episode"], r["t0"]))
        json.dump(dict(task_id=tid, window_steps=WINDOW, threshold=thr, n_chunks=len(es),
                       n_frames=sum(len(r["frames"]) for r in es), chunks=es),
                  open(f"{d}/early_success_chunks.json", "w"), indent=1)
        gate_rule = bool(rule["tpr"] >= GATE and rule["tnr"] >= GATE and rule["fail_terminal"] is not None
                         and rule["fail_terminal"]["tnr"] >= GATE)
        block = dict(
            definition="negative generated frames with |s - s_ref| <= 12 steps (one action chunk either side) are "
                       "excluded from the generated row's TNR and reported separately; s_ref = first simulated success "
                       "step of the chunk's own continuation, else the source episode's first success step",
            rationale="inside the window an imagination running ahead of physics shows a genuinely successful frame "
                      "while the simulator-step label says 'not yet'; the predicate reads frame content and is not "
                      "charged for the generator's time offset (ruling 1, 2026-10-15; fixed before computing)",
            window_steps=WINDOW, raw=raw, rule=rule, window_negatives=wneg, gate_rule=gate_rule,
            early_success_chunks=f"libero/out/predicate_v3/t{tid}/early_success_chunks.json")
        cal["generated_row_timing_window_rule"] = block
        cal["model_generated_1chunk_timing_rule"] = dict(rule, note="generated row under the timing-window rule")
        others = all(cal["gate_85"][k] for k in cal["gate_85"] if not k.startswith("model_generated_1chunk"))
        cal["gate_pass_with_timing_rule"] = bool(others and gate_rule)
        json.dump(cal, open(f"{d}/calibration.json", "w"), indent=1)
        summary[tid] = dict(raw=(raw["tpr"], raw["tnr"], raw["fail_terminal"]["tnr"]),
                            rule=(rule["tpr"], rule["tnr"], rule["fail_terminal"]["tnr"]),
                            window=wneg, early_success_chunks=len(es), gate_rule=gate_rule,
                            gate_pass_with_timing_rule=cal["gate_pass_with_timing_rule"])
        print(tid, json.dumps(summary[tid]), flush=True)


if __name__ == "__main__":
    main()
