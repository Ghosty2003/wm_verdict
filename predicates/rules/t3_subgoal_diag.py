"""t3 sub-goal diagnosis of trajectory-level false alarms (2026-10-20 ruling 1). Read-only: uses the per-step BDDL
sub-goal traces written by content_review.py (bit-exact replays) and the stored per-frame predicate readouts on the
REAL frames (real_pred, predicate v3, threshold_m).

Population: t3 real-failure units of lingbot_famA / lingbot_famP (all are false-safe under rule v1). For each unit the
longest run of consecutive real frames with p >= threshold is the false alarm; "long" = run >= 50 frames.
Class of a unit = the majority sub-goal state over the frames of that run:
  in_not_close   In(bowl, cabinet_bottom) and not Close(cabinet_bottom)
  close_not_in   Close and not In
  neither        neither holds
  both           In and Close at the same frame -- impossible in a failure trajectory (check_success is their
                 conjunction); counted as a consistency check
Also per unit: whether In and Close each held at some step (never at the same step), and whether In held and was
later lost ("exited").

  python t3_subgoal_diag.py      -> out/content_review/t3_subgoal_diag.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import numpy as np
from collections import Counter

ROOT = _wmvp.P("$WMV_ROOT")
CR = f"{ROOT}/libero/out/content_review"
LONG = 50


def longest_run(h):
    best, start, run = (0, None), None, 0
    for i, x in enumerate(h):
        if x:
            run += 1
            if run == 1:
                start = i
            if run > best[0]:
                best = (run, start)
        else:
            run = 0
    return best


def main():
    thr = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t3/calibration.json"))["threshold"])
    log = {u["unit"]: u for u in json.load(open(f"{CR}/replay_log.json"))["units"] if u["task"] == "t3"}
    rows = []
    for uid, L in log.items():
        assert L["replay_bit_exact"] and L["success_trace_match"], uid
        run = "lingbot_famA" if L["family"] == "famA" else "lingbot_famP"
        tr = np.load(f"{ROOT}/libero/out/runs/{run}/trajectories/{uid}.npz")
        z = np.load(f"{CR}/units/{uid}.npz")
        names = [str(n) for n in z["goal_names"]]
        i_in = next(i for i, n in enumerate(names) if n.startswith("in "))
        i_cl = next(i for i, n in enumerate(names) if n.startswith("close "))
        sub = z["subgoals"]                                   # [521, n_goals], step 0..520
        IN, CL = sub[:, i_in], sub[:, i_cl]
        h = tr["real_pred"] >= thr; h[0] = False              # frames 1..520 only
        n, s = longest_run(h)
        st = Counter()
        if n:
            for t in range(s, s + n):
                st["both" if IN[t] and CL[t] else "in_not_close" if IN[t] else "close_not_in" if CL[t] else "neither"] += 1
        cls = st.most_common(1)[0][0] if n else None
        in_ever = bool(IN[1:].any()); cl_ever = bool(CL[1:].any())
        in_exit = bool(in_ever and not IN[-1])
        rows.append(dict(unit=uid, family=L["family"], setting=L["setting"], longest_false_alarm=n,
                         run_start=s, run_state_counts=dict(st), run_class=cls, in_ever=in_ever, close_ever=cl_ever,
                         in_and_close_each_but_never_together=bool(in_ever and cl_ever and not (IN & CL).any()),
                         in_then_lost=in_exit, final_in=bool(IN[-1]), final_close=bool(CL[-1])))

    def table(sel):
        return dict(n=len(sel), run_class=dict(Counter(r["run_class"] for r in sel)),
                    in_and_close_each_but_never_together=sum(r["in_and_close_each_but_never_together"] for r in sel),
                    in_then_lost=sum(r["in_then_lost"] for r in sel),
                    final_state=dict(Counter(("In" if r["final_in"] else "notIn") + "/" + ("Close" if r["final_close"] else "Open")
                                             for r in sel)))
    out = dict(threshold=thr, long_run_min=LONG, goal_names=names,
               long=table([r for r in rows if r["longest_false_alarm"] >= LONG]),
               short=table([r for r in rows if 0 < r["longest_false_alarm"] < LONG]),
               no_fire=table([r for r in rows if r["longest_false_alarm"] == 0]),
               by_setting_long={s: table([r for r in rows if r["longest_false_alarm"] >= LONG and r["setting"] == s])
                                for s in sorted({r["setting"] for r in rows})},
               units=rows)
    json.dump(out, open(f"{CR}/t3_subgoal_diag.json", "w"), indent=1)
    for k in ("long", "short", "no_fire"):
        print(k, json.dumps(out[k]))
    for s, v in out["by_setting_long"].items():
        print("  long", s, json.dumps(v))


if __name__ == "__main__":
    main()
