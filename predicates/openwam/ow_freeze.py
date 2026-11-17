"""Freeze the OpenWAM-alpha-view predicate into protocol.yaml (add-only) and declare the OpenWAM runs for verify_run.
Run only after the gate check (driver.sh): both tasks' frame-level gates passed and a trajectory-level k was selected.

t5: single success detector. t3: conjunction of two detectors (Close AND NOT bowl-outside-visible). verify_run reads one
predicate per task, so for t3 a manifest (both .pt files + SHAs + thresholds) stands in for the predicate file, and a
combined calibration file carries, per gate row, the WORSE of the two detectors' numbers (the row passes only if both do).
The scalar readout written into records ('feat_dist') is the success margin: t5 1 - p (success iff <= 1 - thr);
t3 -min(p_close - thr_close, thr_bowl - p_bowl) (success iff <= 0).

  python ow_freeze.py [tasks, default 5,3]     # e.g. 3 now; 5 later adds t5 to the frozen section (add-only)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import re
import json
import math
import shutil
import hashlib
import datetime
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
PD = f"{HERE}/predicate"
ROWS = ("holdout_real", "holdout_vae_roundtrip", "generated_timing_rule")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT)


def worse(a, b):
    """per gate row: the detector with the lower min(TPR, TNR) and the lower failure-terminal TNR."""
    out = {}
    for k in ("agreement", "tpr", "tnr"):
        vals = [x[k] for x in (a, b) if x.get(k) is not None]
        out[k] = min(vals) if vals else None
    fts = [x["fail_terminal"]["tnr"] for x in (a, b) if x.get("fail_terminal")]
    out["fail_terminal"] = {"tnr": min(fts)} if fts else None
    out["n"] = min(a["n"], b["n"])
    return out


def task_entry(tid):
    d = f"{PD}/t{tid}"
    traj = json.load(open(f"{d}/trajectory_eval.json"))
    k = int(traj["selected_k"]); k_frames = math.ceil(k / 4)
    if tid == 5:
        cal = json.load(open(f"{d}/calibration_success.json"))
        thr = cal["threshold"]; cal_path = f"{d}/calibration_success.json"; pred_path = f"{ROOT}/{cal['artifact']['path']}"
        fthr = 1 - thr
    else:
        cc, cb = (json.load(open(f"{d}/calibration_{n}.json")) for n in ("close", "bowl_out"))
        thr = {"close": cc["threshold"], "bowl_out": cb["threshold"]}
        man = dict(kind="t3 conjunction: success frame iff p_close >= thr.close AND p_bowl_out < thr.bowl_out",
                   detectors={n: dict(path=c["artifact"]["path"], sha256=c["artifact"]["sha256"], threshold=c["threshold"])
                              for n, c in (("close", cc), ("bowl_out", cb))})
        pred_path = f"{d}/predicate_manifest.json"; json.dump(man, open(pred_path, "w"), indent=1)
        comb = dict(threshold=thr, note="per gate row the worse of the two detectors (the row passes only if both do)",
                    **{r: worse(cc[r], cb[r]) for r in ROWS}, drawer_closed=cb.get("drawer_closed"),
                    detectors={"close": rel(f"{d}/calibration_close.json"), "bowl_out": rel(f"{d}/calibration_bowl_out.json")})
        cal_path = f"{d}/calibration_verify.json"; json.dump(comb, open(cal_path, "w"), indent=1)
        fthr = 0.0
    bank = yaml.safe_load(open(f"{ROOT}/protocol.yaml"))["libero_lingbot_v3"]["tasks"][f"t{tid}"]["bank"]
    return dict(task_id=tid, H_env=400, bank=bank,
                predicate=dict(kind=("OpenWAM-view v3-recipe success detector" if tid == 5 else "OpenWAM-view v4 conjunction (manifest)")
                               + " on ow_view.pred_view of the OpenWAM canvas (agentview | wrist, 128 x 256)",
                               path=rel(pred_path), sha256=sha(pred_path), calibration_path=rel(cal_path), calibration_sha256=sha(cal_path),
                               threshold_m=thr, feat_dist_threshold=fthr,
                               gate=dict(evaluations=list(ROWS), min_tpr=0.85, min_tnr=0.85, min_fail_terminal_tnr=0.85),
                               trajectory_rule=dict(k_steps=k, k_imagined_frames=k_frames,
                                                    note="imagined frames are 4 env steps apart (video stride 4): k steps -> ceil(k/4) consecutive imagined frames",
                                                    eval=rel(f"{d}/trajectory_eval.json"), eval_sha256=sha(f"{d}/trajectory_eval.json"))))


def add_task(p, proto, tasks):
    """section already frozen: add the new task entries and extend the OpenWAM runs' task lists (add-only)."""
    have = proto["libero_openwam"]["tasks"]
    new = [t for t in tasks if f"t{t}" not in have]
    if not new:
        raise SystemExit(f"tasks {tasks} already frozen -- refusing to overwrite (add-only)")
    text = open(p).read()
    entries = yaml.safe_dump({f"t{t}": task_entry(t) for t in new}, sort_keys=False, allow_unicode=True, width=140)
    anchor = "libero_openwam:\n  tasks:\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "".join("    " + l + "\n" for l in entries.splitlines()))
    for run in ("openwam_base", "openwam_famA_obj", "openwam_famP_pose"):
        i = text.index(f"\n  {run}:\n"); m = re.compile(r"    tasks:\n((?:    - t\d+\n)+)").search(text, i)
        cur = {int(x) for x in re.findall(r"- t(\d+)", m.group(1))}
        text = text[:m.start()] + "    tasks:\n" + "".join(f"    - t{t}\n" for t in sorted(cur | set(new), reverse=True)) + text[m.end():]
    open(p, "w").write(text); yaml.safe_load(open(p))
    print("added tasks", new)


def main(tasks):
    p = f"{ROOT}/protocol.yaml"
    shutil.copy(p, f"{ROOT}/libero/out/protocol_backup_{datetime.date.today():%Y%m%d}_openwam.yaml")
    proto = yaml.safe_load(open(p))
    if "libero_openwam" in proto:
        return add_task(p, proto, tasks)
    perts = {t: dict(none={"family": None, "severity": 0, "spec": {}}, **v)          # 'none' = the main-bank baseline
             for t, v in proto["libero_lingbot_v3"]["perturbations"].items()}
    sec = dict(tasks={f"t{t}": task_entry(t) for t in tasks}, perturbations=perts)
    runs = {}
    common = dict(tasks=[f"t{t}" for t in tasks], seeds_per_start=10, env_adapter="libero_env_adapter:replay_unit", check7_rule="v2",
                  task_section="libero_openwam", model="OpenWAM/OpenWAM-Alpha-Sim-LIBERO",
                  chaining="calibrated controller (injected component; libero.openwam_alpha.chaining)",
                  closed_loop="eval_design section 28 protocol (reference column; never in the paired verdict table)",
                  machine="$NODE_D")
    runs["openwam_base"] = dict(records="libero/out/runs/openwam_base/records_{task}.jsonl", settings=["none"], baseline_run=None, **common)
    runs["openwam_famA_obj"] = dict(records="libero/out/runs/openwam_famA/records_{task}.jsonl", settings=["A0", "A1", "A2", "A3", "A4"],
                                    baseline_run=None, banks={"t5": "famA_obj_t5", "t3": "famA_obj_t3"}, **common)
    runs["openwam_famP_pose"] = dict(records="libero/out/runs/openwam_famP/records_{task}.jsonl", settings=["P0", "P1", "P2", "P3", "P4"],
                                     baseline_run=None, banks={"t5": "famP_pose_t5", "t3": "famP_pose_t3"}, **common)
    sec["perturbations_note"] = "family A / P settings shared with libero_lingbot_v3 (same anchors, same specs); 'none' = main bank"
    text = open(p).read().rstrip("\n")
    block = ("\n\n# 2026-10-24 OpenWAM-alpha-view predicate (frozen by libero/openwam/ow_freeze.py after the gate check) and the\n"
             "# OpenWAM-alpha runs. Add-only.\n" + yaml.safe_dump({"libero_openwam": sec}, sort_keys=False, allow_unicode=True, width=140))
    runs_text = yaml.safe_dump(runs, sort_keys=False, allow_unicode=True, width=140)
    # insert the run entries at the end of the top-level runs: block
    i = text.index("\nruns:\n") + len("\nruns:\n"); j = text.index("\nverify:\n")
    text = text[:j] + "\n" + "".join("  " + l + "\n" if l else "\n" for l in runs_text.splitlines()) + text[j:]
    open(p, "w").write(text + block)
    yaml.safe_load(open(p))                                         # must still parse
    print("frozen", json.dumps({t: sec["tasks"][t]["predicate"]["trajectory_rule"] for t in sec["tasks"]}))


if __name__ == "__main__":
    import sys
    main([int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "5,3").split(",")])
