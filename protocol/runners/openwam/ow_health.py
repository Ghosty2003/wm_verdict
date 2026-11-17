"""Health check of the OpenWAM-alpha runs while the driver is running (read-only; any interpreter).

Per run x task: units per level, duplicates, truth (a) replay identical, closed-loop rerun identical, action / trajectory
files present, machine, H. Prints one summary line per run x task and a final ANOMALIES line (unit ids).

  python ow_health.py            # human-readable
  python ow_health.py --key      # one line: counts that a watcher compares between polls
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import collections

ROOT = _wmvp.P("$WMV_ROOT")
RUNS = ("openwam_base", "openwam_famA", "openwam_famP")


def check():
    out, anomalies = [], collections.defaultdict(list)
    for run in RUNS:
        for t in ("t5", "t3"):
            p = f"{ROOT}/libero/out/runs/{run}/records_{t}.jsonl"
            if not os.path.exists(p):
                continue
            R = [json.loads(l) for l in open(p)]
            ids = [r["unit_id"] for r in R]
            for r in R:
                u = r["unit_id"]
                if not r["replay_check_identical"]:
                    anomalies["truth_replay_not_identical"].append(u)
                cl = r["closed_loop"]
                if not cl["deterministic_rerun_identical"]:          # ruling 2026-10-24: conclusion level is what counts
                    if "rerun" in cl or "exemption" in cl:
                        others = [cl["rerun"]] if "rerun" in cl else cl["exemption"].get("reruns", [])
                        ok = all(bool(o["success"]) == bool(cl["success"]) for o in others)   # revision 2026-10-25
                        anomalies["closed_loop_bit_level_only (exempt)" if ok else "CLOSED_LOOP_CONCLUSION_MISMATCH"].append(u)
                    else:
                        anomalies["closed_loop_not_identical_pending_rerun"].append(u)
                if not all(os.path.exists(f"{ROOT}/{r[k]}") for k in ("action_file", "trajectory_file")):
                    anomalies["missing_files"].append(u)
                if r["machine"] != "$NODE_D" or r["H_env"] != 400:
                    anomalies["machine_or_H"].append(u)
            if len(ids) != len(set(ids)):
                anomalies["duplicate_units"] += [u for u, n in collections.Counter(ids).items() if n > 1]
            lv = collections.Counter(r["setting_id"] for r in R)
            out.append(f"{run} {t}: {len(R)} units {dict(lv)}; real success {sum(r['real_verdict'] for r in R)}, "
                       f"closed-loop success {sum(bool(r['closed_loop']['success']) for r in R)}, "
                       f"{sum(r['seconds_total'] for r in R) / max(len(R), 1):.1f} s/unit")
    return out, dict(anomalies)


if __name__ == "__main__":
    lines, an = check()
    if "--key" in sys.argv:
        print(json.dumps({k: len(v) for k, v in sorted(an.items())}))
    else:
        print("\n".join(lines)); print("ANOMALIES", json.dumps(an))
