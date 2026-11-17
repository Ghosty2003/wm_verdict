"""verify_run.py -- integrity / validity gate for one WM-Verdict run. No check may FAIL before analysis.

Principle (2026-10-13 rule revision, out/spec_revisions.md #1): the gate guards PROCESS correctness. The direction and
monotonicity of a perturbation's effect are measured properties, not gate conditions -- a perturbation that makes
reality better (more false-unsafes) is as valid as one that makes it worse (more false-safes), if recorded faithfully.

    python verify_run.py --run-dir out/ --protocol protocol.yaml [--run step7] [--report PATH]

Model-family agnostic: every check works on the records / artifacts on disk. The only simulator contact is check 2
(truth-replay spot check), which goes through the adapter named in protocol.yaml (verify.env_adapter).

Checks
  1 threshold fixedness      every unit's threshold / tol / H / perturbation spec == protocol; unit set == protocol
                             (settings x bank states, no gaps / duplicates); bank + calibration SHA-256 == protocol
  2 restore exactness        n bank states (fixed seed): restore -> recorded actions -> compare to recorded truth
  3 predicate calibration    step-4 holdout agreement exists and >= protocol floor
  4 replay determinism       each unit's two truth replays identical (verdict, final dist, final state)
  5 sham                     sham units == baseline units on verdict, final dist, final state, final feat dist
                             (SKIP when the run has no sham setting)
                             A check-4 FAIL likewise needs a human decision (precedent #3:
                             mmbench2/out/verify_waiver_mmb_og_cal_check4.md, cross-instance render nondeterminism).
  6 dose monotonicity        per family, real success count non-increasing with severity (baseline = severity 0);
                             per-state final-distance exceptions are listed. Since 2026-10-13 a non-monotone family is
                             a WARN, not a FAIL: it does not fail the run and needs no written release, but it is listed
                             at the top of the report, and the analysis report must give one line per non-monotone
                             family: genuine response of the environment, or suspected process problem.
                             Applies forward only: reports written before 2026-10-13 keep their FAIL, and precedents
                             #1 (out/verify_waiver_stage2_pert_check6.md) and #2
                             (mmbench2/out/verify_waiver_mmb_pert_check6.md) stand as recorded.
                             A task with no perturbations entry in the protocol (e.g. a calibration-tier task) is
                             evaluated as unperturbed (implicit 'none'); checks 5 and 6 then say so explicitly in
                             their notes and the report lists the task under implicit_unperturbed_tasks.
  7 false-safe attribution   predicate error vs prediction error per false-safe unit (statistics only; FAIL only if a
                             false-safe unit lacks the real-final feature distance needed to attribute it).
                             Spec revision #2 (2026-10-15, forward only, runs with check7_rule: v2): two layers --
                             layer 1 imagined vs real endpoint CONTENT (content_attribution file, manual frame review
                             where needed; unreviewed units -> WARN), layer 2 predicate reliability on the real endpoint
                             distribution over all units. The v1 split is kept as legacy_rule_split.
  8 parameter effect         (2026-10-13) the process-error detector that replaces outcome-based dose screening: per
                             perturbed setting, restore one bank state and replay the SAME action sequence (a recorded
                             one of that setting) with the perturbation on and off; the stepwise states must differ
                             somewhere (any difference; outcomes are not looked at, so it does not depend on whether
                             the chosen start happens to be immune). Sham settings must give zero difference.
                             Deterministic, one pair of replays per setting, through the run's env adapter.
                             Observation-channel perturbations (sensor noise, image degradation) cannot be asserted on
                             the state; their rule is an observation-level assertion (same state, same render: the
                             observation the policy actually receives must differ with the perturbation on vs off, and
                             be identical under sham). Not implemented here (all current families act on the action /
                             dynamics channel); workflows with observation families must implement it as a must-pass
                             check. SKIP without perturbed settings.
  10 archive minimum          (2026-11-13, forward only) rows carrying min_artifacts (protocol.yaml min_artifacts): per
                             (task, setting) cell, a fixed-seed sample of completed units is reopened -- the npz must exist
                             and open, every required key must be present for each pass, and action / success lengths
                             must match the recorded steps and outcome (libero/min_artifacts.py). Any failing cell is a
                             FAIL; such a cell must not be marked complete. Rows without the field (data before the rule)
                             are not retro-checked: SKIP.
Together with check 5 (sham == baseline) and the hook-default bit-identity test of each hook check, check 8 is the
process-error defence; "the lightest level must not wipe out everything" is a design recommendation, not a gate.
Exit code 1 if any check FAILs (WARN does not fail the run).
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import hashlib
import argparse
import importlib
import numpy as np
import yaml

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


# ------------------------------------------------------------------------------------------------ helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["_line"] = i
                out.append(r)
    return out


def setting_of(r):
    return r.get("setting_id", r.get("perturbation"))


def ukey(r):
    """Unit key within a setting: (s0, planner-seed index); single-seed runs have seed index 0."""
    return (r["s0_id"], r.get("seed_index", 0))


def klabel(k):
    return k[0] if k[1] == 0 else f"{k[0]}#k{k[1]}"


def loc(r, path):
    return f"{os.path.basename(path)}:{r['_line']} ({r.get('unit_id')})"


def result(cid, name, status, values=None, failures=None, notes=None):
    return dict(id=cid, name=name, status=status, values=values or {}, failures=failures or [], notes=notes or [])


class Run:
    def __init__(self, run_dir, proto, run):
        self.dir, self.proto, self.run = run_dir, proto, run
        self.spec = proto["runs"][run]
        # a run may cover a subset of tasks and may take its task configs from another section (e.g. mmbench2)
        self.section = proto[self.spec["task_section"]] if self.spec.get("task_section") else None
        self.tcfgs = self.section["tasks"] if self.section else proto["tasks"]
        self.tasks = list(self.spec.get("tasks", self.tcfgs))
        self.records, self.paths, self.missing = {}, {}, []
        self.implicit_none = set()            # tasks without a perturbations entry -> treated as unperturbed
        for t in self.tasks:
            p = os.path.join(run_dir, self.spec["records"].format(task=t))
            self.paths[t] = p
            if os.path.exists(p):
                self.records[t] = load_jsonl(p)
            else:
                self.missing.append(p)
        self.baseline = None
        b = self.spec.get("baseline_run")
        if b:
            self.baseline = {}
            for t in self.tasks:
                p = os.path.join(run_dir, proto["runs"][b]["records"].format(task=t))
                if os.path.exists(p):
                    self.baseline[t] = {ukey(r): r for r in load_jsonl(p)}

    def bank(self, task):
        """Bank entry for this run: runs.<run>.banks.<task> -> banks.<name>, else the task config's bank."""
        name = self.spec.get("banks", {}).get(task)
        return self.proto["banks"][name] if name else self.tcfgs[task]["bank"]

    def perts(self, task):
        sec = self.section if self.section and "perturbations" in self.section else self.proto   # a section without its own
        P = sec["perturbations"]                                                               # perturbations shares the top-level ones
        if task not in P:                     # unperturbed-only task (e.g. calibration tier): implicit 'none' only
            self.implicit_none.add(task)
            return {"none": {"family": None, "severity": 0, "spec": {}}}
        return P[task]

    def tcfg(self, task):
        return self.tcfgs[task]

    def threshold(self, task):
        c = self.tcfgs[task]
        return c["predicate"]["threshold_m"] if "predicate" in c else c["threshold"]

    def H(self, task):
        return self.tcfgs[task].get("H_env", self.proto["H_env"])

    def cal_files(self, task):
        c = self.tcfgs[task]
        if "predicate" in c:
            pr = c["predicate"]
            return [(pr["path"], pr["sha256"]), (pr["calibration_path"], pr["calibration_sha256"])]
        return [(c["calibration"]["summary_path"], c["calibration"]["summary_sha256"]),
                (c["calibration"]["pairs_path"], c["calibration"]["pairs_sha256"])]

    def s0_ids(self, task):
        bj = os.path.join(self.dir, self.bank(task)["summary_path"])
        ids = [e["s0_id"] for e in json.load(open(bj))["entries"]] if os.path.exists(bj) else []
        return ids[:int(self.spec["max_starts"])] if self.spec.get("max_starts") else ids

    def settings(self, task):
        P = self.perts(task)
        return [s for s in self.spec["settings"] if s in P]


# ------------------------------------------------------------------------------------------------ checks
def _release_hash_ok(rel, want, got):
    """Release copy (2026-11-16): a file may carry the hash of its sanitized release copy. Accepted only if the protocol's
    hash is the recorded ORIGINAL hash of that path and the file matches the recorded release-copy hash
    (protocol/release_hashes.yaml next to the protocol file)."""
    p = os.path.join(os.path.dirname(os.environ.get("WMV_PROTOCOL", "")), "release_hashes.yaml")
    if not os.path.exists(p):
        return False
    rh = yaml.safe_load(open(p))
    orig = (rh.get("authoritative_original") or {}).get(rel)
    rel_copy = {**(rh.get("release_copy") or {}), **(rh.get("release_copy_with_path_module") or {})}.get(rel) or {}
    return orig == want and rel_copy.get("sha256") == got


def check1(R):
    P, fails, vals = R.proto, [], {}
    for p in R.missing:
        fails.append(dict(where=p, problem="records file missing"))
    hashes_ok = {}
    for t in R.tasks:
        tp, bk = R.tcfg(t), R.bank(t)
        files = [(bk["path"], bk["sha256"]), (bk["summary_path"], bk["summary_sha256"])] + R.cal_files(t)
        hashes_ok[t] = False  # True only if the bank .pkl itself matches (check 2 loads it)
        for rel, want in files:
            p = os.path.join(R.dir, rel)
            if not os.path.exists(p):
                fails.append(dict(where=rel, problem="file missing"))
                continue
            got = sha256(p)
            if got != want and _release_hash_ok(rel, want, got):
                vals.setdefault("release_copy_hashes_used", []).append(rel)
                if rel == bk["path"]:
                    hashes_ok[t] = True
            elif got != want:
                fails.append(dict(where=rel, problem="SHA-256 mismatch", expected=want, found=got))
            elif rel == bk["path"]:
                hashes_ok[t] = True
        cal_rel = R.cal_files(t)[1][0] if "predicate" in tp else tp["calibration"]["summary_path"]
        cal_p = os.path.join(R.dir, cal_rel)
        if os.path.exists(cal_p):
            cj = json.load(open(cal_p))
            cal_thr = cj.get("threshold_m", cj.get("threshold"))
            if cal_thr != R.threshold(t):
                fails.append(dict(where=cal_rel, problem="calibration threshold != protocol",
                                  expected=R.threshold(t), found=cal_thr))
        if t not in R.records:
            continue
        perts = R.perts(t)
        n_checked = 0
        for r in R.records[t]:
            n_checked += 1
            w = loc(r, R.paths[t])
            checks_ = [("threshold", R.threshold(t)), ("H_env", R.H(t))] + ([("tol", tp["success_tol"])] if "success_tol" in tp else [])
            for field, want in checks_:
                if r.get(field) != want:
                    fails.append(dict(where=w, field=field, expected=want, found=r.get(field)))
            sid = setting_of(r)
            if sid not in R.settings(t):
                fails.append(dict(where=w, field="setting_id", problem="setting not declared for this run", found=sid))
            else:
                want_spec = perts[sid]["spec"]
                got_spec = r.get("perturbation_spec", {})
                if got_spec != want_spec:
                    fails.append(dict(where=w, field="perturbation_spec", expected=want_spec, found=got_spec))
        # coverage: settings x bank states, exactly once each
        s0_ids = R.s0_ids(t)
        K = int(R.spec.get("seeds_per_start", 1))
        seen = {}
        for r in R.records[t]:
            seen.setdefault((setting_of(r), r["s0_id"], r.get("seed_index", 0)), []).append(r)
        for sid in R.settings(t):
            for s0 in s0_ids:
                for k in range(K):
                    n = len(seen.get((sid, s0, k), []))
                    if n != 1:
                        fails.append(dict(where=os.path.basename(R.paths[t]), problem="unit coverage",
                                          unit=f"{s0}__k{k}__{sid}" if K > 1 else f"{s0}__{sid}", count=n))
        extra = [k for k in seen if k[1] not in s0_ids or not (0 <= k[2] < K)]
        for k in extra:
            fails.append(dict(where=os.path.basename(R.paths[t]), problem="unit not in frozen bank x seeds", unit=k))
        vals[t] = dict(units_checked=n_checked, expected_units=len(R.settings(t)) * len(s0_ids) * K, seeds_per_start=K)
    R.hashes_ok = hashes_ok
    return result(1, "threshold fixedness", FAIL if fails else PASS, vals, fails)


def check2(R):
    P = R.proto
    cfg = P["verify"]["restore_spot_check"]
    atol = float(cfg["atol"])
    mod_name, fn_name = R.spec.get("env_adapter", P["verify"]["env_adapter"]).split(":")
    adapter = getattr(importlib.import_module(mod_name), fn_name)
    rs = np.random.RandomState(cfg["seed"])
    fails, vals, notes = [], {}, []
    for t in R.tasks:
        bk = R.bank(t)
        if t not in R.records:
            continue
        if not R.hashes_ok.get(t, False):
            fails.append(dict(where=bk["path"], problem="bank SHA-256 does not match protocol; spot check not run"))
            continue
        s0_ids = R.s0_ids(t)
        picked = [s0_ids[i] for i in sorted(rs.choice(len(s0_ids), size=cfg["n_states"], replace=False))]
        rows = []
        for r in R.records[t]:
            if r["s0_id"] not in picked:
                continue
            w = loc(r, R.paths[t])
            acts = np.load(os.path.join(R.dir, r["action_file"]))["env"]
            if acts.shape[0] != R.H(t):
                fails.append(dict(where=w, problem="action sequence length != H_env", found=int(acts.shape[0])))
                continue
            spec = R.perts(t).get(setting_of(r), {}).get("spec", {})
            out = adapter(t, os.path.join(R.dir, bk["path"]), r["s0_id"], acts, spec)
            A, B = out["states_a"], out["states_b"]
            row = dict(unit=r["unit_id"], two_restores_stepwise_maxdiff=float(np.max(np.abs(A - B))))
            if r.get("trajectory_file"):
                tr = np.load(os.path.join(R.dir, r["trajectory_file"]))["replay1"]
                row["mode"] = "stepwise vs recorded trajectory"
                row["vs_record_stepwise_maxdiff"] = float(np.max(np.abs(A - tr))) if tr.shape == A.shape else float("inf")
            elif r.get("real_final_state_full") is not None:
                row["mode"] = "final state (full precision) vs record"
                row["vs_record_final_maxdiff"] = float(np.max(np.abs(A[-1] - np.asarray(r["real_final_state_full"]))))
            else:
                row["mode"] = "final state (recorded rounded to 1e-6) vs record"
                row["vs_record_final_maxdiff"] = 0.0 if np.round(A[-1], 6).tolist() == r["real_final_state"] else float(
                    np.max(np.abs(np.round(A[-1], 6) - np.asarray(r["real_final_state"]))))
            row["final_dist_diff"] = abs(out["dist"] - r["real_final_dist"])
            row["verdict_match"] = bool(out["verdict"]) == bool(r["real_verdict"])
            bad = [k for k in ("two_restores_stepwise_maxdiff", "vs_record_stepwise_maxdiff", "vs_record_final_maxdiff",
                               "final_dist_diff") if k in row and row[k] > atol] + ([] if row["verdict_match"] else ["verdict"])
            if bad:
                fails.append(dict(where=w, problem="replay differs from record", fields=bad, **row))
            rows.append(row)
        vals[t] = dict(states=picked, units_replayed=len(rows),
                       max_diff=max([max(v for k, v in x.items() if k.endswith(("maxdiff", "_diff"))) for x in rows] or [0.0]),
                       modes=sorted({x["mode"] for x in rows}))
        if any(x["mode"] != "stepwise vs recorded trajectory" for x in rows):
            notes.append(f"{t}: records carry no per-step trajectory; compared the final state / final distance / "
                         f"verdict with the record, and every step between two independent restores (fresh vs dirtied env)")
    return result(2, "restore exactness (spot check)", FAIL if fails else PASS, vals, fails, notes)


def check3(R):
    fails, vals, notes = [], {}, []
    for t in (R.tasks if R.section else R.proto["tasks"]):
        tp = R.tcfg(t)
        if "predicate" in tp:   # frame predicate: TPR / TNR gate on the declared evaluations (all three numbers reported)
            pr = tp["predicate"]; g = pr["gate"]
            cj = json.load(open(os.path.join(R.dir, pr["calibration_path"])))
            vals[t] = {ev: {k: cj[ev][k] for k in ("agreement", "tpr", "tnr")} for ev in g["evaluations"]}
            for ev in g["evaluations"]:
                if cj[ev]["tpr"] < g["min_tpr"] or cj[ev]["tnr"] < g["min_tnr"]:
                    fails.append(dict(where=pr["calibration_path"], evaluation=ev, problem="TPR/TNR below gate",
                                      tpr=cj[ev]["tpr"], tnr=cj[ev]["tnr"], min=[g["min_tpr"], g["min_tnr"]]))
                # 2026-10-15 (LIBERO recalibration ruling): optional failure-terminal TNR floor per evaluation
                if g.get("min_fail_terminal_tnr") is not None:
                    ft = (cj[ev].get("fail_terminal") or {}).get("tnr")
                    vals[t][ev]["fail_terminal_tnr"] = ft
                    if ft is None or ft < g["min_fail_terminal_tnr"]:
                        fails.append(dict(where=pr["calibration_path"], evaluation=ev,
                                          problem="failure-terminal TNR below gate (or missing)", fail_terminal_tnr=ft,
                                          min=g["min_fail_terminal_tnr"]))
            if "monitor_generated_drift" in pr:
                notes.append(f"{t}: generated-frame drift monitor (not a gate): {pr['monitor_generated_drift']}")
            continue
        p = os.path.join(R.dir, tp["calibration"]["summary_path"])
        floor = tp.get("holdout_agreement_floor")
        if not os.path.exists(p):
            fails.append(dict(where=tp["calibration"]["summary_path"], problem="calibration artifact missing"))
            continue
        hold = (json.load(open(p)).get("holdout") or {}).get("agreement")
        vals[t] = dict(holdout_agreement=hold, floor=floor)
        if hold is None or floor is None:
            fails.append(dict(where=tp["calibration"]["summary_path"], problem="holdout agreement or floor missing"))
        elif hold < floor:
            fails.append(dict(where=tp["calibration"]["summary_path"], problem="holdout agreement below floor",
                              found=hold, floor=floor))
    return result(3, "predicate calibration validity", FAIL if fails else PASS, vals, fails, notes)


def check4(R):
    fails, vals, notes = [], {}, []
    for t, recs in R.records.items():
        kinds = {"replay2_fields": 0, "legacy_boolean": 0}
        for r in recs:
            w = loc(r, R.paths[t])
            if isinstance(r.get("replay2"), dict):
                kinds["replay2_fields"] += 1
                r2 = r["replay2"]
                first_state = r.get("real_final_state_full")
                diffs = [k for k, a, b in (("real_verdict", r["real_verdict"], r2.get("real_verdict")),
                                            ("real_final_dist", r["real_final_dist"], r2.get("real_final_dist")),
                                            ("real_final_state", first_state, r2.get("real_final_state_full")))
                         if a != b]
                if diffs or r.get("replay_check_identical") is False:
                    fails.append(dict(where=w, problem="two truth replays differ", fields=diffs or ["replay_check_identical"]))
            elif "replay_check_identical" in r:
                kinds["legacy_boolean"] += 1
                if r["replay_check_identical"] is not True:
                    fails.append(dict(where=w, problem="two truth replays differ (replay_check_identical=false)"))
            else:
                fails.append(dict(where=w, problem="only one truth replay recorded (no second-replay evidence)"))
        vals[t] = kinds
        if kinds["legacy_boolean"]:
            notes.append(f"{t}: {kinds['legacy_boolean']} units carry only the legacy boolean replay_check_identical "
                         f"(computed with array_equal over the full H-step trajectory of both replays)")
    return result(4, "replay determinism", FAIL if fails else PASS, vals, fails, notes)


def implicit_note(R, cid):
    return [f"run {R.run}: task {t} has no perturbations entry in the protocol -> treated as unperturbed (implicit "
            f"'none' setting only); check {cid} handled as for an unperturbed run (SKIP = not applicable, not omitted)"
            for t in sorted(R.implicit_none)]


def check5(R):
    sham = {t: [s for s in R.settings(t) if R.perts(t)[s].get("sham")] for t in R.tasks}
    if not any(sham.values()):
        return result(5, "sham", SKIP, notes=["run has no sham setting"] + implicit_note(R, 5))
    if R.baseline is None:
        return result(5, "sham", FAIL, failures=[dict(problem="sham present but protocol names no baseline_run")])
    fails, vals = [], {}
    for t, sids in sham.items():
        for sid in sids:
            n = 0
            for r in R.records.get(t, []):
                if setting_of(r) != sid:
                    continue
                n += 1
                b = R.baseline.get(t, {}).get(ukey(r))
                w = loc(r, R.paths[t])
                if b is None:
                    fails.append(dict(where=w, problem="no baseline unit for this s0"))
                    continue
                diffs = [k for k in ("real_verdict", "real_final_dist", "real_final_state", "real_final_feat_dist")
                         if r.get(k) != b.get(k)]
                if r.get("real_final_state_full") is not None and b.get("real_final_state_full") is not None \
                        and r["real_final_state_full"] != b["real_final_state_full"]:
                    diffs.append("real_final_state_full")
                if diffs:
                    fails.append(dict(where=w, problem="sham != baseline", fields=diffs,
                                      sham={k: r.get(k) for k in diffs}, baseline={k: b.get(k) for k in diffs}))
            vals[f"{t}/{sid}"] = dict(units=n)
    return result(5, "sham", FAIL if fails else PASS, vals, fails)


def check6(R):
    fails, vals, notes = [], {}, []
    any_family = False
    for t in R.tasks:
        P = R.perts(t)
        fams = {}
        for sid in R.settings(t):
            if P[sid].get("family"):
                fams.setdefault(P[sid]["family"], {})[P[sid]["severity"]] = sid
        for fam, levels in fams.items():
            any_family = True
            by_level = {}
            if R.baseline and t in R.baseline:
                by_level[0] = R.baseline[t]
            for sev, sid in levels.items():
                by_level[sev] = {ukey(r): r for r in R.records.get(t, []) if setting_of(r) == sid}
            sevs = sorted(by_level)
            counts = [sum(bool(r["real_verdict"]) for r in by_level[s].values()) for s in sevs]
            ok = all(a >= b for a, b in zip(counts, counts[1:]))
            states = sorted(set.intersection(*[set(by_level[s]) for s in sevs]))
            exc = {klabel(s0): [round(by_level[s][s0]["real_final_dist"], 4) for s in sevs] for s0 in states
                   if not all(by_level[a][s0]["real_final_dist"] <= by_level[b][s0]["real_final_dist"]
                              for a, b in zip(sevs, sevs[1:]))}
            vals[f"{t}/{fam}"] = dict(severities=sevs, real_success_counts=counts, nonincreasing=ok,
                                      final_dist_exceptions=exc, n_exceptions=len(exc))
            if not ok:
                fails.append(dict(where=f"{t}/{fam}", problem="real success count increases with severity",
                                  severities=sevs, counts=counts))
    if not any_family:
        return result(6, "dose monotonicity", SKIP, notes=["run has no perturbation family"] + implicit_note(R, 6))
    notes.append("final-distance exceptions are listed for information only (not a FAIL condition)")
    if fails:
        notes.append("non-monotone family -> WARN (rule of 2026-10-13): not a gate condition; the analysis report must "
                     "state, per non-monotone family, genuine response vs suspected process problem")
    return result(6, "dose monotonicity", WARN if fails else PASS, vals, fails, notes)


def check8(R):
    P = R.proto
    mod_name, fn_name = R.spec.get("env_adapter", P["verify"]["env_adapter"]).split(":")
    adapter = None
    fails, vals, notes = [], {}, []
    for t in R.tasks:
        if t not in R.records:
            continue
        perts = R.perts(t)
        targets = [s for s in R.settings(t) if s != "none" and (perts[s].get("spec") or perts[s].get("sham")
                                                                 or perts[s].get("sham_hook"))]
        if not targets:
            continue
        if not R.hashes_ok.get(t, False):
            fails.append(dict(where=R.bank(t)["path"], problem="bank SHA-256 does not match protocol; assertion not run"))
            continue
        adapter = adapter or getattr(importlib.import_module(mod_name), fn_name)
        order = {s: i for i, s in enumerate(R.s0_ids(t))}
        bank_path = os.path.join(R.dir, R.bank(t)["path"])
        rows = {}
        for sid in targets:
            recs = sorted((r for r in R.records[t] if setting_of(r) == sid), key=lambda r: (order.get(r["s0_id"], 1e9), ukey(r)))
            if not recs:
                continue
            r = recs[0]
            acts = np.load(os.path.join(R.dir, r["action_file"]))["env"]
            spec = perts[sid].get("spec", {})
            on = adapter(t, bank_path, r["s0_id"], acts, spec)["states_a"]
            off = adapter(t, bank_path, r["s0_id"], acts, {})["states_a"]
            d = np.max(np.abs(on - off), axis=tuple(range(1, on.ndim))) if on.shape == off.shape else None
            # sham_hook (2026-10-15, LIBERO family A): the hook path with a zero perturbation must equal the untouched
            # path; it is asserted here (zero difference) instead of check 5, which needs a separate baseline run
            sham = bool(perts[sid].get("sham") or perts[sid].get("sham_hook"))
            row = dict(unit=r["unit_id"], spec=spec, sham=sham,
                       max_diff=(float(d.max()) if d is not None else None),
                       first_divergent_step=(int(np.argmax(d > 0)) if d is not None and d.max() > 0 else None))
            if d is None:
                fails.append(dict(where=f"{t}/{sid}", problem="perturbed and unperturbed replays have different shapes"))
            elif sham and d.max() > 0:
                fails.append(dict(where=f"{t}/{sid}", problem="sham setting changes the trajectory", **row))
            elif not sham and d.max() == 0:
                fails.append(dict(where=f"{t}/{sid}", problem="perturbation has no effect on the trajectory "
                                                               "(hook not applied?)", **row))
            rows[sid] = row
        vals[t] = rows
    if not vals and not fails:
        return result(8, "parameter effect", SKIP, notes=["run has no perturbed setting"])
    notes.append("same restored state and same recorded action sequence, perturbation on vs off; any stepwise state "
                 "difference passes (outcomes are not used); sham must give zero difference")
    return result(8, "parameter effect", FAIL if fails else PASS, vals, fails, notes)


def check7(R):
    if R.spec.get("check7_rule") == "v2":
        return check7_v2(R)
    return check7_v1(R)


def check7_v1(R):
    fails, vals, notes = [], {}, []
    for t, recs in R.records.items():
        thr = R.threshold(t)
        # a probability-type predicate (LIBERO frame classifier) records feat_dist = 1 - max p and declares the success
        # boundary in distance units (feat_dist_threshold = 1 - p threshold; success iff feat_dist <= it). Without the
        # key, the original rule (feat_dist < threshold) applies unchanged.
        fthr = R.tcfg(t).get("predicate", {}).get("feat_dist_threshold")
        is_succ = (lambda fd: fd <= fthr) if fthr is not None else (lambda fd: fd < thr)
        per = {}
        for r in recs:
            if not (r["model_verdict"] and not r["real_verdict"]):
                continue
            sid = setting_of(r)
            d = per.setdefault(sid, dict(false_safe=0, predicate_error=0, prediction_error=0))
            d["false_safe"] += 1
            fd = r.get("real_final_feat_dist")
            if not isinstance(fd, (int, float)):
                fails.append(dict(where=loc(r, R.paths[t]), problem="false-safe unit lacks real-final feature distance"))
                continue
            if is_succ(fd):
                d["predicate_error"] += 1     # predicate also calls the REAL final observation a success
            else:
                d["prediction_error"] += 1    # predicate is right on the real final obs; the imagined endpoint was wrong
            if "predicate_on_real_final" in r and bool(r["predicate_on_real_final"]) != bool(is_succ(fd)):
                notes.append(f"{loc(r, R.paths[t])}: stored predicate_on_real_final disagrees with protocol threshold")
        tot = dict(false_safe=sum(v["false_safe"] for v in per.values()),
                   predicate_error=sum(v["predicate_error"] for v in per.values()),
                   prediction_error=sum(v["prediction_error"] for v in per.values()))
        vals[t] = dict(total=tot, by_setting=per)
    return result(7, "false-safe attribution", FAIL if fails else PASS, vals, fails, notes)



def check7_v2(R):
    """Spec revision #2 (2026-10-15, forward only; runs that set check7_rule: v2).
    Layer 1 (essential attribution): per false-safe unit, the imagined endpoint CONTENT vs the real endpoint content --
      prediction_error = the imagination shows a success the real run did not reach; predicate_error = the imagined
      content does not show success and the predicate misread it. Taken from the run's content_attribution file
      (manual frame review where needed). Unreviewed false-safe units -> WARN (listed), never silently defaulted.
    Layer 2 (separate): reliability of the predicate on the REAL endpoint distribution, over all units of the run:
      predicate on the real frames vs the simulator verdict (reach: any frame 1..H; final: frame H). The v1 rule's
      dichotomy is kept as legacy_rule_split for comparability with citations of earlier check-7 numbers."""
    fails, vals, notes = [], {}, []
    ca_path = R.spec.get("content_attribution")
    ca = {}
    if ca_path:
        p = os.path.join(R.dir, ca_path)
        if not os.path.exists(p):
            fails.append(dict(where=ca_path, problem="content_attribution file missing"))
        else:
            if R.spec.get("content_attribution_sha256") and sha256(p) != R.spec["content_attribution_sha256"]:
                fails.append(dict(where=ca_path, problem="content_attribution SHA-256 mismatch"))
            ca = {e["unit_id"]: e for e in json.load(open(p))}
    unreviewed = []
    for t, recs in R.records.items():
        thr = R.threshold(t)
        fthr = R.tcfg(t).get("predicate", {}).get("feat_dist_threshold")
        is_succ = (lambda fd: fd <= fthr) if fthr is not None else (lambda fd: fd < thr)
        l1 = dict(false_safe=0, prediction_error=0, predicate_error=0, unreviewed=0)
        legacy = dict(predicate_error=0, prediction_error=0)
        for r in recs:
            if not (r["model_verdict"] and not r["real_verdict"]):
                continue
            l1["false_safe"] += 1
            e = ca.get(r["unit_id"])
            if e is None or e.get("layer1") not in ("prediction_error", "predicate_error"):
                l1["unreviewed"] += 1; unreviewed.append(r["unit_id"])
            else:
                l1[e["layer1"]] += 1
            fd = r.get("real_final_feat_dist")
            if not isinstance(fd, (int, float)):
                fails.append(dict(where=loc(r, R.paths[t]), problem="false-safe unit lacks real-final feature distance"))
                continue
            legacy["predicate_error" if is_succ(fd) else "prediction_error"] += 1
        # layer 2: predicate on real frames vs simulator, all units
        l2 = {}
        for crit, pk, rk in (("reach", "predicate_on_real_final", "real_verdict"),
                             ("final", "predicate_on_real_final_step", "real_verdict_final")):
            rows = [(bool(r[pk]), bool(r[rk])) for r in recs if pk in r]
            if not rows:
                continue
            pos = [p_ for p_, y in rows if y]; neg = [p_ for p_, y in rows if not y]
            l2[crit] = dict(n=len(rows), real_pos=len(pos), real_neg=len(neg),
                            tpr=(sum(pos) / len(pos)) if pos else None,
                            tnr=(sum(not x for x in neg) / len(neg)) if neg else None,
                            agreement=sum(p_ == y for p_, y in rows) / len(rows))
        vals[t] = dict(layer1_content=l1, layer2_predicate_on_real=l2, legacy_rule_split=legacy)
    if unreviewed:
        fails.append(dict(problem="false-safe units without a layer-1 content attribution", units=unreviewed))
    notes.append("layer 1 = imagined vs real endpoint content (essential attribution); layer 2 = predicate reliability "
                 "on the real endpoint distribution (separate); legacy_rule_split = the pre-2026-10-15 rule, for "
                 "comparability only (spec revision #2)")
    hard = [f for f in fails if f.get("problem") != "false-safe units without a layer-1 content attribution"]
    status = FAIL if hard else (WARN if fails else PASS)
    return result(7, "false-safe attribution (two-layer, rev. #2)", status, vals, fails, notes)


# ------------------------------------------------------------------------------------------------ main
def check9(R):
    """closed-loop reference column (runs whose records carry closed_loop.deterministic_rerun_identical).
    Ruling 2026-10-24 (rule clarification) + revision 2026-10-25: bit-exact determinism is the hard stop of the PAIRED
    track only (imagination, truth replays; checks 1-8). For the model-in-the-loop reference column the only FAIL
    condition is a different SUCCESS VERDICT between the two runs. The first-success-step difference has no hard
    tolerance (it is a continuous quantity contaminated by amplified GPU noise; the earlier +-10 step tolerance is
    superseded): it is recorded for every unit and its distribution (median, max) is reported as the closed-loop
    measurement noise. Bit-level differences are exemptions. Units recorded before the runner kept the second run carry
    closed_loop.exemption (reruns); without it they are WARN (pending)."""
    import statistics
    fails, exempt, pending, diffs, n, any_cl = [], [], [], [], 0, False
    for t in R.tasks:
        for r in R.records.get(t, []):
            cl = r.get("closed_loop")
            if not isinstance(cl, dict) or "deterministic_rerun_identical" not in cl:
                continue
            any_cl = True; n += 1
            f1 = cl.get("first_success_step")
            if cl["deterministic_rerun_identical"]:
                if f1 is not None:
                    diffs.append(0)
                continue
            if "rerun" in cl:
                others = [cl["rerun"]]
                info = dict(unit=r["unit_id"], first_divergence_step=cl.get("first_divergence_step"))
            elif "exemption" in cl:
                others = cl["exemption"].get("reruns", [])
                info = dict(unit=r["unit_id"], exemption=cl["exemption"].get("id"))
            else:
                pending.append(dict(unit=r["unit_id"], problem="bit-level mismatch, second run not stored, no rerun yet"))
                continue
            ok = all(bool(o["success"]) == bool(cl["success"]) for o in others)
            d = [abs(f1 - o["first_success_step"]) for o in others if f1 is not None and o.get("first_success_step") is not None]
            diffs += d
            info["first_success_steps"] = [f1] + [o.get("first_success_step") for o in others]
            if ok:
                exempt.append(info)
            else:
                fails.append(dict(info, where=t, problem="closed-loop success verdict differs between runs"))
    if not any_cl:
        return result(9, "closed-loop success-verdict consistency", SKIP, notes=["no closed-loop rerun fields in the records"])
    noise = dict(units_with_steps=len(diffs), median=statistics.median(diffs) if diffs else None, max=max(diffs) if diffs else None,
                 nonzero=sum(1 for x in diffs if x), mean=round(statistics.mean(diffs), 3) if diffs else None)
    vals = dict(units=n, bit_identical=n - len(exempt) - len(fails) - len(pending), exempt_bit_level=exempt,
                success_verdict_mismatch=len(fails), pending=len(pending), first_success_step_abs_diff=noise)
    notes = ["reference column only; FAIL = success verdict differs (revision 2026-10-25); step differences are measurement "
             "noise (first_success_step_abs_diff), the closed-loop time metric uses the FIRST run"]
    if fails:
        return result(9, "closed-loop success-verdict consistency", FAIL, vals, fails, notes)
    return result(9, "closed-loop success-verdict consistency", WARN if pending else PASS, vals, pending, notes)


def check10(R):
    """archive minimum (protocol.yaml min_artifacts, 2026-11-13, forward only)."""
    rows = [r for t in R.tasks for r in R.records.get(t, []) if r.get("min_artifacts")]
    if not rows:
        return result(10, "archive minimum (min_artifacts)", SKIP, notes=["no rows carry min_artifacts (data before 2026-11-13 is not retro-checked)"])
    _wmvp.bootstrap()
    import min_artifacts as MA
    res = MA.verify_rows(rows, sample=int(R.proto.get("min_artifacts", {}).get("verify_sample_per_cell", 5)))
    bad = [dict(cell=k, **v) for k, v in res.items() if v["status"] != "pass"]
    vals = dict(cells=len(res), cells_failing=len(bad), units_checked=sum(v["checked"] for v in res.values()))
    notes = ["a failing cell must not be marked complete (make_snapshot reads out/artifact_verify/*.json)"]
    return result(10, "archive minimum (min_artifacts)", FAIL if bad else PASS, vals, bad, notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--run", default=None, help="run name from protocol.runs (auto if only one run's records exist)")
    ap.add_argument("--report", default=None, help="default: <run-dir>/verify_report_<run>.json")
    a = ap.parse_args()
    proto_path = os.path.abspath(a.protocol)
    proto = yaml.safe_load(open(proto_path))
    sys.path.insert(0, os.path.dirname(proto_path))  # adapter module lives next to the protocol file
    os.environ.setdefault("WMV_PROTOCOL", proto_path)

    run = a.run
    if run is None:
        present = [n for n, s in proto["runs"].items()
                   if s.get("records") and all(os.path.exists(os.path.join(a.run_dir, s["records"].format(task=t)))
                          for t in s.get("tasks", proto["tasks"]))]
        if len(present) != 1:
            sys.exit(f"cannot pick a run automatically (found {present}); pass --run")
        run = present[0]
    R = Run(a.run_dir, proto, run)

    checks = [check1(R)]
    for fn in (check2, check3, check4, check5, check6, check7, check8, check9, check10):
        try:
            checks.append(fn(R))
        except Exception as ex:  # a crashing check is a failing check
            cid = int(fn.__name__[len("check"):])
            checks.append(result(cid, fn.__name__, FAIL, failures=[dict(problem=f"check raised {type(ex).__name__}: {ex}")]))

    overall = FAIL if any(c["status"] == FAIL for c in checks) else PASS
    warnings = [dict(check=c["id"], name=c["name"], **f) for c in checks if c["status"] == WARN for f in c["failures"]]
    report = dict(run=run, run_dir=os.path.abspath(a.run_dir), protocol=proto_path,
                  protocol_sha256=sha256(proto_path), overall=overall, warnings=warnings,
                  implicit_unperturbed_tasks=sorted(R.implicit_none), checks=checks)
    out = a.report or os.path.join(a.run_dir, f"verify_report_{run}.json")
    json.dump(report, open(out, "w"), indent=1, default=float)

    print(f"verify_run  run={run}  dir={a.run_dir}")
    for w in warnings:
        print(f"  WARN (check {w['check']}, {w['name']}): {json.dumps({k: v for k, v in w.items() if k not in ('check', 'name')}, default=float)[:300]}")
    for c in checks:
        brief = json.dumps(c["values"], default=float)
        print(f"  [{c['status']}] {c['id']}. {c['name']}: {brief[:400]}")
        for f in c["failures"][:10]:
            print(f"        FAIL -> {json.dumps(f, default=float)[:400]}")
        if len(c["failures"]) > 10:
            print(f"        ... {len(c['failures']) - 10} more failures in the report")
        for n in c["notes"][:5]:
            print(f"        note: {n}")
    if R.implicit_none:
        print(f"  NOTE: no perturbations entry for {sorted(R.implicit_none)} -> evaluated as unperturbed (implicit 'none')")
    print(f"OVERALL {overall}   report: {out}")
    sys.exit(0 if overall == PASS else 1)


if __name__ == "__main__":
    main()
