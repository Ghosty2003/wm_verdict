"""Write the family-P (pose, batch 2) protocol entries ADD-ONLY: banks.famP_pose_t5/t3, the P levels inside
libero_lingbot_v3.perturbations.<task>, runs.lingbot_famP_pose.   python familyP_protocol.py"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import hashlib
import yaml

_wmvp.bootstrap()
from lingbot_famA import level_spec

ROOT = _wmvp.P("$WMV_ROOT")
BANK = "libero/out/bank/familyP/familyP_anchors.json"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main():
    proto_p = f"{ROOT}/protocol.yaml"; txt = open(proto_p).read(); P = yaml.safe_load(txt)
    assert "lingbot_famP_pose" not in P["runs"]
    ver = json.load(open(f"{ROOT}/libero/out/bank/familyP/familyP_verify.json"))
    assert ver["all_pass"] and not ver["coverage_stop"], "family P bank verification / coverage did not pass"
    bank = json.load(open(f"{ROOT}/{BANK}"))
    banks = []
    for tid in (5, 3):
        ent = f"libero/out/bank/familyP/familyP_entries_t{tid}.json"
        banks.append(f"  famP_pose_t{tid}:                     # 2026-10-15 LIBERO family P native anchors (5 base seeds x P0..P4)\n"
                     f"    path: {BANK}\n    sha256: {sha(BANK)}\n    summary_path: {ent}\n    summary_sha256: {sha(ent)}\n"
                     f"    verify: {{path: libero/out/bank/familyP/familyP_verify.json, all_pass: true, tolerance: 0.0, coverage_all_inside: true}}")
    for tid in (5, 3):
        lines = []
        for sev, L in enumerate(("P0", "P1", "P2", "P3", "P4")):
            a = next(x for x in bank["anchors"] if x["task_id"] == tid and x["level"] == L)
            extra = ", sham_hook: true, sham_rule: \"self-consistency (P0 twice, same seed, bit-identical); P0 vs the native path differs by a known RNG-order artifact\"" if L == "P0" else ""
            lines.append(f"      {L}: {{family: P_pose, severity: {sev}{extra}, coverage: \"{a['coverage']}\", hook: \"{a['hook']}\",\n"
                         f"           spec: {json.dumps(level_spec(a))}}}")
        key = f"\n    t{tid}:\n"
        i = txt.index("\n  perturbations:", txt.index("\nlibero_lingbot_v3:\n"))   # the v3 section's block only
        j = txt.index(key, i) + len(key)
        txt = txt[:j] + "\n".join(lines) + "\n" + txt[j:]
    run = ("  lingbot_famP_pose:           # 2026-10-15 LIBERO family P (robot initial pose, batch 2), LingBot-VA; P0 = family baseline\n"
           "    records: \"libero/out/runs/lingbot_famP/records_{task}.jsonl\"\n"
           "    settings: [P0, P1, P2, P3, P4]\n"
           "    baseline_run: null\n"
           "    task_section: libero_lingbot_v3\n"
           "    tasks: [t5, t3]\n"
           "    banks: {t5: famP_pose_t5, t3: famP_pose_t3}\n"
           "    seeds_per_start: 10          # imagination seed = RandomState(20261033).randint(0, 2**31-1, (10 base anchors, 10))[base, k], shared by levels\n"
           "    env_adapter: \"libero_env_adapter:replay_unit\"\n"
           "    check7_rule: v2\n"
           "    table_note: \"family-P anchors are native samples (init_states=False, uniform initialization_noise); not the main-bank or family-A anchors; compare within the family against P0\"\n")
    i = txt.index("\nverify:"); txt = txt[:i + 1] + run + "\n" + txt[i + 1:]
    j = txt.index("\nbanks:\n") + len("\nbanks:\n"); txt = txt[:j] + "\n".join(banks) + "\n" + txt[j:]
    yaml.safe_load(txt); open(proto_p, "w").write(txt); print("protocol updated (family P)")


if __name__ == "__main__":
    main()
