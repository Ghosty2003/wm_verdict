"""Write the family-A (object) protocol entries ADD-ONLY: banks.famA_obj_t5/t3, libero_lingbot_v3.perturbations,
runs.lingbot_famA_obj. Specs are taken from the frozen bank (must equal what lingbot_famA.level_spec writes).

  python familyA_protocol.py
"""
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
BANK = "libero/out/bank/familyA/familyA_anchors.json"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main():
    proto_p = f"{ROOT}/protocol.yaml"
    txt = open(proto_p).read()
    P = yaml.safe_load(txt)
    assert "lingbot_famA_obj" not in P["runs"] and "perturbations" not in P["libero_lingbot_v3"]
    bank = json.load(open(f"{ROOT}/{BANK}"))
    ver = json.load(open(f"{ROOT}/libero/out/bank/familyA/familyA_verify.json"))
    assert ver["all_pass"], "family A bank verification did not pass"
    lines_banks, lines_pert = [], ["  perturbations:                       # 2026-10-15 family A, object displacement (batch 1); native hooks"]
    for tid in (5, 3):
        ent = f"libero/out/bank/familyA/familyA_entries_t{tid}.json"
        lines_banks.append(f"  famA_obj_t{tid}:                      # 2026-10-15 LIBERO family A native anchors (5 base seeds x 5 levels)\n"
                           f"    path: {BANK}\n    sha256: {sha(BANK)}\n"
                           f"    summary_path: {ent}\n    summary_sha256: {sha(ent)}\n"
                           f"    verify: {{path: libero/out/bank/familyA/familyA_verify.json, all_pass: true, tolerance: 0.0}}")
        lines_pert.append(f"    t{tid}:")
        for sev, L in enumerate(("A0", "A1", "A2", "A3", "A4")):
            a = next(x for x in bank["anchors"] if x["task_id"] == tid and x["level"] == L)
            spec = level_spec(a)
            cover = {"A0": "known (native region, sham)", "A1": "known (native region boundary)",
                     "A2": "out-of-fine-tuning (pre-training coverage not excluded)",
                     "A3": "out-of-fine-tuning (pre-training coverage not excluded)",
                     "A4": "out-of-fine-tuning (pre-training coverage not excluded)"}[L]
            extra = ", sham_hook: true" if L == "A0" else ""
            lines_pert.append(f"      {L}: {{family: A_obj, severity: {sev}{extra}, coverage: \"{cover}\", hook: \"{a['hook']}\",\n"
                              f"           spec: {json.dumps(spec)}}}")
    run = ("  lingbot_famA_obj:            # 2026-10-15 LIBERO family A (object displacement, batch 1), LingBot-VA; A0 = family baseline\n"
           "    records: \"libero/out/runs/lingbot_famA/records_{task}.jsonl\"\n"
           "    settings: [A0, A1, A2, A3, A4]\n"
           "    baseline_run: null           # A0 is the family baseline; the sham is asserted by check 8 (sham_hook)\n"
           "    task_section: libero_lingbot_v3\n"
           "    tasks: [t5, t3]\n"
           "    banks: {t5: famA_obj_t5, t3: famA_obj_t3}\n"
           "    seeds_per_start: 10          # imagination seed = RandomState(20261031).randint(0, 2**31-1, (10 base anchors, 10))[base, k], shared by levels\n"
           "    env_adapter: \"libero_env_adapter:replay_unit\"\n"
           "    check7_rule: v2\n"
           "    directions: {t5: \"+y (book; away from caddy and mug)\", t3: \"+x (bowl; away from the robot)\"}\n"
           "    a4_note: \"t3 A4 = 7 cm beyond the boundary (downgraded from 10 cm: at the Panda reach limit); t5 A4 = 10 cm\"\n"
           "    table_note: \"family-A anchors are native samples (init_states=False), not the frozen main-bank anchors; compare within the family against A0\"\n")
    i = txt.index("\nverify:")
    txt = txt[:i + 1] + run + "\n" + txt[i + 1:]
    j = txt.index("\nbanks:\n") + len("\nbanks:\n")
    txt = txt[:j] + "\n".join(lines_banks) + "\n" + txt[j:]
    k = txt.index("\nlibero_lingbot_v3:\n") + len("\nlibero_lingbot_v3:\n")
    txt = txt[:k] + "\n".join(lines_pert) + "\n" + txt[k:]
    yaml.safe_load(txt)
    open(proto_p, "w").write(txt)
    print("protocol updated")


if __name__ == "__main__":
    main()
