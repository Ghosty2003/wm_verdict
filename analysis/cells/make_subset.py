"""LingBot-VA family O review subset (2026-10-31), after the agent first pass (ruling 2026-10-29: disagreement <= 15 % AND
kappa >= 0.6 on a stratified blind subset of 30, per task; kappa undefined -> full human review).

  t3  stratified blind 30: proportional to the agent's labels (largest remainder), levels balanced round-robin inside each
      label, so the agreement on the subset estimates the agreement on all 50 without reweighting.
  t5  all 38: the agent used a single label on t5, so kappa is undefined and the rule sends t5 to full human review.

The sheet carries no agent fields. The previous sheet (random 15 + 15, never filled) is kept in .agent_firstpass/.

  python make_subset.py   -> jasmine/audit_sheet.json, jasmine/.agent_firstpass/subset_design.json
"""
import os
import json
import numpy as np

D = os.path.dirname(os.path.abspath(__file__)); J = f"{D}/jasmine"; A = f"{J}/.agent_firstpass"
SEED, N3 = 20261031, 30

old = json.load(open(f"{J}/audit_sheet.json"))
assert not any(x.get("jasmine_label") or x.get("jasmine_note") for x in old), "the reviewer has started: do not replace the sheet"
agent = json.load(open(f"{A}/agent_labels.json")); units = {u["unit"]: u for u in json.load(open(f"{D}/units.json"))}
rs = np.random.RandomState(SEED)

t3, t5 = [a for a in agent if a["task"] == "t3"], [a for a in agent if a["task"] == "t5"]
assert len({a["agent_label"] for a in t5}) == 1, "t5 has more than one agent label: draw a stratified subset instead"
strata = {}
for a in t3:
    strata.setdefault(a["agent_label"], []).append(a)
raw = {k: len(v) * N3 / len(t3) for k, v in strata.items()}; quota = {k: int(x) for k, x in raw.items()}
for k in sorted(raw, key=lambda k: (round(quota[k] - raw[k], 9), len(strata[k])))[:N3 - sum(quota.values())]:   # ties: the smaller label
    quota[k] += 1

pick = []
for k, v in sorted(strata.items()):
    by = {}
    for a in v:
        by.setdefault(a["level"], []).append(a)
    for L in by:
        by[L] = [by[L][i] for i in rs.permutation(len(by[L]))]
    levels = [sorted(by)[i] for i in rs.permutation(len(by))]; got = []
    while len(got) < quota[k]:
        for L in levels:
            if by[L] and len(got) < quota[k]:
                got.append(by[L].pop())
    pick += got


def row(a):
    u = units[a["unit"]]
    return dict(unit=u["unit"], task=u["task"], level=u["level"], card=u["card"], jasmine_label=None, jasmine_note=None)


sheet = [row(t5[i]) for i in rs.permutation(len(t5))] + [row(pick[i]) for i in rs.permutation(len(pick))]
if not os.path.exists(f"{A}/audit_sheet_random30_20261028.json"):
    json.dump(old, open(f"{A}/audit_sheet_random30_20261028.json", "w"), indent=1)
json.dump(sheet, open(f"{J}/audit_sheet.json", "w"), indent=1)
design = dict(seed=SEED, t3=dict(n=len(pick), population={k: len(v) for k, v in strata.items()}, quota=quota,
                                 levels={L: sum(a["level"] == L for a in pick) for L in sorted({a["level"] for a in pick})}),
              t5=dict(n=len(t5), design="all units: single agent label, kappa undefined, full human review by rule"))
json.dump(design, open(f"{A}/subset_design.json", "w"), indent=1)
print(design)
