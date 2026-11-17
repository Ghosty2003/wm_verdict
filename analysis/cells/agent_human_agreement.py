"""Agent judge vs human judge on a blind subset (2026-10-28): agreement and Cohen's kappa per labelled field.

Each package keeps the agent's first-pass labels in <dir>/.agent_firstpass/agent_labels.json (hidden from the
reviewer) and the reviewer's labels in <dir>/audit_sheet.json (blind: no agent fields). Per field:
  - agreement = share of items with the same label, Clopper-Pearson 95 %
  - Cohen's kappa (unweighted); for ordinal fields also linear-weighted kappa; 95 % CI by bootstrap over items
    (2000 resamples, seed 0). Kappa is reported as undefined when either side used a single label on the subset
    (no variance: agreement is then the only meaningful number)
  - confusion matrix (agent rows x human columns)
  - pass rule (ruling 2026-10-29): disagreement <= 15 % AND Cohen's kappa (unweighted) >= 0.6, on a stratified
    blind subset of 30; otherwise full human review (or a revised agent pass and a new subset). Kappa undefined -> fail
  - control items (deck_key.json "answer"), when present: accuracy of each judge against the known answer of that package
Items the reviewer left empty are skipped and counted.

  python agent_human_agreement.py [t5_contentcheck|t3_attribution|<dir> ...]   -> <dir>/agreement.json, printed table
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
from scipy.stats import beta

ROOT = _wmvp.P("$WMV_ROOT/libero")
PACKAGES = {
    "t5_contentcheck": dict(dir=f"{ROOT}/openwam/predicate/t5/contentcheck/jasmine", key="id",
                            fields={"label": ["out", "uncertain", "in"]}),                      # ordinal
    "t3_attribution": dict(dir=f"{ROOT}/out/content_review/openwam_t3/jasmine", key="card",          # deck of 30 (10 controls)
                           fields={"label": ["not_success_but_triggered", "borderline", "looks_success"],   # ordinal
                                   "sublabel": None, "motion": None}),
    # one package per task: t5 and t3 use disjoint label sets, so a pooled kappa would count the task itself as agreement
    "lingbot_famO_t3": dict(dir=f"{ROOT}/out/content_review/lingbot_famO/jasmine", key="unit", task="t3",
                            fields={"label": ["not_success_but_triggered", "borderline", "looks_success"]}),  # ordinal
    "lingbot_famO_t5": dict(dir=f"{ROOT}/out/content_review/lingbot_famO/jasmine", key="unit", task="t5",
                            fields={"label": ["out", "uncertain", "in"]}),                      # ordinal
}
RULE = 0.15
KAPPA_MIN = 0.6


def cp(x, n):
    return [float(beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0, float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0]


def kappa(a, h, cats, weighted):
    k = len(cats); idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((k, k))
    for x, y in zip(a, h):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    if n == 0 or len(set(a)) < 2 or len(set(h)) < 2:
        return None
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / max(k - 1, 1) if weighted else (1 - np.eye(k))
    E = np.outer(M.sum(1), M.sum(0)) / n
    d = (W * E).sum()
    return None if d == 0 else float(1 - (W * M).sum() / d)


def boot(a, h, cats, weighted, B=2000):
    rng = np.random.RandomState(0); a, h = np.array(a), np.array(h); v = []
    for _ in range(B):
        i = rng.randint(0, len(a), len(a)); k = kappa(list(a[i]), list(h[i]), cats, weighted)
        if k is not None:
            v.append(k)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > B // 2 else None


def run(name, P):
    agent = {x[P["key"]]: x for x in json.load(open(f"{P['dir']}/.agent_firstpass/agent_labels.json"))}
    human = {x[P["key"]]: x for x in json.load(open(f"{P['dir']}/audit_sheet.json"))}
    agent = {k: x for k, x in agent.items() if k in human and (not P.get("task") or x["task"] == P["task"])}   # the blind subset of this task
    out = dict(package=name, rule=f"pass iff disagreement <= {RULE:.0%} and kappa >= {KAPPA_MIN}", fields={})
    for f, order in P["fields"].items():
        pairs = [(agent[k][f"agent_{f}"], human[k][f"jasmine_{f}"]) for k in agent if human.get(k, {}).get(f"jasmine_{f}")]
        skipped = len(agent) - len(pairs)
        if not pairs:
            out["fields"][f] = dict(n=0, skipped=skipped, note="not filled yet"); continue
        a, h = [p[0] for p in pairs], [p[1] for p in pairs]
        cats = order or sorted(set(a) | set(h)); cats += [c for c in sorted(set(a) | set(h)) if c not in cats]
        same = sum(x == y for x, y in pairs); n = len(pairs)
        r = dict(n=n, skipped=skipped, agreement=round(same / n, 3), agreement_cp95=[round(x, 3) for x in cp(same, n)],
                 disagreement=round(1 - same / n, 3), full_review_triggered=(1 - same / n) > RULE,
                 kappa=kappa(a, h, cats, False), kappa_boot95=boot(a, h, cats, False),
                 agent_label_counts={c: a.count(c) for c in cats}, human_label_counts={c: h.count(c) for c in cats},
                 confusion={c: {d: sum(1 for x, y in pairs if x == c and y == d) for d in cats} for c in cats})
        if order:
            r.update(kappa_linear_weighted=kappa(a, h, cats, True), kappa_linear_weighted_boot95=boot(a, h, cats, True))
        if r["kappa"] is None:
            r["kappa_note"] = "undefined: one side used a single label on this subset (no variance); report agreement only"
        r["pass"] = bool(not r["full_review_triggered"] and r["kappa"] is not None and r["kappa"] >= KAPPA_MIN)
        out["fields"][f] = r
    kp = f"{P['dir']}/.agent_firstpass/deck_key.json"
    if os.path.exists(kp):                                       # control items with a known answer
        key = {k["card"]: k for k in json.load(open(kp)) if k.get("answer")}
        acc = {}
        for who, src in (("agent", agent), ("human", human)):
            got = [(src[c].get(f"{who if who == 'agent' else 'jasmine'}_label"), k["answer"]) for c, k in key.items() if c in src]
            got = [(g, a_) for g, a_ in got if g]
            acc[who] = dict(n=len(got), correct=sum(g == a_ for g, a_ in got))
        out["controls"] = acc
    json.dump(out, open(f"{P['dir']}/agreement{'_' + P['task'] if P.get('task') else ''}.json", "w"), indent=1)
    return out


def main():
    names = sys.argv[1:] or list(PACKAGES)
    for nm in names:
        P = PACKAGES.get(nm) or dict(dir=nm, key="id", fields={"label": None})
        o = run(nm, P)
        for f, r in o["fields"].items():
            if r.get("n"):
                print(f"{nm}/{f}: n={r['n']} agreement {r['agreement']} {r['agreement_cp95']} kappa {r['kappa']} "
                      f"{r.get('kappa_boot95')} weighted {r.get('kappa_linear_weighted')} PASS={r['pass']}")
            else:
                print(f"{nm}/{f}: not filled yet ({r['skipped']} items)")


if __name__ == "__main__":
    main()
