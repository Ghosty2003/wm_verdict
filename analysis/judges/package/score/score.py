#!/usr/bin/env python3
"""Score a judge submission against the answer key (Python 3 standard library only).

  python3 score.py SUBMISSION.csv KEY.csv [--out report.md] [--json report.json] [--by-family]

SUBMISSION.csv  columns frame_id,label,confidence,note (header required; extra columns ignored)
                label in {looks_success, borderline, not_success, uncertain} (S/B/N/U letters are also accepted)
KEY.csv         _key/key.csv (distributed separately from the blind package)

Reported:
 1. Coverage: frames labelled per model x task x set (failure = real closed-loop failure, sanity = real success).
 2. Optimism (failure items only), per model x task x part (other / F family / B family = image corruption), plus the
    model x task total; --by-family adds model x task x family letter.
      main = looks_success / (judged - uncertain)        wide = (looks_success + borderline) / (judged - uncertain)
    Wilson 95% CI. "judged" = failure frames that have a label in the submission (missing frames are not counted).
    For B-family rows prefer the main rate (borderline in corrupted frames is often "can't see", see README).
 3. Judge sanity: on real-SUCCESS items the share judged looks_success (TPR, main) and looks_success+borderline (TPR, wide);
    on real-failure items the share NOT judged looks_success (= 1 - optimism; NOT a true TNR, because an imagined
    frame of a failed run may legitimately show success — that is exactly the optimism being measured).
 4. Agreement with every existing method in the key, on the frames both have labelled:
      agent (4-class)                    exact 4-class agreement + Cohen's kappa, and binary agreement + kappa
      agent / CNN / $COLLAB_COSMOS / human       binary: success vs not (submission: looks_success vs borderline/not_success;
                                         uncertain/unknown on either side is dropped)
    and, against the human labels, the judge's TPR/TNR taking the human label as ground truth.
"""
import argparse
import collections
import csv
import json
import math
import sys

LABELS = ("looks_success", "borderline", "not_success", "uncertain")
LETTER = {"S": "looks_success", "B": "borderline", "N": "not_success", "U": "uncertain"}
MODELS = ("lingbot", "openwam", "cosmos")
MNAME = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α", "cosmos": "Cosmos Policy"}
PARTS = ("other", "F", "B", "all")


def wilson(k, n):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def fmt(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}–{100 * hi:.0f}%)" + (" ⚠n<10" if n < 10 else "")


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) category pairs."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def sub_binary(lab):
    return None if lab == "uncertain" else ("success" if lab == "looks_success" else "failure")


# existing methods in the key -> binary mapping (None = drop)
def m_agent(v):
    return sub_binary(v) if v in LABELS else None


def m_cnn(v):
    return v if v in ("success", "failure") else None


def m_simple(v):      # $COLLAB_COSMOS / jasmine40 / rater2 / rater3 / F160 human
    v = (v or "").strip()
    if v in ("success",):
        return "success"
    if v in ("failure", "borderline"):     # human "borderline" = not counted as success (main convention)
        return "failure"
    return None                            # uncertain / unknown / success_dist_ambiguous / empty


METHODS = [  # (column, display name, mapping, is_human)
    ("agent_label", "agent 代判 (2026-11-14/15, uncalibrated)", m_agent, False),
    ("cnn_verdict", "CNN/predicate (LingBot+OpenWAM T0/T1: optimism_auto v7.1; Cosmos: CNN v1)", m_cnn, False),
    ("collab_cosmos_label", "$COLLAB_COSMOS initial review (Cosmos)", m_simple, False),
    ("f160_agent_read", "F160 agent re-read (OpenWAM T1 F160)", m_agent, False),
    ("jasmine40_task", "human: Jasmine, 40-frame set", m_simple, True),
    ("rater2_task", "human: rater 2, 40-frame set", m_simple, True),
    ("rater3_task", "blind rater 3, 40-frame set", m_simple, True),
    ("f160_jasmine_label", "human: Jasmine, OpenWAM T1 F160 (17)", m_simple, True),
]


def read_submission(path):
    sub, warn = {}, []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or "frame_id" not in rd.fieldnames or "label" not in rd.fieldnames:
            sys.exit("submission must have a header with at least frame_id,label")
        for i, r in enumerate(rd, 2):
            fid = (r.get("frame_id") or "").strip(); lab = (r.get("label") or "").strip()
            lab = LETTER.get(lab.upper(), lab.lower()) if lab else lab
            if not fid:
                continue
            if lab not in LABELS:
                warn.append(f"line {i}: frame {fid}: invalid label {r.get('label')!r} -> ignored"); continue
            if fid in sub:
                warn.append(f"line {i}: duplicate frame {fid} -> last row wins")
            sub[fid] = dict(label=lab, confidence=r.get("confidence", ""), note=r.get("note", ""))
    return sub, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission"); ap.add_argument("key")
    ap.add_argument("--out", help="write the markdown report here (default: stdout)")
    ap.add_argument("--json", help="also write machine-readable numbers here")
    ap.add_argument("--by-family", action="store_true", help="add model x task x family-letter optimism rows")
    a = ap.parse_args()

    key = {r["frame_id"]: r for r in csv.DictReader(open(a.key, newline="", encoding="utf-8"))}
    sub, warn = read_submission(a.submission)
    unknown = sorted(f for f in sub if f not in key)
    if unknown:
        warn.append(f"{len(unknown)} frame ids not in the key (ignored), e.g. {unknown[:5]}")
    L = []
    out = {"warnings": warn, "coverage": [], "optimism": [], "optimism_by_family": [], "sanity": [], "agreement": []}

    # 1. coverage
    L.append("# Judge submission report\n")
    L.append(f"submission: `{a.submission}`; frames in key: {len(key)}; labelled & valid: {sum(f in key for f in sub)}\n")
    if warn:
        L.append("**Warnings**\n"); L += [f"- {w}" for w in warn[:50]]; L.append("")
    L.append("## 1. Coverage (labelled / frames in package)\n")
    L.append("| model | task | failure items | sanity items |"); L.append("|---|---|---|---|")
    for m in MODELS:
        for T in ("T0", "T1", "T2", "T3", "T4"):
            f = [k for k, r in key.items() if r["model"] == m and r["task"] == T and r["set"] == "failure"]
            s = [k for k, r in key.items() if r["model"] == m and r["task"] == T and r["set"] == "sanity"]
            if not f and not s:
                continue
            cf, cs = sum(x in sub for x in f), sum(x in sub for x in s)
            L.append(f"| {MNAME[m]} | {T} | {cf}/{len(f)} | {cs}/{len(s)} |")
            out["coverage"].append(dict(model=m, task=T, failure_labelled=cf, failure_total=len(f), sanity_labelled=cs, sanity_total=len(s)))
    L.append("")

    # 2. optimism
    def opt_rows(group_fn, title, store):
        L.append(title)
        L.append("| model | task | group | real-failure frames | judged | uncertain | borderline | optimism main | optimism wide |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        G = collections.defaultdict(list)
        for fid, r in key.items():
            if r["set"] != "failure":
                continue
            for g in group_fn(r):
                G[(r["model"], r["task"], g)].append(fid)
        order = lambda k: (MODELS.index(k[0]), k[1], PARTS.index(k[2]) if k[2] in PARTS else 9, k[2])
        for k in sorted(G, key=order):
            fids = G[k]; labs = [sub[f]["label"] for f in fids if f in sub]
            c = collections.Counter(labs); valid = len(labs) - c["uncertain"]
            km, kw = c["looks_success"], c["looks_success"] + c["borderline"]
            L.append(f"| {MNAME[k[0]]} | {k[1]} | {k[2]} | {len(fids)} | {len(labs)} | {c['uncertain']} | {c['borderline']} | {fmt(km, valid)} | {fmt(kw, valid)} |")
            lo, hi = wilson(km, valid) if valid else (None, None); wlo, whi = wilson(kw, valid) if valid else (None, None)
            store.append(dict(model=k[0], task=k[1], group=k[2], n_frames=len(fids), n_judged=len(labs), uncertain=c["uncertain"],
                              borderline=c["borderline"], n_valid=valid, k_main=km, k_wide=kw,
                              main=km / valid if valid else None, main_ci=[lo, hi], wide=kw / valid if valid else None, wide_ci=[wlo, whi]))
        L.append("")

    opt_rows(lambda r: [r["part"], "all"], "## 2. Optimism on real-failure frames (part: other / F / B; all = total)\n", out["optimism"])
    if a.by_family:
        opt_rows(lambda r: [r["family"]], "## 2b. Optimism by family letter\n", out["optimism_by_family"])

    # 3. sanity
    L.append("## 3. Judge sanity (real outcome as reference)\n")
    L.append("| model | task | real-success frames judged | judged looks_success (TPR main) | + borderline (TPR wide) | uncertain | real-failure frames judged: NOT looks_success (1 − optimism, not a true TNR) |")
    L.append("|---|---|---|---|---|---|---|")
    for m in MODELS + ("ALL",):
        for T in ("T0", "T1", "T2", "T3", "T4", "ALL"):
            if (m == "ALL") != (T == "ALL"):
                continue
            sel = lambda r: (m == "ALL" or r["model"] == m) and (T == "ALL" or r["task"] == T)
            s = [sub[f]["label"] for f, r in key.items() if f in sub and r["set"] == "sanity" and sel(r)]
            f = [sub[x]["label"] for x, r in key.items() if x in sub and r["set"] == "failure" and sel(r)]
            if not s and not f:
                continue
            cs = collections.Counter(s); vs = len(s) - cs["uncertain"]; cf = collections.Counter(f); vf = len(f) - cf["uncertain"]
            L.append(f"| {MNAME.get(m, m)} | {T} | {len(s)} | {fmt(cs['looks_success'], vs)} | {fmt(cs['looks_success'] + cs['borderline'], vs)} | "
                     f"{cs['uncertain']} | {fmt(vf - cf['looks_success'], vf)} |")
            out["sanity"].append(dict(model=m, task=T, sanity_judged=len(s), sanity_valid=vs, tpr_main=cs["looks_success"] / vs if vs else None,
                                      tpr_wide=(cs["looks_success"] + cs["borderline"]) / vs if vs else None, failure_valid=vf,
                                      fail_not_success=(vf - cf["looks_success"]) / vf if vf else None))
    L.append("")

    # 4. agreement
    L.append("## 4. Agreement with existing methods (frames labelled by both)\n")
    L.append("Binary = success vs not (submission looks_success vs borderline/not_success; uncertain/unknown dropped on either side). "
             "For human rows, TPR/TNR treat the human binary label as ground truth.\n")
    L.append("| method | n (binary) | agreement | Cohen κ | judge TPR vs method | judge TNR vs method | 4-class n | 4-class agreement | 4-class κ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for col, name, mp, human in METHODS:
        pairs, p4 = [], []
        for fid, r in key.items():
            if fid not in sub or not r.get(col):
                continue
            mb, sb = mp(r[col]), sub_binary(sub[fid]["label"])
            if mb and sb:
                pairs.append((sb, mb))
            if col in ("agent_label", "f160_agent_read") and r[col] in LABELS:
                p4.append((sub[fid]["label"], r[col]))
        if not pairs and not p4:
            L.append(f"| {name} | 0 | — | — | — | — | — | — | — |"); continue
        n = len(pairs); ag = sum(x == y for x, y in pairs)
        pos = [x for x, y in pairs if y == "success"]; neg = [x for x, y in pairs if y == "failure"]
        tpr = sum(x == "success" for x in pos) / len(pos) if pos else None
        tnr = sum(x == "failure" for x in neg) / len(neg) if neg else None
        k = kappa(pairs)
        a4 = f"{sum(x == y for x, y in p4)}/{len(p4)} = {100 * sum(x == y for x, y in p4) / len(p4):.0f}%" if p4 else "—"
        k4 = f"{kappa(p4):.2f}" if p4 else "—"
        L.append(f"| {name} | {n} | {ag}/{n} = {100 * ag / n:.0f}% | {k:.2f} | "
                 f"{'—' if tpr is None else f'{tpr:.2f} (n={len(pos)})'} | {'—' if tnr is None else f'{tnr:.2f} (n={len(neg)})'} | {len(p4) or '—'} | {a4} | {k4} |")
        cm = collections.Counter(pairs)
        out["agreement"].append(dict(method=col, name=name, human=human, n=n, agree=ag, kappa=k, tpr=tpr, n_pos=len(pos), tnr=tnr, n_neg=len(neg),
                                     confusion={f"sub={x}|method={y}": v for (x, y), v in sorted(cm.items())},
                                     n4=len(p4), agree4=sum(x == y for x, y in p4), kappa4=kappa(p4) if p4 else None,
                                     confusion4={f"sub={x}|method={y}": v for (x, y), v in sorted(collections.Counter(p4).items())}))
    L.append("")
    L.append("Notes: the agent 代判 labels were made with the 2026-11-14/15 rules (see prompt/judge_prompt.md and README); "
             "CNN/predicate verdicts did not pass their validation gates on generated frames (see key columns cnn_gate). "
             "Neither is ground truth.")
    text = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    else:
        sys.stdout.write(text)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
