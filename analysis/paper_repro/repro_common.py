"""Shared helpers for the VerdictBench paper reproduction scripts (Table 3, Fig. 3, Fig. 4).

Paths: every data path is resolved under $WMV_ROOT (the project root in the original layout: libero/out/...,
libero/perturbation_registry.md). Outputs go to $REPRO_OUT (default: out/ next to this file).
All inputs are opened read-only. Every script records the files it read in out/_manifest_<script>.json, from which
make_inputs.py writes inputs.md (path, size, sha256, mtime, dependent outputs).
"""
import hashlib
import json
import math
import os

ROOT = os.environ.get("WMV_ROOT", "$WMV_ROOT")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("REPRO_OUT", os.path.join(HERE, "out"))

LIBERO = os.path.join(ROOT, "libero")
CELLS_A = os.path.join(LIBERO, "out", "cells", "compare_lingbot_openwam_H400.json")   # base + A + P units (2026-10-26)
CELLS_B = os.path.join(LIBERO, "out", "cells", "compare_new_H400.json")               # E/G/N/O/B units (2026-11-13)
MAIN_TABLE = os.path.join(LIBERO, "out", "closed_main", "主表_长表_20261122.tsv")       # closed-loop main table
REGISTRY = os.path.join(LIBERO, "perturbation_registry.md")                            # perturbation registry v2
RUNS = os.path.join(LIBERO, "out", "runs")                                             # original 50-unit open-loop records

# the three judge-qualified model/task pairs of Table 3 (internal model key, internal task id)
PAIRS = [("LingBot-VA", "T0", "lingbot", "t5"), ("LingBot-VA", "T1", "lingbot", "t3"), ("OpenWAM-α", "T1", "openwam", "t3")]
MODEL_NAME = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
TASK_NAME = {"t5": "T0", "t3": "T1", "t2": "T2", "t4": "T3", "t6": "T4"}

_READ = {}


def track(path, outputs):
    """Register that `path` was read and feeds `outputs` (list of output file names)."""
    _READ.setdefault(os.path.abspath(path), set()).update(outputs)
    return path


def write_manifest(script):
    os.makedirs(OUT, exist_ok=True)
    m = {p: sorted(o) for p, o in sorted(_READ.items())}
    with open(os.path.join(OUT, f"_manifest_{script}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(script=script, root=ROOT, inputs=m), f, ensure_ascii=False, indent=1)


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_json(path, outputs):
    with open(track(path, outputs), encoding="utf-8") as f:
        return json.load(f)


def load_main_table(outputs):
    import csv
    rows = {}
    with open(track(MAIN_TABLE, outputs), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[(r["model"], r["task"], r["level"])] = r
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(x, n, z=1.959963984540054):
    """Wilson score 95% interval for x successes out of n; returns (lo, hi) as fractions."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pct(x, n):
    """Percentage rounded half-to-even to one decimal, as the paper prints (None if n == 0)."""
    return round(100 * x / n, 1) if n else None


def ranks(x):
    """Average ranks (ties averaged), 1-based."""
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rho = Pearson correlation of average ranks (same as scipy.stats.spearmanr)."""
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    vx = sum((p - mx) ** 2 for p in rx)
    vy = sum((q - my) ** 2 for q in ry)
    return cov / math.sqrt(vx * vy)
