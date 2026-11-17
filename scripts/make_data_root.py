#!/usr/bin/env python3
"""Build a WMV_ROOT data root in the legacy layout the code expects (libero/out/..., protocol.yaml, banks, predicate
weights), out of (1) this repository's protocol files and frozen banks, (2) the predicate release assets and (3) run
data you downloaded. Only symlinks are created; nothing is copied or modified.

  python scripts/make_data_root.py --out /path/to/wmv_root --assets /path/to/release_assets \
      [--data /path/to/run_data --subtree libero/out/runs/openwam_base ...]

The code directories of this repository are put on sys.path by wmv_paths.bootstrap(); the data root contains no code:
every --subtree is refused if it holds a .py / .sh file. Then: export WMV_ROOT=/path/to/wmv_root
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def link(dst, src, made):
    if os.path.lexists(dst):
        if os.path.realpath(dst) != os.path.realpath(src):
            sys.exit(f"conflict: {dst} already points to {os.path.realpath(dst)}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    made.append((dst, src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default=None, help="release_assets directory (predicate weights, node map)")
    ap.add_argument("--data", default=None, help="root of the run data (legacy layout)")
    ap.add_argument("--subtree", action="append", default=[], help="data subtree to expose, relative to --data")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    made, missing = [], []
    for r in csv.DictReader(open(os.path.join(REPO, "scripts", "legacy_layout.tsv"), encoding="utf-8"), delimiter="\t"):
        rel, rp = r["legacy_path"], r["release_path"]
        if rp.startswith("upload_copy/"):
            src = os.path.join(REPO, rp[len("upload_copy/"):])
        elif a.assets:
            src = os.path.join(os.path.abspath(a.assets), rp[len("release_assets/"):])
        else:
            continue
        if not os.path.exists(src):
            missing.append(src); continue
        link(os.path.join(out, rel), src, made)
    link(os.path.join(out, "release_hashes.yaml"), os.path.join(REPO, "protocol", "release_hashes.yaml"), made)
    wrapper = os.environ.get("LIBERO_PLUS_ENV_WRAPPER")
    if wrapper and os.path.exists(wrapper):
        link(os.path.join(out, "libero/out/b_family/healthcheck/src/env_wrapper.py"), wrapper, made)
    for s in a.subtree:
        src = os.path.join(os.path.abspath(a.data), s)
        code = [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs if f.endswith((".py", ".sh"))]
        if code:
            sys.exit(f"refusing data subtree {s}: it contains code, e.g. {code[0]}")
        link(os.path.join(out, s), src, made)
    with open(os.path.join(out, "DATA_ROOT_LINKS.tsv"), "w", encoding="utf-8") as f:
        f.write("link\ttarget\n")
        for d, s in made:
            f.write(f"{os.path.relpath(d, out)}\t{s}\n")
    print(f"{len(made)} links under {out}; {len(missing)} release files missing"
          + (f" (e.g. {missing[0]})" if missing else ""))


if __name__ == "__main__":
    main()
