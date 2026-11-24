#!/usr/bin/env python3
"""Merge every judged imagined frame (11-17 package, 11-24 gap package, UVA ext) into one blind package with
fresh shuffled ids. Writes blind/ (frames, items, prompts, refs, README) and key/ (id map + full key rows)."""
import csv, os, random, shutil, hashlib, json, collections
OUT = os.path.join(os.environ.get("WMV_ROOT", "$WMV_ROOT"), "libero", "out"); PKG = f"{OUT}/judge_package_20261117"; GAP = f"{OUT}/judge_gap_20261124"; JB = f"{OUT}/judge_binary_20261124"
D = f"{OUT}/judge_package_all_20261124"; BL = f"{D}/blind"; KY = f"{D}/key"
for p in (f"{BL}/frames", f"{BL}/prompt/refs", KY): os.makedirs(p, exist_ok=True)
src = []
for r in csv.DictReader(open(f"{PKG}/items_blind.csv")): src.append(("pkg20261117", r, f"{PKG}/frames/{r['frame_id']}.png"))
for r in csv.DictReader(open(f"{GAP}/items/items_blind_gap.csv")): src.append(("gap20261124", r, f"{GAP}/frames/{r['frame_id']}.png"))
for r in csv.DictReader(open(f"{PKG}/uva_ext/items_blind_uva.csv")): src.append(("uva_ext", r, f"{PKG}/uva_ext/frames/{r['frame_id']}.png"))
keys = {"pkg20261117": {r["frame_id"]: r for r in csv.DictReader(open(f"{PKG}/_key/key.csv"))},
        "gap20261124": {r["frame_id"]: r for r in csv.DictReader(open(f"{GAP}/key/key_gap.csv"))},
        "uva_ext": {r["frame_id"]: r for r in csv.DictReader(open(f"{PKG}/uva_ext/key_uva.csv"))}}
rng = random.Random(20261124); rng.shuffle(src)
items, kmap = [], []
for i, (pk, r, path) in enumerate(src, 1):
    nid = f"v{i:05d}"; shutil.copy2(path, f"{BL}/frames/{nid}.png")
    items.append([nid, r["task"], r["task_title"], r["task_criterion"]])
    k = keys[pk].get(r["frame_id"], {})
    kmap.append(dict(new_id=nid, src_package=pk, src_frame_id=r["frame_id"], model=k.get("model", ""), task=r["task"],
                     level=k.get("level", k.get("condition", "")), real_success=k.get("real_success", ""), set=k.get("set", ""),
                     gap_part=k.get("gap_part", ""), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()))
items.sort(key=lambda x: (x[1], x[0]))
with open(f"{BL}/items_blind.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame_id", "task", "task_title", "task_criterion"]); w.writerows(items)
with open(f"{KY}/key_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(kmap[0])); w.writeheader(); w.writerows(kmap)
for fn in os.listdir(f"{PKG}/prompt/refs"): shutil.copy2(f"{PKG}/prompt/refs/{fn}", f"{BL}/prompt/refs/{fn}")
p4 = open(f"{PKG}/prompt/judge_prompt.md", encoding="utf-8").read().replace("(agent 代判, 2026-11-14/15)", "(agent judging, 2026-11-14/15)")
p2 = open(f"{JB}/prompt/judge_prompt.md", encoding="utf-8").read().replace('("强制二选一，选更可能的")', '("forced choice: pick the more likely outcome")')
open(f"{BL}/prompt/judge_prompt.md", "w", encoding="utf-8").write(p4)
open(f"{BL}/prompt/judge_prompt_binary.md", "w", encoding="utf-8").write(p2)
shutil.copy2(f"{PKG}/prompt/rubric_examples.md", f"{BL}/prompt/rubric_examples.md")
run = open(f"{PKG}/prompt/run_agent_judge_all.md", encoding="utf-8").read().replace("The full package has 6314 frames", f"The full package has {len(items)} frames")
open(f"{BL}/prompt/run_agent_judge_all.md", "w", encoding="utf-8").write(run)
c = collections.Counter(x[1] for x in items)
json.dump(dict(n=len(items), per_task=c, per_source=collections.Counter(k["src_package"] for k in kmap)), open(f"{D}/_build/build_summary.json", "w"), indent=1)
print(len(items), dict(c), dict(collections.Counter(k["src_package"] for k in kmap)))
