"""Round2 补充：Cosmos Policy T4（libero t6）闭环想象链卡片，口径与 OpenWAM round2 相同。
每档按种子 20261109 抽 5 个单元；卡片 = 第一遍闭环每次规划查询预测的未来主视角帧（future_primary）横条。
族与 OpenWAM t6 round2 对齐：E/G/K/N/B（Cosmos 无 NLw）。只读队友目录。
  python render_cosmos_r2.py [--probe]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

_wmvp.bootstrap()
import ext_models  # noqa: E402

ROOT = _wmvp.P("$COSMOS_DATA_ROOT/cosmos_policy")
OUT = _wmvp.P("$WMV_ROOT/libero/out/annotation/round2_closed/cards_cosmos")
SEED, K, ROWH = 20261109, 5, 168
FAMS = ("E", "G", "K", "N", "B")
Q6 = ("Closed-loop deployment under perturbation. The strip shows the END frame of each planning call — what the model "
      "imagined while executing. Judge the LAST (rightmost) frame: is the white mug ON the plate AND the chocolate pudding "
      "to the RIGHT of the plate?")
FAM = lambda lv: ("E" if lv[:2] in ("Em", "Ef") else "N" if lv[:2] in ("N1", "NR", "NL") else
                  "B" if lv[:2] in ("Bg", "Bf", "Bz") or lv == "B0" else lv[0])


def strip(unit_dir, flip):
    P = f"{unit_dir}/closed_loop1_model_predictions"
    al = json.load(open(f"{P}/alignment.json"))["queries"]
    imgs = []
    for q in al:
        fr = np.load(f"{P}/{q['prediction_file']}")["future_primary"]
        fr = np.asarray(fr)
        if fr.ndim == 4:
            fr = fr[-1]
        if fr.dtype != np.uint8:
            fr = np.clip(fr * (255 if fr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        if flip:
            fr = fr[::-1, ::-1]
        im = Image.fromarray(fr)
        imgs.append(im.resize((int(im.width * ROWH / im.height), ROWH)))
    cat = Image.new("RGB", (sum(i.width for i in imgs), ROWH))
    x = 0
    for im in imgs:
        cat.paste(im, (x, 0)); x += im.width
    return cat


def main():
    probe = "--probe" in sys.argv
    flip = "--flip" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    cells = {}
    for r in csv.DictReader(open(f"{ROOT}/analysis/closed_loop_audit_20261111/unit_outcomes.csv", encoding="utf-8-sig")):
        if r["task"] != "T4" or r["reused_baseline"] == "True":
            continue
        lv = ext_models.cosmos_level(r["condition"])
        if lv is None or lv == "none" or FAM(lv) not in FAMS:
            continue
        cells.setdefault(lv, []).append(os.path.dirname(f"{ROOT}/{r['source']}"))
    if probe:
        lv = sorted(cells)[0]; d = sorted(cells[lv])[0]
        P = f"{d}/closed_loop1_model_predictions"
        z = np.load(f"{P}/query_0016.npz")
        print({k: (z[k].shape, str(z[k].dtype), float(z[k].min()), float(z[k].max())) for k in z.files})
        ob = sorted(os.listdir(f"{d}/closed_loop1_observations"))[:5]
        print("observations:", ob)
        strip(d, flip).save(f"{OUT}/../probe_cosmos_strip.png")
        print("probe ->", f"{OUT}/../probe_cosmos_strip.png", lv, d)
        return
    rng = np.random.RandomState(SEED)
    manifest = []
    order = [l for l in ext_models_order() if l in cells]
    for lv in order:
        dirs = sorted(cells[lv])
        pick = sorted(rng.choice(len(dirs), size=min(K, len(dirs)), replace=False).tolist())
        for i in pick:
            uid = f"cosmos_libero10_t6_{os.path.basename(dirs[i])}__{lv}"
            png = f"{OUT}/{uid}.png"
            try:
                if not os.path.exists(png):
                    strip(dirs[i], flip).save(png, optimize=True)
            except Exception as e:  # noqa: BLE001
                print("SKIP", uid, e, flush=True); continue
            manifest.append(dict(model="cosmos", task="t6", family=FAM(lv), level=lv, unit_id=uid, png=png,
                                 source=dirs[i], question=Q6,
                                 labels=["looks_success", "borderline", "not_success", "uncertain"]))
        print(lv, len(manifest), flush=True)
    json.dump(manifest, open(f"{OUT}/../manifest_cosmos.json", "w"), ensure_ascii=False)
    print("cards:", len(manifest))


def ext_models_order():
    return ([f"Em{i}" for i in range(1, 5)] + [f"Ef{i}" for i in range(1, 5)] + [f"G{i}" for i in range(1, 5)]
            + [f"K{i}" for i in range(1, 10)] + ["N1_1"] + [f"NR{i}" for i in range(1, 6)] + [f"NL{i}" for i in range(1, 6)]
            + ["B0"] + [f"B{t}{s}" for t in "gfz" for s in (2, 5, 9)])


if __name__ == "__main__":
    main()
