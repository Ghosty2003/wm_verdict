"""成功单元参照条（只读存档 read_agentview，每 32 步一帧；不重放）→ out/f_family/cases/sheets/OW_<T>_<lv>_success_ref.png"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob, json, numpy as np
from PIL import Image, ImageDraw
ROOT = _wmvp.P("$WMV_ROOT"); RUNS = f"{ROOT}/libero/out/runs"; OUT = f"{ROOT}/libero/out/f_family/cases/sheets"
for T, tid, lv in (("T1", 3, "F160"), ("T3", 4, "F60")):
    rows = sorted([json.loads(l) for f in glob.glob(f"{RUNS}/openwam_famF/closed_t{tid}.*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["unit_id"])
    S = [r for r in rows if r["setting_id"] == lv and r["closed_loop"]["success"]][::5][:7]
    n = 13; I = Image.new("RGB", (n * 98, len(S) * 116), (255, 255, 255)); d = ImageDraw.Draw(I)
    for i, r in enumerate(S):
        z = np.load(f"{ROOT}/{r['closed_loop_file']}"); rav = z["read_agentview"]
        d.text((2, i * 116 + 1), f"{r['unit_id'].split('libero10_')[1]} success, first success step {r['closed_loop']['first_success_step']}  (frames = steps 0,32,...,384)", fill=(0, 0, 0))
        for c in range(n):
            I.paste(Image.fromarray(np.ascontiguousarray(rav[c][::-1])).resize((96, 96)), (c * 98, i * 116 + 16))
    I.save(f"{OUT}/OW_{T}_{lv}_success_ref.png"); print("REF", T, len(S))
