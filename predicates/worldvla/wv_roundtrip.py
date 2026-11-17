"""WorldVLA-view predicate, VQGAN round-trip row (run with the worldvla interpreter; tokenizer only, no 7B model).
For every episode: the every-6th 256-px WorldVLA-view frames (recal_wv.py render_rt) -> VQGAN encode -> decode
(worldvla_unit.make_processor: the same tokenizer path the imagination uses) -> 128x128. Labels from the 128-px render.
  python wv_roundtrip.py <task_id>
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import numpy as np
from PIL import Image

_wmvp.bootstrap()
import worldvla_unit as W

OUT = _wmvp.P("$WMV_ROOT/libero/out/predicate_wv")


def fail_terminal_mask(labels):
    m = np.zeros(len(labels), bool)
    if not labels.any():
        m[len(labels) // 2:] = True
    return m


def main(task_id):
    proc = W.make_processor("cuda")
    d = f"{OUT}/t{task_id}"
    F, L, M, E = [], [], [], []
    for f in sorted(os.listdir(f"{d}/frames_rt")):
        z = np.load(f"{d}/frames_rt/{f}"); lab = np.load(f"{d}/frames/{f}")["labels"]; ft = fail_terminal_mask(lab)
        for img, i in zip(z["frames"], z["idx"]):
            tok = proc.process_image(Image.fromarray(img))["input_ids"]
            dec = np.asarray(proc.decode_image(list(tok)))
            F.append(np.asarray(Image.fromarray(dec).resize((128, 128), Image.BILINEAR))); L.append(bool(lab[i])); M.append(bool(ft[i])); E.append(f[:-4])
    np.savez_compressed(f"{d}/roundtrip_all.npz", frames=np.stack(F), labels=np.array(L), fail_terminal=np.array(M), episode=np.array(E))
    print(f"t{task_id} roundtrip_all {len(F)} fail_terminal {int(np.sum(M))}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
