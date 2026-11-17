"""LingBot B 族 T2 九格（2026-11-10 决定 2b，桥通过后才跑）：原样调用 bfam/lingbot_tx_famB.py 的生产循环（不跑 sham 步，
跨机器由锚点桥替代，见 exemptions.md），外加闭环逐块想象 latent + 读入的真实帧存档 -> famB/closed_frames/<uid>.npz。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os, sys
import numpy as np
os.environ.setdefault("WMV_TID", "2")
_wmvp.bootstrap()
import lb_capture
import lingbot_unit as LU
import lingbot_tx_famB as M
lb_capture.install()
PENDING = []
_orig_cl = LU.LingBotRig.closed_loop
def closed_loop(self, *a, **k):
    with lb_capture.Capture() as cap:
        res = _orig_cl(self, *a, **k)
    PENDING.append(cap); return res
LU.LingBotRig.closed_loop = closed_loop
_orig_save = np.savez_compressed
def savez(path, *a, **k):
    _orig_save(path, *a, **k)
    p = str(path)
    if "/famB/actions/" in p:
        uid = os.path.basename(p)[:-4]; d = os.path.join(os.path.dirname(os.path.dirname(p)), "closed_frames"); os.makedirs(d, exist_ok=True)
        arrs = {}
        for i, cap in enumerate(PENDING[-2:], 1):
            arrs.update(cap.arrays(prefix="" if i == 1 else "run2_"))
        tmp = f"{d}/{uid}.tmp.npz"; _orig_save(tmp, **arrs); os.replace(tmp, f"{d}/{uid}.npz")
        PENDING.clear()
np.savez_compressed = savez
sys.argv = ["lingbot_tx_famB.py", os.environ.get("WMV_SHARD", "0"), os.environ.get("WMV_NSH", "1")]
M.main()
