"""终局想象乐观率 · 第 2 步：对"有判定器且有帧"的单元判想象末帧（成功单元也判）→ 现实成败 × 想象判定 2×2 混淆矩阵。
当前唯一满足条件的格：OpenWAM T1（t3），判定器 = OpenWAM 视角 v4 联合（p_close >= thr.close 且 p_bowl_out < thr.bowl_out，
manifest libero/openwam/predicate/t3/predicate_manifest.json；轨迹规则 k=1，单帧即部署判定）。
想象末帧 = 最后一次规划调用的最后一帧 imagined[-1]（主口径，按指令）。OpenWAM 第 13 次调用覆盖第 384–416 步，只执行到 400，
imagined[-1] 对应名义第 416 步；另记第 400 步那一帧（最后一次调用的第 4 帧，步距 4）作敏感性对照。
不建新判定器，不补采帧。lerobot_libero 解释器，需 GPU（lingbot_predicate.predict 写死 .cuda()）。
  python judge_ow_t1.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from lingbot_predicate import Net, predict  # noqa: E402
from ow_view import pred_view  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n):
    if not n:
        return None, None
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - s) / d, 4), round((c + s) / d, 4)


def main():
    man = json.load(open(f"{ROOT}/libero/openwam/predicate/t3/predicate_manifest.json"))
    nets, thr = {}, {}
    for k, d in man["detectors"].items():
        m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{d['path']}")); m.eval(); nets[k] = m; thr[k] = d["threshold"]
    U = [u for u in json.load(open(f"{HERE}/units.json")) if u["model"] == "openwam" and u["T"] == "T1"]
    out = []
    for u in U:
        z = np.load(u["frames"])
        im, cl = z["imagined"], z["imagined_call_lens"]
        last = im[-1]; s400 = im[len(im) - int(cl[-1]) + 4]
        X = np.stack([pred_view(last), pred_view(s400)])
        with torch.no_grad():
            pc, pb = predict(nets["close"], X).ravel(), predict(nets["bowl_out"], X).ravel()
        ok = (pc >= thr["close"]) & (pb < thr["bowl_out"])
        out.append(dict(unit_id=u["unit_id"], level=u["level"], family=u["family"], source=u["source"], real_success=u["success"],
                        imag_success=bool(ok[0]), p_close=float(pc[0]), p_bowl_out=float(pb[0]),
                        imag_success_step400=bool(ok[1]), p_close_step400=float(pc[1]), p_bowl_out_step400=float(pb[1]),
                        n_calls=int(len(cl)), frames_per_call=int(cl[-1])))
    with open(f"{HERE}/judged_openwam_T1.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    res = {}
    for tag, key in (("末帧(主口径)", "imag_success"), ("第400步帧(敏感性)", "imag_success_step400")):
        cm = {(rs, im): sum(1 for r in out if r["real_success"] == rs and r[key] == im) for rs in (True, False) for im in (True, False)}
        nf = cm[(False, True)] + cm[(False, False)]; ns = cm[(True, True)] + cm[(True, False)]
        res[tag] = dict(model="openwam", task="T1", judge="OpenWAM t3 v4 联合", n_units=len(out),
                        real_fail__imag_success=cm[(False, True)], real_fail__imag_fail=cm[(False, False)],
                        real_success__imag_success=cm[(True, True)], real_success__imag_fail=cm[(True, False)],
                        optimism_rate=round(cm[(False, True)] / nf, 4) if nf else None, optimism_ci95=wilson(cm[(False, True)], nf), n_real_fail=nf,
                        imag_agrees_on_success=round(cm[(True, True)] / ns, 4) if ns else None, success_ci95=wilson(cm[(True, True)], ns), n_real_success=ns)
    json.dump(res, open(f"{HERE}/confusion_openwam_T1.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
