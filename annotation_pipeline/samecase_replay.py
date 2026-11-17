"""同卷对比标注 · 第 3 步：被抽中、但缺真实帧的 LingBot / OpenWAM 单元，按存档动作在模拟器里 CPU 回放补帧（不跑模型）。
复用 libero/rerun/replay_frames.py 的 replay()（环境、扰动施加、落盘格式都与 round 2 相同），只是单元清单换成 pages.json 里
real.src == "replay" 的单元；OpenWAM F 族另加：动作取闭环存档 npz 的 executed，环境与无扰动相同（F 族只遮模型输入）。
落盘：runs/<run>/replay_frames/<uid>.npz 与 runs/<run>/replay_frames.samecase<shard>.jsonl（与原记录逐单元比对成败）。
  python samecase_replay.py list
  python samecase_replay.py <shard> <nshards>      （lerobot_libero；MUJOCO_GL=osmesa 纯 CPU）
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import os
import re
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
import replay_frames as RF   # noqa: E402

TID = {"T0": 5, "T1": 3, "T2": 2, "T3": 4, "T4": 6}


def wanted():
    P = json.load(open(f"{HERE}/pages.json"))["pages"]
    return {r["real"]["unit_id"]: (p["T"], p["level"], r["model"]) for p in P for r in p["rows"] if r["real"] and r["real"]["src"] == "replay"}


def jobs():
    W = wanted()
    J = {j["uid"]: j for j in RF.jobs() + RF.ow_t2rerun_jobs((2, 4, 6)) if j["uid"] in W and os.path.exists(j["act"][0])}
    for uid, (T, lv, m) in W.items():
        if uid in J or m != "openwam" or not lv.startswith("F"):
            continue
        tid = TID[T]; a = re.search(r"libero10_t\d_a(\d+)__k", uid).group(1)
        bank = RF.MAIN if tid in (3, 5) else RF.tx_bank(tid, "base")
        npz = f"{ROOT}/libero/out/runs/openwam_famF/closed/{uid}.npz"
        z = np.load(npz)
        J[uid] = dict(model="openwam", run="openwam_famF", uid=uid, tid=tid, level=lv, bank=bank, anchor_id=f"libero10_t{tid}_a{a}",
                      res=256, act=(npz, "executed"), setting=None, base_unit=None,
                      rec=dict(success=bool(np.asarray(z["success"]).any()),
                               first_success_step=int(np.argmax(z["success"])) + 1 if np.asarray(z["success"]).any() else None))
    miss = sorted(set(W) - set(J))
    return [J[k] for k in sorted(J)], miss


def main():
    J, miss = jobs()
    if sys.argv[1] == "list":
        from collections import Counter
        print("replay units", len(J), sorted(Counter(j["run"] for j in J).items()), "missing", len(miss), miss[:5])
        print("already done", sum(os.path.exists(f"{RF.RUNS}/{j['run']}/replay_frames/{j['uid']}.npz") for j in J))
        return
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    mine = [j for j in J if zlib.crc32(j["uid"].encode()) % nsh == shard]
    lim = int(os.environ.get("WMV_LIMIT", "0")); n = 0
    print("REPLAY", len(J), "shard", shard, len(mine), flush=True)
    for j in mine:
        if lim and n >= lim:
            break
        od = f"{RF.RUNS}/{j['run']}/replay_frames"
        if os.path.exists(f"{od}/{j['uid']}.npz"):
            continue
        t0 = time.time()
        try:
            acts, av, wr, succ, S = RF.replay(j)
        except Exception as e:  # noqa: BLE001
            print("FAIL", j["uid"], repr(e), flush=True); continue
        os.makedirs(od, exist_ok=True)
        tmp = f"{od}/{j['uid']}.tmp.npz"
        np.savez_compressed(tmp, agentview=av, wrist=wr, success=succ, states=S, actions=acts)
        os.replace(tmp, f"{od}/{j['uid']}.npz")
        fs = int(np.argmax(succ)) + 1 if succ.any() else None
        rc = j["rec"] or {}
        row = dict(unit_id=j["uid"], run=j["run"], model=j["model"], task=f"t{j['tid']}", level=j["level"], purpose="samecase_20261115",
                   n_actions=int(len(acts)), res=j["res"], replay=dict(success=bool(succ.any()), first_success_step=fs),
                   recorded=dict(success=rc.get("success"), first_success_step=rc.get("first_success_step")),
                   matches_recorded_success=bool(succ.any()) == bool(rc.get("success")),
                   matches_recorded_step=fs == rc.get("first_success_step"),
                   frames_file=f"libero/out/runs/{j['run']}/replay_frames/{j['uid']}.npz",
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gl=os.environ.get("MUJOCO_GL"))
        with open(f"{RF.RUNS}/{j['run']}/replay_frames.samecase{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n += 1
        print(j["uid"], "rec", rc.get("success"), rc.get("first_success_step"), "replay", bool(succ.any()), fs, row["seconds"], flush=True)
    print("REPLAY_DONE shard", shard, n, flush=True)


if __name__ == "__main__":
    main()
