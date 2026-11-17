"""Round2 真实画面卡片 · LingBot-VA / OpenWAM-α（2026-11-12）。与 real_cards.py 同一 make_card（13 帧等距、最右 = 最后一帧）。
画面来源（都是卡片单元，抽样规则与 Cosmos/UVA 相同：seed 20261112，每格 5 个）：
  replay  : rerun/replay_frames.py 按记录的已执行动作在模拟器里重放 → runs/<run>/replay_frames/<uid>.npz
            agentview[i] = 第 i+1 步执行后的画面（无初始帧）
  lb_rerun: rerun/lb_rerun.py 在 GPU 上重跑 LingBot T2–T4 闭环（原 run 没存动作）→ runs/lingbot_t{tid}/rerun_closed/<src>/<uid>.npz
            agentview[0] = 初始观测，agentview[i] = 第 i 步后
卡片成败标签 = 画面所属那次执行自己的结果（重放 / 重跑），同时记下原记录是否一致（matches_recorded）。
  python ours_cards.py list                -> units_ours.json（只收画面已存在的单元；补跑完再 list 一次即可追加）
  python ours_cards.py probe               -> probe_ours_*.png（每种来源一张，核对方向）
  python ours_cards.py render <shard> <n>
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_wmvp.bootstrap()
from real_cards import H, OUT, make_card  # noqa: E402

RUNS = _wmvp.P("$WMV_ROOT/libero/out/runs")
ROOT = _wmvp.P("$WMV_ROOT")
TN = {"t5": "T0", "t3": "T1", "t2": "T2", "t4": "T3", "t6": "T4"}


def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pat}")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def units():
    sel, seen = [], set()
    for r in rows("**/replay_frames.*.jsonl") + rows("*/*/replay_frames.*.jsonl"):
        if not r.get("card_unit") or (r["run"], r["unit_id"]) in seen:
            continue
        seen.add((r["run"], r["unit_id"]))
        rp = r["replay"]
        sel.append(dict(model=r["model"], T=TN[r["task"]], level=r["level"], source="replay", run=r["run"],
                        frames=f"{ROOT}/{r['frames_file']}", success=bool(rp["success"]),
                        first_success_step=rp["first_success_step"],
                        matches_recorded=bool(r["matches_recorded_success"] and r["matches_recorded_step"]),
                        unit_id=r["unit_id"]))
    for tid in (2, 4, 6):
        for r in rows(f"lingbot_t{tid}/rerun_closed/*.jsonl"):
            run = f"lingbot_t{tid}/rerun_closed/{r['src']}"
            if not r.get("card_unit") or (run, r["unit_id"]) in seen:
                continue
            seen.add((run, r["unit_id"]))
            rr = r["rerun"]
            sel.append(dict(model="lingbot", T=TN[r["task"]], level=r["level"], source="lb_rerun", run=run,
                            frames=f"{ROOT}/{r['frames_file']}", success=bool(rr["success"]),
                            first_success_step=rr["first_success_step"],
                            matches_recorded=bool(r["matches_recorded_success"] and r["matches_recorded_step"]),
                            unit_id=r["unit_id"]))
    sel = [u for u in sel if os.path.exists(u["frames"])]
    ids = [u["unit_id"] for u in sel]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate unit ids across runs: {sorted(dup)[:5]}"
    return sel


def frames_and_last(u):
    fr = np.load(u["frames"])["agentview"][:, ::-1]   # 原始渲染上下颠倒：只做上下翻转（转 180° 会左右镜像，T3 左/右盘判错）；与 Cosmos/UVA 视频同向，已用 T3 成功单元核对
    ok, fs = u["success"], u["first_success_step"]
    if u["source"] == "replay":                       # 第 i 帧 = 第 i+1 步后
        li = (fs - 1) if ok and fs else min(len(fr), H) - 1
    else:                                             # 第 0 帧 = 初始观测
        li = fs if ok and fs else min(len(fr) - 1, H)
    return fr, max(0, min(li, len(fr) - 1))


def main():
    cmd = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if cmd == "list":
        sel = units()
        json.dump(sel, open(f"{HERE}/units_ours.json", "w"), ensure_ascii=False, indent=0)
        from collections import Counter
        print("selected", len(sel), sorted(Counter((u["model"], u["T"], u["source"]) for u in sel).items()))
        print("label differs from original record:", sum(not u["matches_recorded"] for u in sel))
        return
    sel = json.load(open(f"{HERE}/units_ours.json"))
    if cmd == "probe":
        picks = {}
        for u in sel:
            k = (u["model"], u["source"], u["T"])
            if u["success"] and k not in picks:
                picks[k] = u
        for k, u in picks.items():
            fr, li = frames_and_last(u)
            print(k, u["unit_id"], fr.shape, "last", li, u["first_success_step"], flush=True)
            make_card(fr, li).save(f"{HERE}/probe_ours_{'_'.join(k)}.png")
        return
    shard, nsh = int(sys.argv[2]), int(sys.argv[3])
    done = 0
    for u in sel:
        if zlib.crc32(u["unit_id"].encode()) % nsh != shard:
            continue
        png = f"{OUT}/{u['unit_id']}.png"
        if os.path.exists(png):
            continue
        try:
            fr, li = frames_and_last(u)
            make_card(fr, li).save(png, optimize=True)
            done += 1
        except Exception as e:  # noqa: BLE001
            print("SKIP", u["unit_id"], e, flush=True)
    print("RENDER_DONE shard", shard, done)


if __name__ == "__main__":
    main()
