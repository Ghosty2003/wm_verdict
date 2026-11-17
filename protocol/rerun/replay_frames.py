"""有闭环动作、但没存真实画面的单元：在模拟器里按原样重放第一遍闭环动作，补出每一步的真实画面（2026-11-12）。
不跑模型。环境与原 runner 一致（LeRobot LiberoEnv；LingBot 128 px，OpenWAM 256 px，B 族 128 px）；
物理扰动（Em/Ef/G）重放前施加，K 推动在同一步施加；动作取已执行动作（N 族不再重复施加动作流）。
每个单元逐步比对重放的 success 与原记录（成败 + 首次成功步），不一致的标记出来。
落盘：runs/<run>/replay_frames/<uid>.npz：agentview / wrist（每步执行后，uint8 原分辨率）、success、physics states；
      runs/<run>/replay_frames.<shard>.jsonl：比对结果。
  python replay_frames.py list | <shard> <nshards>     (lerobot_libero 解释器；EGL)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys
import time
import zlib

import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
RUNS = f"{ROOT}/libero/out/runs"
CARD_SEED, CARD_K = 20261112, 5
TNAME = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}


def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pat}")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def bank(path):
    A = {}
    for a in json.load(open(path if path.startswith("/") else f"{ROOT}/{path}"))["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        A[a["anchor_id"]] = a
    return A


_B = {}


def B(path):
    if path not in _B:
        _B[path] = bank(path)
    return _B[path]


MAIN = "libero/out/bank/libero_anchors.json"
FAM_BANK = {"famA": "libero/out/bank/familyA/familyA_anchors.json", "famP": "libero/out/bank/familyP/familyP_anchors.json"}


def tx_bank(tid, fam):
    if fam == "famA":
        return f"libero/out/bank/familyA_t{tid}/familyA_t{tid}_anchors.json"
    if fam == "famP":
        return f"libero/out/bank/familyP_t{tid}/familyP_t{tid}_anchors.json"
    return f"libero/out/bank/libero_anchors_t{tid}.json"


def level_of(r):
    s = r.get("setting_id") or "none"
    return s.split("__")[-1] if "__" in s else s


def jobs():
    """dict(model, run, uid, tid, level, anchor_bank, anchor_id, res, actions=(file,key), spec_setting, base_unit, rec)."""
    J = []
    for m in ("lingbot", "openwam"):
        res = 128 if m == "lingbot" else 256
        base = {r["unit_id"]: r for r in rows(f"{m}_base/records_t*.jsonl")}
        # T0/T1 phys
        for r in rows(f"{m}_phys/closed_t*.jsonl"):
            b = base[r["base_unit"]]
            J.append(dict(model=m, run=f"{m}_phys", uid=r["unit_id"], tid=b["task_id"], level=level_of(r), bank=MAIN,
                          anchor_id=b["s0_id"], res=res, act=(f"{ROOT}/{r['closed_loop_file']}", "executed"),
                          setting=r["setting_id"], base_unit=r["base_unit"], rec=r["closed_loop"]))
        # T0/T1 base / A / P / O / famB
        for fam in ("base", "famA", "famP", "famO", "famB"):
            for r in rows(f"{m}_{fam}/records_t*.jsonl"):
                bk, aid = (FAM_BANK[fam], r["anchor_id"]) if fam in FAM_BANK else (MAIN, r["s0_id"])
                J.append(dict(model=m, run=f"{m}_{fam}", uid=r["unit_id"], tid=r["task_id"], level=level_of(r), bank=bk,
                              anchor_id=aid, res=128 if fam == "famB" else res, act=(f"{ROOT}/{r['action_file']}", "closed_loop"),
                              setting=None, base_unit=None, rec=r["closed_loop"]))
    for tid in (2, 4, 6):
        # LingBot T2–T4 famB（其余 LingBot T2–T4 无动作，走 lb_rerun.py GPU 重跑）
        for r in rows(f"lingbot_t{tid}/famB/records*.jsonl"):
            J.append(dict(model="lingbot", run=f"lingbot_t{tid}/famB", uid=r["unit_id"], tid=tid, level=level_of(r),
                          bank=tx_bank(tid, "base"), anchor_id=r["anchor_id"], res=128,
                          act=(f"{RUNS}/lingbot_t{tid}/famB/actions/{r['unit_id']}.npz", "closed_loop"),
                          setting=None, base_unit=None, rec=r["closed_loop"]))
        R = f"openwam_t{tid}"
        for fam in ("base", "famA", "famP", "famO"):
            for r in rows(f"{R}/records_{fam}.jsonl"):
                J.append(dict(model="openwam", run=R, uid=r["unit_id"], tid=tid, level=level_of(r), bank=tx_bank(tid, fam),
                              anchor_id=r["anchor_id"], res=256, act=(f"{RUNS}/{R}/actions/{r['unit_id']}.npz", "closed_loop"),
                              setting=None, base_unit=None, rec=r["closed_loop"]))
        for r in rows(f"{R}/famB/records*.jsonl"):
            J.append(dict(model="openwam", run=f"{R}/famB", uid=r["unit_id"], tid=tid, level=level_of(r), bank=tx_bank(tid, "base"),
                          anchor_id=r["anchor_id"], res=128, act=(f"{RUNS}/{R}/famB/actions/{r['unit_id']}.npz", "closed_loop"),
                          setting=None, base_unit=None, rec=r["closed_loop"]))
        baseT = {r["unit_id"]: r for r in rows(f"{R}/records_base.jsonl")}
        for r in rows(f"{R}/phys/closed.*.jsonl"):
            npz = f"{RUNS}/{R}/closed/{r['unit_id']}.npz"
            if not os.path.exists(npz):
                continue                                  # 无动作 → ow_rerun.py GPU 重跑
            J.append(dict(model="openwam", run=f"{R}/phys", uid=r["unit_id"], tid=tid, level=level_of(r), bank=tx_bank(tid, "base"),
                          anchor_id=baseT[r["base_unit"]]["anchor_id"], res=256, act=(npz, "executed"),
                          setting=r["setting_id"], base_unit=r["base_unit"], rec=r["closed_loop"]))
    seen, out = set(), []
    for j in J:
        k = (j["run"], j["uid"])
        if k not in seen:
            seen.add(k); out.append(j)
    return out


def ow_t2rerun_jobs(tids=(2,)):
    """ow_rerun.py 的重跑单元（只存了动作/物理状态）→ 重放出画面。T2 = 卡片单元；T3/T4 = GH200 未回传的 142 个（×2 的第一遍）。"""
    J = []
    for tid in tids:
        R = f"openwam_t{tid}"
        baseT = {r["unit_id"]: r for r in rows(f"{R}/records_base.jsonl")}
        ph = {r["unit_id"]: r for r in rows(f"{R}/phys/closed.*.jsonl")}       # base_unit / anchor 从原 phys 记录取
        for r in rows(f"{R}/rerun_closed/phys.*.jsonl"):
            npz = f"{RUNS}/{R}/rerun_closed/phys/{r['unit_id']}.npz"
            if not os.path.exists(npz):
                continue
            p = ph[r["unit_id"]]
            J.append(dict(model="openwam", run=f"{R}/rerun_closed", uid=r["unit_id"], tid=tid, level=r["level"], bank=tx_bank(tid, "base"),
                          anchor_id=baseT[p["base_unit"]]["anchor_id"], res=256, act=(npz, "executed"), setting=r["setting_id"],
                          base_unit=p["base_unit"], rec=r["rerun"]))
    return J


def card_first(J):
    cells = {}
    for j in J:
        cells.setdefault((j["model"], j["tid"], j["level"]), []).append(j)
    pri = set()
    for (m, tid, lv), js in cells.items():
        js = sorted(js, key=lambda j: j["uid"])
        rng = np.random.RandomState(zlib.crc32(f"{CARD_SEED}/{m}/{TNAME[tid]}/{lv}".encode()))
        for i in rng.choice(len(js), size=min(CARD_K, len(js)), replace=False):
            pri.add((js[i]["run"], js[i]["uid"]))
    return sorted(J, key=lambda j: ((j["run"], j["uid"]) not in pri, j["run"], j["uid"])), pri


def replay(j):
    from libero_env_adapter import _make_env
    from phys_perturb import spec_of, apply_physics, maybe_push
    import mujoco
    acts = np.asarray(np.load(j["act"][0])[j["act"][1]], np.float32)
    a = dict(B(j["bank"])[j["anchor_id"]], resolution=j["res"])
    env = _make_env(a, H=len(acts) + 10); u = env.envs[0]
    spec = spec_of(j["setting"], j["tid"], j["base_unit"]) if j["setting"] else None
    if spec:
        apply_physics(u, spec)
    m, d = u._env.sim.model._model, u._env.sim.data._data

    def state():
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return st
    av, wr, succ, S = [], [], [], [state()]          # 画面 = 每一步执行后的观测（第 i 帧 = 第 i+1 步后）；S[0] = 初始状态

    def grab(ob):
        av.append(np.asarray(ob["pixels"]["image"]).copy()); wr.append(np.asarray(ob["pixels"]["image2"]).copy())
    for t, e in enumerate(acts):
        if spec:
            maybe_push(u, spec, t)
        ob, _, term, trunc, _ = u.step(e)
        grab(ob); succ.append(bool(u._env.check_success())); S.append(state())
    env.close()
    return acts, np.stack(av), np.stack(wr), np.array(succ), np.stack(S)


def main():
    J = jobs()
    if sys.argv[1] == "list":
        J, pri = card_first(J)
        from collections import Counter
        print(len(J), "units;", len(pri), "card units first;", sorted(Counter(j["run"] for j in J).items()))
        done = lambda j: os.path.exists(f"{RUNS}/{j['run']}/replay_frames/{j['uid']}.npz")
        for m in ("lingbot", "openwam"):
            c = [j for j in J if j["model"] == m and (j["run"], j["uid"]) in pri]
            r = [j for j in J if j["model"] == m and (j["run"], j["uid"]) not in pri]
            print(f"WEB {m}: card units via replay {len(c)} (done {sum(map(done, c))}); other replay {len(r)} (done {sum(map(done, r))})")
        json.dump([dict(model=j["model"], run=j["run"], uid=j["uid"], tid=j["tid"], level=j["level"], source="replay")
                   for j in J if (j["run"], j["uid"]) in pri], open(f"{os.path.dirname(os.path.abspath(__file__))}/web_units_replay.json", "w"))
        miss = [j for j in J if not os.path.exists(j["act"][0])]
        print("missing action files:", len(miss), [j["uid"] for j in miss[:5]])
        return
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    J, pri = card_first(J)
    # 2026-11-12 优先级拆分：lb_card = LingBot 网页卡片单元（高优先）；ow = OpenWAM 全部；lb_rest = LingBot 其余（垫底）
    sel = os.environ.get("WMV_SET")
    if sel == "ow_t2rerun":                             # OpenWAM T2 卡片单元：ow_rerun.py 只存了动作/物理状态，这里重放出画面
        J = ow_t2rerun_jobs(); pri = {(j["run"], j["uid"]) for j in J}
    elif sel == "ow_rerun_all":                         # OpenWAM T2–T4 全部重跑单元（T2 已做的自动跳过）
        J = ow_t2rerun_jobs((2, 4, 6)); pri = {(j["run"], j["uid"]) for j in J if j["tid"] == 2}
    elif sel == "lb_card":
        J = [j for j in J if j["model"] == "lingbot" and (j["run"], j["uid"]) in pri]
    elif sel == "ow":
        J = [j for j in J if j["model"] == "openwam"]
    elif sel == "lb_rest":
        J = [j for j in J if j["model"] == "lingbot" and (j["run"], j["uid"]) not in pri]
    elif sel:
        raise SystemExit(f"unknown WMV_SET {sel}")
    mine = [j for j in J if zlib.crc32(f"{j['run']}/{j['uid']}".encode()) % nsh == shard]
    if os.environ.get("WMV_TEST"):                      # 冒烟：每个 run（× K 档优先）各取 2 个
        by = {}
        for j in J:
            k = (j["run"], j["level"][:1] == "K")
            if len(by.setdefault(k, [])) < 2:
                by[k].append(j)
        mine = [x for v in by.values() for x in v]
    limit = int(os.environ.get("WMV_LIMIT", "0")); n = 0
    print("REPLAY all", len(J), "this shard", len(mine), flush=True)
    for j in mine:
        if limit and n >= limit:
            break
        od = f"{RUNS}/{j['run']}/replay_frames"
        if os.path.exists(f"{od}/{j['uid']}.npz"):
            continue
        t0 = time.time()
        try:
            acts, av, wr, succ, S = replay(j)
        except Exception as e:  # noqa: BLE001
            print("FAIL", j["uid"], repr(e), flush=True); continue
        os.makedirs(od, exist_ok=True)
        tmp = f"{od}/{j['uid']}.tmp.npz"
        np.savez_compressed(tmp, agentview=av, wrist=wr, success=succ, states=S, actions=acts)
        os.replace(tmp, f"{od}/{j['uid']}.npz")
        fs = int(np.argmax(succ)) + 1 if succ.any() else None
        rc = j["rec"] or {}
        row = dict(unit_id=j["uid"], run=j["run"], model=j["model"], task=f"t{j['tid']}", level=j["level"],
                   card_unit=(j["run"], j["uid"]) in pri, n_actions=int(len(acts)), res=j["res"],
                   replay=dict(success=bool(succ.any()), first_success_step=fs),
                   recorded=dict(success=rc.get("success"), first_success_step=rc.get("first_success_step")),
                   matches_recorded_success=bool(succ.any()) == bool(rc.get("success")),
                   matches_recorded_step=fs == rc.get("first_success_step"),
                   frames_file=f"libero/out/runs/{j['run']}/replay_frames/{j['uid']}.npz",
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        with open(f"{RUNS}/{j['run']}/replay_frames.{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n += 1
        print(j["uid"], "rec", rc.get("success"), rc.get("first_success_step"), "replay", bool(succ.any()), fs, row["seconds"], flush=True)
    print("REPLAY_DONE shard", shard)


if __name__ == "__main__":
    main()
