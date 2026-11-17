"""OpenWAM-α 闭环：只补算【最后一次规划调用】的想象帧（2026-11-15）。不重跑整条闭环。
依据：每次调用之间没有历史（dit_cache / compile 关，SDPA），调用 c 只依赖 (读入画面, proprio, seed + c)。
做法：同一模拟器、同一锚点 / 扰动 / H，把【已记录的 commanded 动作】按原 runner 的块边界（每块 BLOCK=32）逐块喂给
sim_server（N 族的延迟 / 丢帧、K 族推动由 sim_server 自己施加，与原 runner 相同），读出最后一次调用开始时的真实观测，
按该族 runner 的同一条输入路径（B 族腐蚀 / O 族只遮第 0 次调用 / F 族只黑一次调用）调用一次 eng.generate(seed + last)。
闸门：重算出的最后一块 commanded 动作必须与记录里同一窗口一致（max_abs_diff <= 1e-6）；不过闸的单元照样落盘、标记，不用。
另外顺手比对：前缀物理状态（有 closed_loop_states 的 T0/T1 单元）、N 族 executed 前缀、整条 success 曲线 / 首次成功步。

各族镜像的 runner（最后一次调用 c = 12，第 384 步）：
  T0/T1 base / famA / famP   openwam/ow_runner.py:closed_loop            sim.reset(anchor) -> 256 px，原样读入
  T0/T1 famO                 phys/ow_famO.py:closed_occ                  只遮第 0 次调用 -> 最后一次调用原样
  T0/T1 E/G/K/N              phys/phys_closed_openwam.py:closed          reset 带 perturb=spec_of(setting, tid, base_unit)
  T0/T1 famB                 bfam/ow_famB.py:closed_b                    reset128；agentview 用 Corruptor(setting, uid, step=384)
  T2 base / famA / famP / famO  out/task2/ow_tx_run.py -> ow_famO.closed_occ（ratio 只作用于第 0 次调用）
  T2 famB Bxx                bfam/ow_tx_famB.py -> ow_famB.closed_b
  T2 famB B0                 bfam/ow_tx_famB.py -> ow_runner.closed_loop(dict(a, resolution=128))
                             （注意：Sim.reset 不传 resolution，sim_server 用默认 256 px —— 原样镜像，实际读入分辨率记在行里）
  F（全部已有想象帧，不会进清单；代码仍按 ffam/ow_famF.py:closed_f 支持：置黑只发生在第一次 32c >= t 的调用）
  T2 E/G/K/N：原 runner（phys/ow_phys_closed_tx.py）没留下闭环动作（openwam_t2/closed/ 不存在），
             卡片单元已由 ow_rerun.py 重跑并存了全部想象帧；其余单元无动作来源 -> 清单里标 recoverable=false。
动作来源 / 锚点映射与 replay_frames.jobs() 相同（run 时逐单元交叉核对），但动作一律取 commanded（phys npz 的
"commanded"；其余族的 "closed_loop" 本身就是 commanded，这些族 sim_server 无动作流，commanded == executed）。

  python3 ow_lastcall.py list                               -> ow_lastcall_units.json（纯标准库，登录节点可跑）
  python  ow_lastcall.py pilot                              -> ow_lastcall_pilot.json（需要 numpy）
  python  ow_lastcall.py run <shard> <nshards> [--units F]  (openwam 解释器，GPU)
     -> runs/<run>/lastcall/<uid>.npz + runs/<run>/lastcall.<shard>.jsonl
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

ROOT = _wmvp.P("$WMV_ROOT")
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{ROOT}/libero/out/runs"
MAIN_TSV = f"{ROOT}/libero/out/closed_main/主表_长表_20261114.tsv"
SAVED_JSON = f"{ROOT}/libero/out/optimism/units.json"
UNITS_JSON = f"{HERE}/ow_lastcall_units.json"
PILOT_JSON = f"{HERE}/ow_lastcall_pilot.json"
PILOT_SEED = "20261115/ow_lastcall/pilot"
PILOT_N = 20
TNAME = {5: "T0", 3: "T1", 2: "T2"}
H, BLOCK = 400, 32                                      # = ow_runner.H / BLOCK（run 时核对）
GATE_TOL = 1e-6
# 与 replay_frames.MAIN / FAM_BANK / tx_bank 相同（那边模块顶层 import numpy，list 要在无 numpy 的登录节点跑，故抄一份；run 时交叉核对）
MAIN = "libero/out/bank/libero_anchors.json"
FAM_BANK = {"famA": "libero/out/bank/familyA/familyA_anchors.json", "famP": "libero/out/bank/familyP/familyP_anchors.json"}
RATIOS = {"O1": 1 / 16, "O2": 1 / 8, "O3": 3 / 16, "O4": 1 / 4, "O5": 5 / 16}   # = lingbot_famO.RATIOS[1:] = ow_tx_run.RATIOS


def tx_bank(tid, fam):
    if fam == "famA":
        return f"libero/out/bank/familyA_t{tid}/familyA_t{tid}_anchors.json"
    if fam == "famP":
        return f"libero/out/bank/familyP_t{tid}/familyP_t{tid}_anchors.json"
    return f"libero/out/bank/libero_anchors_t{tid}.json"


def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pat}")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def level_of(r):
    s = r.get("setting_id") or "none"
    return s.split("__")[-1] if "__" in s else s


def main_levels():
    """{(T, level): family} of the OpenWAM-α rows of the main long table, T0–T2, NLw excluded."""
    L = {}
    lines = open(MAIN_TSV, encoding="utf-8").read().splitlines()
    hd = lines[0].split("\t")
    for line in lines[1:]:
        d = dict(zip(hd, line.split("\t")))
        if d["model"] == "OpenWAM-α" and d["task"] in ("T0", "T1", "T2") and not d["level"].startswith("NLw"):
            L[(d["task"], d["level"])] = d["family"]
    return L


def saved_ids():
    return {u["unit_id"] for u in json.load(open(SAVED_JSON)) if u.get("model") == "openwam"}


def _u(r, run, tid, kind, mirror, bank, anchor_id, seed, act, **kw):
    rc = r.get("closed_loop") or {}
    d = dict(unit_id=r["unit_id"], task=TNAME[tid], tid=tid, level=level_of(r), run=run, kind=kind, mirror=mirror,
             bank=bank, anchor=anchor_id, seed=seed, action_file=act[0] if act else None, action_key=act[1] if act else None,
             n_calls=-(-int(r.get("H_env") or H) // BLOCK), H_env=int(r.get("H_env") or H),
             recorded_success=rc.get("success"), recorded_first_success_step=rc.get("first_success_step"),
             recorded_machine=r.get("machine"), recorded_checkpoint=r.get("checkpoint"),
             setting=None, base_unit=None, ratio=None, t2_anchor_res=None, reset_res=None, perturbation=None,
             trajectory_file=r.get("trajectory_file"), recoverable=act is not None, skip_reason=None)
    d.update(kw)
    return d


def all_units():
    U = []
    base = {r["unit_id"]: r for r in rows("openwam_base/records_t*.jsonl")}
    for fam in ("base", "famA", "famP", "famO", "famB"):
        for r in rows(f"openwam_{fam}/records_t*.jsonl"):
            tid = r["task_id"]
            bk, aid = (FAM_BANK[fam], r["anchor_id"]) if fam in FAM_BANK else (MAIN, r["s0_id"])
            if fam == "famB":
                kind, mir, kw = "famB", "bfam/ow_famB.py:closed_b", dict(setting=r["setting_id"], reset_res=128)
            elif fam == "famO":
                kind, mir, kw = "occ0", "phys/ow_famO.py:closed_occ", dict(ratio=RATIOS[r["setting_id"]])
            else:
                kind, mir, kw = "plain", "openwam/ow_runner.py:closed_loop", {}
            U.append(_u(r, f"openwam_{fam}", tid, kind, mir, bk, aid, r["imagination_seed"],
                        (f"{ROOT}/{r['action_file']}", "closed_loop"), **kw))
    for r in rows("openwam_phys/closed_t*.jsonl"):
        b = base[r["base_unit"]]; tid = b["task_id"]
        U.append(_u(r, "openwam_phys", tid, "phys", "phys/phys_closed_openwam.py:closed", MAIN, b["s0_id"], b["imagination_seed"],
                    (f"{ROOT}/{r['closed_loop_file']}", "commanded"), setting=r["setting_id"], base_unit=r["base_unit"],
                    perturbation=r.get("perturbation"), trajectory_file=None))
    for r in rows("openwam_famF/closed_t[35].*.jsonl") + rows("openwam_famF/closed_t2.*.jsonl"):
        tid = int(r["task"][1:])
        bk, aid = (MAIN, r["anchor_id"]) if tid in (5, 3) else (tx_bank(tid, "base"), r["anchor_id"])
        U.append(_u(r, "openwam_famF", tid, "famF", "ffam/ow_famF.py:closed_f", bk, aid, r["imagination_seed"],
                    (f"{ROOT}/{r['closed_loop_file']}", "commanded"), setting=r["setting_id"], base_unit=r["base_unit"],
                    t2_anchor_res=256 if tid == 2 else None, trajectory_file=None))
    tid, R = 2, "openwam_t2"
    for fam in ("base", "famA", "famP", "famO"):
        for r in rows(f"{R}/records_{fam}.jsonl"):
            U.append(_u(r, R, tid, "occ0" if fam == "famO" else "plain", "out/task2/ow_tx_run.py -> phys/ow_famO.py:closed_occ",
                        tx_bank(tid, fam), r["anchor_id"], r["imagination_seed"],
                        (f"{RUNS}/{R}/actions/{r['unit_id']}.npz", "closed_loop"),
                        ratio=RATIOS.get(r["setting_id"]) if fam == "famO" else None, t2_anchor_res=256))
    for r in rows(f"{R}/famB/records*.jsonl"):
        b0 = r["setting_id"] == "B0"
        U.append(_u(r, f"{R}/famB", tid, "plain" if b0 else "famB",
                    "bfam/ow_tx_famB.py -> ow_runner.closed_loop(dict(a, resolution=128)) [sim_server renders 256]" if b0
                    else "bfam/ow_tx_famB.py -> ow_famB.closed_b", tx_bank(tid, "base"), r["anchor_id"], r["imagination_seed"],
                    (f"{RUNS}/{R}/famB/actions/{r['unit_id']}.npz", "closed_loop"),
                    setting=r["setting_id"], t2_anchor_res=128 if b0 else None, reset_res=None if b0 else 128))
    baseT = {r["unit_id"]: r for r in rows(f"{R}/records_base.jsonl")}
    for r in rows(f"{R}/phys/closed.*.jsonl"):
        b = baseT[r["base_unit"]]
        npz = f"{RUNS}/{R}/closed/{r['unit_id']}.npz"
        rer = f"{RUNS}/{R}/rerun_closed/phys/{r['unit_id']}.npz"
        act = (npz, "commanded") if os.path.exists(npz) else ((rer, "commanded") if os.path.exists(rer) else None)
        U.append(_u(r, R, tid, "phys", "phys/ow_phys_closed_tx.py -> phys_closed_openwam.closed", tx_bank(tid, "base"),
                    b["anchor_id"], b["imagination_seed"], act, setting=r["setting_id"], base_unit=r["base_unit"],
                    perturbation=r.get("perturbation"), t2_anchor_res=256,
                    action_note=None if act is None or act[0] == npz else "rerun_closed (ow_rerun.py) trajectory, not the recorded one",
                    skip_reason=None if act else "no closed-loop actions recorded (openwam_t2/closed/ never written); needs full rerun (ow_rerun.py)"))
    seen, out = set(), []
    for u in U:
        k = (u["run"], u["unit_id"])
        if k not in seen:
            seen.add(k); out.append(u)
    return out


def cmd_list():
    L = main_levels(); S = saved_ids()
    U = [u for u in all_units() if (u["task"], u["level"]) in L and u["unit_id"] not in S]
    for u in U:
        u["family"] = L[(u["task"], u["level"])]
    json.dump(U, open(UNITS_JSON, "w"), indent=0)
    from collections import Counter
    rec = [u for u in U if u["recoverable"]]
    print("units without saved imagined frames:", len(U), "recoverable:", len(rec))
    print("per task (recoverable):", sorted(Counter(u["task"] for u in rec).items()))
    print("per task (NOT recoverable):", sorted(Counter(u["task"] for u in U if not u["recoverable"]).items()))
    print("per task x family (recoverable):", sorted(Counter((u["task"], u["family"]) for u in rec).items()))
    allu = [u for u in all_units() if (u["task"], u["level"]) in L]
    cnt = Counter((u["task"], u["level"]) for u in allu)
    bad = {k: cnt.get(k, 0) for k in L if cnt.get(k, 0) != 50}
    print("main-table cells with != 50 units found:", bad or "none")
    print("->", UNITS_JSON)


def cmd_pilot():
    import numpy as np
    if not os.path.exists(UNITS_JSON):
        cmd_list()
    U = sorted([u for u in json.load(open(UNITS_JSON)) if u["recoverable"]], key=lambda u: (u["run"], u["unit_id"]))
    rng = np.random.RandomState(zlib.crc32(PILOT_SEED.encode()))
    cells = sorted({(u["task"], u["family"]) for u in U})
    pick = []
    for c in cells:                                     # 每个 (任务, 族) 一个
        js = [u for u in U if (u["task"], u["family"]) == c]
        pick.append(js[int(rng.choice(len(js)))])
    # 余额：先补 T2 B0（sim_server 实际 256 px 渲染、GH200 原机器 —— 最可能不过闸），其余随机
    b0 = [u for u in U if u["task"] == "T2" and u["level"] == "B0"]
    if b0 and not any(p["level"] == "B0" and p["task"] == "T2" for p in pick) and len(pick) < PILOT_N:
        pick.append(b0[int(rng.choice(len(b0)))])
    rest = [u for u in U if u not in pick]
    while len(pick) < PILOT_N and rest:
        pick.append(rest.pop(int(rng.choice(len(rest)))))
    json.dump(pick[:PILOT_N], open(PILOT_JSON, "w"), indent=0)
    for u in pick[:PILOT_N]:
        print(u["task"], u["family"], u["level"], u["run"], u["unit_id"], u["recorded_machine"])
    print("cells", len(cells), "->", PILOT_JSON)


# ------------------------------------------------------------------------------------------------ GPU part
def _check_mapping(units):
    """与 replay_frames.jobs() 的锚点 / 动作文件映射逐单元比对（phys 的动作键不同：那边用 executed 重放，这里用 commanded）。"""
    import replay_frames as RF
    J = {(j["run"], j["uid"]): j for j in RF.jobs() if j["model"] == "openwam"}
    out = {}
    for u in units:
        j = J.get((u["run"], u["unit_id"]))
        out[(u["run"], u["unit_id"])] = None if j is None else bool(
            j["bank"] == u["bank"] and j["anchor_id"] == u["anchor"] and os.path.abspath(j["act"][0]) == os.path.abspath(u["action_file"]))
    return out


def lastcall(W, eng, sim, corr, u, A):
    import numpy as np
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    from phys_perturb import spec_of
    from ow_famB import reset128, corrupt
    from ow_rerun import PROMPT
    from libero_lang import LANG
    prompt = LANG[u["tid"]] if u["tid"] in (5, 3) else PROMPT[u["tid"]]
    rec_cmd = np.load(u["action_file"])[u["action_key"]]
    n_calls = -(-len(rec_cmd) // W.BLOCK); last = n_calls - 1; t_last = last * W.BLOCK
    anchor = dict(A[u["anchor"]])
    if u["t2_anchor_res"]:
        anchor = dict(anchor, resolution=u["t2_anchor_res"])
    spec = None
    # ---- reset exactly as the producing runner
    if u["kind"] == "phys":
        spec = spec_of(u["setting"], u["tid"], u["base_unit"])
        f = f"{sim.scratch}/obs.npz"; sim.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, perturb=spec); o = dict(np.load(f))
    elif u["kind"] == "famB":
        o = reset128(sim, anchor)
    else:
        o = sim.reset(anchor)
    # ---- recorded commanded actions, same chunk boundaries (one sim.step per planning call)
    ex, succ, S = [], [], []
    for c in range(last):
        r = sim.step(list(rec_cmd[c * W.BLOCK:(c + 1) * W.BLOCK])); o = r
        ex += list(r["executed"]); succ += r["success"].tolist(); S.append(r["states"])
    # ---- the last planning call, same input path as the family's runner
    av_raw, wr_raw = o["agentview_image"], o["robot0_eye_in_hand_image"]
    av_in = av_raw
    if u["kind"] == "occ0" and last == 0:
        from lingbot_famO import occlude
        av_in = occlude(av_raw, u["ratio"])
    elif u["kind"] == "famB" and u["setting"] != "B0":
        av_in = corrupt(o, corr, u["setting"], u["unit_id"], t_last, None)["agentview_image"]
    elif u["kind"] == "famF":
        import ffam_def as FD
        tn = FD.LEVELS[u["setting"]]
        if tn is not None and t_last >= tn and not any(c * W.BLOCK >= tn for c in range(last)):
            av_in = FD.blackout(av_raw)
    canvas = W.canvas_from_raw(av_in, wr_raw)
    proprio = W._eef10(o)
    out = eng.generate({"first_frame_image": [Image.fromarray(canvas)], "prompt": prompt, "proprio": proprio, "seed": u["seed"] + last})
    imagined = np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]])
    act10 = np.asarray(out["actions"], np.float32)[:, :10]
    A7 = [native_eef10_to_libero7d(x) for x in act10][: W.H - t_last]
    recomputed = np.stack(A7)
    window = np.asarray(rec_cmd[t_last:t_last + len(A7)])
    same_shape = recomputed.shape == np.asarray(rec_cmd[t_last:]).shape
    mad = float(np.max(np.abs(recomputed.astype(np.float64) - window.astype(np.float64)))) if same_shape else float("inf")
    exact = bool(same_shape and np.array_equal(recomputed.astype(window.dtype), window))
    gate = bool(mad <= GATE_TOL and len(act10) == W.BLOCK)    # 块边界假设 = 每次调用 BLOCK 个动作；不成立则不过闸
    # ---- finish the episode with the RECORDED last window (whole-trajectory reproduction check only)
    r = sim.step(list(rec_cmd[t_last:])); ex += list(r["executed"]); succ += r["success"].tolist(); S.append(r["states"])
    ex, succ, S = np.stack(ex), np.array(succ, bool), np.concatenate(S)
    chk = dict(replay_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None, replay_success=bool(succ.any()))
    srcz = np.load(u["action_file"])
    if "success" in srcz.files:
        chk["success_curve_identical"] = bool(np.array_equal(succ, np.asarray(srcz["success"], bool)))
    if "executed" in srcz.files:
        rex = np.asarray(srcz["executed"], np.float32)
        chk["executed_identical"] = bool(rex.shape == ex.shape and np.array_equal(ex.astype(np.float32), rex))
    if u.get("trajectory_file"):
        tz = np.load(f"{ROOT}/{u['trajectory_file']}")
        if "closed_loop_states" in tz.files:
            rs = tz["closed_loop_states"]
            chk["prefix_states_max_abs_diff"] = float(np.max(np.abs(S[:t_last] - rs[:t_last]))) if rs.shape == S.shape else None
            chk["states_identical"] = bool(rs.shape == S.shape and np.array_equal(S, rs))
        if "closed_loop_success" in tz.files:
            chk["success_curve_identical"] = bool(np.array_equal(succ, np.asarray(tz["closed_loop_success"], bool)))
    d = dict(imagined=imagined, last_call_index=np.array(last), last_call_env_step=np.array(t_last),
             read_agentview=np.asarray(av_raw, np.uint8), read_wrist=np.asarray(wr_raw, np.uint8),
             model_agentview=np.asarray(av_in, np.uint8), canvas=np.asarray(canvas, np.uint8), proprio=np.asarray(proprio),
             actions10=act10, recomputed_commanded=recomputed, recorded_commanded_window=window,
             max_abs_diff=np.array(mad), gate_pass=np.array(gate), exact_equal=np.array(exact),
             state_at_last_call=S[t_last - 1] if t_last else np.zeros(0), replay_success=succ, replay_executed=ex)
    meta = dict(last_call_index=last, last_call_env_step=t_last, n_calls=n_calls, n_model_actions=int(len(act10)),
                max_abs_diff=mad, exact_equal=exact, gate_pass=gate, read_px=int(np.asarray(av_raw).shape[0]),
                imagined_shape=list(imagined.shape), prompt=prompt,
                spec_matches_record=None if spec is None or u.get("perturbation") is None
                else json.loads(json.dumps(spec)) == u["perturbation"], **chk)
    return d, meta


def cmd_run(shard, nsh, units_file):
    import numpy as np
    _wmvp.bootstrap()
    import ow_runner as W
    from replay_frames import B
    assert (W.H, W.BLOCK) == (H, BLOCK), (W.H, W.BLOCK)
    U = [u for u in json.load(open(units_file)) if u["recoverable"]]
    mine = [u for u in U if zlib.crc32(f"{u['run']}/{u['unit_id']}".encode()) % nsh == shard
            and not os.path.exists(f"{RUNS}/{u['run']}/lastcall/{u['unit_id']}.npz")]
    print("OW_LASTCALL units", len(U), "this shard todo", len(mine), flush=True)
    if not mine:
        print("OW_LASTCALL_DONE shard", shard); return
    mp = _check_mapping(mine)
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    import torch
    gpu = torch.cuda.get_device_name(0)
    sim = W.Sim(f"/dev/shm/ow_lastcall_{os.getpid()}")
    corr = None
    if any(u["kind"] == "famB" and u["setting"] != "B0" for u in mine):
        from bcorrupt import Corruptor
        corr = Corruptor()
    limit = int(os.environ.get("WMV_LIMIT", "0")); n = 0
    for u in mine:
        if limit and n >= limit:
            break
        od = f"{RUNS}/{u['run']}/lastcall"
        t0 = time.time()
        try:
            d, meta = lastcall(W, eng, sim, corr, u, B(u["bank"]))
        except Exception as e:  # noqa: BLE001
            print("FAIL", u["unit_id"], repr(e), flush=True)
            row = dict(unit_id=u["unit_id"], run=u["run"], task=u["task"], family=u.get("family"), level=u["level"], error=repr(e),
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=gpu)
            with open(f"{RUNS}/{u['run']}/lastcall.{shard}.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            continue
        os.makedirs(od, exist_ok=True)
        tmp = f"{od}/{u['unit_id']}.tmp.npz"; np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{u['unit_id']}.npz")
        row = dict(unit_id=u["unit_id"], run=u["run"], task=u["task"], family=u.get("family"), level=u["level"], kind=u["kind"],
                   mirror=u["mirror"], seed=u["seed"], **meta,
                   recorded=dict(success=u["recorded_success"], first_success_step=u["recorded_first_success_step"],
                                 machine=u["recorded_machine"], checkpoint=u["recorded_checkpoint"]),
                   matches_recorded_first_success_step=meta["replay_first_success_step"] == u["recorded_first_success_step"],
                   mapping_matches_replay_frames=mp.get((u["run"], u["unit_id"])),
                   action_source=dict(file=os.path.relpath(u["action_file"], ROOT), key=u["action_key"], note=u.get("action_note")),
                   lastcall_file=f"libero/out/runs/{u['run']}/lastcall/{u['unit_id']}.npz", checkpoint=ckname,
                   checkpoint_matches_record=None if not u["recorded_checkpoint"] else ckname == u["recorded_checkpoint"],
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=gpu)
        with open(f"{RUNS}/{u['run']}/lastcall.{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n += 1
        print(u["unit_id"], "gate", row["gate_pass"], "diff", row["max_abs_diff"], "fs", meta["replay_first_success_step"],
              "rec", u["recorded_first_success_step"], row["seconds"], flush=True)
    if corr is not None:
        corr.close()
    sim.close()
    print("OW_LASTCALL_DONE shard", shard)


def main():
    a = sys.argv[1:]
    if a[0] == "list":
        cmd_list()
    elif a[0] == "pilot":
        cmd_pilot()
    elif a[0] == "run":
        uf = a[a.index("--units") + 1] if "--units" in a else UNITS_JSON
        cmd_run(int(a[1]), int(a[2]), uf)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
