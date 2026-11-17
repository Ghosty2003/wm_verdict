"""OpenWAM-α 闭环 x86 全量重跑（2026-11-15 用户裁决）。

为什么：在 GH200（ARM）上跑出的原闭环在 x86 上无法逐位重放（轨迹在中途分叉），且很多单元原本没存想象帧。
裁决：对需要的单元在 x86 上把【整条闭环】重新跑一遍（每次规划调用都真正调用模型，与原 runner 相同），全部落盘，
并把重跑结果与原记录结果并排报告。不做"只补想象帧"的重算。

各族镜像的 runner（与 ow_lastcall.py 相同的映射，扩到 T3/T4 与全部 F；H = 400，每 32 步一次规划调用，
第 c 次调用 seed = 单元种子 + c，最后一块截到 400 - 32*12 = 16 步）：
  T0/T1 base / famA / famP   openwam/ow_runner.py:closed_loop        sim.reset(anchor) -> 256 px，读入原样
  T0/T1 famO                 phys/ow_famO.py:closed_occ              只遮第 0 次调用（中心方块，ratio × 原始边长）
  T0/T1 E/G/K/N              phys/phys_closed_openwam.py:closed      reset 带 perturb=spec_of(setting, tid, base_unit)
                                                                    （N 族动作流 / K 推动由 sim_server 在 step 内施加）
  T0/T1 famB（含 B0）        bfam/ow_famB.py:closed_b                reset128；每次调用读入的主视角 Corruptor(setting, uid, step)
                                                                    （B0：同一 128 px 路径，不腐蚀）；proprio 取未腐蚀观测
  T2–T4 base/famA/famP/famO  out/task2/ow_tx_run.py -> ow_famO.closed_occ（ratio 只作用于第 0 次调用；非 O 族 ratio=None）
  T2–T4 famB Bxx             bfam/ow_tx_famB.py -> ow_famB.closed_b
  T2–T4 famB B0              bfam/ow_tx_famB.py -> ow_runner.closed_loop(dict(a, resolution=128))
                             （Sim.reset 不传 resolution，sim_server 实际按默认 256 px 渲染 —— 原样镜像）
  T2–T4 E/G/K/N              phys/ow_phys_closed_tx.py -> phys_closed_openwam.closed（anc = dict(A[...], resolution=256)）
  F（T0–T4）                 ffam/ow_famF.py:closed_f                第一次满足 32c >= t 的调用把主视角整帧置黑（只这一次）
原 runner 每单元跑两遍（第二遍只用于一致性检查）；这里只跑第一遍（重跑结果与原记录第一遍比）。

逐步真实画面：原 sim_server 每块只回传最后一帧观测。这里用本文件的 `_simserver` 子命令（lerobot_libero 解释器）
作模拟器：reset / step / 动作流 / K 推动 / 物理状态与 openwam/sim_server.py 逐行相同（直接 import 它的 _make_env /
raw_obs / _phys），只在 step 里多一个 frames 开关：每执行一步后再调一次 raw_obs（强制渲染两路相机，纯读，不碰物理）。
WMV_STEP_FRAMES=0 关掉（npz 里 step_frames=False，之后可由 states / executed 在 x86 上重放渲染）。

  python3 ow_x86_rerun.py list                              -> ow_x86_rerun_units.json（纯标准库，登录节点可跑）
  python3 ow_x86_rerun.py pilot                             -> ow_x86_rerun_pilot.json（纯标准库）
  python  ow_x86_rerun.py run <shard> <nshards> [--units F] (openwam 解释器，GPU)
     -> runs/<run>/x86_rerun/<uid>.npz + runs/<run>/x86_rerun.<shard>.jsonl
  python3 ow_x86_rerun.py compare                           -> ow_x86_rerun_compare.json / .tsv（纯标准库）
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
MAIN_TSV = f"{ROOT}/libero/out/closed_main/主表_长表_20261115.tsv"
SAVED_JSON = f"{ROOT}/libero/out/optimism/units.json"
GH200_JSON = f"{ROOT}/libero/out/annotation/samecase_20261115/gh200_units.json"
PAGE_JSON = f"{ROOT}/libero/out/annotation/samecase_20261115/ow_full_rerun_units.json"
UNITS_JSON = f"{HERE}/ow_x86_rerun_units.json"
PILOT_JSON = f"{HERE}/ow_x86_rerun_pilot.json"
CMP_JSON = f"{HERE}/ow_x86_rerun_compare.json"
CMP_TSV = f"{HERE}/ow_x86_rerun_compare.tsv"
SUB = "x86_rerun"
TNAME = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}
TORDER = ["T0", "T1", "T2", "T3", "T4"]
H, BLOCK = 400, 32                                      # = ow_runner.H / BLOCK（run 时核对）
# 与 replay_frames.MAIN / FAM_BANK / tx_bank、ow_tx_run.RATIOS / PROMPT 相同（run 时交叉核对）
MAIN = "libero/out/bank/libero_anchors.json"
FAM_BANK = {"famA": "libero/out/bank/familyA/familyA_anchors.json", "famP": "libero/out/bank/familyP/familyP_anchors.json"}
RATIOS = {"O1": 1 / 16, "O2": 1 / 8, "O3": 3 / 16, "O4": 1 / 4, "O5": 5 / 16}
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}


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


def family_of_level(lv):
    if lv == "none":
        return "base"
    for p, f in (("Em", "E"), ("Ef", "E"), ("N", "N"), ("A", "A"), ("P", "P"), ("O", "O"), ("B", "B"), ("G", "G"),
                 ("K", "K"), ("F", "F")):
        if lv.startswith(p):
            return f
    return "?"


def h400(fs):
    """记录的首次成功步按 H=400 截断：成功 ⇔ first_success_step <= 400。"""
    return fs if (fs is not None and int(fs) <= H) else None


def is_gh200(machine, uid, gh):
    return bool((machine and "gh200" in str(machine).lower()) or uid in gh)


def main_levels():
    """{(T, level): family}：主长表 OpenWAM-α 行（T0–T4，NLw 已不在表里；防御性再排除一次）。"""
    L = {}
    lines = open(MAIN_TSV, encoding="utf-8").read().splitlines()
    hd = lines[0].split("\t")
    for line in lines[1:]:
        d = dict(zip(hd, line.split("\t")))
        if d["model"] == "OpenWAM-α" and d["task"] in TORDER and not d["level"].startswith("NLw"):
            L[(d["task"], d["level"])] = d["family"]
    return L


def _u(r, run, tid, kind, mirror, bank, anchor_id, seed, rec_act, rec_states, **kw):
    rc = r.get("closed_loop") or {}
    fs = rc.get("first_success_step")
    d = dict(unit_id=r["unit_id"], task=TNAME[tid], tid=tid, level=level_of(r), family=None, run=run, kind=kind, mirror=mirror,
             bank=bank, anchor=anchor_id, seed=int(seed), H_env=int(r.get("H_env") or H),
             setting=None, base_unit=None, ratio=None, anchor_res=None, reset_res=None, render_px=256,
             spec_recorded=r.get("perturbation_spec") if r.get("perturbation_spec") is not None else r.get("perturbation"),
             recorded_success=rc.get("success"), recorded_first_success_step=fs,
             recorded_first_success_step_h400=h400(fs), recorded_success_h400=h400(fs) is not None,
             recorded_success_inconsistent=bool(rc.get("success")) != (fs is not None),
             recorded_machine=r.get("machine"), recorded_checkpoint=r.get("checkpoint"),
             recorded_action_file=rec_act[0] if rec_act else None, recorded_action_key=rec_act[1] if rec_act else None,
             recorded_states_file=rec_states[0] if rec_states else None, recorded_states_key=rec_states[1] if rec_states else None)
    d.update(kw)
    return d


def all_units():
    """所有 OpenWAM-α 闭环单元（与 ow_lastcall.all_units 同一映射，扩到 T3/T4 与 T0–T4 全部 F）。"""
    U = []
    base = {r["unit_id"]: r for r in rows("openwam_base/records_t*.jsonl")}
    for fam in ("base", "famA", "famP", "famO", "famB"):
        for r in rows(f"openwam_{fam}/records_t*.jsonl"):
            tid = r["task_id"]
            bk, aid = (FAM_BANK[fam], r["anchor_id"]) if fam in FAM_BANK else (MAIN, r["s0_id"])
            if fam == "famB":
                kind, mir, kw = "famB", "bfam/ow_famB.py:closed_b", dict(setting=r["setting_id"], reset_res=128, render_px=128)
            elif fam == "famO":
                kind, mir, kw = "occ0", "phys/ow_famO.py:closed_occ", dict(ratio=RATIOS[r["setting_id"]], setting=r["setting_id"])
            else:
                kind, mir, kw = "plain", "openwam/ow_runner.py:closed_loop", {}
            U.append(_u(r, f"openwam_{fam}", tid, kind, mir, bk, aid, r["imagination_seed"],
                        (f"{ROOT}/{r['action_file']}", "closed_loop"),
                        (f"{ROOT}/{r['trajectory_file']}", "closed_loop_states") if r.get("trajectory_file") else None, **kw))
    for r in rows("openwam_phys/closed_t*.jsonl"):
        b = base[r["base_unit"]]; tid = b["task_id"]
        f = f"{ROOT}/{r['closed_loop_file']}"
        U.append(_u(r, "openwam_phys", tid, "phys", "phys/phys_closed_openwam.py:closed", MAIN, b["s0_id"], b["imagination_seed"],
                    (f, "commanded"), None, setting=r["setting_id"], base_unit=r["base_unit"]))
    for r in rows("openwam_famF/closed_t*.jsonl"):
        tid = int(r["task"][1:])
        bk = MAIN if tid in (5, 3) else tx_bank(tid, "base")
        f = f"{ROOT}/{r['closed_loop_file']}"
        U.append(_u(r, "openwam_famF", tid, "famF", "ffam/ow_famF.py:closed_f", bk, r["anchor_id"], r["imagination_seed"],
                    (f, "commanded"), (f, "states"), setting=r["setting_id"], base_unit=r["base_unit"],
                    anchor_res=256 if tid not in (5, 3) else None))
    for tid in (2, 4, 6):
        R = f"openwam_t{tid}"
        for fam in ("base", "famA", "famP", "famO"):
            for r in rows(f"{R}/records_{fam}.jsonl"):
                U.append(_u(r, R, tid, "occ0" if fam == "famO" else "plain", "out/task2/ow_tx_run.py -> phys/ow_famO.py:closed_occ",
                            tx_bank(tid, fam), r["anchor_id"], r["imagination_seed"],
                            (f"{RUNS}/{R}/actions/{r['unit_id']}.npz", "closed_loop"), None,
                            ratio=RATIOS.get(r["setting_id"]) if fam == "famO" else None,
                            setting=r["setting_id"] if fam == "famO" else None, anchor_res=256))
        for r in rows(f"{R}/famB/records*.jsonl"):
            b0 = r["setting_id"] == "B0"
            U.append(_u(r, f"{R}/famB", tid, "plain" if b0 else "famB",
                        "bfam/ow_tx_famB.py -> ow_runner.closed_loop(dict(a, resolution=128)) [sim_server renders 256]" if b0
                        else "bfam/ow_tx_famB.py -> ow_famB.closed_b", tx_bank(tid, "base"), r["anchor_id"], r["imagination_seed"],
                        (f"{RUNS}/{R}/famB/actions/{r['unit_id']}.npz", "closed_loop"), None,
                        setting=r["setting_id"], anchor_res=128 if b0 else None, reset_res=None if b0 else 128,
                        render_px=256 if b0 else 128))
        baseT = {r["unit_id"]: r for r in rows(f"{R}/records_base.jsonl")}
        for r in rows(f"{R}/phys/closed.*.jsonl"):
            b = baseT[r["base_unit"]]
            U.append(_u(r, R, tid, "phys", "phys/ow_phys_closed_tx.py -> phys_closed_openwam.closed", tx_bank(tid, "base"),
                        b["anchor_id"], b["imagination_seed"], (f"{RUNS}/{R}/closed/{r['unit_id']}.npz", "commanded"), None,
                        setting=r["setting_id"], base_unit=r["base_unit"], anchor_res=256))
    seen, out = {}, []
    for u in U:
        if u["unit_id"] in seen:
            raise SystemExit(f"duplicate unit_id across runs: {u['unit_id']} ({seen[u['unit_id']]} / {u['run']})")
        seen[u["unit_id"]] = u["run"]; out.append(u)
    return out


def _cross_check_lastcall(U):
    """与 ow_lastcall.all_units() 的同一 (run, unit) 逐字段比对（锚点 / 种子 / 库 / 族路径 / ratio / setting / 分辨率）。"""
    _wmvp.bootstrap()
    import ow_lastcall as LC
    ref = {(u["run"], u["unit_id"]): u for u in LC.all_units()}
    n, bad = 0, []
    for u in U:
        v = ref.get((u["run"], u["unit_id"]))
        if v is None:
            continue
        n += 1
        pairs = [("anchor", "anchor"), ("seed", "seed"), ("bank", "bank"), ("kind", "kind"), ("ratio", "ratio"),
                 ("reset_res", "reset_res"), ("anchor_res", "t2_anchor_res")]
        diff = [a for a, b in pairs if u[a] != v[b]]
        if u["kind"] in ("phys", "famB", "famF") and u["setting"] != v["setting"]:
            diff.append("setting")
        if diff:
            bad.append((u["unit_id"], diff))
    return n, bad


def cmd_list():
    from collections import Counter
    L = main_levels()
    S = {u["unit_id"] for u in json.load(open(SAVED_JSON)) if u.get("model") == "openwam"}
    gh = set(json.load(open(GH200_JSON)))
    page = json.load(open(PAGE_JSON))
    allu = all_units()
    by = {u["unit_id"]: u for u in allu}
    out = {}
    for u in allu:
        if (u["task"], u["level"]) not in L:
            continue
        x86_imag = u["unit_id"] in S and not is_gh200(u["recorded_machine"], u["unit_id"], gh)
        if not u["recorded_success_h400"] and not x86_imag:
            out[u["unit_id"]] = dict(u, reason=["fail_no_x86_imag"])
    missing_page = []
    for uid in page:
        if uid not in by:
            missing_page.append(uid); continue
        if uid in out:
            out[uid]["reason"].append("page")
        else:
            out[uid] = dict(by[uid], reason=["page"])
    U = sorted(out.values(), key=lambda u: (TORDER.index(u["task"]), u["run"], u["unit_id"]))
    for u in U:
        u["family"] = L.get((u["task"], u["level"])) or family_of_level(u["level"])
        u["in_main_table"] = (u["task"], u["level"]) in L
        u["recorded_machine_class"] = "gh200" if is_gh200(u["recorded_machine"], u["unit_id"], gh) else "x86"
        u["has_saved_imagined_frames"] = u["unit_id"] in S
        u["recorded_actions_saved"] = bool(u["recorded_action_file"] and os.path.exists(u["recorded_action_file"]))
        u["reason"] = "+".join(u["reason"])
        u["output_file"] = f"libero/out/runs/{u['run']}/{SUB}/{u['unit_id']}.npz"
    json.dump(U, open(UNITS_JSON, "w"), indent=0)
    # ---- report
    n_chk, bad = _cross_check_lastcall(U)
    cnt = Counter((u["task"], u["level"]) for u in allu if (u["task"], u["level"]) in L)
    off = {f"{k[0]}/{k[1]}": cnt.get(k, 0) for k in sorted(L) if cnt.get(k, 0) != 50}
    print("units:", len(U), "| reasons:", dict(Counter(u["reason"] for u in U)))
    print("page ids not found in any OpenWAM record:", len(missing_page), missing_page[:5])
    print("page units outside the main table levels:", Counter(f"{u['task']}/{u['level']}" for u in U if not u["in_main_table"]))
    print("recorded success/fs inconsistent:", sum(u["recorded_success_inconsistent"] for u in U),
          "| recorded fs > 400:", sum(1 for u in U if u["recorded_first_success_step"] and u["recorded_first_success_step"] > H))
    print("cross-check vs ow_lastcall.all_units:", n_chk, "overlapping units,", len(bad), "mismatches", bad[:5])
    print("main-table cells with != 50 units found:", off or "none")
    print()
    fams = ["base", "A", "P", "O", "B", "E", "G", "K", "N", "F"]
    reasons = ["fail_no_x86_imag", "page", "fail_no_x86_imag+page"]
    C = Counter((u["task"], u["family"], u["reason"]) for u in U)
    print("task × family × reason   (columns: " + " / ".join(reasons) + ")")
    print("task fam " + " ".join(f"{f:>12}" for f in fams))
    for t in TORDER:
        print(f"{t:<8} " + " ".join(f"{'/'.join(str(C.get((t, f, r), 0)) for r in reasons):>12}" for f in fams))
    print()
    print("per task: total | gh200 | x86 | no recorded actions")
    for t in TORDER:
        T = [u for u in U if u["task"] == t]
        print(t, len(T), sum(u["recorded_machine_class"] == "gh200" for u in T), sum(u["recorded_machine_class"] == "x86" for u in T),
              sum(not u["recorded_actions_saved"] for u in T))
    print("->", UNITS_JSON)


def cmd_pilot():
    """6 个单元，按 unit_id 排序取每个槽的第一个。清单里没有 T0 base/A（都已有 x86 想象帧或成功、且不在页面清单）时，
    从全部 T0 base/A 单元里取第一个（x86 原记录 + 有存动作 → 逐位复现检查：重跑 commanded 应与原记录完全一致），
    标 pilot_not_in_list=True。T3 B 优先取腐蚀档（走 Corruptor 路径；B0 走的是 plain 路径，T0 base/A 已覆盖）。"""
    if not os.path.exists(UNITS_JSON):
        cmd_list()
    U = sorted(json.load(open(UNITS_JSON)), key=lambda u: u["unit_id"])
    gh = set(json.load(open(GH200_JSON)))
    t0 = lambda u: u["task"] == "T0" and u["family"] in ("base", "A")
    want = [("T0 base-or-A", t0),
            ("T1 O", lambda u: u["task"] == "T1" and u["family"] == "O"),
            ("T2 E/G/K/N no saved actions", lambda u: u["task"] == "T2" and u["family"] in ("E", "G", "K", "N")
             and not u["recorded_actions_saved"]),
            ("T3 B gh200", lambda u: u["task"] == "T3" and u["family"] == "B" and u["recorded_machine_class"] == "gh200"
             and u["kind"] == "famB"),
            ("T4 K gh200", lambda u: u["task"] == "T4" and u["family"] == "K" and u["recorded_machine_class"] == "gh200"),
            ("T4 F", lambda u: u["task"] == "T4" and u["family"] == "F"),
            ("T4 NR (no T4 F)", lambda u: u["task"] == "T4" and u["level"].startswith("NR"))]
    pick = []
    for name, f in want:
        if name == "T4 NR (no T4 F)" and any(p["_pilot_slot"] == "T4 F" for p in pick):
            continue
        js = [u for u in U if f(u)]
        if js:
            pick.append(dict(js[0], _pilot_slot=name, pilot_not_in_list=False)); continue
        if name == "T0 base-or-A":
            L = main_levels()
            A = sorted([u for u in all_units() if (u["task"], u["level"]) in L], key=lambda u: u["unit_id"])
            A = [dict(u, family=L[(u["task"], u["level"])]) for u in A]
            A = [u for u in A if t0(u)]
            if A:
                u = A[0]
                u.update(_pilot_slot=name, pilot_not_in_list=True, reason="pilot_only", in_main_table=True,
                         recorded_machine_class="gh200" if is_gh200(u["recorded_machine"], u["unit_id"], gh) else "x86",
                         has_saved_imagined_frames=None, recorded_actions_saved=os.path.exists(u["recorded_action_file"]),
                         output_file=f"libero/out/runs/{u['run']}/{SUB}/{u['unit_id']}.npz")
                pick.append(u); print("slot", name, "empty in the list -> first T0 base/A unit overall (pilot only)"); continue
        print("no unit for slot", name)
    json.dump(pick, open(PILOT_JSON, "w"), indent=0)
    for u in pick:
        print(f"{u['_pilot_slot']:<28}", u["unit_id"], u["level"], u["kind"], u["recorded_machine"],
              "rec_fs", u["recorded_first_success_step"], "actions_saved", u["recorded_actions_saved"],
              "in_list", not u["pilot_not_in_list"])
    print("->", PILOT_JSON)


# ------------------------------------------------------------------------------------------------ simulator (lerobot_libero)
def cmd_simserver():
    """openwam/sim_server.py 的同一循环（import 它的 _make_env / raw_obs / _phys），step 多一个 frames 开关。"""
    _wmvp.bootstrap()
    import numpy as np
    import sim_server as SS
    _make_env, raw_obs, _phys = SS._make_env, SS.raw_obs, SS._phys
    env = u = None; stream = None; tstep = 0; pert = None
    for line in sys.stdin:
        c = json.loads(line)
        if c["cmd"] == "close":
            break
        if c["cmd"] in ("reset", "replay"):
            if env is not None:
                env.close()
            env = _make_env(dict(c["anchor"], resolution=c.get("resolution", 256)), H=c.get("H", 600)); u = env.envs[0]
            stream = None; tstep = 0; pert = c.get("perturb")
            if pert:
                from phys_perturb import apply_physics, ActionStream
                apply_physics(u, pert); stream = ActionStream(pert, c.get("H", 600))
        if c["cmd"] == "reset":
            o = raw_obs(u); np.savez(c["out"], phys=_phys(u), **o)
            print(json.dumps(dict(ok=True)), flush=True)
        elif c["cmd"] in ("step", "replay"):
            if stream is not None:
                from phys_perturb import maybe_push
            acts = np.load(c["actions"])["actions"]
            frames = bool(c.get("frames"))
            succ, S, EX, AV, WR = [], [], [], [], []
            for a in acts:
                e = stream(tstep, a) if stream is not None else np.asarray(a, np.float32); tstep += 1
                if stream is not None:
                    maybe_push(u, pert, tstep - 1)
                u.step(e); EX.append(e)
                succ.append(bool(u._env.check_success())); S.append(_phys(u))
                if frames:                                  # 纯读：强制渲染当前状态的两路相机，不碰物理
                    f = raw_obs(u); AV.append(f["agentview_image"]); WR.append(f["robot0_eye_in_hand_image"])
            o = raw_obs(u)
            extra = dict(agentview_steps=np.stack(AV), wrist_steps=np.stack(WR)) if frames else {}
            np.savez(c["out"], success=np.array(succ), states=np.stack(S), executed=np.stack(EX), **o, **extra)
            print(json.dumps(dict(ok=True, n=len(acts))), flush=True)
    if env is not None:
        env.close()


# ------------------------------------------------------------------------------------------------ GPU part
def _sim_class(W):
    import subprocess
    import numpy as np

    class XSim(W.Sim):
        def __init__(self, scratch):                       # 同 W.Sim，只换成本文件的 _simserver
            self.scratch = scratch; os.makedirs(scratch, exist_ok=True)
            self.p = subprocess.Popen([W.LIBERO_PY, "-u", os.path.abspath(__file__), "_simserver"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=open(f"{scratch}/sim_server.err", "a"), text=True, bufsize=1,
                                      env=dict(os.environ, MUJOCO_GL="egl"))

        def reset_x(self, anchor, **kw):
            f = f"{self.scratch}/obs.npz"; self.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, **kw); return dict(np.load(f))

        def step_x(self, acts, frames):
            a, f = f"{self.scratch}/acts.npz", f"{self.scratch}/step.npz"
            np.savez(a, actions=np.asarray(acts, np.float32)); self.call(cmd="step", actions=a, out=f, frames=frames)
            return dict(np.load(f))
    return XSim


def closed_full(W, eng, sim, corr, u, A, frames):
    """该族原 runner 的同一闭环（每次调用都调模型），全部落盘。"""
    import numpy as np
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    from phys_perturb import spec_of
    from ow_famB import corrupt
    from lingbot_famO import occlude
    import ffam_def as FD
    from libero_lang import LANG
    tid, kind = u["tid"], u["kind"]
    prompt = LANG[tid] if tid in (5, 3) else PROMPT[tid]
    anchor = dict(A[u["anchor"]])
    if u["anchor_res"]:
        anchor = dict(anchor, resolution=u["anchor_res"])   # 原 runner 的 dict(a, resolution=…)；sim_server 用命令里的 resolution
    spec = None
    if kind == "phys":
        spec = spec_of(u["setting"], tid, u["base_unit"])
        o = sim.reset_x(anchor, perturb=spec)
    elif kind == "famB":
        o = sim.reset_x(anchor, resolution=128)
        assert o["agentview_image"].shape == (128, 128, 3), o["agentview_image"].shape
    else:
        o = sim.reset_x(anchor)
    t_nom = FD.LEVELS[u["setting"]] if kind == "famF" else None
    init_state = o["phys"]
    cmd, ex, succ, S, imag, rav, rwr, canv, prop, a10, AV, WR = [], [], [], [], [], [], [], [], [], [], [], []
    applied, call_b = None, None
    for c in range(W.N_CALLS):
        step = len(cmd)
        av_raw, wr_raw = o["agentview_image"], o["robot0_eye_in_hand_image"]
        av_in = av_raw
        if kind == "occ0" and c == 0 and u["ratio"] is not None:
            av_in = occlude(av_raw, u["ratio"])
        elif kind == "famB":
            av_in = corrupt(o, corr, u["setting"], u["unit_id"], step, None)["agentview_image"]
        elif kind == "famF" and t_nom is not None and applied is None and step >= t_nom:
            av_in = FD.blackout(av_raw); applied, call_b = step, c
        rav.append(np.asarray(av_raw, np.uint8).copy()); rwr.append(np.asarray(wr_raw, np.uint8).copy())
        canvas = W.canvas_from_raw(av_in, wr_raw); canv.append(canvas)
        proprio = W._eef10(o); prop.append(np.asarray(proprio, np.float64))
        out = eng.generate({"first_frame_image": [Image.fromarray(canvas)], "prompt": prompt, "proprio": proprio, "seed": u["seed"] + c})
        imag.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A10 = np.asarray(out["actions"], np.float32)[:, :10]; a10.append(A10)
        Ac = [native_eef10_to_libero7d(x) for x in A10][: W.H - len(cmd)]
        r = sim.step_x(Ac, frames); cmd += Ac; ex += list(r["executed"]); succ += r["success"].tolist(); S.append(r["states"])
        if frames:
            AV.append(r["agentview_steps"]); WR.append(r["wrist_steps"])
        o = r
        if os.environ.get("WMV_STOP_ON_SUCCESS", "1") == "1" and any(succ):   # 2026-11-15 用户：成功了就可以停（停在成功所在的那一块末尾）
            break
    succ = np.array(succ, bool)
    d = dict(commanded=np.stack(cmd), executed=np.stack(ex), success=succ, states=np.concatenate(S), initial_state=init_state,
             read_agentview=np.stack(rav), read_wrist=np.stack(rwr), model_canvas=np.stack(canv), proprio=np.stack(prop),
             actions10=np.concatenate(a10), actions10_call_lens=np.array([len(x) for x in a10]),
             imagined=np.concatenate(imag), imagined_call_lens=np.array([len(x) for x in imag]),
             call_env_steps=np.arange(W.N_CALLS) * W.BLOCK, step_frames=np.array(bool(frames)),
             final_agentview=np.asarray(o["agentview_image"], np.uint8), final_wrist=np.asarray(o["robot0_eye_in_hand_image"], np.uint8))
    if frames:
        d["agentview_steps"] = np.concatenate(AV); d["wrist_steps"] = np.concatenate(WR)
    if kind == "famF":
        d["blackout_step"] = np.array(-1 if applied is None else applied); d["blackout_call"] = np.array(-1 if call_b is None else call_b)
    meta = dict(prompt=prompt, read_px=int(np.asarray(rav[0]).shape[0]), n_calls=W.N_CALLS, n_steps=int(len(succ)),
                imagined_shape=list(d["imagined"].shape), step_frames=bool(frames),
                spec_matches_record=None if spec is None or u.get("spec_recorded") is None
                else json.loads(json.dumps(spec)) == u["spec_recorded"],
                blackout_step=applied if kind == "famF" else None, blackout_call=call_b if kind == "famF" else None)
    return d, meta


def _first_diff(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    x = np.where(np.any(a[:n].reshape(n, -1) != b[:n].reshape(n, -1), axis=1))[0]
    return int(x[0]) + 1 if len(x) else (None if len(a) == len(b) else n + 1)


def _vs_recorded(u, d):
    """与原记录的动作 / 物理状态逐步比对（原记录有存时）。"""
    import numpy as np
    out = dict(recorded_actions_available=False, commanded_identical_to_recorded=None, first_commanded_divergence_step=None,
               recorded_states_available=False, states_identical_to_recorded=None, first_state_divergence_step=None)
    f, k = u.get("recorded_action_file"), u.get("recorded_action_key")
    if f and os.path.exists(f):
        z = np.load(f)
        if k in z.files:
            rc = np.asarray(z[k], np.float32)
            out.update(recorded_actions_available=True, first_commanded_divergence_step=_first_diff(d["commanded"].astype(np.float32), rc),
                       commanded_identical_to_recorded=bool(rc.shape == d["commanded"].shape
                                                            and np.array_equal(d["commanded"].astype(np.float32), rc)))
    f, k = u.get("recorded_states_file"), u.get("recorded_states_key")
    if f and os.path.exists(f):
        z = np.load(f)
        if k in z.files:
            rs = z[k]
            out.update(recorded_states_available=True, first_state_divergence_step=_first_diff(d["states"], rs),
                       states_identical_to_recorded=bool(rs.shape == d["states"].shape and np.array_equal(d["states"], rs)))
    return out


def cmd_run(shard, nsh, units_file):
    import numpy as np
    _wmvp.bootstrap()
    import ow_runner as W
    from replay_frames import B
    import replay_frames as RF
    import ow_tx_run as TX
    assert (W.H, W.BLOCK, W.N_CALLS) == (H, BLOCK, 13), (W.H, W.BLOCK, W.N_CALLS)
    assert (RF.MAIN, RF.FAM_BANK) == (MAIN, FAM_BANK) and TX.RATIOS == RATIOS and TX.PROMPT == PROMPT
    assert all(RF.tx_bank(t, f) == tx_bank(t, f) for t in (2, 4, 6) for f in ("base", "famA", "famP"))
    frames = os.environ.get("WMV_STEP_FRAMES", "1") != "0"
    U = json.load(open(units_file))
    mine = [u for u in U if zlib.crc32(u["unit_id"].encode()) % nsh == shard
            and not os.path.exists(f"{RUNS}/{u['run']}/{SUB}/{u['unit_id']}.npz")]
    print("OW_X86_RERUN units", len(U), "this shard todo", len(mine), "step_frames", frames, flush=True)
    if not mine:
        print("OW_X86_RERUN_DONE shard", shard); return
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    import torch
    gpu = torch.cuda.get_device_name(0)
    sim = _sim_class(W)(f"/dev/shm/ow_x86_rerun_{os.getpid()}")
    corr = None
    if any(u["kind"] == "famB" and u["setting"] != "B0" for u in mine):
        from bcorrupt import Corruptor
        corr = Corruptor()
    limit = int(os.environ.get("WMV_LIMIT", "0")); n = 0
    for u in mine:
        if limit and n >= limit:
            break
        od = f"{RUNS}/{u['run']}/{SUB}"; jl = f"{RUNS}/{u['run']}/{SUB}.{shard}.jsonl"
        t0 = time.time()
        try:
            d, meta = closed_full(W, eng, sim, corr, u, B(u["bank"]), frames)
        except Exception as e:  # noqa: BLE001   出错不写 npz（下次重跑会再做），只记一行错误
            print("FAIL", u["unit_id"], repr(e), flush=True)
            row = dict(unit_id=u["unit_id"], run=u["run"], task=u["task"], family=u["family"], level=u["level"], error=repr(e),
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=gpu, checkpoint=ckname)
            os.makedirs(os.path.dirname(jl), exist_ok=True)
            with open(jl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            try:                                            # 模拟器可能已坏：重启
                sim.close()
            except Exception:  # noqa: BLE001
                pass
            sim = _sim_class(W)(f"/dev/shm/ow_x86_rerun_{os.getpid()}")
            continue
        os.makedirs(od, exist_ok=True)
        tmp = f"{od}/{u['unit_id']}.tmp.npz"; np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{u['unit_id']}.npz")
        s = d["success"]; ok = bool(s.any()); fs = int(np.argmax(s)) + 1 if ok else None
        chk = _vs_recorded(u, d)
        row = dict(unit_id=u["unit_id"], run=u["run"], task=u["task"], level=u["level"], family=u["family"], kind=u["kind"],
                   mirror=u["mirror"], reason=u["reason"], anchor=u["anchor"], seed=u["seed"], setting=u["setting"],
                   rerun_success=ok, rerun_first_success_step=fs,
                   recorded_success_h400=u["recorded_success_h400"], recorded_first_success_step=u["recorded_first_success_step"],
                   recorded_first_success_step_h400=u["recorded_first_success_step_h400"],
                   outcome_matches_recorded=ok == u["recorded_success_h400"],
                   step_matches_recorded=fs == u["recorded_first_success_step_h400"],
                   recorded_machine=u["recorded_machine"], recorded_machine_class=u["recorded_machine_class"], **chk, **meta,
                   checkpoint=ckname, checkpoint_matches_record=None if not u["recorded_checkpoint"] else ckname == u["recorded_checkpoint"],
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=gpu, H_env=H,
                   files=dict(npz=f"libero/out/runs/{u['run']}/{SUB}/{u['unit_id']}.npz",
                              keys=sorted(d), npz_bytes=os.path.getsize(f"{od}/{u['unit_id']}.npz")))
        with open(jl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n += 1
        print(u["unit_id"], "rerun", ok, fs, "rec", u["recorded_success_h400"], u["recorded_first_success_step"],
              "cmd_identical", chk["commanded_identical_to_recorded"], "div", chk["first_commanded_divergence_step"],
              row["seconds"], flush=True)
    if corr is not None:
        corr.close()
    sim.close()
    print("OW_X86_RERUN_DONE shard", shard)


# ------------------------------------------------------------------------------------------------ compare (stdlib)
def cmd_compare():
    R = {}
    for f in sorted(glob.glob(f"{RUNS}/**/{SUB}.*.jsonl", recursive=True)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    R[r["unit_id"]] = r                     # 同一单元多行时取最后一行
    G = {}
    for r in R.values():
        for cls in (r["recorded_machine_class"], "all"):
            g = G.setdefault((r["task"], r["level"], cls), dict(task=r["task"], level=r["level"], family=r["family"], machine_class=cls,
                                                               n_rerun=0, rerun_success=0, recorded_success_h400=0,
                                                               outcome_mismatch=0, rerun_succ_recorded_fail=0,
                                                               rerun_fail_recorded_succ=0, step_mismatch=0,
                                                               commanded_identical=0, commanded_compared=0))
            g["n_rerun"] += 1; g["rerun_success"] += r["rerun_success"]; g["recorded_success_h400"] += r["recorded_success_h400"]
            g["outcome_mismatch"] += not r["outcome_matches_recorded"]; g["step_mismatch"] += not r["step_matches_recorded"]
            g["rerun_succ_recorded_fail"] += r["rerun_success"] and not r["recorded_success_h400"]
            g["rerun_fail_recorded_succ"] += (not r["rerun_success"]) and r["recorded_success_h400"]
            if r.get("commanded_identical_to_recorded") is not None:
                g["commanded_compared"] += 1; g["commanded_identical"] += r["commanded_identical_to_recorded"]
    lv = {(t, l): i for i, (t, l) in enumerate(main_levels())}
    out = sorted(G.values(), key=lambda g: (TORDER.index(g["task"]), lv.get((g["task"], g["level"]), 999), g["level"],
                                            ["gh200", "x86", "all"].index(g["machine_class"])))
    json.dump(dict(n_units=len(R), rows=out), open(CMP_JSON, "w"), indent=1, ensure_ascii=False)
    cols = list(out[0]) if out else []
    with open(CMP_TSV, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for g in out:
            fh.write("\t".join(str(g[c]) for c in cols) + "\n")
    tot = [g for g in out if g["machine_class"] != "all"]
    for cls in ("gh200", "x86"):
        T = [g for g in tot if g["machine_class"] == cls]
        print(cls, "units", sum(g["n_rerun"] for g in T), "rerun succ", sum(g["rerun_success"] for g in T),
              "recorded succ", sum(g["recorded_success_h400"] for g in T), "outcome mismatches", sum(g["outcome_mismatch"] for g in T))
    print(len(R), "units ->", CMP_JSON, CMP_TSV)


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "list":
        cmd_list()
    elif a[0] == "pilot":
        cmd_pilot()
    elif a[0] == "run":
        uf = a[a.index("--units") + 1] if "--units" in a else UNITS_JSON
        cmd_run(int(a[1]), int(a[2]), uf)
    elif a[0] == "compare":
        cmd_compare()
    elif a[0] == "_simserver":
        cmd_simserver()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
