"""扰动族 F（单次全帧遮挡）· OpenWAM-α 闭环，T0–T4（定义见 ffam_def.py；注册表 v2，2026-11-12）。

挂点：与 O 族（phys/ow_famO.closed_occ）同一位置——每次规划调用读入的原始 256 px 主视角 o["agentview_image"]，
在 ow_view.canvas_from_raw（旋转 180° + LANCZOS 缩放 + 拼 L 形画布）之前。O 族在第 0 次调用遮挡中心方块；
F 族在【第一次满足 32c ≥ t 的调用 c】把主视角整帧换成全 0 新数组（t=60/160/260 → 第 2/5/9 次调用 = 第 64/160/288 步）。
腕部 o["robot0_eye_in_hand_image"] 原对象原样传入；proprio 不动；生成的想象帧不碰。每次调用只读当前一帧，
没有视觉记忆：置黑只影响这一次调用产生的 32 步动作（之后的影响全部经由真实状态的改变传递）。
闭环循环与 ow_famO.closed_occ / ow_runner.closed_loop 相同（sim.reset 无物理扰动，H=400，13 次调用），F0（sham）走同一循环不置黑。
配对协议：每单元两遍（同种子），硬要求 = 两遍成败一致。成功判据 reach_anytime（调用全部跑满，不提前停）。
落盘（store-all）：commanded / executed、逐步 success、逐步全物理状态、每次调用读入的原始主视角 + 腕部、实际喂给模型的画布、全部想象帧。

  python ow_famF.py <shard> <nshards> [tids]      正式跑（openwam 解释器，GPU）
  python ow_famF.py sham [n]                      F0 与基线存档闭环动作逐位比对（T0/T1 前 n 个单元）
  python ow_famF.py hc                            单条 T0 健康检查 → out/f_family/healthcheck/openwam/
输出：out/runs/openwam_famF/closed_t<tid>.<shard>.jsonl、out/runs/openwam_famF/closed/<uid>.npz
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
import ffam_def as FD  # noqa: E402
import min_artifacts as MA  # noqa: E402  存档最低要求（protocol.yaml min_artifacts，2026-11-13）

RUNS = f"{ROOT}/libero/out/runs"
OUT = f"{RUNS}/openwam_famF"
HC = f"{ROOT}/libero/out/f_family/healthcheck/openwam"
TIDS = (5, 3, 2, 4, 6)
TNAME = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}
PROMPT_TX = {2: "turn on the stove and put the moka pot on it",
             4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
             6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}


def anchors(path):
    A = {}
    for a in json.load(open(path))["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        A[a["anchor_id"]] = a
    return A


def prompt_of(tid):
    if tid in (5, 3):
        from libero_lang import LANG
        return LANG[tid]
    return PROMPT_TX[tid]


def base_units(tid):
    """[(base_row, anchor, ref_closed_actions_path, key)]：T0/T1 = openwam_base；T2–T4 = openwam_t{tid}/records_base（同 ow_tx_run）。"""
    if tid in (5, 3):
        A = anchors(f"{ROOT}/libero/out/bank/libero_anchors.json")
        R = [json.loads(l) for l in open(f"{RUNS}/openwam_base/records_t{tid}.jsonl")]
        return [(r, A[r["s0_id"]], f"{ROOT}/{r['action_file']}") for r in R]
    A = anchors(f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json")
    R = [json.loads(l) for l in open(f"{RUNS}/openwam_t{tid}/records_base.jsonl")]
    return [(r, dict(A[r["anchor_id"]], resolution=256), f"{RUNS}/openwam_t{tid}/actions/{r['unit_id']}.npz") for r in R]


def closed_f(eng, sim, anchor, prompt, seed, t_nom):
    """ow_famO.closed_occ 的同一循环 + F 挂点。t_nom=None → sham。"""
    import ow_runner as W
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    o = sim.reset(anchor)
    cmd, ex, succ, S, imag, rav, rwr, canv = [], [], [], [], [], [], [], []
    applied, call_b, wr_same = None, None, True
    for c in range(W.N_CALLS):
        step = len(cmd)                                          # 本次调用读入的是第 step 步之后的观测（= 32c）
        av_raw, wr_raw = o["agentview_image"], o["robot0_eye_in_hand_image"]
        av_in = av_raw
        if t_nom is not None and applied is None and step >= t_nom:
            av_in = FD.blackout(av_raw); applied, call_b = step, c      # 进 canvas_from_raw（旋转/缩放/拼画布）之前置黑
        rav.append(np.asarray(av_raw, np.uint8).copy()); rwr.append(np.asarray(wr_raw, np.uint8).copy())
        canvas = W.canvas_from_raw(av_in, wr_raw)
        wr_same = wr_same and (o["robot0_eye_in_hand_image"] is wr_raw)
        canv.append(canvas)
        img = Image.fromarray(canvas)
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        imag.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(cmd)]
        r = sim.step(A); cmd += A; ex += list(r["executed"]); succ += r["success"].tolist(); S.append(r["states"]); o = r
    succ = np.array(succ)
    return dict(commanded=np.stack(cmd), executed=np.stack(ex), success=succ, states=np.concatenate(S),
                read_agentview=np.stack(rav), read_wrist=np.stack(rwr), model_canvas=np.stack(canv),
                imagined=np.concatenate(imag), imagined_call_lens=np.array([len(x) for x in imag]),
                blackout_step=applied, blackout_call=call_b, wrist_same_object=wr_same,
                ok=bool(succ.any()), first=int(np.argmax(succ)) + 1 if succ.any() else None)


def engine():
    import ow_runner as W
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_famF_{os.getpid()}")
    return eng, sim, ckname


def _gpu_name():
    import torch
    return torch.cuda.get_device_name(0)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    tids = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(TIDS)
    eng, sim, ckname = engine()
    os.makedirs(f"{OUT}/closed", exist_ok=True)
    for tid in tids:
        done = {json.loads(l)["unit_id"] for f in glob.glob(f"{OUT}/closed_t{tid}.*.jsonl") for l in open(f)}
        jobs = [(r, a, s) for r, a, _ in base_units(tid) for s in FD.SETTINGS
                if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
        print("OW_FAMF", f"t{tid}", "jobs", len(jobs), flush=True)
        path = f"{OUT}/closed_t{tid}.{shard}.jsonl"
        for r, a, s in jobs:
            uid = f"{r['unit_id']}__{s}"; t0 = time.time(); seed = r["imagination_seed"]
            c1 = closed_f(eng, sim, a, prompt_of(tid), seed, FD.LEVELS[s])
            c2 = closed_f(eng, sim, a, prompt_of(tid), seed, FD.LEVELS[s])
            n = min(len(c1["executed"]), len(c2["executed"]))
            dd = np.where(np.any(c1["executed"][:n] != c2["executed"][:n], axis=1))[0]
            div = int(dd[0]) + 1 if len(dd) else None
            d = {}
            for p, c in (("", c1), ("run2_", c2)):
                for k in ("commanded", "executed", "success", "states", "read_agentview", "read_wrist", "model_canvas",
                          "imagined", "imagined_call_lens"):
                    d[f"{p}{k}"] = c[k]
                d[f"{p}blackout_step"] = np.array(-1 if c["blackout_step"] is None else c["blackout_step"])
            tmp = f"{OUT}/closed/{uid}.tmp.npz"
            np.savez_compressed(tmp, **d); os.replace(tmp, f"{OUT}/closed/{uid}.npz")
            row = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, family="F", task=f"t{tid}", paper_task=TNAME[tid],
                       anchor_id=a["anchor_id"], imagination_seed=seed, H_env=FD.H, checkpoint=ckname,
                       perturbation=dict(FD.spec("openwam", s, c1["blackout_step"]), blackout_call=c1["blackout_call"]),
                       blackout_applied=c1["blackout_step"] is not None,
                       blackout_after_success=bool(c1["ok"] and c1["blackout_step"] is not None and c1["first"] <= c1["blackout_step"]),
                       closed_loop=dict(success=c1["ok"], first_success_step=c1["first"], steps=int(len(c1["success"])),
                                        deterministic_rerun_identical=div is None,
                                        rerun=dict(success=c2["ok"], first_success_step=c2["first"], blackout_step=c2["blackout_step"]),
                                        first_divergence_step=div, conclusion_consistent=c1["ok"] == c2["ok"]),
                       closed_loop_file=os.path.relpath(f"{OUT}/closed/{uid}.npz", ROOT),
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=_gpu_name(), min_artifacts=1)
            MA.assert_unit(f"{OUT}/closed/{uid}.npz", row)        # 缺任何一项抛错停跑，不写这一行
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(uid, c1["ok"], c2["ok"], "blackout", c1["blackout_step"], row["seconds"], flush=True)
    sim.close()
    print("OW_FAMF_DONE shard", shard)


def sham(n=5):
    eng, sim, _ = engine(); ok = []
    for tid in (5, 3):
        for r, a, refp in base_units(tid)[:n]:
            ref = np.load(refp)["closed_loop"]
            c = closed_f(eng, sim, a, prompt_of(tid), r["imagination_seed"], None)
            same = bool(c["commanded"].shape == ref.shape and np.array_equal(c["commanded"], ref))
            ok.append(same); print("F sham", r["unit_id"], same, c["commanded"].shape, ref.shape, flush=True)
    sim.close()
    print("F_SHAM", "PASS" if all(ok) else "FAIL", sum(ok), "/", len(ok))
    return all(ok)


if __name__ == "__main__":
    if sys.argv[1] == "sham":
        sys.exit(0 if sham(int(sys.argv[2]) if len(sys.argv) > 2 else 5) else 1)
    elif sys.argv[1] == "hc":
        import hc_openwam
        hc_openwam.main()
    else:
        main()
