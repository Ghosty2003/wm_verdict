"""F 族失败案例分析（用户 2026-11-14）：OpenWAM-α T1 F160（t3）与 T3 F60（t4）的全部闭环失败单元。只读现有存档，不跑模型、不跑新单元。
  prep  （lerobot_libero，节点上需 EGL）对每个失败单元：用存档的 commanded 动作在同一锚点上重放（不调模型），逐步渲染主视角真实帧；
        校验：逐步 full-physics state 与存档 states 逐位相同、每次调用步的渲染帧与存档 read_agentview 逐位相同，任一不符则该单元不出图。
        → out/f_family/cases/cache/<uid>.npz（real 每步主视角原始帧、imagined 置黑那次调用 9 帧 + 前一次调用 9 帧、校验结果）
        → out/f_family/cases/stats.json（逐单元：想象帧亮度、想象 vs 真实同步 PSNR，置黑调用与前一次调用对比）
  sheets  候选总览（每单元：置黑调用想象第 1/4/8 帧 | 同步真实帧 | 第 400 步），供挑典型
  figure <uid> ...  对比图（格式同乐观率案例图）：上排 = 置黑那次调用的想象帧条（j=0 为读入条件帧，已置黑），中排 = 同步真实帧条，
                    下排 = 之后的真实帧条（每 32 步一帧到第 400 步）
方向：原始渲染只做上下翻转为正视；OpenWAM 画布顶部 256 行 = 原始转 180°（横向拉伸到 320），左右翻回并缩放为正方形。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys

import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
OUT = f"{ROOT}/libero/out/f_family/cases"
CELLS = (("T1", 3, "F160"), ("T3", 4, "F60"))
H = 400


def failures():
    U = []
    for T, tid, lv in CELLS:
        for f in sorted(glob.glob(f"{RUNS}/openwam_famF/closed_t{tid}.*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r["setting_id"] == lv and not r["closed_loop"]["success"]:
                    U.append(dict(T=T, tid=tid, level=lv, uid=r["unit_id"], base_unit=r["base_unit"],
                                  b_step=r["perturbation"]["actual_blackout_step"], b_call=r["perturbation"]["blackout_call"],
                                  npz=f"{ROOT}/{r['closed_loop_file']}"))
    return sorted(U, key=lambda u: (u["T"], u["uid"]))


def up_real(x):
    return np.ascontiguousarray(np.asarray(x)[::-1])


def up_imag(c):
    from PIL import Image
    a = np.ascontiguousarray(np.asarray(c)[:256, ::-1])
    return np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR))


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def prep():
    _wmvp.bootstrap()
    import mujoco
    from libero_env_adapter import _make_env
    from ow_view import raw_obs
    from ow_famF import base_units
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    stats = []
    for u in failures():
        anc = {r["unit_id"]: a for r, a, _ in base_units(u["tid"])}[u["base_unit"]]
        z = np.load(u["npz"])
        acts, S0, rav = z["commanded"], z["states"], z["read_agentview"]
        env = _make_env(dict(anc, resolution=256), H=H + 100); e = env.envs[0]   # 同 sim_server：一律 256 px（锚点自带的 resolution 不用）
        m, d = e._env.sim.model._model, e._env.sim.data._data
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        real = [np.asarray(raw_obs(e)["agentview_image"], np.uint8).copy()]       # real[k] = 第 k 步执行后的画面（k=0 初始）
        state_ok, succ = True, []
        for k, a in enumerate(acts):
            e.step(np.asarray(a, np.float32)); succ.append(bool(e._env.check_success()))
            mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            state_ok = state_ok and bool(np.array_equal(st, S0[k]))
            real.append(np.asarray(raw_obs(e)["agentview_image"], np.uint8).copy())
        env.close()
        real = np.stack(real)
        frame_ok = all(np.array_equal(real[32 * c], rav[c]) for c in range(len(rav)))
        ok = state_ok and frame_ok and not any(succ)
        im, cl = z["imagined"], z["imagined_call_lens"]
        off = np.concatenate([[0], np.cumsum(cl)])
        b = u["b_call"]
        imb = np.stack([up_imag(x) for x in im[off[b]:off[b + 1]]])
        imp = np.stack([up_imag(x) for x in im[off[b - 1]:off[b]]])
        realU = np.stack([up_real(x) for x in real])
        sb = [32 * b + 4 * j for j in range(len(imb))]; sp = [32 * (b - 1) + 4 * j for j in range(len(imp))]
        row = dict(u, replay_state_identical=state_ok, replay_frames_identical=frame_ok, replay_fail=not any(succ), usable=ok,
                   imag_mean_blackout=[round(float(x.mean()), 1) for x in imb], imag_mean_prev=[round(float(x.mean()), 1) for x in imp],
                   psnr_blackout=[round(psnr(imb[j], realU[min(s, H)]), 2) for j, s in enumerate(sb)],
                   psnr_prev=[round(psnr(imp[j], realU[min(s, H)]), 2) for j, s in enumerate(sp)],
                   act_blackout=np.asarray(acts[32 * b:32 * b + 32]).round(4).tolist())
        stats.append(row)
        np.savez_compressed(f"{OUT}/cache/{u['uid']}.npz", real=realU, imag_blackout=imb, imag_prev=imp,
                            steps_blackout=np.array(sb), steps_prev=np.array(sp), commanded=acts, b_call=b, ok=ok)
        print("PREP", u["uid"], "state", state_ok, "frames", frame_ok, flush=True)
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
    print("PREP_DONE", len(stats), "usable", sum(r["usable"] for r in stats), flush=True)


def _tile(a, h=112):
    from PIL import Image
    return Image.fromarray(a).resize((h, h))


def sheets():
    from PIL import Image, ImageDraw
    os.makedirs(f"{OUT}/sheets", exist_ok=True)
    S = {r["uid"]: r for r in json.load(open(f"{OUT}/stats.json"))}
    idx = {}
    for T, _, lv in CELLS:
        us = [u for u in failures() if u["T"] == T]
        cw, ch = 6 * 114 + 10, 112 + 20
        I = Image.new("RGB", (cw, len(us) * ch), (255, 255, 255)); d = ImageDraw.Draw(I)
        for i, u in enumerate(us):
            c = np.load(f"{OUT}/cache/{u['uid']}.npz"); sb = c["steps_blackout"]; y = i * ch
            ims = [c["imag_blackout"][1], c["imag_blackout"][4], c["imag_blackout"][8],
                   c["real"][sb[4]], c["real"][min(sb[8], H)], c["real"][H]]
            for k, a in enumerate(ims):
                I.paste(_tile(a), (k * 114 + (10 if k >= 3 else 0), y + 18))
            r = S[u["uid"]]
            d.text((2, y + 2), f"#{i} {u['uid'].split('__')[0][-6:]}/{u['uid'].split('__')[1]}  imag j1/j4/j8 | real @{sb[4]}/@{min(sb[8], H)}/@400"
                   f"  usable={r['usable']}", fill=(0, 0, 0))
            idx[f"{T}#{i}"] = u["uid"]
        I.save(f"{OUT}/sheets/OW_{T}_{lv}.png")
    json.dump(idx, open(f"{OUT}/sheets/index.json", "w"), indent=0)
    print("SHEETS done")


def figure(uids):
    from PIL import Image, ImageDraw
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    M = {u["uid"]: u for u in failures()}
    for uid in uids:
        u = M[uid]; c = np.load(f"{OUT}/cache/{uid}.npz")
        assert bool(c["ok"]), f"{uid}: replay check failed, no figure"
        sb = [int(s) for s in c["steps_blackout"]]
        later = list(range(sb[-1] + 32, H + 1, 32)) or [H]
        if later[-1] != H:
            later.append(H)
        T = 128
        top = [_tile(x, T) for x in c["imag_blackout"]]
        mid = [_tile(c["real"][min(s, H)], T) for s in sb]
        bot = [_tile(c["real"][s], T) for s in later]
        n = max(len(top), len(bot))
        W = 130 + n * (T + 4); Ht = 36 + 3 * (T + 22) + 10
        NF = f"{OUT}/notes.json"
        note = json.load(open(NF)).get(uid, "") if os.path.exists(NF) else ""
        F = Image.new("RGB", (W, Ht), (255, 255, 255)); d = ImageDraw.Draw(F)
        d.text((6, 4), f"openwam {u['T']} {u['level']}  {uid}   blackout at step {u['b_step']} (call {u['b_call']})   real closed loop: FAIL (H=400)",
               fill=(0, 0, 0))
        d.text((6, 18), note, fill=(150, 30, 30))
        rows = [(top, sb, "imagined\n(blackout call;\n j=0 = input,\n blacked out)"), (mid, sb, "real\n(same steps)"),
                (bot, later, "real\n(after, every\n 32 steps)")]
        for ri, (ims, steps, lab) in enumerate(rows):
            y = 36 + ri * (T + 22); x = 130
            d.text((6, y + 40), lab, fill=(0, 0, 0))
            for j, im in enumerate(ims):
                F.paste(im, (x, y)); d.text((x + 2, y + T + 4), f"step {steps[j]}" + (" (input)" if ri == 0 and j == 0 else ""), fill=(80, 80, 80))
                x += T + 4
            if ri == 0:
                d.rectangle([130 - 2, y - 2, 130 + T + 1, y + T + 1], outline=(0, 0, 0), width=3)
        F.save(f"{OUT}/figures/{uid}.png")
        print("FIG", uid)


if __name__ == "__main__":
    {"prep": lambda: prep(), "sheets": lambda: sheets(), "figure": lambda: figure(sys.argv[2:]),
     "list": lambda: print(json.dumps(failures(), indent=0)[:3000])}[sys.argv[1]]()
