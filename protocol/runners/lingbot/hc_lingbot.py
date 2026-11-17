"""F 族健康检查 · LingBot-VA，单条 T0（t5）单元 lingbot_libero10_t5_a0__k1__none（基线闭环失败、跑满，三个置黑点都会到达）。由 `lingbot_famF.py hc` 调用（lerobot_libero，GPU）。
跑：未挂钩原版闭环（LingBotRig.closed_loop）、F0（sham，同挂点不置黑）、F60 / F160 / F260（各一遍，F60 再跑第二遍查确定性）。
检查：
 (a) 只有目标规划调用那一步的主视角被置黑：逐步比对"实际喂给模型的主视角"与"环境原始主视角"，不同的步恰好 1 个且等于预期步；
     置黑步之前的动作与 F0 逐位相同；模型侧读入（lb_capture）在目标块之前的每一块与 F0 逐位相同，目标块只有最后一帧主视角不同且全 0。
 (b) 置黑发生在模型预处理之前：原始观测非零，送进 obs_to_batch 的数组全 0，预处理后的模型侧张量也全 0；腕部张量与 F0 相同。
 (c) 腕部与想象帧未被碰：腕部是原对象原样传入；目标块之前的每块想象 latent 与 F0 逐位相同（挂点不写 latent）。
 sham：F0 动作 == 未挂钩原版闭环动作 == 基线存档 closed_loop[:400]（后者依赖机器，单列）。
输出：out/f_family/healthcheck/lingbot/{hc_lingbot.json, evidence_*.npz, sheet_F60.png, sheet_F160.png, sheet_F260.png}
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import ffam_def as FD
import lingbot_famF as LF

AV, WR = "observation.images.image", "observation.images.image2"


def tensor_img(x):
    """模型侧张量 [1,3,h,w] (float16, [0,1]) → HWC uint8。"""
    x = np.asarray(x, np.float32).reshape(-1, *np.asarray(x).shape[-3:])[0]
    return (np.clip(x.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


def sheet(rows, path, title):
    S, pad, lw = 128, 4, 200
    ncol = max(len(r[1]) for r in rows)
    W = lw + ncol * (S + pad); Hh = 24 + len(rows) * (S + 18)
    im = Image.new("RGB", (W, Hh), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.text((4, 4), title, fill=(0, 0, 0))
    y = 24
    for label, imgs, caps in rows:
        d.text((4, y + S // 2 - 6), label, fill=(0, 0, 0))
        for j, (a, c) in enumerate(zip(imgs, caps)):
            a = Image.fromarray(np.ascontiguousarray(a)).resize((int(S * a.shape[1] / a.shape[0]), S)) if a.shape[0] != S else Image.fromarray(np.ascontiguousarray(a))
            x = lw + j * (S + pad)
            im.paste(a.crop((0, 0, S, S)) if a.width > S else a, (x, y))
            d.text((x, y + S + 2), c, fill=(0, 0, 0))
        y += S + 18
    im.save(path)


def main():
    import torch
    import lb_capture
    lb_capture.install()
    os.makedirs(LF.HC, exist_ok=True)
    tid = 5
    # 选基线闭环失败、跑满 400 步的单元：三个置黑点（60/172/268）都会到达（成功单元在成功那步就结束了，晚档位不会触发）
    r, a = next((r, a) for r, a in LF.base_units(tid) if r["unit_id"] == "lingbot_libero10_t5_a0__k1__none")
    LF.set_H(tid)
    rig = LF.make_rig_class()(tid)
    seed = r["imagination_seed"]
    out = dict(unit=r["unit_id"], anchor=a["anchor_id"], seed=seed, machine=os.uname().nodename,
               gpu=torch.cuda.get_device_name(0), checks={})

    # 未挂钩原版闭环
    plain = rig.closed_loop(a, seed, np.zeros((600, 7), np.float32))["actions"][:FD.H]
    ref = np.load(f"{LF.ROOT}/{r['action_file']}")["closed_loop"][:FD.H]

    def run(setting):
        rec = LF.Recorder(); rig.rec = rec
        with lb_capture.Capture() as cap:
            c = LF.closed_f(rig, a, seed, FD.LEVELS[setting], keep_passed=True)
        rig.rec = None
        return c, rec, cap

    R = {s: run(s) for s in ("F0", "F60", "F160", "F260")}
    R2_F60 = run("F60")
    c0, rec0, cap0 = R["F0"]
    av0 = np.stack(rec0.av)
    out["checks"]["sham"] = dict(
        F0_vs_unhooked_closed_loop=bool(c0["actions"].shape == plain.shape and np.array_equal(c0["actions"], plain)),
        F0_vs_baseline_archive=bool(c0["actions"].shape == ref.shape and np.array_equal(c0["actions"], ref)),
        F0_steps=int(c0["steps"]), F0_success=c0["success"], F0_first_success_step=c0["first_success_step"],
        F0_blackout_step=c0["blackout_step"],
        F0_passed_equals_raw_all_steps=bool(np.array_equal(c0["passed_av"], av0[:len(c0["passed_av"])])),
        baseline_archive_machine_note="基线存档在别的机器/时间生成，逐位相同依赖机器；同进程未挂钩原版闭环的比对才是代码路径检查")
    ev = dict(F0_actions=c0["actions"], unhooked_actions=plain, baseline_archive_actions=ref, F0_passed_av=c0["passed_av"])
    decoded = {}
    for s in ("F60", "F160", "F260"):
        c, rec, cap = R[s]
        b = c["blackout_step"]; exp = FD.EXPECTED["lingbot"][s]
        av = np.stack(rec.av); passed = c["passed_av"]
        diff_steps = [int(t) for t in range(len(passed)) if not np.array_equal(passed[t], av[t])]
        n = min(len(c["actions"]), len(c0["actions"]))
        dd = np.where(np.any(c["actions"][:n] != c0["actions"][:n], axis=1))[0]
        first_act_diff = int(dd[0]) if len(dd) else None             # 0-based 步号
        idx = c["plan_steps"].index(b) if b in c["plan_steps"] else None
        # 模型侧读入（lb_capture）：块 0 = 初始观测；块 i>0 = 该次规划前缓冲的关键帧
        same_before = all(set(cap.real[i]) == set(cap0.real[i]) and all(np.array_equal(cap.real[i][k], cap0.real[i][k]) for k in cap.real[i])
                          for i in range(idx))
        lat_before = all(np.array_equal(cap.latents[i], cap0.latents[i]) for i in range(idx))
        tgt, tgt0 = cap.real[idx], cap0.real[idx]
        av_t, av_t0 = np.asarray(tgt[AV], np.float32), np.asarray(tgt0[AV], np.float32)
        wr_t, wr_t0 = np.asarray(tgt[WR], np.float32), np.asarray(tgt0[WR], np.float32)
        out["checks"][s] = dict(
            nominal_t=FD.LEVELS[s], expected_blackout_step=exp, actual_blackout_step=b, matches_expected=b == exp,
            blackout_is_planning_call=b in c["plan_steps"], planning_calls_first12=c["plan_steps"][:12],
            a_steps_where_model_input_differs_from_raw=diff_steps, a_exactly_one_step=diff_steps == [b],
            a_passed_frame_all_zero=bool(passed[b].max() == 0), a_raw_frame_at_step_mean=float(av[b].mean()),
            a_actions_identical_before_blackout=bool(first_act_diff is None or first_act_diff >= b),
            a_first_action_diff_step_0based=first_act_diff,
            a_model_reads_identical_in_all_chunks_before_target=bool(same_before), target_chunk_index=idx,
            a_target_chunk_frames=int(av_t.shape[0]),
            a_target_chunk_only_last_agentview_frame_differs=bool(np.array_equal(av_t[:-1], av_t0[:-1]) and not np.array_equal(av_t[-1], av_t0[-1])),
            b_model_side_agentview_tensor_last_frame_max=float(av_t[-1].max()), b_model_side_agentview_last_frame_all_zero=bool(av_t[-1].max() == 0),
            b_sham_model_side_agentview_last_frame_mean=float(av_t0[-1].mean()),
            c_wrist_same_object=c["wrist_same_object"], c_model_side_wrist_target_chunk_identical_to_sham=bool(np.array_equal(wr_t, wr_t0)),
            c_raw_wrist_identical_to_sham_up_to_blackout=bool(np.array_equal(np.stack(rec.wr)[:b + 1], np.stack(rec0.wr)[:b + 1])),
            c_imagined_latents_identical_in_chunks_before_target=bool(lat_before),
            success=c["success"], first_success_step=c["first_success_step"], steps=c["steps"])
        ev.update({f"{s}_actions": c["actions"], f"{s}_passed_av": passed, f"{s}_raw_av": av, f"{s}_raw_wr": np.stack(rec.wr),
                   f"{s}_target_chunk_model_av": av_t, f"{s}_target_chunk_model_wr": wr_t, f"sham_target_chunk_model_av_{s}": av_t0})
        # 解码目标块的想象（展示用）：F 与 sham 的第 idx 块
        try:
            for tag, cc in (("F", cap), ("sham", cap0)):
                z = torch.from_numpy(np.asarray(cc.latents[idx], np.float32))
                decoded[(s, tag)] = np.asarray(rig.decode(z))
        except Exception as e:  # noqa: BLE001
            out["checks"][s]["decode_error"] = repr(e)
        # 联系表
        ts = [t for t in range(b - 2, b + 3) if 0 <= t < len(passed)]
        rows = [("env raw agentview", [av[t] for t in ts], [f"t={t}" for t in ts]),
                ("fed to model (pre-proc)", [passed[t] for t in ts], [f"t={t}" + (" BLACK" if t == b else "") for t in ts]),
                ("model-side tensor, F", [tensor_img(av_t[j]) for j in range(-5, 0)], [f"clip[{j}]" for j in range(-5, 0)]),
                ("model-side tensor, sham", [tensor_img(av_t0[j]) for j in range(-5, 0)], [f"clip[{j}]" for j in range(-5, 0)]),
                ("wrist fed (untouched)", [np.stack(rec.wr)[t] for t in ts], [f"t={t}" for t in ts])]
        for tag in ("F", "sham"):
            if (s, tag) in decoded:
                fr = decoded[(s, tag)]; js = np.linspace(0, len(fr) - 1, 5).round().astype(int)
                rows.append((f"imagined chunk {idx}, {tag}", [fr[j][:, :128] for j in js], [f"f{j}" for j in js]))
        sheet(rows, f"{LF.HC}/sheet_{s}.png",
              f"LingBot-VA T0 {r['unit_id']}  {s}: nominal t={FD.LEVELS[s]}, blackout at planning call step {b} (chunk {idx})")
    c2 = R2_F60[0]; c1 = R["F60"][0]
    out["checks"]["F60_pass2"] = dict(identical_actions=bool(np.array_equal(c1["actions"], c2["actions"])),
                                      blackout_step=c2["blackout_step"], success=c2["success"],
                                      conclusion_consistent=c1["success"] == c2["success"])
    np.savez_compressed(f"{LF.HC}/evidence_{r['unit_id']}.npz", **ev)
    ok = out["checks"]["sham"]["F0_vs_unhooked_closed_loop"] and out["checks"]["sham"]["F0_passed_equals_raw_all_steps"] and all(
        v["matches_expected"] and v["a_exactly_one_step"] and v["a_passed_frame_all_zero"] and v["a_actions_identical_before_blackout"]
        and v["a_model_reads_identical_in_all_chunks_before_target"] and v["a_target_chunk_only_last_agentview_frame_differs"]
        and v["b_model_side_agentview_last_frame_all_zero"] and v["c_wrist_same_object"]
        and v["c_model_side_wrist_target_chunk_identical_to_sham"] and v["c_imagined_latents_identical_in_chunks_before_target"]
        for k, v in out["checks"].items() if k.startswith("F") and k != "F60_pass2")
    out["PASS"] = bool(ok)
    json.dump(out, open(f"{LF.HC}/hc_lingbot.json", "w"), ensure_ascii=False, indent=1, default=str)
    print("HC_LINGBOT", "PASS" if ok else "FAIL", json.dumps(out, ensure_ascii=False, default=str)[:3000], flush=True)
