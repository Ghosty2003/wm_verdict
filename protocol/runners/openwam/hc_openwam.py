"""F 族健康检查 · OpenWAM-α，单条 T0（t5）单元 openwam_libero10_t5_a0__k1（与 LingBot 健康检查同锚点同种子序号）。
由 `ow_famF.py hc` 调用（openwam 解释器，GPU）。
跑：未挂钩原版闭环（ow_famO.closed_occ，ratio=None）、F0、F60 / F160 / F260（各一遍，F60 再跑第二遍查确定性）。
检查：
 (a) 只有目标调用的主视角被置黑：13 次调用里，喂给模型的画布 == canvas_from_raw(原始主视角, 原始腕部) 的调用恰好 12 次，
     唯一不同的是预期调用（第 2/5/9 次 = 第 64/160/288 步）；置黑步之前的动作与 F0 逐位相同；目标调用之前每次调用的画布与想象帧与 F0 逐位相同。
 (b) 置黑在模型预处理（旋转 180° + LANCZOS + 拼画布）之前：该调用原始主视角非零，画布主视角区 [:256] 全 0；
     画布腕部区与 canvas_from_raw(原始) 的腕部区逐位相同。
 (c) 腕部与想象帧未被碰：腕部区同上；目标调用之前的想象帧与 F0 逐位相同（挂点不写想象帧）。
 sham：F0 动作 == 未挂钩原版闭环 == 基线存档 closed_loop。
输出：out/f_family/healthcheck/openwam/{hc_openwam.json, evidence_*.npz, sheet_F60.png, sheet_F160.png, sheet_F260.png}
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import ffam_def as FD
import ow_famF as OF

TOP = 256


def sheet(rows, path, title, S=160):
    pad, lw = 4, 210
    ncol = max(len(r[1]) for r in rows)
    W = lw + ncol * (int(S * 320 / 384) + pad + 40); Hh = 24 + len(rows) * (S + 18)
    im = Image.new("RGB", (W, Hh), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.text((4, 4), title, fill=(0, 0, 0)); y = 24
    for label, imgs, caps in rows:
        d.text((4, y + S // 2 - 6), label, fill=(0, 0, 0)); x = lw
        for a, c in zip(imgs, caps):
            a = Image.fromarray(np.ascontiguousarray(a)); a = a.resize((int(S * a.width / a.height), S))
            im.paste(a, (x, y)); d.text((x, y + S + 2), c, fill=(0, 0, 0)); x += a.width + pad
        y += S + 18
    im.save(path)


def main():
    import ow_runner as W
    from ow_famO import closed_occ
    os.makedirs(OF.HC, exist_ok=True)
    tid = 5
    r, a, refp = next(x for x in OF.base_units(tid) if x[0]["unit_id"] == "openwam_libero10_t5_a0__k1")
    eng, sim, ck = OF.engine()
    prompt, seed = OF.prompt_of(tid), r["imagination_seed"]
    out = dict(unit=r["unit_id"], anchor=a["anchor_id"], seed=seed, machine=os.uname().nodename, checkpoint=ck, checks={})
    plain_a, plain_s, _ = closed_occ(eng, sim, a, prompt, seed, None)
    ref = np.load(refp)["closed_loop"]
    R = {s: OF.closed_f(eng, sim, a, prompt, seed, FD.LEVELS[s]) for s in ("F0", "F60", "F160", "F260")}
    R2 = OF.closed_f(eng, sim, a, prompt, seed, FD.LEVELS["F60"])
    sim.close()
    c0 = R["F0"]
    recon0 = [W.canvas_from_raw(c0["read_agentview"][i], c0["read_wrist"][i]) for i in range(len(c0["model_canvas"]))]
    out["checks"]["sham"] = dict(
        F0_vs_unhooked_closed_loop=bool(np.array_equal(c0["commanded"], np.stack(plain_a))),
        F0_vs_baseline_archive=bool(c0["commanded"].shape == ref.shape and np.array_equal(c0["commanded"], ref)),
        F0_canvas_equals_raw_canvas_all_calls=all(np.array_equal(c0["model_canvas"][i], recon0[i]) for i in range(len(recon0))),
        F0_blackout_step=c0["blackout_step"], F0_success=c0["ok"], F0_first_success_step=c0["first"],
        baseline_archive_machine_note="基线存档在别的机器/时间生成，逐位相同依赖机器；同进程未挂钩原版闭环的比对才是代码路径检查")
    ev = dict(F0_commanded=c0["commanded"], unhooked_commanded=np.stack(plain_a), baseline_archive=ref, F0_canvas=c0["model_canvas"])
    L0 = np.cumsum(c0["imagined_call_lens"])
    for s in ("F60", "F160", "F260"):
        c = R[s]; b, cb = c["blackout_step"], c["blackout_call"]
        recon = [W.canvas_from_raw(c["read_agentview"][i], c["read_wrist"][i]) for i in range(len(c["model_canvas"]))]
        differ = [i for i in range(len(recon)) if not np.array_equal(c["model_canvas"][i], recon[i])]
        n = min(len(c["commanded"]), len(c0["commanded"]))
        dd = np.where(np.any(c["commanded"][:n] != c0["commanded"][:n], axis=1))[0]
        first_diff = int(dd[0]) if len(dd) else None
        L = np.cumsum(c["imagined_call_lens"])
        im_before = bool(np.array_equal(c["imagined"][:L[cb - 1]], c0["imagined"][:L0[cb - 1]])) if cb else True
        can_before = bool(np.array_equal(c["model_canvas"][:cb], c0["model_canvas"][:cb]))
        cv = c["model_canvas"][cb]
        out["checks"][s] = dict(
            nominal_t=FD.LEVELS[s], expected_blackout_step=FD.EXPECTED["openwam"][s], actual_blackout_step=b, blackout_call=cb,
            matches_expected=b == FD.EXPECTED["openwam"][s],
            a_calls_where_model_input_differs_from_raw=differ, a_exactly_one_call=differ == [cb],
            a_actions_identical_before_blackout=bool(first_diff is None or first_diff >= b), a_first_action_diff_step_0based=first_diff,
            a_canvases_identical_to_sham_before_target=can_before, a_imagined_identical_to_sham_before_target=im_before,
            b_raw_agentview_mean_at_target=float(c["read_agentview"][cb].mean()),
            b_canvas_agent_slot_max=int(cv[:TOP].max()), b_canvas_agent_slot_all_zero=bool(cv[:TOP].max() == 0),
            b_sham_canvas_agent_slot_mean=float(c0["model_canvas"][cb][:TOP].mean()),
            c_canvas_wrist_slot_equals_raw_wrist_canvas=bool(np.array_equal(cv[TOP:], recon[cb][TOP:])),
            c_canvas_wrist_slot_equals_sham=bool(np.array_equal(cv[TOP:], c0["model_canvas"][cb][TOP:])),
            c_raw_wrist_equals_sham_at_target=bool(np.array_equal(c["read_wrist"][cb], c0["read_wrist"][cb])),
            success=c["ok"], first_success_step=c["first"], blackout_after_success=bool(c["ok"] and c["first"] <= b))
        ev.update({f"{s}_commanded": c["commanded"], f"{s}_canvas": c["model_canvas"], f"{s}_read_agentview": c["read_agentview"],
                   f"{s}_read_wrist": c["read_wrist"], f"{s}_imagined": c["imagined"], f"{s}_imagined_call_lens": c["imagined_call_lens"]})
        cs = [i for i in (cb - 1, cb, cb + 1) if 0 <= i < len(recon)]
        i0, i1 = (L[cb - 1] if cb else 0), L[cb]
        j0, j1 = (L0[cb - 1] if cb else 0), L0[cb]
        fi = np.linspace(i0, i1 - 1, 5).round().astype(int); fj = np.linspace(j0, j1 - 1, 5).round().astype(int)
        rows = [("env raw agentview 256", [c["read_agentview"][i] for i in cs], [f"call {i} (t={32 * i})" for i in cs]),
                ("model canvas, F", [c["model_canvas"][i] for i in cs], [f"call {i}" + (" BLACK" if i == cb else "") for i in cs]),
                ("model canvas, sham", [c0["model_canvas"][i] for i in cs], [f"call {i}" for i in cs]),
                ("env raw wrist", [c["read_wrist"][i] for i in cs], [f"call {i}" for i in cs]),
                (f"imagined call {cb}, F", [c["imagined"][k] for k in fi], [f"#{k - i0}" for k in fi]),
                (f"imagined call {cb}, sham", [c0["imagined"][k] for k in fj], [f"#{k - j0}" for k in fj])]
        sheet(rows, f"{OF.HC}/sheet_{s}.png",
              f"OpenWAM-a T0 {r['unit_id']}  {s}: nominal t={FD.LEVELS[s]}, blackout at call {cb} = step {b} (raw obs before rotate/resize/canvas)")
    c1 = R["F60"]
    out["checks"]["F60_pass2"] = dict(identical_actions=bool(np.array_equal(c1["commanded"], R2["commanded"])),
                                      blackout_step=R2["blackout_step"], success=R2["ok"], conclusion_consistent=c1["ok"] == R2["ok"])
    np.savez_compressed(f"{OF.HC}/evidence_{r['unit_id']}.npz", **ev)
    ok = out["checks"]["sham"]["F0_vs_unhooked_closed_loop"] and out["checks"]["sham"]["F0_canvas_equals_raw_canvas_all_calls"] and all(
        v["matches_expected"] and v["a_exactly_one_call"] and v["a_actions_identical_before_blackout"]
        and v["a_canvases_identical_to_sham_before_target"] and v["a_imagined_identical_to_sham_before_target"]
        and v["b_canvas_agent_slot_all_zero"] and v["c_canvas_wrist_slot_equals_raw_wrist_canvas"] and v["c_canvas_wrist_slot_equals_sham"]
        for k, v in out["checks"].items() if k.startswith("F") and k != "F60_pass2")
    out["PASS"] = bool(ok)
    json.dump(out, open(f"{OF.HC}/hc_openwam.json", "w"), ensure_ascii=False, indent=1, default=str)
    print("HC_OPENWAM", "PASS" if ok else "FAIL", json.dumps(out, ensure_ascii=False, default=str)[:3000], flush=True)
