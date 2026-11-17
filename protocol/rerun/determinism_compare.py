"""H100 vs x86 5090 同 seed 确定性核对（在 H100 上用 lerobot_libero 解释器跑；只读 npz）。

参照：集群 x86 5090 上已跑完的 npz（LingBot rerun_closed、OpenWAM x86_rerun，拷到 H100 的 ref/ 下）；
H100：同一单元、同一 seed 用 lb_rerun_h100.py / ow_rerun_h100.py 重跑（out-root = h100/determinism/runs）。
逐步比较：指令动作（第一次不一致步、最大差）、执行动作、states（OpenWAM 有；LingBot 参照没存 states，用逐步真实主视角画面代替）、
success 与首次成功步、想象（OpenWAM 每次调用的想象帧；LingBot 每块想象 latent）。
  python determinism_compare.py <determinism_units.json> <ref_dir> <h100_out_root> <report_prefix>
"""
import json
import os
import sys
import time

import numpy as np


def first_div(a, b):
    """1-based 第一个不一致的步（逐行比较）；完全一致 None；前缀一致但长度不同 = 短者长度 + 1。"""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    if n == 0:
        return None if len(a) == len(b) else 1
    x = np.where(np.any(a[:n].reshape(n, -1) != b[:n].reshape(n, -1), axis=1))[0]
    return int(x[0]) + 1 if len(x) else (None if len(a) == len(b) else n + 1)


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max()) if n else None


def fs(s):
    s = np.asarray(s, bool)
    return int(np.argmax(s)) + 1 if s.any() else None


def cmp_seq(a, b, name):
    d = first_div(a, b)
    return {f"{name}_identical": d is None, f"{name}_first_divergence_step": d, f"{name}_max_abs_diff": maxdiff(a, b),
            f"{name}_len": [int(len(a)), int(len(b))]}


def lingbot(e, ref, h):
    out = dict(model="lingbot", unit_id=e["unit_id"], tid=e["tid"], src=e["src"], level=e["level"], ref_machine=e["ref_row"]["machine"],
               ref_gpu=e["ref_row"].get("gpu"))
    for p, tag in (("", "pass1"), ("run2_", "pass2")):
        o = {}
        if f"{p}commanded" not in ref or f"{p}commanded" not in h:
            out[tag] = None; continue
        o.update(cmp_seq(h[f"{p}commanded"], ref[f"{p}commanded"], "commanded"))
        o.update(cmp_seq(h[f"{p}executed"], ref[f"{p}executed"], "executed"))
        o.update(cmp_seq(h[f"{p}agentview"], ref[f"{p}agentview"], "agentview_frames"))   # 参照没存 states，用逐步真实画面代替
        o.update(cmp_seq(h[f"{p}wrist"], ref[f"{p}wrist"], "wrist_frames"))
        sh, sr = h[f"{p}success"], ref[f"{p}success"]
        o.update(success_h100=bool(np.asarray(sh).any()), success_x86=bool(np.asarray(sr).any()), first_success_h100=fs(sh), first_success_x86=fs(sr))
        o["success_flipped"] = o["success_h100"] != o["success_x86"]
        nh, nr = int(h[f"{p}imag_n_chunks"]), int(ref[f"{p}imag_n_chunks"])
        ch = []
        for i in range(min(nh, nr)):
            a, b = h[f"{p}imag_latents_{i:02d}"], ref[f"{p}imag_latents_{i:02d}"]
            ch.append(dict(chunk=i, identical=bool(a.shape == b.shape and np.array_equal(a, b)),
                           max_abs_diff=float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max()) if a.shape == b.shape else None))
        firstc = next((c["chunk"] for c in ch if not c["identical"]), None)
        o.update(imag_chunks=[nh, nr], imag_first_differing_chunk=firstc, imag_all_identical=firstc is None and nh == nr,
                 imag_max_abs_diff_chunk0=ch[0]["max_abs_diff"] if ch else None,
                 imag_max_abs_diff_any=max((c["max_abs_diff"] or 0) for c in ch) if ch else None)
        out[tag] = o
    if "run2_commanded" in h:
        out["h100_pass1_vs_pass2_commanded_identical"] = first_div(h["commanded"], h["run2_commanded"]) is None
        out["h100_pass1_vs_pass2_states_identical"] = first_div(h["states"], h["run2_states"]) is None if "states" in h else None
    if "run2_commanded" in ref:
        out["x86_pass1_vs_pass2_commanded_identical"] = first_div(ref["commanded"], ref["run2_commanded"]) is None
    p1 = out.get("pass1") or {}
    out["fully_identical"] = bool(p1 and p1["commanded_identical"] and p1["executed_identical"] and p1["agentview_frames_identical"]
                                  and p1["imag_all_identical"] and not p1["success_flipped"] and p1["first_success_h100"] == p1["first_success_x86"])
    return out


def openwam(u, ref, h):
    out = dict(model="openwam", unit_id=u["unit_id"], run=u["run"], task=u["task"], family=u["family"], kind=u["kind"], level=u["level"],
               ref_machine=u["ref_row"]["machine"], ref_gpu=u["ref_row"].get("gpu"))
    out.update(cmp_seq(h["commanded"], ref["commanded"], "commanded"))
    out.update(cmp_seq(h["executed"], ref["executed"], "executed"))
    out.update(cmp_seq(h["states"], ref["states"], "states"))
    out.update(cmp_seq(h["actions10"], ref["actions10"], "actions10"))
    if "agentview_steps" in h and "agentview_steps" in ref:
        out.update(cmp_seq(h["agentview_steps"], ref["agentview_steps"], "agentview_frames"))
    out.update(success_h100=bool(h["success"].any()), success_x86=bool(ref["success"].any()), first_success_h100=fs(h["success"]),
               first_success_x86=fs(ref["success"]))
    out["success_flipped"] = out["success_h100"] != out["success_x86"]
    # 想象帧：按调用切分
    def calls(z):
        L = np.asarray(z["imagined_call_lens"]); st = np.concatenate([[0], np.cumsum(L)])
        return [z["imagined"][st[i]:st[i + 1]] for i in range(len(L))]
    ch, cr = calls(h), calls(ref)
    per = []
    for i in range(min(len(ch), len(cr))):
        a, b = ch[i], cr[i]
        same = a.shape == b.shape and np.array_equal(a, b)
        per.append(dict(call=i, identical=bool(same), max_abs_diff=int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) if a.shape == b.shape else None,
                        frac_pixels_diff=float((a != b).mean()) if a.shape == b.shape else None))
    firstc = next((c["call"] for c in per if not c["identical"]), None)
    out.update(n_calls=[len(ch), len(cr)], imag_first_differing_call=firstc, imag_all_identical=firstc is None and len(ch) == len(cr),
               imag_call0_max_abs_diff=per[0]["max_abs_diff"] if per else None, imag_call0_frac_pixels_diff=per[0]["frac_pixels_diff"] if per else None,
               imag_per_call=per)
    out["canvas_call0_identical"] = bool(np.array_equal(h["model_canvas"][0], ref["model_canvas"][0]))
    out["fully_identical"] = bool(out["commanded_identical"] and out["states_identical"] and out["imag_all_identical"] and not out["success_flipped"]
                                  and out["first_success_h100"] == out["first_success_x86"])
    return out


def main():
    D = json.load(open(sys.argv[1])); refdir, hroot, prefix = sys.argv[2], sys.argv[3], sys.argv[4]
    R = []
    for e in D["lingbot"]:
        hp = f"{hroot}/lingbot_t{e['tid']}/rerun_closed/{e['src']}/{e['unit_id']}.npz"
        rp = f"{refdir}/{os.path.basename(e['ref_npz'])}"
        if not (os.path.exists(hp) and os.path.exists(rp)):
            R.append(dict(model="lingbot", unit_id=e["unit_id"], missing=dict(h100=os.path.exists(hp), ref=os.path.exists(rp)))); continue
        R.append(lingbot(e, np.load(rp), np.load(hp)))
    for u in D["openwam"]:
        hp = f"{hroot}/{u['run']}/x86_rerun/{u['unit_id']}.npz"
        rp = f"{refdir}/{os.path.basename(u['ref_npz'])}"
        if not (os.path.exists(hp) and os.path.exists(rp)):
            R.append(dict(model="openwam", unit_id=u["unit_id"], missing=dict(h100=os.path.exists(hp), ref=os.path.exists(rp)))); continue
        R.append(openwam(u, np.load(rp), np.load(hp)))
    def summ(model):
        X = [r for r in R if r["model"] == model and "missing" not in r]
        p = (lambda r: r["pass1"]) if model == "lingbot" else (lambda r: r)
        X = [r for r in X if p(r)]
        divs = [p(r)["commanded_first_divergence_step"] for r in X if p(r)["commanded_first_divergence_step"] is not None]
        return dict(n=len(X), fully_identical=sum(r["fully_identical"] for r in X),
                    commanded_identical=sum(p(r)["commanded_identical"] for r in X),
                    commanded_first_divergence_steps=sorted(divs),
                    commanded_first_divergence_min=min(divs) if divs else None, commanded_first_divergence_median=float(np.median(divs)) if divs else None,
                    commanded_max_abs_diff_max=max((p(r)["commanded_max_abs_diff"] or 0) for r in X) if X else None,
                    success_flips=sum(p(r)["success_flipped"] for r in X),
                    first_success_step_changed=sum((p(r)["first_success_h100"] != p(r)["first_success_x86"]) for r in X),
                    imag_identical=sum(p(r)["imag_all_identical"] for r in X))
    S = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), h100="mhcl-h100-3 (NVIDIA H100 96GB, driver 560.35.05)",
             reference="cluster x86 RTX 5090 ($NODE_C/585/627)", lingbot=summ("lingbot"), openwam=summ("openwam"),
             missing=[r for r in R if "missing" in r])
    json.dump(dict(summary=S, units=R), open(f"{prefix}.json", "w"), indent=1)
    md = [f"# H100 确定性核对（{S['generated']}）", "", f"H100：{S['h100']}；参照：{S['reference']}。同单元、同 seed、同脚本（H100 版只改选单元/输出路径/多存数据）。", ""]
    for m in ("lingbot", "openwam"):
        s = S[m]
        md += [f"## {m}", "", f"- 单元数 {s['n']}；完全一致 {s['fully_identical']}；指令动作逐位一致 {s['commanded_identical']}；想象逐位一致 {s['imag_identical']}",
               f"- 指令动作第一次不一致的步（1 起）：{s['commanded_first_divergence_steps']}（最早 {s['commanded_first_divergence_min']}，中位 {s['commanded_first_divergence_median']}）",
               f"- 指令动作最大绝对差（各单元共同前缀内的最大值）：{s['commanded_max_abs_diff_max']}",
               f"- 成败翻转 {s['success_flips']}；首次成功步变化 {s['first_success_step_changed']}", ""]
        md += ["| 单元 | 族/档 | 指令动作首个不一致步 | 最大差 | states/画面首个不一致步 | 想象首个不一致块/调用 | x86 成败(步) | H100 成败(步) |", "|---|---|---|---|---|---|---|---|"]
        for r in R:
            if r["model"] != m or "missing" in r:
                continue
            p = r["pass1"] if m == "lingbot" else r
            if not p:
                continue
            st = p.get("states_first_divergence_step", p.get("agentview_frames_first_divergence_step"))
            im = p.get("imag_first_differing_chunk", p.get("imag_first_differing_call"))
            fam = f"{r.get('src', r.get('family'))}/{r['level']}"
            md.append(f"| {r['unit_id']} | {fam} | {p['commanded_first_divergence_step']} | {p['commanded_max_abs_diff']:.3g} | {st} | {im} | "
                      f"{p['success_x86']}({p['first_success_x86']}) | {p['success_h100']}({p['first_success_h100']}) |")
        md.append("")
        if m == "lingbot":
            md.append("LingBot 的 states 列 = 逐步真实主视角画面首个不一致步（x86 参照没存 states）。H100 两遍之间是否逐位一致："
                      + ", ".join(f"{r['unit_id']}={r.get('h100_pass1_vs_pass2_commanded_identical')}" for r in R if r["model"] == m and "missing" not in r))
            md.append("")
    open(f"{prefix}.md", "w").write("\n".join(md))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
