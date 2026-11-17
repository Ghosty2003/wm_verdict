"""660 runner 冒烟核对（H100 上 lerobot_libero 解释器，只读）：对 smoke660 每个单元检查
  1) 扰动确实施加：O 首帧遮挡（模型输入中心方块全黑、方块外与真实帧相同）；B 每步都腐蚀（腐蚀帧数 = 步数，且帧确实被改）；
     K 推动发生在 push_after_step；N 动作流改写了执行动作；Em/Ef/G apply_physics 有改动记录；base/A/P 无扰动记录；
  2) 两遍逐位一致；3) 字段齐全；4) 成败与原记录是否一致（只记录）。
输出：out/h100/smoke660/smoke660_check.json + 打印表格。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import numpy as np

O = _wmvp.P("$WMV_ROOT/libero/out/h100/smoke660")
NEED = ["commanded", "executed", "agentview", "wrist", "success", "states", "imag_n_chunks", "imag_latents_00", "imag_decoded",
        "imag_decoded_chunk_start", "meta_json"]
rows = [json.loads(l) for f in glob.glob(f"{O}/runs/**/rerun_closed.h100_*.jsonl", recursive=True) for l in open(f) if l.strip()]
out = []
for r in sorted(rows, key=lambda r: (r["tid"], r["kind"], r["unit_id"])):
    z = np.load(f"{O}/runs/{r['run']}/rerun_closed/{r['unit_id']}.npz")   # 冒烟输出在 smoke660/runs（行里 frames_file 写的是生产路径）
    keys = set(z.files)
    missing = [k for k in NEED + ["run2_" + k for k in NEED if k != "meta_json"] if k not in keys]
    p1 = r["perturbation_applied"]["pass1"]; kind = r["kind"]; ok_pert, note = True, ""
    n = int(r["rerun"]["steps"])
    if kind == "famO":
        occ = z["pert_occluded_t0_model_input"]; raw = z["agentview"][0]; s = p1["side_px"]; o = (128 - s) // 2
        inside_black = bool((occ[o:o + s, o:o + s] == 0).all()); m = np.ones(occ.shape[:2], bool); m[o:o + s, o:o + s] = False
        outside_same = bool((occ[m] == raw[m]).all())
        ok_pert = inside_black and outside_same; note = f"side {s}px, 方块全黑 {inside_black}, 方块外与真实帧相同 {outside_same}"
    elif kind == "famB":
        mi = z["pert_model_input_agentview"]; stp = z["pert_corrupted_steps"]
        changed = int(sum(not np.array_equal(mi[i], z["agentview"][int(t)]) for i, t in enumerate(stp)))
        ok_pert = len(mi) == n and changed == len(mi); note = f"腐蚀帧 {len(mi)}/{n} 步，与真实帧不同 {changed}"
    elif kind == "phys":
        sp = p1["spec"]; k = sp["kind"]
        if k == "push":
            ok_pert = p1["push_steps"] == [sp["push_after_step"]] or (n <= sp["push_after_step"] and p1["push_steps"] == [])
            note = f"推动步 {p1['push_steps']}（应为 {sp['push_after_step']}，闭环步数 {n}）"
        elif k in ("drop1", "dropr", "delay"):
            ms = p1["action_stream_modified_steps"]; ok_pert = len(ms) > 0; note = f"动作流改写 {len(ms)} 步，前几步 {ms[:5]}"
        else:
            ok_pert = bool(p1["apply_physics"]); note = f"apply_physics: {json.dumps(p1['apply_physics'], ensure_ascii=False)[:160]}"
    else:
        ok_pert = p1 == {}; note = "无扰动"
    out.append(dict(unit_id=r["unit_id"], task=r["task"], kind=kind, setting=r["setting"], perturbation_ok=ok_pert, note=note,
                    run2_bit_identical=r["run2_bit_identical"], fields_missing=missing, steps=n,
                    h100=(r["rerun"]["success"], r["rerun"]["first_success_step"]),
                    recorded=(r["recorded_reference_only"]["success"], r["recorded_reference_only"]["first_success_step"]),
                    matches_recorded_success=r["matches_recorded_success"], seconds=r["seconds"], machine=r["machine"]))
json.dump(out, open(f"{O}/smoke660_check.json", "w"), indent=1, ensure_ascii=False)
for e in out:
    print(f"{e['task']} {e['kind']:<5} {str(e['setting']):<6} 扰动OK={e['perturbation_ok']} 两遍一致={e['run2_bit_identical']} 缺字段={e['fields_missing']} "
          f"H100={e['h100']} 原记录={e['recorded']} | {e['note']}")
print("ALL_OK", all(e["perturbation_ok"] and e["run2_bit_identical"] and not e["fields_missing"] for e in out), len(out))
