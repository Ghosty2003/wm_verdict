"""合并清单（只读，按需运行；头节点 python3 纯标准库）。口径见 libero/out/h100/口径决定_20261116.md：
一个单元的“真实成败”和“想象帧”必须来自同一次运行；禁止“成败取集群、想象取 H100”。

对每个 LingBot（T2–T4 闭环重跑 + 网页 660：T0/T1 各族与 T3/T4 famB）/ OpenWAM（x86 全量闭环重跑）单元，给出唯一来源：
  * 集群 x86 重跑（runs/<run>/{rerun_closed|x86_rerun}，machine=$NODE_*）：成败 = 该次重跑的 rerun 字段，想象 = 同一 npz；
  * H100（out/h100/runs/<run>/...，machine=H100）：成败 = H100 本次运行的 rerun 字段，想象 = 同一 npz。
  两边都有（例如确定性核对单元不在生产目录里，生产目录理论上不重叠）：取集群 x86 重跑，另一份记为 alternate（不混用）。
  原记录成败（recorded）只作参考，不作为任何单元的成败来源。
检查：
  1. outcome_source != imag_source 的单元数（必须为 0）；
  2. 行里的成败 与 同一 npz 里逐步 success 曲线是否一致（npz 在 NAS 上才查；用 zipfile 直接读 success.npy，不需要 numpy）；
  3. npz 里是否真的有想象（LingBot: imag_latents_00 / imag_decoded；OpenWAM: imagined）。
输出：libero/out/h100/merge_manifest/units_source.jsonl、summary.json
  python3 merge_manifest.py [--no-npz-check]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import struct
import sys
import time
import zipfile
from collections import Counter

ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
H100 = f"{ROOT}/libero/out/h100/runs"
OUT = f"{ROOT}/libero/out/h100/merge_manifest"
CHECK = "--no-npz-check" not in sys.argv


def npz_success_any(path, key="success"):
    """不用 numpy：读 npz（zip）里的 <key>.npy（bool 一维），返回 (any, first_true_1based, len)。"""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        b = z.read(f"{key}.npy")
        hl = struct.unpack("<H", b[8:10])[0] if b[6] == 1 else struct.unpack("<I", b[8:12])[0]
        off = (10 if b[6] == 1 else 12) + hl
        data = b[off:]
        first = next((i + 1 for i, x in enumerate(data) if x), None)
        return first is not None, first, len(data), names


def lb_rows(base, pat):
    R = {}
    for f in sorted(glob.glob(f"{base}/lingbot_t*/rerun_closed/{pat}")):
        tid = int(f.split("/lingbot_t")[1].split("/")[0])
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                R[(tid, r["src"], r["unit_id"])] = r
    return R


def ow_rows(base, pat):
    R = {}
    for f in sorted(glob.glob(f"{base}/**/{pat}", recursive=True)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    R[(r["run"], r["unit_id"])] = r
    return R


def main():
    os.makedirs(OUT, exist_ok=True)
    out = []
    # ---------------- LingBot
    cl = lb_rows(RUNS, "*.jsonl"); h1 = lb_rows(H100, "*.h100_*.jsonl")
    for k in sorted(set(cl) | set(h1)):
        tid, src, uid = k
        use, alt = ("cluster_x86", cl[k]) if k in cl else ("H100", h1[k]), ("H100" if (k in cl and k in h1) else None)
        r = use[1]
        npz = (f"{RUNS}/lingbot_t{tid}/rerun_closed/{src}/{uid}.npz" if use[0] == "cluster_x86"
               else f"{H100}/lingbot_t{tid}/rerun_closed/{src}/{uid}.npz")
        e = dict(model="lingbot", unit_id=uid, tid=tid, src=src, level=r.get("level"), source=use[0], machine=r.get("machine"), gpu=r.get("gpu"),
                 success=r["rerun"]["success"], first_success_step=r["rerun"]["first_success_step"],
                 outcome_source=f"{use[0]}:{os.path.relpath(npz, ROOT)}#pass1", imag_source=f"{use[0]}:{os.path.relpath(npz, ROOT)}#pass1",
                 imag_keys="imag_latents_XX" + (" + imag_decoded" if use[0] == "H100" else ""),
                 recorded_success_reference_only=r["recorded"]["success"], alternate_run=alt)
        out.append((e, npz, "success", ("imag_latents_00.npy",)))
    # ---------------- LingBot 网页 660（T0/T1 各族、T3/T4 famB；只有 H100 跑过这套全量闭环）
    for f in sorted(glob.glob(f"{H100}/**/rerun_closed.h100_*.jsonl", recursive=True)):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            npz = f"{ROOT}/{r['frames_file']}"
            e = dict(model="lingbot", unit_id=r["unit_id"], tid=r["tid"], src=r["kind"], run=r["run"], level=r.get("level"), source="H100",
                     machine=r.get("machine"), gpu=r.get("gpu"), success=r["rerun"]["success"], first_success_step=r["rerun"]["first_success_step"],
                     outcome_source=f"H100:{r['frames_file']}#pass1", imag_source=f"H100:{r['frames_file']}#pass1",
                     imag_keys="imag_latents_XX + imag_decoded", recorded_success_reference_only=r["recorded_reference_only"]["success"],
                     alternate_run=None, scope="web660")
            out.append((e, npz, "success", ("imag_latents_00.npy",)))
    # ---------------- OpenWAM
    cl = ow_rows(RUNS, "x86_rerun.*.jsonl"); h1 = ow_rows(H100, "x86_rerun.h100_*.jsonl")
    for k in sorted(set(cl) | set(h1)):
        run, uid = k
        use = ("cluster_x86", cl[k]) if k in cl else ("H100", h1[k])
        r = use[1]
        npz = f"{RUNS if use[0] == 'cluster_x86' else H100}/{run}/x86_rerun/{uid}.npz"
        e = dict(model="openwam", unit_id=uid, run=run, task=r.get("task"), family=r.get("family"), level=r.get("level"), source=use[0],
                 machine=r.get("machine"), gpu=r.get("gpu"), success=r["rerun_success"], first_success_step=r["rerun_first_success_step"],
                 outcome_source=f"{use[0]}:{os.path.relpath(npz, ROOT)}", imag_source=f"{use[0]}:{os.path.relpath(npz, ROOT)}", imag_keys="imagined",
                 recorded_success_reference_only=r.get("recorded_success_h400"), alternate_run="H100" if (k in cl and k in h1) else None)
        out.append((e, npz, "success", ("imagined.npy",)))
    # ---------------- 检查
    rows, bad_src, n_checked, bad_curve, no_imag, missing_npz = [], [], 0, [], [], 0
    for e, npz, key, imag_members in out:
        if e["outcome_source"] != e["imag_source"]:
            bad_src.append(e["unit_id"])
        if CHECK:
            if os.path.exists(npz):
                try:
                    anyv, first, n, names = npz_success_any(npz, key)
                    n_checked += 1
                    e["npz_success"], e["npz_first_success_step"] = anyv, first
                    if anyv != bool(e["success"]) or first != e["first_success_step"]:
                        bad_curve.append(e["unit_id"])
                    if not all(m in names for m in imag_members):
                        no_imag.append(e["unit_id"])
                except Exception as ex:  # noqa: BLE001
                    e["npz_check_error"] = repr(ex); bad_curve.append(e["unit_id"])
            else:
                missing_npz += 1; e["npz_on_nas"] = False      # H100 上已完成但还没拉回的
        rows.append(e)
    with open(f"{OUT}/units_source.jsonl.part", "w") as fh:
        for e in rows:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/units_source.jsonl.part", f"{OUT}/units_source.jsonl")
    S = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rule=__doc__,
             units_by_model_source=dict(Counter(f"{e['model']}|{e['source']}" for e in rows)),
             units_by_model_machine=dict(Counter(f"{e['model']}|{e['machine']}" for e in rows)),
             outcome_source_ne_imag_source=len(bad_src), outcome_source_ne_imag_source_units=bad_src[:50],
             npz_checked=n_checked, npz_not_on_nas_yet=missing_npz,
             row_vs_npz_success_mismatch=len(bad_curve), row_vs_npz_success_mismatch_units=bad_curve[:50],
             npz_without_imagination=len(no_imag), npz_without_imagination_units=no_imag[:50],
             units_with_alternate_run=sum(1 for e in rows if e.get("alternate_run")))
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in S.items() if k != "rule"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
