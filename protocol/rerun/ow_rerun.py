"""OpenWAM T2–T4 物理扰动（E/G/N/K）闭环中没有落盘的单元全量重跑（2026-11-12 用户裁决）。
缺失：openwam_t2/closed/ 整个不存在（1850 行）；T3/T4 在 GH200 上跑的部分 npz 没回传（126 + 16 行）。
与原 runner（phys/ow_phys_closed_tx.py → phys_closed_openwam.closed）同一循环、同锚点/种子/扰动、×2，
额外落盘：每步完整物理状态（sim_server step 本来就返回，原脚本丢弃）、每次规划读入的真实主视角/腕部画面、
全部想象帧。逐步真实画面之后由物理状态精确渲染。原记录不动。
输出：runs/openwam_t{tid}/rerun_closed/phys/<uid>.npz + phys.<shard>.jsonl（含与原记录比对）
  WMV_LIMIT=n python ow_rerun.py <shard> <nshards>   (openwam interpreter, GPU)
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
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}
CARD_SEED, CARD_K = 20261112, 5


def level_of(r):
    s = r.get("setting_id") or "none"
    return s.split("__")[-1] if "__" in s else s


def jobs():
    J = []
    for tid in (2, 4, 6):
        R = f"{RUNS}/openwam_t{tid}"
        for f in sorted(glob.glob(f"{R}/phys/closed.*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if not os.path.exists(f"{R}/closed/{r['unit_id']}.npz"):
                    J.append((tid, r))
    return J


def card_first(J):
    T = {2: "T2", 4: "T3", 6: "T4"}
    cells = {}
    for j in J:
        cells.setdefault((j[0], level_of(j[1])), []).append(j)
    pri = set()
    for (tid, lv), js in cells.items():
        js = sorted(js, key=lambda j: j[1]["unit_id"])
        rng = np.random.RandomState(zlib.crc32(f"{CARD_SEED}/openwam/{T[tid]}/{lv}".encode()))
        for i in rng.choice(len(js), size=min(CARD_K, len(js)), replace=False):
            pri.add((js[i][0], js[i][1]["unit_id"]))
    return sorted(J, key=lambda j: ((j[0], j[1]["unit_id"]) not in pri, j[0], j[1]["unit_id"])), pri


def closed_full(W, eng, sim, anchor, prompt, seed, spec):
    """phys_closed_openwam.closed 的同一循环，另存 states / 读入画面。"""
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    f = f"{sim.scratch}/obs.npz"; sim.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, perturb=spec); o = dict(np.load(f))
    cmd, ex, succ, imag, S, rav, rwr = [], [], [], [], [], [], []
    for c in range(W.N_CALLS):
        rav.append(np.asarray(o["agentview_image"], np.uint8)); rwr.append(np.asarray(o["robot0_eye_in_hand_image"], np.uint8))
        img = Image.fromarray(W.canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        imag.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(cmd)]
        r = sim.step(A); cmd += A; ex += list(r["executed"]); succ += r["success"].tolist(); S.append(r["states"]); o = r
    return dict(commanded=np.stack(cmd), executed=np.stack(ex), success=np.array(succ), states=np.concatenate(S),
                read_agentview=np.stack(rav), read_wrist=np.stack(rwr),
                imagined=np.concatenate(imag), imagined_call_lens=np.array([len(x) for x in imag]))


def main():
    J = jobs()
    if sys.argv[1] == "list":
        J, pri = card_first(J)
        from collections import Counter
        print(len(J), "rows;", len(pri), "card units first;", Counter(j[0] for j in J))
        return
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import ow_runner as W
    from phys_perturb import spec_of
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_rerun_{os.getpid()}")
    J, pri = card_first(J)
    # 2026-11-12 范围收缩：T2（原 runner 未落盘）只跑卡片单元、1 遍；T3/T4 GH200 未回传的是真丢失数据，全部按原协议 ×2 补回
    J = [j for j in J if j[0] != 2 or (j[0], j[1]["unit_id"]) in pri]
    mine = [j for j in J if zlib.crc32(f"{j[0]}/{j[1]['unit_id']}".encode()) % nsh == shard]
    print("OW_RERUN all", len(J), "this shard", len(mine), flush=True)
    banks = {}
    limit = int(os.environ.get("WMV_LIMIT", "0")); n_done = 0
    for tid, r in mine:
        if limit and n_done >= limit:
            break
        uid = r["unit_id"]; od = f"{RUNS}/openwam_t{tid}/rerun_closed/phys"
        if os.path.exists(f"{od}/{uid}.npz"):
            continue
        if tid not in banks:
            banks[tid] = (
                {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json"))["anchors"]},
                {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{RUNS}/openwam_t{tid}/records_base.jsonl")})
        A, B = banks[tid]
        b = B[r["base_unit"]]
        spec = spec_of(r["setting_id"], tid, r["base_unit"])
        anc = dict(A[b["anchor_id"]], resolution=256)
        t0 = time.time()
        R1 = closed_full(W, eng, sim, anc, PROMPT[tid], r["imagination_seed"], spec)
        R2 = closed_full(W, eng, sim, anc, PROMPT[tid], r["imagination_seed"], spec) if tid != 2 else None
        os.makedirs(od, exist_ok=True)
        d = dict(R1)
        if R2 is not None:
            d.update({f"run2_{k}": v for k, v in R2.items()})
        tmp = f"{od}/{uid}.tmp.npz"; np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{uid}.npz")
        v1, v2 = bool(R1["success"].any()), (None if R2 is None else bool(R2["success"].any()))
        fs = int(np.argmax(R1["success"])) + 1 if v1 else None
        rc = r["closed_loop"]
        row = dict(unit_id=uid, src="phys", task=f"t{tid}", setting_id=r["setting_id"], level=level_of(r),
                   card_unit=(tid, uid) in pri, H_env=W.H, spec_matches_record=json.loads(json.dumps(spec)) == r.get("perturbation"),
                   rerun=dict(success=v1, first_success_step=fs, run2=None if R2 is None else dict(success=v2)),
                   passes=1 if R2 is None else 2, purpose="card" if tid == 2 else "recover GH200-lost data",
                   recorded=dict(success=rc.get("success"), first_success_step=rc.get("first_success_step"), run2=rc.get("rerun")),
                   matches_recorded_success=v1 == bool(rc.get("success")),
                   matches_recorded_step=fs == rc.get("first_success_step"),
                   recorded_machine=r.get("machine"), frames_file=f"libero/out/runs/openwam_t{tid}/rerun_closed/phys/{uid}.npz",
                   checkpoint=ckname, seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        with open(f"{RUNS}/openwam_t{tid}/rerun_closed/phys.{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n_done += 1
        print(uid, "rec", rc.get("success"), "now", v1, v2, row["seconds"], flush=True)
    sim.close()
    print("OW_RERUN_DONE shard", shard)


if __name__ == "__main__":
    main()
