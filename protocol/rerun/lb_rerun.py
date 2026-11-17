"""LingBot T2/T3/T4 闭环全量重跑（2026-11-12 用户裁决：所有当时没存的数据全部重跑）。
原 runner（lingbot_tx_closed / lingbot_tx_famO_closed / phys/t2_egn_closed / phys/t2_k_closed）只写了 jsonl，
闭环动作与真实画面全部丢弃。本脚本对 gen_closed_main SPEC 里 LingBot T2–T4 的每一行闭环记录，用与原 runner
完全相同的函数、锚点、种子、扰动、H=400 重跑 ×2，并把全部中间量落盘：
  commanded / executed 动作、每一步真实主视角 + 腕部画面（128 px，含第 0 步初始观测）、逐步 success、
  每块想象 latent（lb_capture）。原记录一个字节不动。
输出：runs/lingbot_t{tid}/rerun_closed/<src>.<shard>.jsonl（含与原记录的逐项比对）
      runs/lingbot_t{tid}/rerun_closed/<src>/<uid>.npz
抽到标注卡片的单元排在最前面。
  python lb_rerun.py <shard> <nshards>          (lerobot_libero, GPU)
  python lb_rerun.py list                        只列任务数
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
H = 400
CARD_SEED, CARD_K = 20261112, 5


def anchors(path):
    A = {}
    for a in json.load(open(f"{ROOT}/{path}" if not path.startswith("/") else path))["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        A[a["anchor_id"]] = a
    return A


def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pat}")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def level_of(r):
    s = r.get("setting_id") or "none"
    return s.split("__")[-1] if "__" in s else s


def jobs():
    """[(tid, src, row)]，src ∈ egn / K / base / famA / famP / famO。"""
    J = []
    for tid in (2, 4, 6):
        R = f"lingbot_t{tid}"
        J += [(tid, "egn", r) for r in rows(f"{R}/phys/closed_egn.*.jsonl")]
        J += [(tid, "K", r) for r in rows(f"{R}/phys/closed_K.*.jsonl")]
        J += [(tid, "base", r) for r in rows(f"{R}/base/closed.*.jsonl")]
        J += [(tid, "famA", r) for r in rows(f"{R}/famA/closed*.jsonl")]
        J += [(tid, "famP", r) for r in rows(f"{R}/famP/closed*.jsonl")]
        J += [(tid, "famO", r) for r in rows(f"{R}/famO/closed_occ.*.jsonl")]
    return J


def card_first(J):
    """与 annotation/round2_real/real_cards.pick 同一抽样（seed 20261112，每格 5 个），抽中的排前面。"""
    T = {2: "T2", 4: "T3", 6: "T4"}
    cells = {}
    for j in J:
        cells.setdefault((j[0], level_of(j[2])), []).append(j)
    pri = set()
    for (tid, lv), js in cells.items():
        js = sorted(js, key=lambda j: j[2]["unit_id"])
        rng = np.random.RandomState(zlib.crc32(f"{CARD_SEED}/lingbot/{T[tid]}/{lv}".encode()))
        for i in rng.choice(len(js), size=min(CARD_K, len(js)), replace=False):
            pri.add((js[i][0], js[i][1], js[i][2]["unit_id"]))
    return sorted(J, key=lambda j: ((j[0], j[1], j[2]["unit_id"]) not in pri, j[0], j[1], j[2]["unit_id"])), pri


class Recorder:
    def __init__(self):
        self.cmd, self.exe, self.av, self.wr, self.succ = [], [], [], [], []


def main():
    J = jobs()
    if sys.argv[1] == "list":
        J, pri = card_first(J)
        from collections import Counter
        print(len(J), "rows;", len(pri), "card units first;", Counter((j[0], j[1]) for j in J))
        return
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import torch
    import lb_capture
    import lingbot_unit as LU
    from phys_perturb import spec_of
    from lingbot_t2_cloud import BANKS as BANKS_T2
    from lingbot_tx_run import banks_of
    import t2_egn_closed as EGN
    import t2_k_closed as KC
    from lingbot_t2_cloud import occlude
    LU.H = H
    EGN.H = KC.H = H
    lb_capture.install()

    class Rig(LU.LingBotRig):
        occ = None
        rec = None

        def make_env(self, anchor):
            env, obs = LU.LingBotRig.make_env(self, anchor)
            rec = self.rec
            if rec is not None:
                rec.av.append(np.asarray(obs["pixels"]["image"][0]).copy())
                rec.wr.append(np.asarray(obs["pixels"]["image2"][0]).copy())
                orig_step = env.step

                def step(a, _o=orig_step, _r=rec):
                    _r.exe.append(np.asarray(a, np.float32).reshape(-1).copy())
                    out = _o(a)
                    ob, inf = out[0], out[4]
                    _r.av.append(np.asarray(ob["pixels"]["image"][0]).copy())
                    _r.wr.append(np.asarray(ob["pixels"]["image2"][0]).copy())
                    s = bool(np.asarray(inf.get("is_success", [False])).reshape(-1)[0]) if "is_success" in inf else False
                    _r.succ.append(s)
                    return out
                env.step = step
            if self.occ is not None:
                obs = dict(obs); obs["pixels"] = dict(obs["pixels"])
                obs["pixels"]["image"] = occlude(obs["pixels"]["image"], self.occ)
                self.occ = None
            return env, obs

        def to_env_action(self, a_norm):
            a = LU.LingBotRig.to_env_action(self, a_norm)
            if self.rec is not None:
                self.rec.cmd.append(np.asarray(a[0], np.float32).copy())
            return a

    bank_cache = {}

    def bank(tid, fam):
        k = (tid, fam)
        if k not in bank_cache:
            # 与原 runner 一致：base/famA/famP 用 runner 的 bank 表；egn/K/famO 用 libero_anchors_t{tid}.json
            p = (BANKS_T2[fam] if tid == 2 else banks_of(tid)[fam]) if fam in ("base", "famA", "famP") else f"libero/out/bank/libero_anchors_t{tid}.json"
            bank_cache[k] = anchors(p)
        return bank_cache[k]

    base_cache = {}

    def base_rec(tid, uid):
        if tid not in base_cache:
            base_cache[tid] = {r["unit_id"]: r for r in rows(f"lingbot_t{tid}/base/records.*.jsonl")}
        return base_cache[tid][uid]

    def imagined(tid, fam, uid):
        for p in (f"{RUNS}/lingbot_t{tid}/{fam}/actions/{uid}.npz",
                  _wmvp.P(f"{ROOT.rsplit('/wm_verdict', 1)[0]}/cloud_results/$NODE_CLOUD/wm_t{tid}/{fam}/actions/{uid}.npz")):
            if os.path.exists(p):
                return np.load(p)["env"][:H]
        raise FileNotFoundError(f"imagined actions for {uid}")

    rigs = {}
    J, pri = card_first(J)
    if os.environ.get("WMV_CARD_ONLY"):        # 2026-11-12 范围收缩：只跑标注卡片抽中的单元
        J = [j for j in J if (j[0], j[1], j[2]["unit_id"]) in pri]
    if not os.environ.get("WMV_INCLUDE_NLW"):   # 2026-11-14 用户："NLw 都先不跑"——NLw 档暂缓（分片按 crc32(键) 划分，过滤不改变其余单元的分片）
        J = [j for j in J if not level_of(j[2]).startswith("NLw")]
    if os.environ.get("WMV_UNITS"):            # 2026-11-15 同卷标注：只补跑指定单元（json 列表 unit_id）——缺 LingBot 的四模型同卷 seed
        want = set(json.load(open(os.environ["WMV_UNITS"])))
        J = [j for j in J if j[2]["unit_id"] in want]
    passes = (1,) if os.environ.get("WMV_ONE_PASS") else (1, 2)   # 卡片只用第一遍
    mine = [j for j in J if zlib.crc32(f"{j[0]}/{j[1]}/{j[2]['unit_id']}".encode()) % nsh == shard]
    print("LB_RERUN all", len(J), "this shard", len(mine), flush=True)
    limit = int(os.environ.get("WMV_LIMIT", "0")); n_done = 0
    for tid, src, r in mine:
        uid = r["unit_id"]
        if limit and n_done >= limit:
            break
        od = f"{RUNS}/lingbot_t{tid}/rerun_closed/{src}"
        if os.path.exists(f"{od}/{uid}.npz"):
            continue
        if tid not in rigs:
            rigs.clear(); torch.cuda.empty_cache()
            rigs[tid] = Rig(tid)
        rig = rigs[tid]
        t0 = time.time()
        res = []
        for run in passes:
            rec = Recorder(); rig.rec = rec
            with lb_capture.Capture() as cap:
                if src == "egn":
                    b = base_rec(tid, r["base_unit"]); spec = spec_of(r["setting_id"], tid, r["base_unit"])
                    c, _ = EGN.closed_pert(rig, bank(tid, "anch")[b["anchor_id"]], r["imagination_seed"], spec,
                                           imagined(tid, "base", r["base_unit"]))
                elif src == "K":
                    b = base_rec(tid, r["base_unit"]); spec = spec_of(r["setting_id"], tid, r["base_unit"])
                    c = KC.closed_push(rig, bank(tid, "anch")[b["anchor_id"]], r["imagination_seed"], spec,
                                       imagined(tid, "base", r["base_unit"]))
                elif src == "famO":
                    rig.occ = r["occlusion_ratio"]
                    c = rig.closed_loop(bank(tid, "anch")[r["anchor_id"]], r["imagination_seed"], imagined(tid, "famO", uid))
                    rig.occ = None
                else:
                    c = rig.closed_loop(bank(tid, src)[r["anchor_id"]], r["imagination_seed"], imagined(tid, src, uid))
            rig.rec = None
            res.append((c, rec, cap))
        os.makedirs(od, exist_ok=True)
        d = {}
        for i, (c, rec, cap) in enumerate(res, 1):
            p = "" if i == 1 else "run2_"
            d.update({f"{p}commanded": np.stack(rec.cmd), f"{p}executed": np.stack(rec.exe),
                      f"{p}agentview": np.stack(rec.av), f"{p}wrist": np.stack(rec.wr),
                      f"{p}success": np.array(rec.succ, bool)})
            d.update(cap.arrays(prefix=f"{p}imag_"))
        tmp = f"{od}/{uid}.tmp.npz"
        np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{uid}.npz")
        rec_cl = r["closed_loop"]
        c1 = res[0][0]
        c2 = res[1][0] if len(res) > 1 else None
        row = dict(unit_id=uid, src=src, task=f"t{tid}", setting_id=r.get("setting_id"), level=level_of(r),
                   card_unit=(tid, src, uid) in pri, H_env=H,
                   rerun=dict(success=c1["success"], first_success_step=c1["first_success_step"], steps=int(c1["steps"]),
                              run2=None if c2 is None else dict(success=c2["success"], first_success_step=c2["first_success_step"],
                                                                steps=int(c2["steps"]))),
                   passes=len(res),
                   recorded=dict(success=rec_cl.get("success"), first_success_step=rec_cl.get("first_success_step"),
                                 run2=rec_cl.get("rerun")),
                   matches_recorded_success=bool(c1["success"]) == bool(rec_cl.get("success")),
                   matches_recorded_step=c1["first_success_step"] == rec_cl.get("first_success_step"),
                   recorded_machine=r.get("machine"), frames_file=f"libero/out/runs/lingbot_t{tid}/rerun_closed/{src}/{uid}.npz",
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=torch.cuda.get_device_name(0))
        with open(f"{RUNS}/lingbot_t{tid}/rerun_closed/{src}.{shard}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n_done += 1
        print(uid, src, "rec", rec_cl.get("success"), "now", c1["success"], None if c2 is None else c2["success"], row["seconds"], flush=True)
    print("LB_RERUN_DONE shard", shard)


if __name__ == "__main__":
    main()
