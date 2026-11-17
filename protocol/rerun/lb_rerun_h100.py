"""LingBot T2/T3/T4 闭环全量重跑 —— H100 版（2026-11-15，复制自 ../lb_rerun.py，原文件不动）。

与 lb_rerun.py 完全相同的闭环（同一函数、锚点、种子、扰动、H=400、两遍），只改了：
  1. 选单元：不按 crc32 分片，而是按 H100 队列文件（有序 unit 列表，h100_assignment.json 生成）逐个认领
     （claims/<uid> 用 O_EXCL 原子创建，多进程/多卡共享同一优先级队列）；
  2. 输出：写到 libero/out/h100/runs/lingbot_t{tid}/rerun_closed/<src>/<uid>.npz（= 集群目录结构外加一层 h100/），
     jsonl 写 rerun_closed/<src>.h100_<worker>.jsonl；
  3. 多存（"这一次一定要把所有的数据都记下来"）：
     - 逐步 MuJoCo 全物理状态（mjSTATE_FULLPHYSICS，含第 0 步初始状态）：states / run2_states —— 纯读，不碰物理；
     - 每块想象 latent 照存（imag_latents_XX，同原版），另在 H100 上把每一遍的全部块拼接后用策略自带 VAE 解码成图片：
       imag_decoded [T,128,256,3] uint8 + imag_decoded_chunk_start（第 c 块在解码序列里的起始帧）
       （与 r3_cards.lb_decode / optimism 相同口径：全部块按时间拼接后一次解码，已验证与 rig.decode 逐位一致）；
     - 元数据：machine=H100、主机名、GPU 型号/编号、torch/CUDA/驱动版本、checkpoint 名与 HF revision、
       脚本 sha256（本文件 + 原 lb_rerun.py + lb_capture.py + lingbot_unit.py）、种子、每遍耗时、起止时间。
  4. 确定性核对模式：--out-root 指向 h100/determinism，--units 给单元列表。

  python lb_rerun_h100.py run --queue Q.json --worker W [--out-root DIR] [--deadline EPOCH]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import hashlib
import json
import os
import socket
import sys
import time
import zlib

import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
RUNS = f"{ROOT}/libero/out/runs"                       # 只读：原记录、锚点、想象动作
OUT_ROOT_DEFAULT = f"{ROOT}/libero/out/h100/runs"       # 写：H100 输出
H = 400
CARD_SEED, CARD_K = 20261112, 5
HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


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
    """与 lb_rerun.jobs() 相同：[(tid, src, row)]。"""
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
    """与 lb_rerun.card_first 相同（集群 2236 的处理顺序）。"""
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


def shard_of(tid, src, uid, nsh=6):
    return zlib.crc32(f"{tid}/{src}/{uid}".encode()) % nsh


class Recorder:
    def __init__(self):
        self.cmd, self.exe, self.av, self.wr, self.succ, self.states = [], [], [], [], [], []


def claim(cdir, key):
    os.makedirs(cdir, exist_ok=True)
    try:
        fd = os.open(f"{cdir}/{key}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}\n".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    a = sys.argv[1:]
    assert a[0] == "run", __doc__
    qf = a[a.index("--queue") + 1]
    worker = a[a.index("--worker") + 1]
    out_root = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT_DEFAULT
    deadline = float(a[a.index("--deadline") + 1]) if "--deadline" in a else float(os.environ.get("WMV_DEADLINE", "0"))
    Q = json.load(open(qf))                       # [{"tid","src","unit_id",...}] 有序
    want = {(q["tid"], q["src"], q["unit_id"]) for q in Q}
    import torch
    import lb_capture
    import lingbot_unit as LU
    import mujoco
    from phys_perturb import spec_of
    from lingbot_t2_cloud import BANKS as BANKS_T2
    from lingbot_tx_run import banks_of
    import t2_egn_closed as EGN
    import t2_k_closed as KC
    from lingbot_t2_cloud import occlude
    LU.H = H
    EGN.H = KC.H = H
    lb_capture.install()
    J = {(j[0], j[1], j[2]["unit_id"]): j for j in jobs() if (j[0], j[1], j[2]["unit_id"]) in want}
    _, pri = card_first(list(jobs()))
    missing = want - set(J)
    print("LB_H100 queue", len(Q), "found", len(J), "missing", len(missing), "worker", worker, "out", out_root, flush=True)
    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    env_meta = dict(machine="H100", hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(0), cuda_visible_devices=gpu_idx,
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    driver=os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0").read().strip(),
                    mujoco=mujoco.__version__, checkpoint=LU.CKPT,
                    script_sha256=dict(lb_rerun_h100=sha(os.path.abspath(__file__)), lb_rerun=sha(_wmvp.code("libero/rerun/lb_rerun.py")),
                                       lb_capture=sha(_wmvp.code("libero/backfill/lb_capture.py")), lingbot_unit=sha(_wmvp.code("libero/lingbot_unit.py")),
                                       t2_egn_closed=sha(_wmvp.code("libero/phys/t2_egn_closed.py")), t2_k_closed=sha(_wmvp.code("libero/phys/t2_k_closed.py")),
                                       phys_perturb=sha(_wmvp.code("libero/phys/phys_perturb.py"))))
    try:
        from huggingface_hub import scan_cache_dir  # noqa: F401
        snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lerobot--lingbot_va_libero_long/snapshots/*"))
        env_meta["checkpoint_revision"] = os.path.basename(snap[0]) if snap else None
    except Exception:  # noqa: BLE001
        pass

    class Rig(LU.LingBotRig):
        occ = None
        rec = None

        def make_env(self, anchor):
            env, obs = LU.LingBotRig.make_env(self, anchor)
            rec = self.rec
            if rec is not None:
                u = env.envs[0]

                def snap(_u=u):
                    m, d = _u._env.sim.model._model, _u._env.sim.data._data
                    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
                    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    return st
                rec.av.append(np.asarray(obs["pixels"]["image"][0]).copy())
                rec.wr.append(np.asarray(obs["pixels"]["image2"][0]).copy())
                rec.states.append(snap())
                orig_step = env.step

                def step(a, _o=orig_step, _r=rec, _snap=snap):
                    _r.exe.append(np.asarray(a, np.float32).reshape(-1).copy())
                    out = _o(a)
                    ob, inf = out[0], out[4]
                    _r.av.append(np.asarray(ob["pixels"]["image"][0]).copy())
                    _r.wr.append(np.asarray(ob["pixels"]["image2"][0]).copy())
                    _r.states.append(_snap())
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
    passes = (1,) if os.environ.get("WMV_ONE_PASS") else (1, 2)
    cdir = f"{out_root}/_claims_lingbot"
    n_done = 0
    def queue_iter():
        # 2026-11-15 v2：每领一个单元前检查队列文件是否被改过（mtime），改过就重读并按新顺序继续（已领过的不再给）
        nonlocal Q, J
        seen, mt = set(), os.path.getmtime(qf)
        while True:
            changed = False
            for q in Q:
                k = (q["tid"], q["src"], q["unit_id"])
                if k in seen:
                    continue
                if os.path.getmtime(qf) != mt:
                    changed = True; break
                seen.add(k); yield q
            if not changed:
                return
            mt = os.path.getmtime(qf); Q = json.load(open(qf))
            w = {(q["tid"], q["src"], q["unit_id"]) for q in Q}
            if w - set(J):
                J = {(j[0], j[1], j[2]["unit_id"]): j for j in jobs() if (j[0], j[1], j[2]["unit_id"]) in w}
            print("LB_H100 queue reloaded", len(Q), flush=True)

    for q in queue_iter():
        key = (q["tid"], q["src"], q["unit_id"])
        if key not in J:
            continue
        if deadline and time.time() > deadline:
            print("LB_H100 deadline reached, stop taking new units", flush=True); break
        if os.path.exists(f"{out_root}/STOP"):
            print("LB_H100 STOP file, stop taking new units", flush=True); break
        tid, src, r = J[key]
        uid = r["unit_id"]
        od = f"{out_root}/lingbot_t{tid}/rerun_closed/{src}"
        if os.path.exists(f"{od}/{uid}.npz") or not claim(cdir, f"{tid}_{src}_{uid}"):
            continue
        t_start = time.time()
        try:
            if tid not in rigs:
                rigs.clear(); torch.cuda.empty_cache()
                rigs[tid] = Rig(tid)
            rig = rigs[tid]
            res, secs = [], []
            for run in passes:
                t0 = time.time()
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
                res.append((c, rec, cap)); secs.append(round(time.time() - t0, 1))
            os.makedirs(od, exist_ok=True)
            d = {}
            t_dec = time.time()
            for i, (c, rec, cap) in enumerate(res, 1):
                p = "" if i == 1 else "run2_"
                d.update({f"{p}commanded": np.stack(rec.cmd), f"{p}executed": np.stack(rec.exe),
                          f"{p}agentview": np.stack(rec.av), f"{p}wrist": np.stack(rec.wr),
                          f"{p}success": np.array(rec.succ, bool), f"{p}states": np.stack(rec.states)})
                d.update(cap.arrays(prefix=f"{p}imag_"))
                if cap.latents:                                   # H100 上解码：全部块按时间拼接后一次解码（r3/optimism 同口径）
                    import torch as _t
                    lat = _t.cat([_t.from_numpy(z.astype(np.float32)) for z in cap.latents], dim=2)
                    fr = rig.decode(lat)
                    fr = fr.numpy() if hasattr(fr, "numpy") else np.asarray(fr)
                    lens = [int(z.shape[2]) for z in cap.latents]
                    lat_start = np.cumsum([0] + lens[:-1])
                    d[f"{p}imag_decoded"] = np.asarray(fr, np.uint8)
                    # 因果 VAE：拼接序列第 0 个 latent 帧 -> 第 0 帧；第 s>0 个 -> 第 1+4(s-1) .. 4s 帧
                    d[f"{p}imag_decoded_chunk_start"] = np.array([(1 + 4 * (s - 1)) if s > 0 else 0 for s in lat_start])
                    d[f"{p}imag_latent_chunk_lens"] = np.array(lens)
            dec_s = round(time.time() - t_dec, 1)
            d["meta_json"] = np.array(json.dumps(dict(env_meta, unit_id=uid, src=src, tid=tid, seed=r["imagination_seed"],
                                                      setting_id=r.get("setting_id"), passes=len(res), seconds_per_pass=secs,
                                                      decode_seconds=dec_s, worker=worker,
                                                      started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)))))
            tmp = f"{od}/{uid}.tmp.npz"
            np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{uid}.npz")
            rec_cl = r["closed_loop"]
            c1 = res[0][0]
            c2 = res[1][0] if len(res) > 1 else None
            run2_identical = None
            if len(res) > 1:
                run2_identical = bool(np.array_equal(d["commanded"], d["run2_commanded"]) and np.array_equal(d["states"], d["run2_states"]))
            row = dict(unit_id=uid, src=src, task=f"t{tid}", setting_id=r.get("setting_id"), level=level_of(r),
                       card_unit=(tid, src, uid) in pri, H_env=H, seed=r["imagination_seed"], tier=q.get("tier"),
                       rerun=dict(success=c1["success"], first_success_step=c1["first_success_step"], steps=int(c1["steps"]),
                                  run2=None if c2 is None else dict(success=c2["success"], first_success_step=c2["first_success_step"],
                                                                    steps=int(c2["steps"]))),
                       passes=len(res), run2_bit_identical=run2_identical,
                       run2_check=("n/a: single pass (WMV_ONE_PASS=1; 2026-11-16 user-approved for C_rest = originally-successful non-web units)"
                                   if len(res) == 1 else "pass1 vs pass2 commanded+states bit-identical check"),
                       recorded=dict(success=rec_cl.get("success"), first_success_step=rec_cl.get("first_success_step"),
                                     run2=rec_cl.get("rerun")),
                       matches_recorded_success=bool(c1["success"]) == bool(rec_cl.get("success")),
                       matches_recorded_step=c1["first_success_step"] == rec_cl.get("first_success_step"),
                       recorded_machine=r.get("machine"),
                       frames_file=f"libero/out/h100/runs/lingbot_t{tid}/rerun_closed/{src}/{uid}.npz",
                       npz_bytes=os.path.getsize(f"{od}/{uid}.npz"), seconds=round(time.time() - t_start, 1),
                       seconds_per_pass=secs, decode_seconds=dec_s, worker=worker,
                       started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)),
                       finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **env_meta)
            with open(f"{out_root}/lingbot_t{tid}/rerun_closed/{src}.h100_{worker}.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            n_done += 1
            print(uid, src, "rec", rec_cl.get("success"), "now", c1["success"], None if c2 is None else c2["success"],
                  "run2_identical", run2_identical, row["seconds"], flush=True)
        except Exception as e:  # noqa: BLE001  出错：记一行错误，释放认领（之后可重试），继续
            import traceback
            traceback.print_exc()
            os.makedirs(f"{out_root}/_errors", exist_ok=True)
            with open(f"{out_root}/_errors/lingbot.h100_{worker}.jsonl", "a") as fh:
                fh.write(json.dumps(dict(unit_id=uid, src=src, tid=tid, error=repr(e), seconds=round(time.time() - t_start, 1),
                                         time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **env_meta)) + "\n")
            try:
                os.remove(f"{cdir}/{tid}_{src}_{uid}")
            except OSError:
                pass
            rigs.clear(); torch.cuda.empty_cache()
    print("LB_H100_DONE worker", worker, "units", n_done, flush=True)


if __name__ == "__main__":
    main()
