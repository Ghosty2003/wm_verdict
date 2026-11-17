"""OpenWAM-α 闭环全量重跑 —— H100 版（2026-11-15）。

不复制闭环代码：直接 import 原 ../ow_x86_rerun.py（原文件不动），调用它的 closed_full / _sim_class / _vs_recorded，
因此每次规划调用、种子 = seed + c、早停（WMV_STOP_ON_SUCCESS=1，默认）、逐步真实画面、states、每次调用的全部想象帧
与集群 x86 版逐行相同。只改：
  1. 选单元：按 H100 队列文件（有序单元列表）逐个认领（claims/<uid> O_EXCL），多进程/多卡共享同一优先级队列；
  2. 输出：libero/out/h100/runs/<run>/x86_rerun/<uid>.npz + <run>/x86_rerun.h100_<worker>.jsonl
     （= 集群目录结构外加一层 h100/；子目录名沿用 x86_rerun 以便合并，行里 machine=H100 区分）；
  3. 多记：machine=H100、主机名、GPU 型号/编号、torch/CUDA/驱动版本、checkpoint 名、脚本 sha256（本文件 + 原脚本 +
     ow_runner / sim_server / openwam_unit）、种子、起止时间、耗时；npz 里另存 meta_json。

  python ow_rerun_h100.py run --queue Q.json --worker W [--out-root DIR] [--deadline EPOCH]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import hashlib
import json
import os
import socket
import sys
import time

ROOT = _wmvp.P("$WMV_ROOT")
RERUN = f"{ROOT}/libero/rerun"
_wmvp.bootstrap()
import ow_x86_rerun as X  # noqa: E402   原脚本（只读 import）

OUT_ROOT_DEFAULT = f"{ROOT}/libero/out/h100/runs"


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def claim(cdir, key):
    os.makedirs(cdir, exist_ok=True)
    try:
        fd = os.open(f"{cdir}/{key}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}\n".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def run(qf, worker, out_root, deadline):
    import numpy as np
    _wmvp.bootstrap()
    import ow_runner as W
    from replay_frames import B
    import replay_frames as RF
    import ow_tx_run as TX
    assert (W.H, W.BLOCK, W.N_CALLS) == (X.H, X.BLOCK, 13), (W.H, W.BLOCK, W.N_CALLS)
    assert (RF.MAIN, RF.FAM_BANK) == (X.MAIN, X.FAM_BANK) and TX.RATIOS == X.RATIOS and TX.PROMPT == X.PROMPT
    assert all(RF.tx_bank(t, f) == X.tx_bank(t, f) for t in (2, 4, 6) for f in ("base", "famA", "famP"))
    frames = os.environ.get("WMV_STEP_FRAMES", "1") != "0"
    stop_on_success = os.environ.get("WMV_STOP_ON_SUCCESS", "1") == "1"
    U = json.load(open(qf))
    print("OW_H100 queue", len(U), "worker", worker, "out", out_root, "step_frames", frames, "stop_on_success", stop_on_success, flush=True)
    W._setup()
    eng, pre, arch, load_s, ckname = W._engine()
    import torch
    gpu = torch.cuda.get_device_name(0)
    env_meta = dict(machine="H100", hostname=socket.gethostname(), gpu=gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    driver=os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0").read().strip(),
                    checkpoint=ckname, stop_on_success=stop_on_success,
                    script_sha256=dict(ow_rerun_h100=sha(os.path.abspath(__file__)), ow_x86_rerun=sha(f"{RERUN}/ow_x86_rerun.py"),
                                       ow_runner=sha(_wmvp.code("libero/openwam/ow_runner.py")), sim_server=sha(_wmvp.code("libero/openwam/sim_server.py")),
                                       openwam_unit=sha(_wmvp.code("libero/openwam/openwam_unit.py")), ow_famB=sha(_wmvp.code("libero/bfam/ow_famB.py")),
                                       phys_perturb=sha(_wmvp.code("libero/phys/phys_perturb.py"))))
    sim = X._sim_class(W)(f"/dev/shm/ow_h100_{worker}_{os.getpid()}")
    corr = None
    if any(u["kind"] == "famB" and u["setting"] != "B0" for u in U):
        from bcorrupt import Corruptor
        corr = Corruptor()
    cdir = f"{out_root}/_claims_openwam"
    n = 0
    for u in U:
        if deadline and time.time() > deadline:
            print("OW_H100 deadline reached", flush=True); break
        if os.path.exists(f"{out_root}/STOP"):
            print("OW_H100 STOP file", flush=True); break
        od = f"{out_root}/{u['run']}/{X.SUB}"
        if os.path.exists(f"{od}/{u['unit_id']}.npz") or not claim(cdir, u["unit_id"]):
            continue
        jl = f"{out_root}/{u['run']}/{X.SUB}.h100_{worker}.jsonl"
        t0 = time.time()
        try:
            d, meta = X.closed_full(W, eng, sim, corr, u, B(u["bank"]), frames)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            os.makedirs(f"{out_root}/_errors", exist_ok=True)
            with open(f"{out_root}/_errors/openwam.h100_{worker}.jsonl", "a") as fh:
                fh.write(json.dumps(dict(unit_id=u["unit_id"], run=u["run"], error=repr(e), seconds=round(time.time() - t0, 1),
                                         time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **env_meta)) + "\n")
            try:
                os.remove(f"{cdir}/{u['unit_id']}")
            except OSError:
                pass
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
            sim = X._sim_class(W)(f"/dev/shm/ow_h100_{worker}_{os.getpid()}")
            continue
        os.makedirs(od, exist_ok=True)
        d["meta_json"] = np.array(json.dumps(dict(env_meta, unit_id=u["unit_id"], run=u["run"], seed=u["seed"], **meta,
                                                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), worker=worker)))
        tmp = f"{od}/{u['unit_id']}.tmp.npz"; np.savez_compressed(tmp, **d); os.replace(tmp, f"{od}/{u['unit_id']}.npz")
        s = d["success"]; ok = bool(s.any()); fs = int(np.argmax(s)) + 1 if ok else None
        chk = X._vs_recorded(u, d)
        row = dict(unit_id=u["unit_id"], run=u["run"], task=u["task"], level=u["level"], family=u["family"], kind=u["kind"],
                   mirror=u["mirror"], reason=u.get("reason"), anchor=u["anchor"], seed=u["seed"], setting=u["setting"], tier=u.get("tier"),
                   rerun_success=ok, rerun_first_success_step=fs,
                   recorded_success_h400=u["recorded_success_h400"], recorded_first_success_step=u["recorded_first_success_step"],
                   recorded_first_success_step_h400=u["recorded_first_success_step_h400"],
                   outcome_matches_recorded=ok == u["recorded_success_h400"],
                   step_matches_recorded=fs == u["recorded_first_success_step_h400"],
                   recorded_machine=u["recorded_machine"], recorded_machine_class=u["recorded_machine_class"], **chk, **meta,
                   checkpoint_matches_record=None if not u["recorded_checkpoint"] else ckname == u["recorded_checkpoint"],
                   seconds=round(time.time() - t0, 1), H_env=X.H, worker=worker,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   files=dict(npz=f"libero/out/h100/runs/{u['run']}/{X.SUB}/{u['unit_id']}.npz",
                              keys=sorted(d), npz_bytes=os.path.getsize(f"{od}/{u['unit_id']}.npz")), **env_meta)
        with open(jl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n += 1
        print(u["unit_id"], "rerun", ok, fs, "rec", u["recorded_success_h400"], u["recorded_first_success_step"],
              "cmd_identical", chk["commanded_identical_to_recorded"], "div", chk["first_commanded_divergence_step"], row["seconds"], flush=True)
    if corr is not None:
        corr.close()
    sim.close()
    print("OW_H100_DONE worker", worker, "units", n, flush=True)


def main():
    a = sys.argv[1:]
    assert a and a[0] == "run", __doc__
    qf = a[a.index("--queue") + 1]
    worker = a[a.index("--worker") + 1]
    out_root = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT_DEFAULT
    deadline = float(a[a.index("--deadline") + 1]) if "--deadline" in a else float(os.environ.get("WMV_DEADLINE", "0"))
    run(qf, worker, out_root, deadline)


if __name__ == "__main__":
    main()
