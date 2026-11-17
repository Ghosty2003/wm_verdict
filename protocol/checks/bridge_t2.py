"""T2 跨机器锚点桥（2026-11-10 决定 2a）：在 5090 上重跑 LingBot T2 无扰动 50 个基线单元（同锚点、同想象种子、同 H），
开环想象 + 真值重放 x2 + 闭环 x2，全部产物落盘；与 A100 原记录（lingbot_t2/base/records, machine=$NODE_CLOUD）的开环真值
成功率做 Wilson 95% 区间比对。只写 lingbot_t2/famB_bridge/，不碰原记录。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob, json, math, os, sys, time
import numpy as np
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
OUT = f"{ROOT}/libero/out/runs/lingbot_t2/famB_bridge"
def wilson(k, n):
    p = k / n; z = 1.96; d = 1 + z*z/n; c = p + z*z/(2*n); s = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - s) / d, (c + s) / d
def main():
    import torch
    import lingbot_unit as LU
    base = sorted((json.loads(l) for f in glob.glob(f"{ROOT}/libero/out/runs/lingbot_t2/base/records.*.jsonl") for l in open(f)),
                  key=lambda r: r["unit_id"])
    assert len(base) == 50, len(base)
    LU.H = base[0]["H_env"]
    A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors_t2.json"))["anchors"]}
    for sub in ("actions", "trajectories", "latents"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    path = f"{OUT}/records.jsonl"
    done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
    rig = LU.LingBotRig(2); gpu = torch.cuda.get_device_name(0); node = os.uname().nodename
    for r in base:
        uid = r["unit_id"]
        if uid in done:
            continue
        t0 = time.time(); a = A[r["anchor_id"]]; seed = r["imagination_seed"]
        I = rig.imagine(a, seed)
        R1 = rig.replay(a, I["env_actions"]); R2 = rig.replay(a, I["env_actions"])
        C1 = rig.closed_loop(a, seed, I["env_actions"]); C2 = rig.closed_loop(a, seed, I["env_actions"])
        s1 = np.asarray(R1["succ"], bool)
        np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=I["env_actions"], closed_loop=C1["actions"], closed_loop2=C2["actions"])
        np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=R1["states"], replay2=R2["states"], real_success=s1)
        np.savez_compressed(f"{OUT}/latents/{uid}.npz", latents=I["latents"].to(torch.float16).cpu().numpy())
        rec = dict(unit_id=uid, a100_unit=uid, anchor_id=r["anchor_id"], imagination_seed=seed, H_env=LU.H,
                   real_verdict=bool(s1.any()), real_first_success_step=int(np.argmax(s1)) + 1 if s1.any() else None,
                   replay_check_identical=bool(np.array_equal(R1["states"], R2["states"])),
                   a100_real_verdict=r["real_verdict"], a100_real_first_success_step=r["real_first_success_step"],
                   closed_loop=dict(success=C1["success"], first_success_step=C1["first_success_step"], steps=C1["steps"],
                                    rerun=dict(success=C2["success"], first_success_step=C2["first_success_step"]),
                                    conclusion_consistent=C1["success"] == C2["success"]),
                   machine=f"{node} (RTX 5090)", gpu=gpu, seconds=round(time.time() - t0, 1))
        open(path, "a").write(json.dumps(rec) + "\n"); print(uid, rec["real_verdict"], "a100", r["real_verdict"], rec["seconds"], flush=True)
    R = [json.loads(l) for l in open(path)]
    k5, ka = sum(x["real_verdict"] for x in R), sum(x["a100_real_verdict"] for x in R); n = len(R)
    w5, wa = wilson(k5, n), wilson(ka, n)
    overlap = w5[0] <= wa[1] and wa[0] <= w5[1]
    rep = dict(n=n, rtx5090=dict(k=k5, rate=k5/n, ci95=w5), a100=dict(k=ka, rate=ka/n, ci95=wa), overlap=overlap,
               verdict="PASS" if overlap else "FAIL", note="open-loop truth success (imagination machine differs); closed loop reported per unit")
    json.dump(rep, open(f"{OUT}/bridge_report.json", "w"), indent=1); print("BRIDGE", rep["verdict"], json.dumps(rep), flush=True)
if __name__ == "__main__":
    main()
