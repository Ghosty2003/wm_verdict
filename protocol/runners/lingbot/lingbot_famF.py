"""扰动族 F（单次全帧遮挡）· LingBot-VA 闭环，T0–T4（定义见 ffam_def.py；注册表 v2，2026-11-12）。

挂点：与 O 族同一位置——LeRobot LiberoEnv 返回的原始观测 dict（128 px HWC uint8），在 rig.obs_to_batch
（preprocess_observation → env_preprocessor 旋转 → policy preprocessor）之前。O 族改 make_env 返回的第 0 步观测；
F 族在闭环逐步循环里改【第一个满足 step ≥ t 且本步会触发规划调用】那一步的观测（规划调用 = select_action 发现动作队列空，
或第 0 步）。只换掉 obs["pixels"]["image"]（新的全 0 数组），obs["pixels"]["image2"]（腕部）是同一个对象原样传入。
LingBot 每一步的观测都会作为关键帧进缓冲区，在下一次规划调用时整段（16 帧）VAE 编码进 KV cache；置黑的这一帧
恰好是规划调用那一步的观测 = 该段关键帧的最后一帧，于是这一次规划直接读到它，之后它作为历史留在 KV cache
（以及因果流式 VAE 的 feat_cache）里——与 O 族 t=0 遮挡帧留在上下文里的方式相同。其余每一步的观测不动。

闭环循环与 lingbot_unit.LingBotRig.closed_loop 逐行相同（同锚点、同 torch 种子、H=400），F0（sham）走同一循环只是不置黑。
配对协议：每单元跑两遍（同种子），硬要求 = 两遍成败一致；首次分歧步记录。
落盘（store-all）：commanded / executed、每一步真实主视角 + 腕部（原始、未置黑，含第 0 步）、实际喂给模型的主视角
只存置黑那一步的索引（其余步与真实画面逐位相同，健康检查已验证）、逐步 success、每块想象 latent 与每块模型侧读入帧（lb_capture）。

  python lingbot_famF.py <shard> <nshards> [tids]      正式跑（tids 默认 5,3,2,4,6）
  python lingbot_famF.py sham [n]                      F0 与基线闭环动作逐位比对（T0/T1 前 n 个单元）
  python lingbot_famF.py hc                            单条 T0 健康检查 → out/f_family/healthcheck/lingbot/
输出：out/runs/lingbot_famF/closed_t<tid>.<shard>.jsonl、out/runs/lingbot_famF/closed/<uid>.npz
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
import ffam_def as FD  # noqa: E402
import min_artifacts as MA  # noqa: E402  存档最低要求（protocol.yaml min_artifacts，2026-11-13）

RUNS = f"{ROOT}/libero/out/runs"
OUT = f"{RUNS}/lingbot_famF"
HC = f"{ROOT}/libero/out/f_family/healthcheck/lingbot"
H = FD.H
TIDS = (5, 3, 2, 4, 6)
TNAME = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}


def anchors(path):
    A = {}
    for a in json.load(open(path))["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        A[a["anchor_id"]] = a
    return A


def base_units(tid):
    """[(base_row, anchor)]：T0/T1 = lingbot_base（与 K/E/G/N 闭环同源），T2–T4 = lingbot_t{tid}/base（与 t2_k_closed 同源）。"""
    if tid in (5, 3):
        A = anchors(f"{ROOT}/libero/out/bank/libero_anchors.json")
        R = [json.loads(l) for l in open(f"{RUNS}/lingbot_base/records_t{tid}.jsonl")]
        return [(r, A[r["s0_id"]]) for r in R]
    A = anchors(f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json")
    R = {}
    for f in sorted(glob.glob(f"{RUNS}/lingbot_t{tid}/base/records.*.jsonl")):
        for l in open(f):
            r = json.loads(l); R[r["unit_id"]] = r
    return [(r, A[r["anchor_id"]]) for r in sorted(R.values(), key=lambda r: r["unit_id"])]


class Recorder:
    def __init__(self):
        self.cmd, self.exe, self.av, self.wr, self.succ = [], [], [], [], []


def make_rig_class():
    import lingbot_unit as LU

    class Rig(LU.LingBotRig):
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
            return env, obs

        def to_env_action(self, a_norm):
            a = LU.LingBotRig.to_env_action(self, a_norm)
            if self.rec is not None:
                self.rec.cmd.append(np.asarray(a[0], np.float32).copy())
            return a
    return Rig


def closed_f(rig, anchor, seed, t_nom, keep_passed=False):
    """lingbot_unit.LingBotRig.closed_loop 的同一循环 + F 挂点。t_nom=None → sham（不置黑）。
    返回 dict(success, first_success_step, steps, actions, blackout_step, plan_steps, passed_av?, wrist_same_object)."""
    import torch
    env, obs = rig.make_env(anchor)
    rig.reset_policy(seed)
    p = rig.policy
    acts, first, applied, plans, passed, wr_same = [], None, None, [], [], True
    with torch.inference_mode():
        for t in range(H):
            will_plan = (not p._started) or len(p._action_queue) == 0       # 本步 select_action 会触发规划调用
            if will_plan:
                plans.append(t)
            if t_nom is not None and applied is None and will_plan and t >= t_nom:
                raw_wr = obs["pixels"]["image2"]
                obs = dict(obs); obs["pixels"] = dict(obs["pixels"])
                obs["pixels"]["image"] = FD.blackout(obs["pixels"]["image"])      # 模型预处理之前置黑
                wr_same = wr_same and (obs["pixels"]["image2"] is raw_wr)
                applied = t
            if keep_passed:
                passed.append(np.asarray(obs["pixels"]["image"][0]).copy())
            batch = rig.obs_to_batch(obs)
            a = rig.to_env_action(p.select_action(batch))
            acts.append(a[0])
            obs, r, term, trunc, info = env.step(a)
            s = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0]) if "is_success" in info else False
            if "final_info" in info:
                fi = info["final_info"]
                if isinstance(fi, dict) and "is_success" in fi:
                    s = s or bool(np.asarray(fi["is_success"]).reshape(-1)[0])
            if s and first is None:
                first = t + 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
    env.close()
    out = dict(success=first is not None, first_success_step=first, steps=len(acts), actions=np.stack(acts),
               blackout_step=applied, plan_steps=plans, wrist_same_object=wr_same)
    if keep_passed:
        out["passed_av"] = np.stack(passed)
    return out


def set_H(tid):
    import lingbot_unit as LU
    LU.H = 520 if tid in (5, 3) else H          # 与各自基线闭环一致（T0/T1 基线 episode_length 520、循环截到 400；T2–T4 = 400）


def run_unit(rig, anchor, r, setting, tid):
    """一遍：Recorder + lb_capture 包住 closed_f。"""
    import lb_capture
    rec = Recorder(); rig.rec = rec
    with lb_capture.Capture() as cap:
        c = closed_f(rig, anchor, r["imagination_seed"], FD.LEVELS[setting])
    rig.rec = None
    return c, rec, cap


def main():
    import torch
    import lb_capture
    lb_capture.install()
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    tids = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(TIDS)
    Rig = make_rig_class()
    os.makedirs(f"{OUT}/closed", exist_ok=True)
    for tid in tids:
        done = {json.loads(l)["unit_id"] for f in glob.glob(f"{OUT}/closed_t{tid}.*.jsonl") for l in open(f)}
        jobs = [(r, a, s) for r, a in base_units(tid) for s in FD.SETTINGS
                if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
        print("LB_FAMF", f"t{tid}", "jobs", len(jobs), flush=True)
        if not jobs:
            continue
        set_H(tid); rig = Rig(tid)
        path = f"{OUT}/closed_t{tid}.{shard}.jsonl"
        for r, a, s in jobs:
            uid = f"{r['unit_id']}__{s}"; t0 = time.time()
            res = [run_unit(rig, a, r, s, tid) for _ in (1, 2)]
            (c1, rec1, cap1), (c2, rec2, cap2) = res
            n = min(len(c1["actions"]), len(c2["actions"]))
            dd = np.where(np.any(c1["actions"][:n] != c2["actions"][:n], axis=1))[0]
            div = int(dd[0]) + 1 if len(dd) else (None if len(c1["actions"]) == len(c2["actions"]) else n + 1)
            d = {}
            for i, (c, rec, cap) in enumerate(res, 1):
                p = "" if i == 1 else "run2_"
                d.update({f"{p}commanded": np.stack(rec.cmd), f"{p}executed": np.stack(rec.exe),
                          f"{p}agentview": np.stack(rec.av), f"{p}wrist": np.stack(rec.wr),
                          f"{p}success": np.array(rec.succ, bool), f"{p}blackout_step": np.array(-1 if c["blackout_step"] is None else c["blackout_step"]),
                          f"{p}plan_steps": np.array(c["plan_steps"])})
                d.update(cap.arrays(prefix=f"{p}imag_"))
            tmp = f"{OUT}/closed/{uid}.tmp.npz"
            np.savez_compressed(tmp, **d); os.replace(tmp, f"{OUT}/closed/{uid}.npz")
            row = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, family="F", task=f"t{tid}", paper_task=TNAME[tid],
                       anchor_id=a["anchor_id"], imagination_seed=r["imagination_seed"], H_env=H,
                       perturbation=FD.spec("lingbot", s, c1["blackout_step"]),
                       blackout_applied=c1["blackout_step"] is not None,
                       blackout_note=None if c1["blackout_step"] is not None else "episode ended (success) before the first planning call at or after nominal t",
                       closed_loop=dict(success=c1["success"], first_success_step=c1["first_success_step"], steps=c1["steps"],
                                        deterministic_rerun_identical=div is None,
                                        rerun=dict(success=c2["success"], first_success_step=c2["first_success_step"], steps=c2["steps"],
                                                   blackout_step=c2["blackout_step"]),
                                        first_divergence_step=div, conclusion_consistent=c1["success"] == c2["success"]),
                       closed_loop_file=os.path.relpath(f"{OUT}/closed/{uid}.npz", ROOT),
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
                       min_artifacts=1)
            MA.assert_unit(f"{OUT}/closed/{uid}.npz", row)        # 缺任何一项抛错停跑，不写这一行
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(uid, c1["success"], c2["success"], "blackout", c1["blackout_step"], c2["blackout_step"], row["seconds"], flush=True)
            torch.cuda.empty_cache()
        del rig; torch.cuda.empty_cache()
    print("LB_FAMF_DONE shard", shard)


def sham(n=5):
    """F0 走 F 挂点路径（不置黑）→ 与基线存档闭环动作逐位相同（T0/T1，基线 closed_loop[:400]）。"""
    Rig = make_rig_class(); ok = []
    for tid in (5, 3):
        set_H(tid); rig = Rig(tid)
        for r, a in base_units(tid)[:n]:
            ref = np.load(f"{ROOT}/{r['action_file']}")["closed_loop"][:H]
            c = closed_f(rig, a, r["imagination_seed"], None)
            same = bool(c["actions"].shape == ref.shape and np.array_equal(c["actions"], ref))
            ok.append(same); print("F sham", r["unit_id"], same, c["actions"].shape, ref.shape, flush=True)
        del rig
    print("F_SHAM", "PASS" if all(ok) else "FAIL", sum(ok), "/", len(ok))
    return all(ok)


if __name__ == "__main__":
    if sys.argv[1] == "sham":
        sys.exit(0 if sham(int(sys.argv[2]) if len(sys.argv) > 2 else 5) else 1)
    elif sys.argv[1] == "hc":
        import hc_lingbot
        hc_lingbot.main()
    else:
        main()
