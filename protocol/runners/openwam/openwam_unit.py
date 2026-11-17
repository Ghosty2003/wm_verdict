"""OpenWAM-alpha precheck (2026-10-23 ruling): one LIBERO-10 anchor, pure imagination to H = 520 by chaining 17 calls.

Model: OpenWAM/OpenWAM-Alpha-Sim-LIBERO (dual_system / joint_self_attn, mutual mask, Wan2.2-TI2V-5B), repo
OpenWAM-Official/OpenWAM @ 1e103e1. One call = joint denoising of 32 actions + 9 video frames (video stride 4; frame k
= after actions 0..4k-1) from ONE first frame + 10-D proprio + task.language.

Chaining (protocol note; training-external use, the code has no multi-chunk rollout): call c+1 starts from call c's
last generated frame (the 320x384 L-shaped canvas, fed back as is) and from an IMAGINED proprio obtained by
integrating call c's own actions with the OSC_POSE semantics (position += 0.05 m * a_xyz; orientation = Exp(0.5 rad *
a_rotvec) @ R, world frame; gripper channel = commanded open-scale). All 32 actions of every call are used
(17 x 32 = 544 >= 520); the official closed-loop eval executes 10 per call and replans from the real observation.
Determinism: dit_cache off, torch.compile off, SDPA attention, seed = base + call index.

Two interpreters (as for WorldVLA):
  env side   (lerobot_libero)  first_frame --anchor A          -> runs/<A>/first_frame.npz
                               replay --anchor A               -> runs/<A>/replay.npz   (truth (a), twice)
  model side (openwam)         imagine --anchor A --seed S     -> runs/<A>/imagined.npz + imagined.json
                               probe --anchor A --seed S       -> runs/<A>/probe.json   (joint-consistency check)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _wmvp.P("$OPENWAM_REPO")
ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{HERE}/runs"
CKPT = f"{REPO}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
H = 520
N_CALLS = 17
OSC_POS, OSC_ROT = 0.05, 0.5          # robosuite OSC_POSE output_max (LIBERO default controller)
RES = 256                              # official client camera size


# ===================================================================================================== env side
def _env(anchor_id):
    _wmvp.bootstrap()
    from libero_env_adapter import _make_env, _anchor
    a = dict(_anchor(f"{ROOT}/libero/out/bank/libero_anchors.json", anchor_id), resolution=RES)
    return _make_env(a, H=H)


def _raw(u):
    o = u._env.env._get_observations(force_update=True)
    return {k: np.asarray(o[k]).copy() for k in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                                                 "robot0_eef_quat", "robot0_gripper_qpos")}


def cmd_first_frame(a):
    from libero.libero import benchmark
    for anchor in a.anchors.split(","):
        os.makedirs(f"{OUT}/{anchor}", exist_ok=True)
        if os.path.exists(f"{OUT}/{anchor}/first_frame.npz"):
            continue
        env = _env(anchor); u = env.envs[0]; o = _raw(u); env.close()
        tid = int(anchor.split("_t")[1].split("_")[0])
        o["language"] = np.array(benchmark.get_benchmark_dict()["libero_10"]().get_task(tid).language)   # = official client prompt
        np.savez_compressed(f"{OUT}/{anchor}/first_frame.npz", **o)
        print("first frame", anchor, {k: v.shape for k, v in o.items()})


def _replay_once(anchor, acts):
    import mujoco
    env = _env(anchor); u = env.envs[0]
    m, d = u._env.sim.model._model, u._env.sim.data._data
    r0 = _raw(u)
    S, succ, frames = [], [], [r0["agentview_image"][::-1, ::-1]]
    eef = [np.concatenate([r0["robot0_eef_pos"], r0["robot0_eef_quat"], r0["robot0_gripper_qpos"]])]
    for t, x in enumerate(acts):
        u.step(np.asarray(x, np.float32)); succ.append(bool(u._env.check_success()))
        r_ = _raw(u); eef.append(np.concatenate([r_["robot0_eef_pos"], r_["robot0_eef_quat"], r_["robot0_gripper_qpos"]]))
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        S.append(st)
        if (t + 1) % 4 == 0:
            frames.append(r_["agentview_image"][::-1, ::-1])
    env.close()
    return np.stack(S), np.array(succ), np.stack(frames), np.stack(eef)


def _rundir(anchor, chain):
    d = f"{OUT}/{anchor}/{chain}"; os.makedirs(d, exist_ok=True); return d


def cmd_replay(a):
    for anchor in a.anchors.split(","):
        for chain in a.chains.split(","):
            d = _rundir(anchor, chain)
            acts = np.load(f"{d}/imagined.npz")["env_actions"][:H]
            S1, s1, F, E = _replay_once(anchor, acts); S2, s2, _, _ = _replay_once(anchor, acts)
            same = bool(np.array_equal(S1, S2) and np.array_equal(s1, s2))
            np.savez_compressed(f"{d}/replay.npz", succ=s1, frames_every4=F, states=S1, eef=E)
            r = dict(anchor=anchor, chain=chain, success=bool(s1.any()),
                     first_success_step=int(np.argmax(s1)) + 1 if s1.any() else None, replay_bit_identical=same)
            json.dump(r, open(f"{d}/replay.json", "w"), indent=1); print("REPLAY", json.dumps(r), flush=True)


# =================================================================================================== model side
def _setup():
    os.environ.setdefault("WAM_ATTENTION_IMPL", "sdpa")
    sys.path[:0] = [REPO, f"{REPO}/third_party"]


def _engine():
    _setup()
    from omegaconf import OmegaConf
    from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    dep = OmegaConf.load(f"{REPO}/configs/deploy.yaml"); OmegaConf.update(dep, "defaults", [], merge=False)
    dep.optimization.decode_video = True; dep.optimization.dit_cache.enabled = False; dep.optimization.compile.enabled = False
    name = sorted(f for f in os.listdir(CKPT) if f.endswith(".safetensors"))[-1]
    t0 = time.time()
    tcfg, arch = load_from_checkpoint_dir(CKPT, device="cuda:0", ckpt_name=name)
    cfg = merge_deploy_cfg(tcfg, dep); eng = JointInferenceEngine(cfg=cfg, architecture=arch)
    return eng, ObsPreprocessor.from_cfg(cfg, eng), arch, time.time() - t0, name


def _canvas(pre, o, state):
    from PIL import Image
    from benchmarks.utils.client import resize_for_lshape_slot
    f = lambda k, slot: Image.fromarray(resize_for_lshape_slot(o[k][::-1, ::-1], slot))
    return pre.preprocess({"images": {"head_camera": f("agentview_image", "head_camera"),
                                      "left_wrist_camera": f("robot0_eye_in_hand_image", "left_wrist_camera"),
                                      "right_wrist_camera": None}, "state": state})["image"]


def _integrate(state, acts10):
    """imagined proprio after executing acts10 (32 x 10 native EEF10) from `state` (raw 10-D)."""
    from scipy.spatial.transform import Rotation as Rot
    from benchmarks.utils.action_conversion import quat_xyzw_to_rot6d, rot6d_to_axis_angle
    pos = state[:3].astype(np.float64).copy()
    r6 = state[3:9]
    # rot6d -> matrix via the repo's own inverse (axis-angle), avoids guessing the column / row convention
    R = Rot.from_rotvec(rot6d_to_axis_angle(r6.astype(np.float32))).as_matrix()
    grip = state[9]
    for a in acts10:
        pos += OSC_POS * np.clip(a[:3], -1, 1)
        rv = np.clip(rot6d_to_axis_angle(a[3:9].astype(np.float32)), -1, 1)
        R = Rot.from_rotvec(OSC_ROT * rv).as_matrix() @ R
        grip = float(np.clip(a[9], -1, 1))
    q = Rot.from_matrix(R).as_quat()                                       # xyzw
    return np.concatenate([pos, quat_xyzw_to_rot6d(q[None].astype(np.float32))[0], [grip]]).astype(np.float32)


class Chainer:
    """imagined proprio between calls (protocol note: chaining options of the 2026-10-23 ruling).
      integrate   naive OSC_POSE integration of the call's own actions (0.05 m / 0.5 rad per unit command)
      reuse       the anchor's initial state for every call
      calibrated  ARX controller model fitted on REAL replays (fit_controller.py; an injected component)"""

    def __init__(self, kind, state0):
        self.kind, self.state0 = kind, state0.copy()
        self.ctx = (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        if kind == "calibrated":
            self.params = json.load(open(f"{HERE}/controller/params.json"))

    def next(self, state, acts10):
        if self.kind == "reuse":
            return self.state0.copy()
        if self.kind == "integrate":
            return _integrate(state, acts10)
        from scipy.spatial.transform import Rotation as Rot
        from benchmarks.utils.action_conversion import quat_xyzw_to_rot6d, rot6d_to_axis_angle
        P = self.params
        p = state[:3].astype(np.float64).copy(); R = Rot.from_rotvec(rot6d_to_axis_angle(state[3:9].astype(np.float32)))
        g = float(state[9]); ux0, dp0, ur0, w0 = self.ctx
        for a in acts10:
            ux = np.clip(a[:3], -1, 1); ur = np.clip(rot6d_to_axis_angle(a[3:9].astype(np.float32)), -1, 1); cg = float(np.clip(a[9], -1, 1))
            dp = np.array([P["pos"][k][0] * ux[k] + P["pos"][k][1] * ux0[k] + P["pos"][k][2] * dp0[k] for k in range(3)])
            w = np.array([P["rot"][k][0] * ur[k] + P["rot"][k][1] * ur0[k] + P["rot"][k][2] * w0[k] for k in range(3)])
            p = p + dp; R = Rot.from_rotvec(w) * R; g = g + P["grip_k"] * (cg - g)
            ux0, dp0, ur0, w0 = ux, dp, ur, w
        self.ctx = (ux0, dp0, ur0, w0)
        return np.concatenate([p, quat_xyzw_to_rot6d(R.as_quat()[None].astype(np.float32))[0], [g]]).astype(np.float32)


def _health(acts10):
    """per-call output health of the policy head on imagined frames (OpenWAM has no token grammar: 'malformed' =
    non-finite, saturated, or a broken rotation)."""
    from benchmarks.utils.action_conversion import rot6d_to_axis_angle
    fin = bool(np.isfinite(acts10).all())
    xyz = acts10[:, :3]; r6 = acts10[:, 3:9]
    a1, a2 = r6[:, :3], r6[:, 3:6]
    ortho = np.abs((a1 * a2).sum(1)) / (np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1) + 1e-9)
    rv = np.stack([rot6d_to_axis_angle(x.astype(np.float32)) for x in r6]) if fin else np.full((len(r6), 3), np.nan)
    sat = float(np.mean(np.abs(np.concatenate([xyz, rv], 1)) > 1.0)) if fin else 1.0
    malformed = (~np.isfinite(acts10).all(1)) | (np.abs(xyz) > 1.5).any(1) | (ortho > 0.5) | (np.abs(acts10[:, 9]) > 1.5)
    return dict(finite=fin, saturated_share=round(sat, 4), malformed_actions=int(malformed.sum()),
                mean_abs_xyz=round(float(np.abs(xyz).mean()), 4), action_std_over_chunk=round(float(acts10[:, :9].std(0).mean()), 4),
                rot6d_nonortho_max=round(float(ortho.max()), 4), gripper_range=[round(float(acts10[:, 9].min()), 3), round(float(acts10[:, 9].max()), 3)])


def cmd_imagine(a):
    import torch
    _setup()
    from benchmarks.utils.action_conversion import libero_obs_to_eef10
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    eng, pre, arch, t_load, ckname = _engine()
    for anchor in a.anchors.split(","):
        for chain in a.chains.split(","):
            _imagine_one(a, eng, pre, t_load, ckname, anchor, chain, libero_obs_to_eef10, native_eef10_to_libero7d)


def _imagine_one(a, eng, pre, t_load, ckname, anchor, chain, libero_obs_to_eef10, native_eef10_to_libero7d):
    import torch
    d = _rundir(anchor, chain)
    o = dict(np.load(f"{OUT}/{anchor}/first_frame.npz"))
    state = libero_obs_to_eef10(o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"])
    img = _canvas(pre, o, state); prompt = str(o.pop("language")); ch = Chainer(chain, state)
    frames, acts10, env7, calls = [np.asarray(img)], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for c in range(N_CALLS):
        torch.cuda.synchronize(); t0 = time.time()
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": state, "seed": a.seed + c})
        torch.cuda.synchronize(); dt = time.time() - t0
        A = np.asarray(out["actions"], np.float32)[:, :10]
        vid = [np.asarray(v) for v in out["video"]]
        hl = _health(A)
        motion = float(np.mean([np.abs(vid[k + 1].astype(float) - vid[k].astype(float)).mean() for k in range(len(vid) - 1)]))
        calls.append(dict(call=c, seconds=round(dt, 2), n_frames=len(vid), video_motion_mean_absdiff=round(motion, 3), **hl,
                          state_in=[round(float(x), 4) for x in state]))
        print(json.dumps(calls[-1]), flush=True)
        frames += vid[1:]; acts10.append(A)
        env7 += [native_eef10_to_libero7d(x) for x in A]
        img = out["video"][-1]
        state = ch.next(state, A)
    acts10 = np.concatenate(acts10); env7 = np.stack(env7)
    np.savez_compressed(f"{d}/imagined.npz", frames=np.stack(frames), actions10=acts10, env_actions=env7)
    info = dict(anchor=anchor, seed=a.seed, chain=chain, checkpoint=ckname, repo_commit="1e103e1", n_calls=N_CALLS, n_actions=len(env7),
                n_frames=len(frames), frame_convention="320x384 canvas (agentview 320x256 top, wrist 160x128 bottom-left); frame g = after 4g actions",
                chaining=f"last generated frame + imagined proprio ({chain}, see Chainer); all 32 actions per call",
                load_seconds=round(t_load, 1), seconds_total=round(sum(x["seconds"] for x in calls), 1),
                seconds_per_call_mean=round(float(np.mean([x["seconds"] for x in calls])), 2),
                gpu_peak_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2), calls=calls)
    json.dump(info, open(f"{d}/imagined.json", "w"), indent=1)
    print("IMAGINE", json.dumps({k: v for k, v in info.items() if k != "calls"}))


def cmd_probe(a):
    """joint-consistency check (test harness only; never used for verdicts): same first frame, proprio and seed;
    actions CLAMPED (action noise level held at 0, initial action latents replaced) to (i) the model's own actions and
    (ii) altered actions (xyz deltas negated). The generated video must differ between (i) and (ii); (i) twice must be
    identical."""
    import torch
    _setup()
    from benchmarks.utils.action_conversion import libero_obs_to_eef10
    eng, pre, arch, _, _ = _engine()
    import openwam.deploy.engine as E
    o = dict(np.load(f"{OUT}/{a.anchor}/first_frame.npz"))
    state = libero_obs_to_eef10(o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"])
    img = _canvas(pre, o, state); prompt = str(o.pop("language"))
    cond = {"first_frame_image": [img], "prompt": prompt, "proprio": state, "seed": a.seed}
    captured = {}
    unnorm = arch.normalizer.unnormalize
    arch.normalizer.unnormalize = lambda x: (captured.__setitem__("norm", np.array(x)), unnorm(x))[1]
    free = eng.generate(cond); arch.normalizer.unnormalize = unnorm
    own = torch.tensor(captured["norm"])[None]                         # (1, 32, 80) normalized
    alt = own.clone(); alt[..., :3] = -alt[..., :3]
    make_schedule = E.make_schedule
    randn = torch.randn

    def run(clamp):
        E.make_schedule = lambda *x, **k: [(tv, 0.0) for tv, _ in make_schedule(*x, **k)]
        torch.randn = lambda *s, **k: (clamp.to(device=k.get("device"), dtype=k.get("dtype")) if tuple(s) == tuple(clamp.shape) else randn(*s, **k))
        try:
            return [np.asarray(v).astype(float) for v in eng.generate(dict(cond))["video"]]
        finally:
            E.make_schedule = make_schedule; torch.randn = randn

    v_own, v_own2, v_alt = run(own), run(own), run(alt)
    psnr = lambda x, y: 99.0 if ((x - y) ** 2).mean() == 0 else float(10 * np.log10(255 ** 2 / ((x - y) ** 2).mean()))
    r = dict(anchor=a.anchor, seed=a.seed,
             own_vs_own_maxabs=float(max(np.abs(x - y).max() for x, y in zip(v_own, v_own2))),
             own_vs_alt_psnr_by_frame=[round(psnr(x, y), 2) for x, y in zip(v_own, v_alt)],
             own_vs_alt_mean_absdiff_by_frame=[round(float(np.abs(x - y).mean()), 3) for x, y in zip(v_own, v_alt)],
             clamped_own_vs_free_psnr_by_frame=[round(psnr(x, np.asarray(y).astype(float)), 2) for x, y in zip(v_own, free["video"])],
             note="frame 0 is the reconstructed first frame in all runs; frames 1..8 must differ between own and altered actions")
    r["pass"] = bool(r["own_vs_own_maxabs"] == 0 and min(r["own_vs_alt_mean_absdiff_by_frame"][2:]) > 1.0)
    json.dump(r, open(f"{OUT}/{a.anchor}/probe.json", "w"), indent=1); print("PROBE", json.dumps(r))


def cmd_compare(a):
    """per anchor x chain: boundary state error (imagined proprio fed to call c vs the REAL end effector after replaying
    that rollout's own actions to step 32c), PSNR of the imagined agentview region vs the paired truth frame at the same
    step, video motion per call, truth outcome; plus a final-frame sheet."""
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation as Rot
    sys.path[:0] = [REPO]
    from benchmarks.utils.action_conversion import rot6d_to_axis_angle
    rows, sheet = {}, []
    for anchor in a.anchors.split(","):
        for chain in a.chains.split(","):
            d = _rundir(anchor, chain); I = json.load(open(f"{d}/imagined.json")); Z = np.load(f"{d}/imagined.npz")
            Rp = np.load(f"{d}/replay.npz"); E = Rp["eef"]; rj = json.load(open(f"{d}/replay.json"))
            perr, rerr = [], []
            for c in range(1, N_CALLS):
                s_ = 32 * c
                if s_ >= len(E):
                    break
                st = np.array(I["calls"][c]["state_in"], np.float64)
                perr.append(float(np.linalg.norm(st[:3] - E[s_, :3])))
                Rm = Rot.from_rotvec(rot6d_to_axis_angle(st[3:9].astype(np.float32))); Rt = Rot.from_quat(E[s_, 3:7])
                rerr.append(float(np.degrees((Rm * Rt.inv()).magnitude())))
            fr = Z["frames"]; tf = Rp["frames_every4"]; ps = []
            for g in range(min(len(fr), len(tf))):
                im = fr[g][:256].astype(float)
                tr = np.asarray(Image.fromarray(tf[g]).resize((320, 256), Image.LANCZOS)).astype(float)
                m = ((im - tr) ** 2).mean(); ps.append(99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m)))
            ps = np.array(ps)
            bnd = [g for g in range(8, len(ps), 8)]
            cliff = [round(float(ps[g - 1] - ps[g + 1]), 2) for g in bnd if g + 1 < len(ps)]
            rows[f"{anchor}/{chain}"] = dict(
                truth_success=rj["success"], replay_bit_identical=rj["replay_bit_identical"],
                boundary_pos_err_m=[round(x, 3) for x in perr], boundary_rot_err_deg=[round(x, 1) for x in rerr],
                psnr_mean_by_call=[round(float(ps[1 + 8 * c:1 + 8 * (c + 1)].mean()), 2) for c in range(N_CALLS) if 1 + 8 * c < len(ps)],
                psnr_drop_across_block_boundary_db=cliff,
                motion_by_call=[x["video_motion_mean_absdiff"] for x in I["calls"]],
                malformed_actions=sum(x["malformed_actions"] for x in I["calls"]))
            np.save(f"{d}/psnr_curve.npy", ps)
            t = Image.fromarray(fr[min(len(fr), len(tf)) - 1][:256]); u_ = Image.fromarray(tf[min(len(fr), len(tf)) - 1]).resize((320, 256))
            c_ = Image.new("RGB", (646, 272), "white"); c_.paste(t, (0, 16)); c_.paste(u_, (326, 16))
            ImageDraw.Draw(c_).text((3, 2), f"{anchor} / {chain}: imagined end (left) vs real (right), truth={'S' if rj['success'] else 'fail'}", fill="black")
            sheet.append(c_)
            print(f"{anchor}/{chain}", json.dumps(rows[f"{anchor}/{chain}"]), flush=True)
    json.dump(rows, open(f"{OUT}/compare_chains.json", "w"), indent=1)
    W = Image.new("RGB", (646 * len(a.chains.split(",")), 272 * len(a.anchors.split(","))), "white")
    for i, c_ in enumerate(sheet):
        W.paste(c_, ((i % len(a.chains.split(","))) * 646, (i // len(a.chains.split(","))) * 272))
    W.save(f"{OUT}/compare_chains_end.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--anchor", default="libero10_t5_a0")
    ap.add_argument("--anchors", default="libero10_t5_a0"); ap.add_argument("--chains", default="integrate")
    ap.add_argument("--seed", type=int, default=20261035)
    a = ap.parse_args()
    {"first_frame": cmd_first_frame, "replay": cmd_replay, "imagine": cmd_imagine, "probe": cmd_probe,
     "compare": cmd_compare}[a.cmd](a)
