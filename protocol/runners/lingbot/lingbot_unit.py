"""LIBERO batch, LingBot-VA (lerobot/lingbot_va_libero_long): one paired unit = (anchor, imagination seed).

Model branch  pure imagination: the official chunk loop with the real-keyframe feedback (`_compute_kv_cache`) skipped,
              so chunk c is conditioned on the model's OWN predicted latents + actions of chunks < c (cached at t=0
              by `_infer`, update_cache=1). 33 chunks: 12 + 16 x 32 = 524 actions, truncated to H = 520.
              Within a chunk the video is denoised first and cached, then the actions attend to it (joint consistency).
Truth (a)     open-loop replay of the SAME imagined (post-processed) env actions from the same anchor, in a fresh env,
              twice (bit-exact replay check). Primary paired-verdict track.
Truth (b)     the model's normal closed-loop deployment (real keyframes fed back) from the same anchor with the same
              torch seed; per-step L2 between executed and imagined actions. Reference column only (not paired).
Observation / action processing is the lerobot-eval chain verbatim:
  preprocess_observation -> env_preprocessor -> preprocessor -> policy -> postprocessor -> env_postprocessor -> env.
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import hashlib
from copy import deepcopy

import numpy as np
import torch
import gymnasium as gym
import mujoco

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

CKPT = "lerobot/lingbot_va_libero_long"
H = 520


class LingBotRig:
    def __init__(self, task_id, device="cuda"):
        self.task_id = task_id
        self.suite = benchmark.get_benchmark_dict()["libero_10"]()
        self.env_cfg = LiberoEnvCfg(task="libero_10", task_ids=[task_id], observation_height=128, observation_width=128)
        cfg = PreTrainedConfig.from_pretrained(CKPT)
        cfg.pretrained_path = CKPT
        cfg.device = device
        cfg.save_predicted_video = True              # keep each chunk's predicted latents (last_predicted_latents)
        self.cfg = cfg
        self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.task_desc = None

    # ---------------------------------------------------------------- env
    def make_env(self, anchor):
        if anchor.get("native"):                 # family A: LIBERO native hooks (native_env.py), init_states=False
            from native_env import make_native
            env = gym.vector.SyncVectorEnv([lambda: make_native(anchor, self.suite, H=H)])
            obs, info = env.reset(seed=[anchor["seed"]])
            self.task_desc = list(env.call("task_description"))
            return env, obs

        def _f():
            return LiberoEnv(task_suite=self.suite, task_id=anchor["task_id"], task_suite_name="libero_10",
                             observation_width=128, observation_height=128, init_states=True,
                             episode_index=anchor["init_index"], n_envs=1, num_steps_wait=anchor["settle_steps"],
                             hard_reset=True, episode_length=H)
        env = gym.vector.SyncVectorEnv([_f])
        obs, info = env.reset(seed=[anchor["seed"]])
        self.task_desc = list(env.call("task_description"))
        return env, obs

    def obs_to_batch(self, obs):
        o = preprocess_observation(obs)
        o["task"] = self.task_desc
        o = self.env_pre(o)
        return self.pre(o)

    def to_env_action(self, a_norm):          # a_norm: [1, n_used] normalized
        a = self.post(a_norm)
        a = self.env_post({ACTION: a})[ACTION]
        return a.to("cpu").numpy()

    def reset_policy(self, seed):
        torch.manual_seed(seed)
        self.policy.reset(); self.pre.reset(); self.post.reset()

    # ---------------------------------------------------------------- model branch
    @torch.inference_mode()
    def imagine(self, anchor, seed):
        env, obs = self.make_env(anchor)
        batch = self.obs_to_batch(obs)
        env.close()
        self.reset_policy(seed)
        p = self.policy
        p._ensure_frozen_modules()
        p._maybe_init_prompt(batch)
        t0 = time.time()
        acts_norm = [p.predict_action_chunk(batch)]            # chunk 0 (12 actions); conditions on the real frame
        lat = [p.last_predicted_latents]
        c = 1
        while sum(a.shape[1] for a in acts_norm) < H:
            fsid = 4 * c                                       # = what _compute_kv_cache would have advanced to
            p._frame_st_id = fsid
            actions, latents = p._infer(None, frame_st_id=fsid)   # NO real-keyframe feedback: own predictions as history
            p._executed_actions = actions
            used = p.config.used_action_channel_ids
            a = actions[:, used].squeeze(-1).flatten(2).transpose(1, 2).contiguous().to(torch.float32)
            acts_norm.append(a)
            lat.append(latents.detach().to("cpu"))
            c += 1
        t_img = time.time() - t0
        A = torch.cat(acts_norm, dim=1)[:, :H]                 # [1, H, n_used] normalized
        env_actions = np.concatenate([self.to_env_action(A[:, t]) for t in range(H)], 0)   # [H, 7]
        latents = torch.cat(lat, dim=2)                        # [1, 48, 4*n_chunks, h, w]
        return dict(env_actions=env_actions, actions_norm=A[0].cpu().numpy(), latents=latents,
                    n_chunks=c, seconds_imagine=t_img)

    # ---------------------------------------------------------------- truth (a): open-loop replay
    def replay(self, anchor, env_actions, record_frames=False):
        env, obs = self.make_env(anchor)
        u = env.envs[0]
        frames, succ, states = [], [], []
        def snap():
            m, d = u._env.sim.model._model, u._env.sim.data._data
            st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
            mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            return st
        states.append(snap())
        if record_frames:
            frames.append((obs["pixels"]["image"][0].copy(), obs["pixels"]["image2"][0].copy()))
        first = None
        for t, a in enumerate(env_actions):
            obs, r, term, trunc, info = u.step(a.astype(np.float32))
            s = bool(info.get("is_success", False))
            succ.append(s)
            states.append(snap())
            if record_frames:
                frames.append((obs["pixels"]["image"].copy(), obs["pixels"]["image2"].copy()))
            if s and first is None:
                first = t + 1
        env.close()
        return dict(success=first is not None, first_success_step=first, succ=np.array(succ), states=np.stack(states),
                    frames=frames)

    # ---------------------------------------------------------------- truth (b): closed loop
    @torch.inference_mode()
    def closed_loop(self, anchor, seed, imagined_env_actions):
        env, obs = self.make_env(anchor)
        self.reset_policy(seed)
        t0 = time.time()
        acts, succ, first = [], [], None
        for t in range(H):
            batch = self.obs_to_batch(obs)
            a = self.to_env_action(self.policy.select_action(batch))
            acts.append(a[0])
            obs, r, term, trunc, info = env.step(a)
            s = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0]) if "is_success" in info else False
            if "final_info" in info:
                fi = info["final_info"]
                if isinstance(fi, dict) and "is_success" in fi:
                    s = s or bool(np.asarray(fi["is_success"]).reshape(-1)[0])
            succ.append(s)
            if s and first is None:
                first = t + 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
        env.close()
        acts = np.stack(acts)
        n = len(acts)
        dev = np.linalg.norm(acts - imagined_env_actions[:n], axis=1)
        return dict(success=first is not None, first_success_step=first, steps=n, actions=acts,
                    action_dev_l2=dev, seconds=time.time() - t0)

    # ---------------------------------------------------------------- decoding
    @torch.no_grad()
    def decode(self, latents):
        return self.policy.decode_predicted_latents(latents)   # [T, H, W, 3] uint8


if __name__ == "__main__":
    # smoke + cost: one unit on anchor libero10_t5_a0, seed 1
    bank = json.load(open(_wmvp.P("$WMV_ROOT/libero/out/bank/libero_anchors.json")))
    a = [x for x in bank["anchors"] if x["anchor_id"] == sys.argv[1] if len(sys.argv) > 1] or [bank["anchors"][5]]
    a = a[0]
    rig = LingBotRig(a["task_id"])
    out = {}
    t0 = time.time(); I1 = rig.imagine(a, seed=1); out["imagine_s"] = time.time() - t0
    I2 = rig.imagine(a, seed=1)
    out["imagine_repeat_identical_actions"] = bool(np.array_equal(I1["env_actions"], I2["env_actions"]))
    out["imagine_repeat_max_action_diff"] = float(np.abs(I1["env_actions"] - I2["env_actions"]).max())
    out["n_chunks"] = I1["n_chunks"]; out["latent_shape"] = list(I1["latents"].shape)
    torch.cuda.synchronize(); t0 = time.time()
    lat1 = I1["latents"][:, :, 4:8]
    f1 = rig.decode(lat1); torch.cuda.synchronize(); out["decode_one_chunk_resident_s"] = time.time() - t0
    out["decode_one_chunk_frames"] = list(f1.shape)
    out["gpu_mem_after_decode_GB"] = torch.cuda.max_memory_allocated() / 1e9
    t0 = time.time(); F = rig.decode(I1["latents"]); torch.cuda.synchronize(); out["decode_full_resident_s"] = time.time() - t0
    out["decode_full_frames"] = list(F.shape)
    t0 = time.time(); R1 = rig.replay(a, I1["env_actions"]); R2 = rig.replay(a, I1["env_actions"]); out["replay2x_s"] = time.time() - t0
    out["replay_identical"] = bool(np.array_equal(R1["states"], R2["states"]))
    out["truth_a_success"] = R1["success"]; out["truth_a_first_step"] = R1["first_success_step"]
    CL = rig.closed_loop(a, 1, I1["env_actions"])
    out["truth_b_success"] = CL["success"]; out["truth_b_first_step"] = CL["first_success_step"]; out["truth_b_s"] = CL["seconds"]
    out["truth_b_dev_l2_first12_mean"] = float(CL["action_dev_l2"][:12].mean()); out["truth_b_dev_l2_mean"] = float(CL["action_dev_l2"].mean())
    out["gpu_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
    os.makedirs(_wmvp.P("$WMV_ROOT/libero/out/cost"), exist_ok=True)
    import imageio
    imageio.mimwrite(_wmvp.P(f"$WMV_ROOT/libero/out/cost/imagined_{a['anchor_id']}_s1.mp4"), F.numpy(), fps=20)
    json.dump(out, open(_wmvp.P(f"$WMV_ROOT/libero/out/cost/smoke_{a['anchor_id']}.json"), "w"), indent=1)
    print("SMOKE", json.dumps(out, indent=1))
