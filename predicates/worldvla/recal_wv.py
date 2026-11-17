"""WorldVLA-view predicate (appendix only; WorldVLA = exclusion E4). Same recipe as v3 (recal_v3.py) on the SAME
episodes, re-rendered in WorldVLA's view: LIBERO agentview at 256 px, rotated 180 deg (worldvla_unit.model_view),
no wrist camera; downsampled to 128x128 for the classifier. Budget: <= 4 h GPU (ruling 2026-10-15), else not done.

  render <task>   replay every v3 episode's recorded actions in a fresh env at 256 px (physics is render-size
                  independent; checked against the stored labels) -> out/predicate_wv/t<task>/frames/<episode>.npz
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
from PIL import Image

_wmvp.bootstrap()
_wmvp.bootstrap()
from recal_v3 import registry, _displace_fixed
from worldvla_unit import model_view

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/predicate_wv"
RES_RENDER = 256
RES_NET = 128


def wv_frame(raw):
    return np.asarray(Image.fromarray(model_view(raw)).resize((RES_NET, RES_NET), Image.BILINEAR))


def render(task_id):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    d = f"{OUT}/t{task_id}/frames"; os.makedirs(d, exist_ok=True)
    mism = 0
    for e in registry(task_id):
        path = f"{d}/{e['id']}.npz"
        if os.path.exists(path):
            continue
        a = e["anc"]; z = np.load(e["path"])
        env = LiberoEnv(task_suite=suite, task_id=a["task_id"], task_suite_name="libero_10", observation_width=RES_RENDER,
                        observation_height=RES_RENDER, init_states=True, episode_index=a["init_index"], n_envs=1,
                        num_steps_wait=a["settle_steps"], hard_reset=True, episode_length=520)
        obs, _ = env.reset(seed=a["seed"])
        if e["disp"]:
            _displace_fixed(env, e["disp"]["object"], e["disp"]["offset_xy_m"])
            for _ in range(10):
                obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
        F, L = [wv_frame(obs["pixels"]["image"])], [bool(env._env.check_success())]
        for act in z["actions"]:
            obs, *_ = env.step(np.asarray(act, np.float32))
            F.append(wv_frame(obs["pixels"]["image"])); L.append(bool(env._env.check_success()))
        env.close()
        L = np.array(L); same = bool(np.array_equal(L, z["labels"][:len(L)]))
        mism += (not same)
        np.savez_compressed(path, frames=np.stack(F), labels=L, labels_match_v3=same)
        print(f"t{task_id} {e['id']}: T={len(L)} labels_match_v3={same}", flush=True)
    print(f"t{task_id} render done; label mismatches {mism}", flush=True)


def render_rt(task_id):
    """every 6th frame at full 256 px (WorldVLA view) for the VQGAN round-trip row."""
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    d = f"{OUT}/t{task_id}/frames_rt"; os.makedirs(d, exist_ok=True)
    for e in registry(task_id):
        path = f"{d}/{e['id']}.npz"
        if os.path.exists(path):
            continue
        a = e["anc"]; z = np.load(e["path"])
        env = LiberoEnv(task_suite=suite, task_id=a["task_id"], task_suite_name="libero_10", observation_width=RES_RENDER,
                        observation_height=RES_RENDER, init_states=True, episode_index=a["init_index"], n_envs=1,
                        num_steps_wait=a["settle_steps"], hard_reset=True, episode_length=520)
        obs, _ = env.reset(seed=a["seed"])
        if e["disp"]:
            _displace_fixed(env, e["disp"]["object"], e["disp"]["offset_xy_m"])
            for _ in range(10):
                obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
        F, I = [model_view(obs["pixels"]["image"])], [0]
        for t, act in enumerate(z["actions"], 1):
            obs, *_ = env.step(np.asarray(act, np.float32))
            if t % RT_EVERY == 0:
                F.append(model_view(obs["pixels"]["image"])); I.append(t)
        env.close()
        np.savez_compressed(path, frames=np.stack(F), idx=np.array(I))
    print(f"t{task_id} render_rt done", flush=True)


RT_EVERY = 6


if __name__ == "__main__":
    {"render": render, "render_rt": render_rt}[sys.argv[1]](int(sys.argv[2]))
