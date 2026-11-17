"""verify_run.py check-2 / check-8 adapter for LIBERO runs (LeRobot LiberoEnv wrapper, hf-libero 0.1.4, robosuite 1.4.0,
mujoco 3.8.1). Must run inside the `lerobot_libero` conda environment.

replay_unit(task, bank_path, s0_id, env_actions, spec) restores the frozen anchor with the bank's recipe (a NEW
LiberoEnv(hard_reset=True, init_states=True) per restore -> reset(seed), settle steps inside reset; a reused env drifts
fixture poses, libero_native_hooks.md section 8) and replays the recorded post-processed env actions open loop, exactly
as the runner's truth (a) branch (lingbot_baseline.run_unit): env.envs[0].step(a), success = _env.check_success(),
state = mj_getState(FULLPHYSICS) after every step. states_b is a second restore made after another LiberoEnv instance
was created and stepped with random actions in the same process (dirtied process). dist = 0.0 (LIBERO success is a
predicate, not a distance; the records carry real_final_dist = 0.0). Family-A banks (native hooks, native_env.py):
s0_id = base anchor id, spec['familyA_level'] selects the level anchor; spec == {} -> untouched native path.
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import numpy as np

_cache = {}


def _anchor(path, s0_id):
    if path not in _cache:
        _cache[path] = {a["anchor_id"]: a for a in json.load(open(path))["anchors"]}
    return _cache[path][s0_id]


def _family_anchor(path, s0_id, spec):
    """family A banks: s0_id = base id, the level comes from spec['familyA_level']. spec == {} -> the UNTOUCHED native
    path (original BDDL, same seed) -- the reference for check 8 and for the A0 sham.
    family P banks (spec revision #3, 2026-10-20; eval_design section 22 rulings 2 and 5): level from
    spec['familyP_level']; spec == {} -> the P0 anchor, the family's own baseline. Check 8 then compares P1..P4 with P0,
    and P0 with a second independent P0 generation (the redefined P sham: same seed, bit-identical). P0 vs the native
    path differs by the known RNG-order artifact and is not a check-8 reference."""
    bank = json.load(open(path))
    if bank.get("family") == "P_pose":
        return _anchor(path, f"{s0_id}_{spec['familyP_level'] if spec else 'P0'}")
    if spec:
        return _anchor(path, f"{s0_id}_{spec['familyA_level']}")
    a = dict(_anchor(path, f"{s0_id}_A0"))
    a["bddl_file"] = bank["bddl"][f"t{a['task_id']}_original"]["path"]
    if "/libero/libero/bddl_files/" in a["bddl_file"]:      # release: the original BDDL ships with hf-libero; the bank
        import os                                                # recorded a machine-specific path (placeholder in the release)
        from libero.libero import get_libero_path
        a["bddl_file"] = os.path.join(get_libero_path("bddl_files"), a["bddl_file"].split("/libero/libero/bddl_files/", 1)[1])
    return a


def _make_env(a, H=520):
    import gymnasium as gym
    if a.get("native"):
        import sys, os
        _wmvp.bootstrap()
        from native_env import make_native
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[a["suite"]]()
        env = gym.vector.SyncVectorEnv([lambda: make_native(a, suite, H=H)])
        env.reset(seed=[a["seed"]])
        return env
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()[a["suite"]]()

    def _f():
        return LiberoEnv(task_suite=suite, task_id=a["task_id"], task_suite_name=a["suite"],
                         observation_width=a["resolution"], observation_height=a["resolution"], init_states=True,
                         episode_index=a["init_index"], n_envs=1, num_steps_wait=a["settle_steps"],
                         hard_reset=True, episode_length=H)
    env = gym.vector.SyncVectorEnv([_f])
    env.reset(seed=[a["seed"]])
    return env


def _run(a, env_actions):
    import mujoco
    env = _make_env(a, H=len(env_actions))
    u = env.envs[0]

    def snap():
        m, d = u._env.sim.model._model, u._env.sim.data._data
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return st
    S, ok_any = [snap()], False
    for act in env_actions:
        u.step(np.asarray(act, np.float32))
        ok_any |= bool(u._env.check_success())
        S.append(snap())
    env.close()
    return np.stack(S), ok_any


def replay_unit(task, bank_path, s0_id, env_actions, spec):
    if json.load(open(bank_path)).get("family"):           # family A (native hooks): level from spec
        a = _family_anchor(bank_path, s0_id, spec)
    else:
        if spec:
            raise NotImplementedError("perturbation spec given for a non-family bank")
        a = _anchor(bank_path, s0_id)
    A, verdict = _run(a, env_actions)
    dirty = _make_env(a, H=len(env_actions))                     # dirtied process: another instance, random steps
    for x in np.random.RandomState(0).uniform(-1, 1, (13, 7)):
        dirty.envs[0].step(x.astype(np.float32))
    B, _ = _run(a, env_actions)
    dirty.close()
    return dict(states_a=A, states_b=B, final_state=A[-1], verdict=verdict, dist=0.0)
