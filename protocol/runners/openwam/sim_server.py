"""LIBERO simulator side for the OpenWAM-alpha runner (lerobot_libero interpreter; the model runs in the openwam
interpreter). One JSON command per stdin line, one JSON reply per stdout line; arrays travel as .npz files under a
scratch directory. The environment is the same LeRobot LiberoEnv recipe as every other run (libero_env_adapter._make_env:
a NEW env per reset; native anchors through native_env), cameras at 256 px for the OpenWAM canvas (B family: 128 px, protocol v1.0).

Commands
  reset   {anchor, perturb?, resolution?} -> new env (cameras at `resolution` px, default 256; B family: 128) (+ optional E/G/N perturbation, libero/phys/phys_perturb.py, applied
                                            right after reset; the action stream then applies to every executed step);
                                            returns the raw first observation        (file: obs)
  step    {actions: file}                -> execute the actions; returns per-step success, the raw observation after
                                            the last action, full-physics states after each action   (file: step)
  replay  {anchor, actions: file}        -> fresh env, execute all actions, per-step success + full-physics states
  close
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
import mujoco

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from libero_env_adapter import _make_env
from ow_view import raw_obs


def _phys(u):
    m, d = u._env.sim.model._model, u._env.sim.data._data
    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return st


def main():
    env = u = None; stream = None; tstep = 0          # optional E/G/N perturbation (2026-10-26); absent -> unchanged
    for line in sys.stdin:
        c = json.loads(line)
        if c["cmd"] == "close":
            break
        if c["cmd"] in ("reset", "replay"):
            if env is not None:
                env.close()
            env = _make_env(dict(c["anchor"], resolution=c.get("resolution", 256)), H=c.get("H", 600)); u = env.envs[0]
            stream = None; tstep = 0; pert = c.get("perturb")
            if pert:
                from phys_perturb import apply_physics, ActionStream, maybe_push
                apply_physics(u, pert); stream = ActionStream(pert, c.get("H", 600))
        if c["cmd"] == "reset":
            o = raw_obs(u); np.savez(c["out"], phys=_phys(u), **o)
            print(json.dumps(dict(ok=True)), flush=True)
        elif c["cmd"] in ("step", "replay"):
            acts = np.load(c["actions"])["actions"]
            succ, S, EX = [], [], []
            for a in acts:
                e = stream(tstep, a) if stream is not None else np.asarray(a, np.float32); tstep += 1
                if stream is not None:
                    maybe_push(u, pert, tstep - 1)             # K: push once `push_after_step` steps are done
                u.step(e); EX.append(e)
                succ.append(bool(u._env.check_success())); S.append(_phys(u))
            o = raw_obs(u)
            np.savez(c["out"], success=np.array(succ), states=np.stack(S), executed=np.stack(EX), **o)
            print(json.dumps(dict(ok=True, n=len(acts))), flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
