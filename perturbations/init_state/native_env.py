"""LIBERO family A: LeRobot LiberoEnv driven through LIBERO's NATIVE initial-state hooks (libero_native_hooks.md).

  object hook  BDDL region sampler (bddl_base_domain._add_placement_initializer -> TableRegionSampler): the task
               object's `:ranges` in the BDDL `:regions` block are rewritten (bddl_region_rewrite) and the env is built
               with init_states=False, so env.reset() samples the placement natively from the rewritten region.
  pose hook    robosuite `initialization_noise` (robots/robot.py:129-141), passed to OffScreenRenderEnv (LeRobot does
               not forward it; this adapter does). Not used by the object levels.
No state vector is written by hand (no set_state / set_init_state). Restore recipe = a NEW env per restore ->
reset(seed) -> settle steps (num_steps_wait), the same recipe as the frozen anchor bank, with init_states=False.
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import re
import hashlib

from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.libero import LiberoEnv

BDDL_HOOK = "BDDL :regions :ranges rewrite -> native TableRegionSampler (bddl_base_domain._add_placement_initializer)"
POSE_HOOK = "robosuite initialization_noise (robots/robot.py:129-141) passed to OffScreenRenderEnv"


class NativeLiberoEnv(LiberoEnv):
    def __init__(self, *args, bddl_file=None, initialization_noise=None, **kwargs):
        kwargs["init_states"] = False
        super().__init__(*args, **kwargs)
        if bddl_file:
            self._task_bddl_file = bddl_file
        self._init_noise = initialization_noise

    def _ensure_env(self) -> None:          # = LiberoEnv._ensure_env + the initialization_noise pass-through
        if self._env is not None:
            return
        extra = {"initialization_noise": self._init_noise} if self._init_noise else {}
        env = OffScreenRenderEnv(bddl_file_name=self._task_bddl_file, camera_heights=self.observation_height,
                                 camera_widths=self.observation_width, control_freq=self.control_freq,
                                 hard_reset=self.hard_reset, **extra)
        env.reset()
        self._env = env


def make_native(anchor, suite, H=None, res=None):
    """LiberoEnv for a family-A anchor (anchor['native'] is True)."""
    root = _wmvp.P("$WMV_ROOT")
    bddl = os.path.join(root, anchor["bddl_file"]) if anchor.get("bddl_file") else None
    r = res or anchor["resolution"]
    return NativeLiberoEnv(task_suite=suite, task_id=anchor["task_id"], task_suite_name=anchor["suite"],
                           observation_width=r, observation_height=r, n_envs=1, num_steps_wait=anchor["settle_steps"],
                           hard_reset=True, episode_length=H, bddl_file=bddl,
                           initialization_noise=anchor.get("initialization_noise"))


def bddl_region_rewrite(src, region, ranges):
    """Return BDDL text with the `:ranges` of `region` replaced by one rectangle `ranges` = (xmin, ymin, xmax, ymax).
    Numbers are written with repr() so a rewrite with the original values is value-identical (the A0 sham)."""
    pat = re.compile(r"(\(%s\s*\n\s*\(:target [^)]*\)\s*\n\s*\(:ranges \(\s*\n\s*\()([^)]*)(\))" % re.escape(region))
    m = pat.search(src)
    if m is None:
        raise ValueError(f"region {region} not found")
    body = " ".join(repr(float(v)) for v in ranges)
    return src[:m.start(2)] + body + src[m.end(2):]


def region_ranges(src, region):
    pat = re.compile(r"\(%s\s*\n\s*\(:target [^)]*\)\s*\n\s*\(:ranges \(\s*\n\s*\(([^)]*)\)" % re.escape(region))
    return tuple(float(v) for v in pat.search(src).group(1).split())


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()
