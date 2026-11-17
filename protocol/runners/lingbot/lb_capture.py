"""LingBot-VA closed-loop capture hook (store-all-data ruling 2026-11-08).

While enabled, every chunk the policy predicts during a closed loop is recorded: the predicted latents (the model's
imagination for that chunk, decodable with rig.decode) and the real camera frames it was conditioned on (the first
observation for chunk 0, the real keyframes fed back into the KV cache for later chunks). Model behaviour is untouched:
the hook only copies tensors around the original predict_action_chunk call.

Usage:
    import lb_capture; lb_capture.install()
    with lb_capture.Capture() as cap:  rig.closed_loop(...)
    cap.arrays()  -> dict of numpy arrays ready for np.savez_compressed
"""
import numpy as np

_ACTIVE = []


def _to_np(t):
    return t.detach().to("cpu").float().numpy().astype(np.float16)


def install():
    import lerobot.policies.lingbot_va.modeling_lingbot_va as M
    cls = M.LingBotVAPolicy
    if getattr(cls, "_wmv_capture_installed", False):
        return
    orig = cls.predict_action_chunk

    def wrapped(self, batch, **kw):
        cap = _ACTIVE[-1] if _ACTIVE else None
        if cap is not None:
            if self._first_chunk:
                raw = self._extract_raw_obs(batch)
                cap.real.append({k: _to_np(v)[None] for k, v in raw.items()})
            else:
                buf = list(getattr(self, "_obs_buffer", []) or [])
                cap.real.append({k: np.concatenate([_to_np(o[k])[None] for o in buf]) for k in buf[0]} if buf else {})
        out = orig(self, batch, **kw)
        if cap is not None and self.last_predicted_latents is not None:
            cap.latents.append(self.last_predicted_latents.to("cpu").float().numpy().astype(np.float16))
        return out

    cls.predict_action_chunk = wrapped
    cls._wmv_capture_installed = True


class Capture:
    def __init__(self):
        self.latents, self.real = [], []

    def __enter__(self):
        _ACTIVE.append(self); return self

    def __exit__(self, *a):
        _ACTIVE.remove(self)

    def arrays(self, prefix=""):
        d = {f"{prefix}n_chunks": np.array(len(self.latents))}
        for i, z in enumerate(self.latents):
            d[f"{prefix}latents_{i:02d}"] = z
        for i, r in enumerate(self.real):
            for k, v in r.items():
                d[f"{prefix}real_{i:02d}__{k.replace('.', '_')}"] = v
        return d
