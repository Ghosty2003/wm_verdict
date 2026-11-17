"""OpenWAM-alpha view helpers shared by the predicate pipeline and the runner (pure numpy / PIL; importable from both
the lerobot_libero and the openwam interpreters).

canvas      the model's 384 x 320 (H x W) L-shaped input / output image: agentview (raw LIBERO frame rotated 180 deg)
            LANCZOS-resized to 320 x 256 on top, wrist LANCZOS-resized to 160 x 128 bottom-left, bottom-right black
            (OpenWAM client resize_for_lshape_slot + server assemble_multiview_layout; the BILINEAR re-fit of the
            assembler is a no-op at these slot sizes).
pred_view   predicate input, same layout as the LingBot-VA predicate: [agentview 128 x 128 | wrist 128 x 128], each
            BILINEAR-resized from its canvas slot. Real canvases and generated canvases go through the same function.
"""
import numpy as np
from PIL import Image

CANVAS_H, CANVAS_W, TOP_H, WRIST_W = 384, 320, 256, 160


def canvas_from_raw(agent_raw, wrist_raw):
    head = Image.fromarray(np.ascontiguousarray(np.asarray(agent_raw)[::-1, ::-1])).resize((CANVAS_W, TOP_H), Image.Resampling.LANCZOS)
    wrist = Image.fromarray(np.ascontiguousarray(np.asarray(wrist_raw)[::-1, ::-1])).resize((WRIST_W, CANVAS_H - TOP_H), Image.Resampling.LANCZOS)
    c = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    c.paste(head, (0, 0)); c.paste(wrist, (0, TOP_H))
    return np.asarray(c, dtype=np.uint8)


def pred_view(canvas):
    c = np.asarray(canvas)
    a = Image.fromarray(np.ascontiguousarray(c[:TOP_H])).resize((128, 128), Image.BILINEAR)
    w = Image.fromarray(np.ascontiguousarray(c[TOP_H:, :WRIST_W])).resize((128, 128), Image.BILINEAR)
    return np.concatenate([np.asarray(a), np.asarray(w)], axis=1)


def raw_obs(u):
    """agentview / wrist raw images + proprio pieces from a LeRobot LiberoEnv (u = env.envs[0])."""
    o = u._env.env._get_observations(force_update=True)
    return {k: np.asarray(o[k]).copy() for k in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                                                 "robot0_eef_quat", "robot0_gripper_qpos")}
