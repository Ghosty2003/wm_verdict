"""4090 侧：解码 LingBot-VA 想象 latent 的最后一帧（预筛乐观率用）。
与 annotation/round3_imagined/r3_cards.py 的 lb_decode 逐行相同：只加载 LingBot 的 VAE（不加载 5B 主干），
把全部 latent 块按时间维拼接后整段解码，取最后一帧的主视角（左 128 列）。
输入：manifest.json + latents/<unit_id>.npz（键 n_chunks、latents_00..）
输出：out/<unit_id>.png（主视角末帧 128×128）、out/<unit_id>_full.png（整幅末帧）、out/decode_log.json（每单元 sha256、耗时、环境）
  python decode_last_frame.py            （lerobot_libero，CUDA）
完成后把 out/ 整个传回 Dropbox：wm_verdict/optimism_prescreen/decode_out_4090/"""
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.lingbot_va.utils import _torch_dtype, denormalize_latents, load_vae
    cfg = PreTrainedConfig.from_pretrained("lerobot/lingbot_va_libero_long")
    vae = load_vae(cfg.wan_pretrained_path, torch_dtype=_torch_dtype(cfg.dtype), torch_device="cuda", subfolder="vae").eval()
    man = json.load(open(f"{HERE}/manifest.json"))
    os.makedirs(f"{HERE}/out", exist_ok=True)
    log = dict(host=platform.node(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0), dtype=str(vae.dtype), units=[])
    for u in man["units"]:
        t0 = time.time()
        z = np.load(f"{HERE}/{u['file']}")
        assert hashlib.sha256(open(f"{HERE}/{u['file']}", "rb").read()).hexdigest() == u["sha256"], u["unit_id"]
        n = int(z["n_chunks"])
        lat = torch.cat([torch.from_numpy(z[f"latents_{i:02d}"].astype(np.float32)) for i in range(n)], dim=2)
        with torch.no_grad():
            lat = lat.to(device="cuda", dtype=vae.dtype)
            lat = denormalize_latents(lat, vae.config.latents_mean, vae.config.latents_std, vae.config.z_dim)
            video = vae.decode(lat, return_dict=False)[0]
            video = (video.float().clamp(-1, 1) + 1.0) / 2.0
            frames = (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
        last = frames[-1]
        Image.fromarray(last[:, :128]).save(f"{HERE}/out/{u['unit_id']}.png")
        Image.fromarray(last).save(f"{HERE}/out/{u['unit_id']}_full.png")
        log["units"].append(dict(unit_id=u["unit_id"], n_frames=int(frames.shape[0]), shape=list(last.shape), seconds=round(time.time() - t0, 2),
                                 png_sha256=hashlib.sha256(open(f"{HERE}/out/{u['unit_id']}.png", "rb").read()).hexdigest()))
        print(u["unit_id"], frames.shape, f"{time.time() - t0:.1f}s", flush=True)
    json.dump(log, open(f"{HERE}/out/decode_log.json", "w"), indent=1)
    print("done", len(log["units"]))


if __name__ == "__main__":
    main()
