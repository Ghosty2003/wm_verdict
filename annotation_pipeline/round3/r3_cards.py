"""Round 3（r3_imagined，2026-11-13 用户裁决）：闭环【想象末帧】卡片 = 终局想象乐观率的人工版。
范围：
  fail_all        —— 终局乐观率案例候选集的全部现实失败单元（optimism/cases.pick_units：其余格全收，
                     OpenWAM T3/T4 失败单元各按固定种子抽 60 个；共 570）；
  success_control —— 每个 模型×任务×扰动族 按固定种子抽 5 个现实成功单元做对照（不足 5 个全收）。
卡片：只放想象末帧（最后一次规划调用的最后一帧）的主视角，正视、正方形；不展示任何真实帧，避免标注者被现实结果锚定。
  OpenWAM：imagined[-1] 画布顶部（原始转 180° 并横向拉到 320）→ 左右翻回正视、缩回正方形。
  LingBot：闭环各块潜变量沿时间拼接后整段解码（与上游一致），取最后一帧的主视角（左半）。
参考图：T0/T1 沿用 wm_verdict_web/cards/ref/t5.png、t3.png；T2–T4 用 refs 子命令按同格式新做。
  python r3_cards.py list      （本机 python3 也可：只读 json）→ units_r3.json
  python r3_cards.py render    （lerobot_libero，GPU：LingBot 需解码）→ cards/<uid>.png + 网页仓库 cards/r3/<uid>.webp
  python r3_cards.py refs      → 网页仓库 cards/ref/t2.png、t4.png、t6.png
  python r3_cards.py manifest  → manifest_r3.json（供 wm_verdict_web/tools/new_round.py 入库）
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
OPT = f"{ROOT}/libero/out/optimism"
WEB = _wmvp.P("$WMV_WEB_ROOT")
SEED, K, SIDE = 20261113, 5, 384
TID = {"T0": "t5", "T1": "t3", "T2": "t2", "T3": "t4", "T4": "t6"}
_wmvp.bootstrap()


def select():
    import numpy as np
    from cases import pick_units
    U = json.load(open(f"{OPT}/units.json"))
    fails = [dict(u, group="fail_all") for u in pick_units()]
    ctrl = []
    for key in sorted({(u["model"], u["T"], u["family"]) for u in U}):
        us = sorted([u for u in U if (u["model"], u["T"], u["family"]) == key and u["success"]], key=lambda u: u["unit_id"])
        if not us:
            continue
        rng = np.random.RandomState(zlib.crc32(f"{SEED}/r3/{key[0]}/{key[1]}/{key[2]}".encode()))
        ctrl += [dict(us[i], group="success_control") for i in sorted(rng.choice(len(us), min(K, len(us)), replace=False))]
    sel = fails + ctrl
    ids = [u["unit_id"] for u in sel]
    assert len(ids) == len(set(ids))
    json.dump(dict(seed=SEED, rule="fail_all = optimism/cases.pick_units()（OpenWAM T3/T4 失败各抽 60，种子 crc32('20261113/<(model,T)>')）；"
                   "success_control = 每 模型×任务×族 RandomState(crc32(f'{SEED}/r3/{model}/{T}/{family}')).choice(n, min(5,n))",
                   units=sel), open(f"{HERE}/units_r3.json", "w"), ensure_ascii=False, indent=0)
    from collections import Counter
    print("selected", len(sel), Counter(u["group"] for u in sel), sorted(Counter((u["model"], u["T"]) for u in sel).items()))


def last_frame(u, rig_box):
    import numpy as np
    c = f"{OPT}/cache/{u['unit_id']}.npz"
    if os.path.exists(c):
        return np.load(c)["imagined"][-1]
    z = np.load(u["frames"])
    if u["model"] == "openwam":
        return np.ascontiguousarray(z["imagined"][-1][:256, ::-1])
    return lb_decode(u, rig_box)[-1][:, :128]


def lb_latents(u):
    import numpy as np
    import torch
    z = np.load(u["frames"])
    p = "imag_latents" if u["key"] == "imag_latents" else "latents"
    n = int(z["imag_n_chunks" if p == "imag_latents" else "n_chunks"])
    return torch.cat([torch.from_numpy(z[f"{p}_{i:02d}"].astype(np.float32)) for i in range(n)], dim=2)


def lb_decode(u, box):
    """只加载 LingBot 的 VAE（不加载 5B 主干和 UMT5 文本编码器，显存约 1–2 GB），解码与
    LingBotVAPolicy._decode_predicted_video 逐行相同；verify 子命令已与 rig.decode 的缓存结果逐位比对。"""
    import torch
    if not box:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.lingbot_va.utils import _torch_dtype, denormalize_latents, load_vae
        cfg = PreTrainedConfig.from_pretrained("lerobot/lingbot_va_libero_long")
        vae = load_vae(cfg.wan_pretrained_path, torch_dtype=_torch_dtype(cfg.dtype), torch_device="cuda", subfolder="vae").eval()
        box += [vae, denormalize_latents]
    vae, denorm = box
    with torch.no_grad():
        lat = lb_latents(u).to(device=next(vae.parameters()).device, dtype=vae.dtype)
        lat = denorm(lat, vae.config.latents_mean, vae.config.latents_std, vae.config.z_dim)
        video = vae.decode(lat, return_dict=False)[0]
        video = (video.float().clamp(-1, 1) + 1.0) / 2.0
        return (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8).cpu().numpy()


def verify():
    """VAE 单独解码 vs 案例缓存（LingBotRig.decode 产出）逐位比对：取 3 个 LingBot 缓存单元。"""
    import numpy as np
    U = {u["unit_id"]: u for u in json.load(open(f"{OPT}/units.json"))}
    box, n = [], 0
    for f in sorted(os.listdir(f"{OPT}/cache")):
        uid = f[:-4]
        if not uid.startswith("lingbot"):
            continue
        a = lb_decode(U[uid], box)[-16:, :, :128]; b = np.load(f"{OPT}/cache/{f}")["imagined"]
        print("VERIFY", uid, a.shape, b.shape, "identical" if np.array_equal(a, b) else f"DIFF max={np.abs(a.astype(int) - b).max()}", flush=True)
        n += 1
        if n == 3:
            break


def card(frame):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame).resize((SIDE, SIDE), Image.BICUBIC)
    ImageDraw.Draw(im).rectangle([0, 0, SIDE - 1, SIDE - 1], outline=(255, 196, 0), width=4)
    return im


def render():
    sel = json.load(open(f"{HERE}/units_r3.json"))["units"]
    os.makedirs(f"{HERE}/cards", exist_ok=True); os.makedirs(f"{WEB}/cards/r3", exist_ok=True)
    rig, n = [], 0
    for u in sel:
        png, webp = f"{HERE}/cards/{u['unit_id']}.png", f"{WEB}/cards/r3/{u['unit_id']}.webp"
        if os.path.exists(png) and os.path.exists(webp):
            continue
        im = card(last_frame(u, rig))
        im.save(png, optimize=True); im.save(webp + ".tmp", "WEBP", quality=85, method=6); os.replace(webp + ".tmp", webp)
        n += 1
        if n % 50 == 0:
            print("render", n, flush=True)
    print("RENDER_DONE", n, "of", len(sel), flush=True)


def refs():
    """T2–T4 参考图（同 t3/t5 格式：真实画面拼图 + 红字说明）：起始帧、真实成功末帧、两张真实失败末帧（LingBot 补跑单元的逐步真实画面）。"""
    import numpy as np
    from PIL import Image, ImageDraw
    U = json.load(open(f"{OPT}/units.json"))
    for T in ("T2", "T3", "T4"):
        lb = sorted([u for u in U if u["model"] == "lingbot" and u["T"] == T and u["source"] == "lb_rerun"], key=lambda u: u["unit_id"])
        ok = next(u for u in lb if u["success"] and u["family"] == "none")
        bad = [u for u in lb if not u["success"] and u["family"] in ("K", "NL")][:2]
        up = lambda u, i: Image.fromarray(np.ascontiguousarray(np.load(u["frames"])["agentview"][i][::-1])).resize((256, 256))
        tiles = [(up(ok, 0), "REAL reset (step 0)"), (up(ok, -1), "REAL SUCCESS final"),
                 (up(bad[0], -1), "REAL not-success final (example)"), (up(bad[1], -1), "REAL not-success final (example)")]
        S = Image.new("RGB", (512, 512), (255, 255, 255)); d = ImageDraw.Draw(S)
        for j, (im, t) in enumerate(tiles):
            x, y = (j % 2) * 256, (j // 2) * 256
            S.paste(im, (x, y)); d.rectangle([x, y, x + 255, y + 12], fill=(255, 255, 255)); d.text((x + 3, y + 1), t, fill=(200, 0, 0))
        S.save(f"{WEB}/cards/ref/{TID[T]}.png")
        print("REF", TID[T], ok["unit_id"], [b["unit_id"] for b in bad])


def manifest():
    from build_manifest import GOAL, FAM
    HOW = ("Closed-loop deployment, IMAGINED view: this is the world model's own predicted camera frame at its LAST planning call "
           "(the final imagined frame, yellow box). It is NOT a real camera frame. Judge only what the imagined image shows: ")
    d = json.load(open(f"{HERE}/units_r3.json"))
    man, miss = [], 0
    for u in d["units"]:
        png = f"{HERE}/cards/{u['unit_id']}.png"
        if not os.path.exists(png):
            miss += 1; continue
        t = TID[u["T"]]
        man.append(dict(model=u["model"], task=t, family=FAM(u["level"]), level=u["level"], unit_id=u["unit_id"], png=png,
                        group=u["group"], sample_seed=d["seed"], sim_success=u["success"], frames_source=u["source"],
                        question=HOW + "does the imagined scene show the task COMPLETED — " + GOAL[t],
                        labels=["looks_success", "borderline", "not_success", "uncertain"],
                        ref=f"cards/ref/{t}.png"))
    json.dump(dict(round_name="r3_imagined", seed=d["seed"], rule=d["rule"], cards=man), open(f"{HERE}/manifest_r3.json", "w"), ensure_ascii=False)
    from collections import Counter
    print("manifest", len(man), "missing png", miss, Counter(m["group"] for m in man))


if __name__ == "__main__":
    {"list": select, "render": render, "refs": refs, "manifest": manifest, "verify": verify}[sys.argv[1]]()
