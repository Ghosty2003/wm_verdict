# WorldVLA environment setup (2026-10-14)

Status: conda env `worldvla` created and checked on CPU only. No model was loaded onto the GPU or the CPU (7B), and no GPU process was started.

## Source pins
- Code: `alibaba-damo-academy/WorldVLA`, branch `worldvla` @ `ebdc36dc5cfd` (2025-10-10, "update: base model path"). This is the code that matches the HF `WorldVLA` weights. `main` @ `e548ccc` is RynnVLA-002 and is used only for reference.
  - Copied (gifs/mp4 excluded) to `third_party/WorldVLA_worldvla_branch/`. The copy is not edited. sha256 of the sorted per-file sha256 list: `7d5692d2…5c5b956757`.
- Weights: HF `Alibaba-DAMO-Academy/WorldVLA` @ `5952d61278d5`, snapshot `~/.cache/huggingface/hub/models--Alibaba-DAMO-Academy--WorldVLA/snapshots/5952d61278d52608b2f141ef5b155d42e237165b/`.
  - `model_256/libero_10/`: 3 safetensors shards. Sizes match the HF index (4989332608 / 4991450608 / 4045582856 B). Download complete.
  - `chameleon/tokenizer/{vqgan.ckpt (281270377 B), vqgan.yaml, text_tokenizer.json}`.
  - `base_model/{tokenizer.json, tokenizer_config.json, special_tokens_map.json}`. Only the text tokenizer files are needed; the base-model weights are not needed.

## What the repo asks for vs what was installed
Repo `environment.yml` gives python 3.10.16, torch 2.5.0+cu124, torchvision 0.20.0+cu124, transformers 4.48.0, tokenizers 0.21.0, accelerate 1.2.1, flash-attn 2.7.3, numpy 1.26.3, mujoco 3.2.7, robosuite 1.4.0, plus about 150 other pins (tensorflow, gradio, wandb, …). LIBERO is installed from GitHub with `pip install -e`.

| package | repo pin | installed | note |
|---|---|---|---|
| python | 3.10.16 | 3.10.16 | |
| torch / torchvision | 2.5.0+cu124 / 0.20.0+cu124 | **2.7.1+cu128 / 0.22.1+cu128** | **DEVIATION.** cu124 wheels have no sm_120 kernels, so they cannot run on the RTX 5090. 2.7.1 is the oldest torch release with cu128 wheels. Compiled arch flags: `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`. |
| transformers / tokenizers | 4.48.0 / 0.21.0 | same | The repo vendors its own `model/chameleon/modeling_chameleon.py` from transformers 4.4x, so this pin matters. |
| huggingface-hub, accelerate, safetensors, sentencepiece, numpy, pillow | 0.27.1, 1.2.1, 0.5.2, 0.2.0, 1.26.3, 10.2.0 | same | |
| fairscale, tensorboard, einops, pyyaml, h5py, opencv-python, scipy, imageio(+ffmpeg), easydict, omegaconf, psutil, tqdm | repo pins | same | `xllmx` imports these. |
| flash-attn | 2.7.3 | **not installed** | Optional (`is_flash_attn_2_available()` guard). The eval never requests flash: `from_pretrained` sets no `attn_implementation`, so transformers 4.48 auto-selects **sdpa** (`_supports_sdpa=True`, modeling_chameleon.py:1114). We get the same selection, and the run records the value as `attn_impl` in imagined.json. A source build for sm_120 is not needed. |
| tensorflow, gradio, wandb (pin), lpips, … | pinned | not installed | Not on the inference/eval path. |
| LIBERO | `pip install -e LIBERO` (GitHub) | **hf-libero 0.1.4** (+ robosuite 1.4.0, mujoco 3.2.7, bddl 1.0.1, robomimic 0.2.0) | Same package family as `lerobot_libero`. mujoco is kept at the repo pin 3.2.7; `lerobot_libero` has 3.8.1. |
| cmake | 3.31.4 | 3.31.4 | Needed to build `hf-egl-probe` / `egl_probe` from source. |

Full freeze: `logs/worldvla_pip_freeze.txt` (127 packages). Install logs: `logs/01..05_*.log`. Pins used: `constraints.txt`.

## lerobot in the same env? No, so the design is two processes
- The local lerobot (`$LEROBOT_ROOT`, 0.6.2) needs `requires-python >=3.12`, `numpy>=2`, `torch>=2.7`, and `transformers>=5.4,<5.6` for its `libero` extra (`pyproject.toml:32,62-64,160,289`).
- WorldVLA needs python 3.10 (env pin), numpy 1.26, and transformers 4.48. Its vendored Chameleon code imports transformers-4.x internals: `modeling_flash_attention_utils._flash_attention_forward`, `LogitsWarper`, and the old `cache_utils`.
- These requirements cannot both be met, so `pip install -e $LEROBOT_ROOT` was **not** run in `worldvla`.
- Instead the unit runs in two processes and passes data through files (`worldvla_unit.py`):
  - **Env side** (`lerobot_libero`, py3.12, lerobot 0.6.2, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1) runs the anchor restore and the open-loop replay. This is the exact `anchor_bank.py` recipe with a fresh `LiberoEnv` per restore.
  - **Model side** (`worldvla`) runs the imagination, the probe, and the official-eval wrapper.
- hf-libero 0.1.4 + robosuite 1.4.0 **are** installed in `worldvla` (py3.10 is compatible). They are needed only by the repo's own eval script (task 1d).
- `~/.libero/config.yaml` is shared through `$HOME` and points to the `lerobot_libero` site-packages copy of the hf-libero 0.1.4 bddl/init files. Both envs read the same files.

## Commands run (in order)
```bash
conda create -n worldvla python=3.10.16 -y
PY=$CONDA_ROOT/envs/worldvla/bin/python
$PY -m pip install --upgrade "pip==24.2"
$PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
  numpy==1.26.3 transformers==4.48.0 tokenizers==0.21.0 huggingface-hub==0.27.1 accelerate==1.2.1 safetensors==0.5.2 \
  sentencepiece==0.2.0 pillow==10.2.0 pyyaml==6.0.2 einops==0.8.0 fairscale==0.4.13 tensorboard==2.18.0 imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 tqdm==4.67.1 h5py==3.12.1 opencv-python==4.11.0.86 scipy==1.15.1 easydict==1.13 omegaconf==2.3.0 psutil==6.1.1
# first attempt at the LIBERO stack failed: hf-egl-probe / egl_probe need cmake (logs/04_libero.log)
PATH=$CONDA_ROOT/envs/worldvla/bin:$PATH $PY -m pip install -c constraints.txt cmake==3.31.4
PATH=$CONDA_ROOT/envs/worldvla/bin:$PATH $PY -m pip install -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
  hf-libero==0.1.4 robosuite==1.4.0 mujoco==3.2.7
$PY -m pip check            # -> No broken requirements found.
```
The repo is **not** pip-installed: `xllmx` and `worldvla/` are put on `sys.path` by `worldvla_unit.py`.

## CPU-only checks done (no GPU context; `CUDA_VISIBLE_DEVICES=""`; rendering through Mesa software EGL)
Script: scratchpad `wv_smoke/smoke_cpu.py`. Results:
- **Imports.** `model`, `libero_util.Chameleon_utils`, `xllmx.solvers.pretrain` and `libero.libero.benchmark` all import. The config loads (32 layers, hidden 4096, vocab 65536). The model is not instantiated, so the attention implementation is set only at load time (sdpa expected).
- **VQGAN.** `vqgan.ckpt` loads under torch 2.7's default `torch.load(weights_only=True)` with no patch needed. On CPU, encode takes 0.45 s and decode 0.50 s for one 256² frame.
- **Frame tokens.** A frame is 276 tokens: `8197, 8812(h=8), 8812(w=8)`, then 16 rows of (16 codes + newline 8803), then `8196`.
  - Real anchor frame (`libero10_t5_a0`) VQ round-trip PSNR: 23.97 dB.
- **Prompt lengths.**
  - VLA prompt (2 frames): 575 tokens.
  - WM prompt (train wording): 304 tokens.
  - WM training sequence: 581 tokens, 277 of them labelled (the next frame + 8710).
  - All prompts end with the separator 8710, which is also the generation eos.
- **Action codec.** token → bin centre → `unnorm_min_max` → re-tokenize gives the identical 9 tokens. The quantization error is ≤ 0.004 in env units.
- **Loop control flow.** `imagine()` with a fake model: chunks are queried at steps 0, 10 and 20, the VLA history is ≤ 1 previous frame, and the shapes are correct.
- **Env side** (`lerobot_libero`, software EGL):
  - `first-frame` for `libero10_t5_a0` at 256 px works.
  - Physics after restore is **bit-identical** to a restore at the bank resolution (128).
  - The model-view frame is upright (raw agentview rotated 180°).
  - `replay` of 30 dummy actions: two fresh-env replays are bit-identical, and also identical to the 128 px replay.
  - `nvidia-smi` showed no process from these checks.
- **LIBERO in `worldvla`** (mujoco 3.2.7): the repo's `get_libero_env(task, 256)`, init state and 10 dummy steps run, and `get_libero_image` has shape (256, 256, 3). An EGL teardown warning appears with software EGL only.
