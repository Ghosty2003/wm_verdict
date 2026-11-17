"""LIBERO batch, WorldVLA (Alibaba-DAMO-Academy/WorldVLA, model_256/libero_10): one paired unit = (anchor, seed).

TWO-PROCESS DESIGN (env and model cannot share a Python env):
  env side   conda env `lerobot_libero` (py3.12, lerobot 0.6.2, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1)
             -> anchor restore (anchor_bank.py recipe) and open-loop replay.
  model side conda env `worldvla` (py3.10.16, torch 2.7.1+cu128, transformers 4.48.0, repo code @ worldvla branch ebdc36d)
             -> alternating VLA <-> world-model imagination, joint-consistency probe, official eval wrapper.
  Hand-off is by files under OUT/<anchor_id>/s<seed>/.  `all` runs the four steps with the right interpreters.

Code references (repo copy: third_party/WorldVLA_worldvla_branch, = github alibaba-damo-academy/WorldVLA@worldvla ebdc36d):
  observation   agentview only (no wrist, no state), rendered natively at 256x256 (libero_utils.py:22), rotated 180 deg
                (img[::-1, ::-1], libero_utils.py:66 eval; regenerate_libero_dataset_save_img_action.py:107 train),
                no resize (libero_utils.py:67 commented out), var_center_crop is a no-op for 256x256 (item_processor.py:272).
  image tokens  VQGAN f=16 -> 16x16 codes + 1 newline token per row + <racm3:break>(8197) <h=8>(8812) <w=8>(8812) ...
                <eoss>(8196) = 276 tokens/frame (item_processor.py:276-298); decode item_processor.py:368-391.
  VLA mode      prompt "What action should the robot take to {task}?<|image|><|image|>" with images
                [previous frame, current frame] (his_type 2h_1a_img_only, Chameleon_utils.py:499-509; training
                action_model_conv_generation.py:110-124), greedy, eos 8710, max_new_tokens = 25*12
                (Chameleon_utils.py:548-559; --action_steps default 25 eval_solver_libero_256.py:76), returns all action
                blocks <10004> 7 tokens <15004> (modeling_xllmx_chameleon_ck.py:329-349) = chunk of ~10 (libero_10
                trained with ck_10, args.json); the whole chunk is executed open-loop (eval_solver_libero_256.py:251-268).
  action codec  norm: 2*(a-min)/(max-min)-1, clip [-1,1], 256 uniform bins, token = digitize + 10005
                (item_processor.py:302-334); decode bin centers (modeling_xllmx_chameleon_ck.py:403-408);
                denorm unnorm_min_max (eval_solver_libero_256.py:125-133); min/max are the same constants for all suites;
                gripper passes through ([-1,1] -> [-1,1]), no invert / binarize (eval_solver_libero_256.py:272-279).
  WM mode       Markov, his=1 (README:188-193, concate_action_world_model_data.py:46): prompt
                "Generate the next image based on the provided sequence of historical images and corresponding actions."
                "<|image|><|action|>" (world_model_conv_generation.py:110-134), answer = next frame; the action is the
                RAW env-unit action a_t taken at frame t (world_model_conv_generation.py:114) re-encoded by
                process_action.  The inference helper (Chameleon_utils.py:598-636, his_type "1a2i") uses a DIFFERENT
                prompt ("Generate the image based on the current image and the action.") -> --wm_prompt helper.
                generate_img does NOT reset model.init_input_ids (modeling_xllmx_chameleon_ck.py:317-327, forward :31-39);
                we reset it before every call (mandatory in an alternating loop, otherwise the attention mask is built on
                the concatenation of the previous call's sequence).
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import glob
import hashlib
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _wmvp.P("$WORLDVLA_REPO")
REPO_WV = os.path.join(REPO, "worldvla")
SNAP = (sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Alibaba-DAMO-Academy--WorldVLA/snapshots/*/")) or [""]))[-1].rstrip("/")
CKPT = os.path.join(SNAP, "model_256", "libero_10")
TOK_DIR = os.path.join(SNAP, "base_model")                 # tokenizer.json / tokenizer_config.json (text tokenizer)
VQ_DIR = os.path.join(SNAP, "chameleon", "tokenizer")      # vqgan.ckpt, vqgan.yaml, text_tokenizer.json
BANK = _wmvp.P("$WMV_ROOT/libero/out/bank/libero_anchors.json")
OUT = os.environ.get("WORLDVLA_OUT", _wmvp.P("$WMV_ROOT/libero/out/worldvla"))
PY_ENV = os.path.expanduser(_wmvp.P("$CONDA_ROOT/envs/lerobot_libero/bin/python"))
PY_MODEL = os.path.expanduser(_wmvp.P("$CONDA_ROOT/envs/worldvla/bin/python"))

H = 520
RES = 256
SUITE = "libero_10"
ACT_MIN = np.array([-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0])   # item_processor.py:327
ACT_MAX = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])              # item_processor.py:328
WM_PROMPTS = {
    "train": "Generate the next image based on the provided sequence of historical images and corresponding actions.",
    "helper": "Generate the image based on the current image and the action.",
    "paper": "Generate the next frame based on the current image and the action.",   # paper §3.3 World Model Data,
}                                    # same layout [BOS]{text}[BOI]{image}[EOI][BOA]{action}[EOA][EOS] -> {image} (Eq. after :284)
ONESTEP_SEED = 20261015              # transition sample for the prompt selection (same frames for every wording)
IMG_START, IMG_END, NEWLINE, GRID8, SEP = 8197, 8196, 8803, 8812, 8710
ACT_START, ACT_END = 10004, 15004
TOK_PER_FRAME = 3 + 16 * 17 + 1          # 276 (+ the trailing 8710 that generate emits = 277)


def unit_dir(anchor_id, seed):
    d = os.path.join(OUT, anchor_id, f"s{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def load_anchor(anchor_id):
    bank = json.load(open(BANK))
    a = [x for x in bank["anchors"] if x["anchor_id"] == anchor_id]
    assert a, f"unknown anchor {anchor_id}"
    return a[0]


def unnorm_min_max(a):                   # eval_solver_libero_256.py:125-133 (verbatim formula)
    a = np.asarray(a, dtype=np.float64)[:7]
    return (a + 1) / 2 * (ACT_MAX - ACT_MIN + 1e-8) + ACT_MIN


def model_view(raw_agentview):
    """LIBERO raw agentview -> WorldVLA convention (rotate 180 deg, libero_utils.py:66). Same as LeRobot LiberoProcessor."""
    return np.ascontiguousarray(np.asarray(raw_agentview)[::-1, ::-1])


# =====================================================================================================================
# ENV SIDE (run with the lerobot_libero interpreter)
# =====================================================================================================================
def _make_env(anchor, res=RES):
    """anchor_bank.py:41-44 recipe (fresh LeRobot LiberoEnv, hard reset), only the render resolution differs."""
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()[anchor["suite"]]()
    return LiberoEnv(task_suite=suite, task_id=anchor["task_id"], task_suite_name=anchor["suite"],
                     observation_width=res, observation_height=res, init_states=True,
                     episode_index=anchor["init_index"], n_envs=1, num_steps_wait=anchor["settle_steps"],
                     hard_reset=True, episode_length=H)


def _phys(env):
    import mujoco
    m, d = env._env.sim.model._model, env._env.sim.data._data
    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return st


def cmd_first_frame(args):
    a = load_anchor(args.anchor)
    d = unit_dir(a["anchor_id"], args.seed)
    env = _make_env(a)
    obs, _ = env.reset(seed=a["seed"])                 # restore = fresh env -> reset(seed) (anchor_bank.py:63-67)
    st = _phys(env)
    raw = np.asarray(obs["pixels"]["image"]).copy()
    wrist = np.asarray(obs["pixels"]["image2"]).copy()
    task = env.task_description
    env.close()
    res_check = None
    if args.check_res:                                 # physics must not depend on render size (bank was frozen at 128)
        e2 = _make_env(a, res=a["resolution"])
        e2.reset(seed=a["seed"])
        res_check = bool(np.array_equal(st, _phys(e2)))
        e2.close()
    np.savez_compressed(os.path.join(d, "first_frame.npz"), model_view=model_view(raw), raw_agentview=raw,
                        raw_wrist=wrist, phys=st)
    meta = dict(anchor=a, task_description=task, res=RES, phys_sha256=hashlib.sha256(st.tobytes()).hexdigest(),
                phys_identical_to_bank_resolution=res_check)
    json.dump(meta, open(os.path.join(d, "first_frame.json"), "w"), indent=1)
    print("FIRST_FRAME", json.dumps(meta)[:400])


def _replay_once(a, actions, res=RES, frames=False):
    env = _make_env(a, res)
    obs, _ = env.reset(seed=a["seed"])
    states, succ, F = [_phys(env)], [], []
    if frames:
        F.append(model_view(obs["pixels"]["image"]))
    for x in actions:
        obs, r, term, trunc, info = env.step(np.asarray(x, dtype=np.float32))
        succ.append(bool(info.get("is_success", False)))
        states.append(_phys(env))
        if frames:
            F.append(model_view(obs["pixels"]["image"]))
    env.close()
    succ = np.array(succ)
    first = int(np.argmax(succ)) + 1 if succ.any() else None
    return dict(states=np.stack(states), succ=succ, first=first, frames=np.stack(F) if frames else None)


def cmd_replay(args):
    """truth (a): open-loop replay of the SAME imagined env actions from the same anchor, fresh env, twice."""
    a = load_anchor(args.anchor)
    d = unit_dir(a["anchor_id"], args.seed)
    I = np.load(os.path.join(d, "imagined.npz"))
    acts = I["env_actions"][:H]
    t0 = time.time()
    R1 = _replay_once(a, acts, frames=True)
    R2 = _replay_once(a, acts)
    out = dict(anchor_id=a["anchor_id"], seed=args.seed, n_actions=int(len(acts)), success=R1["first"] is not None,
               first_success_step=R1["first"], replay_bit_identical=bool(np.array_equal(R1["states"], R2["states"])),
               states_sha256=hashlib.sha256(R1["states"].tobytes()).hexdigest(), seconds=time.time() - t0)
    if args.check_res:
        R3 = _replay_once(a, acts, res=a["resolution"])
        out["identical_to_bank_resolution_replay"] = bool(np.array_equal(R1["states"], R3["states"]))
    np.savez_compressed(os.path.join(d, "replay.npz"), succ=R1["succ"], frames=R1["frames"], states=R1["states"])
    json.dump(out, open(os.path.join(d, "replay.json"), "w"), indent=1)
    print("REPLAY", json.dumps(out))


# =====================================================================================================================
# MODEL SIDE (run with the worldvla interpreter)
# =====================================================================================================================
def _model_imports():
    for p in (REPO, REPO_WV):                          # xllmx (repo root) and model/, data/, libero_util/ (worldvla/)
        if p not in sys.path:
            sys.path.insert(0, p)


class Rig:
    def __init__(self, device="cuda", wm_prompt="train", max_seq_len=8192):
        _model_imports()
        import torch
        from model import ChameleonXLLMXForConditionalGeneration_ck
        self.torch = torch
        self.device = device
        self.wm_prompt = WM_PROMPTS[wm_prompt]
        t0 = time.time()
        # eval_solver_libero_256.py:88-96, 136-139 (+ launch script: --max_seq_len 8192 --unmask_image_logits)
        self.model = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
            CKPT, max_position_embeddings=max_seq_len, mask_image_logits=False, dropout=0.05, z_loss_weight=1e-5,
            torch_dtype=torch.bfloat16, device_map="cpu").to(device).eval()
        self.t_load_model = time.time() - t0
        self.proc = make_processor(device)
        self.t_load_total = time.time() - t0

    # ------------------------------------------------------------------ VLA mode
    def vla(self, cur_img, prev_imgs, task, constrain=False):
        """repo function verbatim (Chameleon_utils.py:386-563, his_type 2h_1a_img_only, action_steps 25).
        constrain=True (ruling 4, 2026-10-15): same prompt construction (the repo function), but model.generate_dis_ma is
        swapped for a copy whose greedy decoding is restricted by ActionStructure to well-formed action blocks."""
        from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck
        if constrain:
            self.model.generate_dis_ma = self._generate_dis_ma_constrained
        try:
            acts = get_action_Chameleon_dis_awm_ck(self.model, cur_img, task, self.proc, list(prev_imgs), [],
                                                   "2h_1a_img_only", 25)
        finally:
            self.model.__dict__.pop("generate_dis_ma", None)
            self.model.init_input_ids = None
        return [a.float().cpu().numpy() for a in acts]

    def _generate_dis_ma_constrained(self, input_ids, generation_config):
        """modeling_xllmx_chameleon_ck.py:328-347 (generate_dis_ma) with logits_processor=ActionStructure."""
        from transformers import LogitsProcessorList
        from model.chameleon import ChameleonForConditionalGeneration
        m = self.model
        m.init_input_ids = None
        res = ChameleonForConditionalGeneration.generate(
            m, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False,
            return_dict_in_generate=True, logits_processor=LogitsProcessorList([ActionStructure(input_ids.shape[1])]))
        dis_tokens = res['sequences'][:, input_ids.shape[1]:][0]
        decoded_actions = m.decode_token_ids_to_actions(dis_tokens)
        action_sequences, start_index = [], None
        for i, token in enumerate(dis_tokens):
            if token == ACT_START:
                start_index = i
            elif token == ACT_END:
                if start_index is not None:
                    action_sequences.append(decoded_actions[start_index + 1:i])
                start_index = None
        return action_sequences

    def vla_checked(self, cur_img, prev_imgs, task, step, st):
        """official call; if malformed (repo mask builder raises, no block, or a block that is not 7 values), up to
        MAX_RETRY consecutive constrained retries. Returns the chunk, or None = generation-invalid (st records why)."""
        st["n_vla_calls"] += 1
        try:
            chunk = self.vla(cur_img, prev_imgs, task)
            ok = bool(chunk) and all(c.shape == (7,) for c in chunk)
            why = "malformed action blocks" if not ok else None
        except ValueError as e:
            chunk, ok, why = None, False, f"repo mask builder: {str(e)[:80]}"
        if ok:
            return chunk
        st["n_vla_malformed"] += 1
        st["vla_malformed_steps"].append(step)
        if st["first_vla_malformed_step"] is None:
            st["first_vla_malformed_step"] = step
        for _ in range(MAX_RETRY):
            st["n_vla_retries"] += 1
            try:
                chunk = self.vla(cur_img, prev_imgs, task, constrain=True)
                if chunk and all(c.shape == (7,) for c in chunk):
                    return chunk
            except ValueError as e:
                why = f"constrained retry: {str(e)[:80]}"
        st["invalid_reason"] = f"VLA at step {step}: {why}; {MAX_RETRY} constrained retries malformed"
        return None

    # ------------------------------------------------------------------ WM mode
    def wm_tokens(self, cur_img, env_action):
        conv = {"conversations": [{"from": "human", "value": self.wm_prompt + "<|image|><|action|>"}],
                "image": [cur_img], "action": [np.asarray(env_action, dtype=np.float64)]}
        return self.proc.process_item(conv, training_mode=False)

    def wm(self, cur_img, env_action, constrain=False, sample=None):
        """next frame from (current frame, raw env-unit action). Returns (PIL image or None, generated tokens, timings)."""
        torch = self.torch
        from transformers import GenerationConfig, LogitsProcessorList
        from model.chameleon import ChameleonForConditionalGeneration
        t0 = time.time()
        tokens = self.wm_tokens(cur_img, env_action)
        _sync(torch); t1 = time.time()
        ids = torch.tensor(tokens, dtype=torch.int64, device=self.model.device).unsqueeze(0)
        gc = GenerationConfig(max_new_tokens=1200, max_length=self.model.config.max_position_embeddings,
                              temperature=1, top_k=None, do_sample=False, eos_token_id=[SEP])   # Chameleon_utils.py:615-621
        if sample:
            gc.do_sample, gc.temperature, gc.top_k = True, sample["temperature"], sample["top_k"]
        lp = LogitsProcessorList([ImageStructure(ids.shape[1], self.proc)]) if constrain else None
        self.model.init_input_ids = None
        try:
            with torch.no_grad():
                res = ChameleonForConditionalGeneration.generate(        # = model.generate_img (+ optional processor)
                    self.model, input_ids=ids, generation_config=gc, output_hidden_states=True, training=False,
                    return_dict_in_generate=True, logits_processor=lp)
        except ValueError as e:      # repo mask builder (generate_att_mask_3) raises when the model, instead of closing the
            # image with <eoss>(8196) + 8710, runs on into stray action tokens (seen with the "train" wording, 2026-10-15)
            # -> counted as a structural failure; the caller retries with the constrained processor (official path would crash)
            self.model.init_input_ids = None
            _sync(torch); t2 = time.time()
            print("WM generate failed:", repr(e)[:120])
            return None, [], dict(tokenize=t1 - t0, generate=t2 - t1, decode=0.0, n_prompt=len(tokens), n_gen=0)
        gen = res["sequences"][0, ids.shape[1]:].cpu().tolist()
        _sync(torch); t2 = time.time()
        img = None
        try:
            img = self.decode(gen[:-1] if gen and gen[-1] == SEP else gen)
        except Exception as e:                       # malformed token grid
            print("WM decode failed:", repr(e)[:200])
        _sync(torch); t3 = time.time()
        return img, gen, dict(tokenize=t1 - t0, generate=t2 - t1, decode=t3 - t2, n_prompt=len(tokens), n_gen=len(gen))

    def decode(self, toks):
        return self.proc.decode_image(list(toks))

    def frame_tokens(self, img):
        return self.proc.process_image(img)["input_ids"]


def make_processor(device):
    """FlexARItemProcessor_Action (item_processor.py:199-334) with explicit paths/device and a print-free decode_image.
    __init__ mirrors item_processor.py:210-250 except for the hard-coded relative paths and device="cuda"."""
    _model_imports()
    import torch
    from data.convertsation import Conversation
    from data.item_processor import FlexARItemProcessor_Action, generate_crop_size_list
    import model.chameleon_vae_ori as cvo
    from xllmx.data.item_processor import MMConvItemProcessor

    class WVProcessor(FlexARItemProcessor_Action):
        def __init__(self):
            MMConvItemProcessor.__init__(self, {"<|image|>": self.process_image, "<|action|>": self.process_action},
                                         ["<|image|>", "<|action|>"], TOK_DIR, Conversation)
            self.patch_size = 32
            self.crop_size_list = generate_crop_size_list((RES // self.patch_size) ** 2, self.patch_size)
            self.chameleon_ori_vocab = cvo.VocabInfo(
                json.load(open(os.path.join(VQ_DIR, "text_tokenizer.json"), encoding="utf8"))["model"]["vocab"])
            self.chameleon_ori_translation = cvo.VocabTranslation(self.chameleon_ori_vocab, device=device)
            self.chameleon_ori_image_tokenizer = cvo.ImageTokenizer(
                cfg_path=os.path.join(VQ_DIR, "vqgan.yaml"), ckpt_path=os.path.join(VQ_DIR, "vqgan.ckpt"), device=device)
            self.n_bins, self.min_action, self.max_action = 256, -1, 1
            self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
            self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        def decode_image(self, tokens):                    # item_processor.py:368-391 without the prints
            if tokens[0] == self.token2id(self.image_start_token):
                tokens = tokens[1:]
            if tokens[-1] == self.token2id(self.image_end_token):
                tokens = tokens[:-1]
            h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
            tokens = list(tokens[2:])
            hl, wl = h_grids * 2, w_grids * 2
            assert (h_grids, w_grids) == (RES // 32, RES // 32), f"grid {h_grids}x{w_grids}"
            assert len(tokens) == hl * (wl + 1), f"{len(tokens)} tokens"
            for i in range(len(tokens)):
                if (i + 1) % (wl + 1) != 0:
                    tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]
            t = torch.tensor(tokens, dtype=torch.int64, device=device).view(hl, wl + 1)[:, :-1].flatten()
            return self.chameleon_ori_image_tokenizer.pil_from_img_toks(t, hl, wl)

    return WVProcessor()


MAX_RETRY = 5          # ruling 4: >= 5 consecutive malformed retries at one step -> unit is generation-invalid


class ActionStructure:
    """Constrained-retry decoding for the VLA mode (ruling 4, 2026-10-15; NOT in the repo): greedy decoding restricted
    to well-formed action blocks <10004> 7 x bin token (10005..10261, item_processor.py:309) <15004>, repeated; the
    end-of-answer token 8710 is allowed only at a block boundary after >= 1 block; at most 33 blocks (= the official
    max_new_tokens 25*12 = 300). Within these constraints the model's own greedy choice is kept."""
    def __init__(self, prompt_len):
        import torch
        self.p = prompt_len
        self.bins = torch.zeros(65536, dtype=torch.bool); self.bins[10005:10262] = True
        self.dev = None

    def __call__(self, input_ids, scores):
        import torch
        k = input_ids.shape[1] - self.p
        if self.dev is None:
            self.bins = self.bins.to(scores.device); self.dev = scores.device
        neg = torch.finfo(scores.dtype).min
        r = k % 9
        if r == 0:
            allowed = [SEP] if k >= 9 * 33 else ([ACT_START, SEP] if k > 0 else [ACT_START])
            out = torch.full_like(scores, neg); out[:, allowed] = scores[:, allowed]
            return out
        if r == 8:
            out = torch.full_like(scores, neg); out[:, ACT_END] = 0.0
            return out
        return scores.masked_fill(~self.bins, neg)


class ImageStructure:
    """Fallback only (not in the repo): constrain greedy decoding to a well-formed 16x16 grid
    (<8197><8812><8812> (16 image codes + <8803>) x16 <8196> <8710>); used when the unconstrained output fails to decode."""
    def __init__(self, prompt_len, proc):
        import torch
        self.p = prompt_len
        img_ids = sorted(proc.chameleon_ori_translation.bpe2img.keys())
        self.allowed = torch.zeros(65536, dtype=torch.bool)
        self.allowed[img_ids] = True
        self.dev = None

    def __call__(self, input_ids, scores):
        import torch
        k = input_ids.shape[1] - self.p
        if self.dev is None:
            self.allowed = self.allowed.to(scores.device); self.dev = scores.device
        forced = {0: IMG_START, 1: GRID8, 2: GRID8, 3 + 16 * 17: IMG_END, 4 + 16 * 17: SEP}.get(k)
        if forced is None and 3 <= k < 3 + 16 * 17 and (k - 3 + 1) % 17 == 0:
            forced = NEWLINE
        neg = torch.finfo(scores.dtype).min
        if forced is not None:
            out = torch.full_like(scores, neg); out[:, forced] = 0.0
            return out
        if 3 <= k < 3 + 16 * 17:
            return scores.masked_fill(~self.allowed, neg)
        return scores


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pil(arr):
    from PIL import Image
    return Image.fromarray(np.asarray(arr, dtype=np.uint8))


def imagine(rig, first_frame, task, seed, n_steps=H, sample_wm=None, log_every=20):
    """(a) model branch: alternating loop from the anchor's first real frame.
    Mirrors the official policy loop (eval_solver_libero_256.py:217-291) with env.step replaced by the world model:
      if the action queue is empty: chunk = VLA(prev frame, cur frame)            (his_img[-1:] + [cur_img])
      a = queue.pop(0); a_env = unnorm_min_max(a)
      next frame = WM(cur frame, a_env); his.append(cur); cur = next
    Frames are kept as decoded pixels (256x256, WorldVLA/LeRobot rotated-180 view) and fed back as pixels (official;
    RynnVLA-002 eval_solver_libero_g_video_512_third_wrist.py:245 does the same)."""
    torch = rig.torch
    torch.manual_seed(seed); np.random.seed(seed)
    import random; random.seed(seed)
    cur = _pil(first_frame)
    frames = [np.asarray(cur)]
    his, queue = [], []
    env_actions, norm_actions, chunk_id, wm_tok = [], [], [], []
    T = dict(vla=[], vla_step=[], wm_tokenize=[], wm_generate=[], wm_decode=[], step=[], n_gen=[])
    n_fallback, status, n_chunk = 0, "ok", 0
    st = dict(n_vla_calls=0, n_vla_malformed=0, n_vla_retries=0, first_vla_malformed_step=None, vla_malformed_steps=[],
              n_wm_calls=0, n_wm_malformed=0, n_wm_retries=0, first_wm_malformed_step=None, wm_malformed_steps=[],
              invalid_reason=None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for t in range(n_steps):
        ts = time.time()
        if not queue:
            _sync(torch); tv = time.time()
            chunk = rig.vla_checked(cur, his[-1:], task, t, st)
            _sync(torch); T["vla"].append(time.time() - tv); T["vla_step"].append(t)
            if chunk is None:                           # ruling 4: generation-invalid, recorded as is, no data made up
                status = f"generation_invalid@{t}"; break
            queue = list(chunk); n_chunk += 1
        a = queue.pop(0)
        a_env = unnorm_min_max(a)
        img, gen, tm = rig.wm(cur, a_env, sample=sample_wm)
        st["n_wm_calls"] += 1
        if img is None:                                 # malformed grid -> constrained greedy retries (ruling 4)
            n_fallback += 1; st["n_wm_malformed"] += 1; st["wm_malformed_steps"].append(t)
            if st["first_wm_malformed_step"] is None:
                st["first_wm_malformed_step"] = t
            for _ in range(MAX_RETRY):
                st["n_wm_retries"] += 1
                img, gen, tm2 = rig.wm(cur, a_env, constrain=True, sample=sample_wm)
                tm = {k: tm[k] + tm2[k] if k in ("tokenize", "generate", "decode") else tm2[k] for k in tm}
                if img is not None:
                    break
            if img is None:
                st["invalid_reason"] = f"WM at step {t}: {MAX_RETRY} constrained retries malformed"
                status = f"generation_invalid@{t}"; break
        his.append(cur)
        cur = img
        frames.append(np.asarray(img))
        env_actions.append(a_env.astype(np.float32)); norm_actions.append(a.astype(np.float32))
        chunk_id.append(n_chunk - 1)
        g = np.full(TOK_PER_FRAME + 1, -1, np.int32); g[:min(len(gen), len(g))] = gen[:len(g)]
        wm_tok.append(g)
        T["wm_tokenize"].append(tm["tokenize"]); T["wm_generate"].append(tm["generate"])
        T["wm_decode"].append(tm["decode"]); T["n_gen"].append(tm["n_gen"]); T["step"].append(time.time() - ts)
        if log_every and (t + 1) % log_every == 0:
            el = time.time() - t_start
            print(f"[imagine] step {t + 1}/{n_steps}  {el:.0f}s  ({el / (t + 1):.2f} s/step, "
                  f"wm gen {np.mean(T['wm_generate'][-log_every:]):.2f}s, chunks {n_chunk})", flush=True)
    return dict(frames=np.stack(frames).astype(np.uint8), env_actions=np.stack(env_actions) if env_actions else
                np.zeros((0, 7), np.float32), norm_actions=np.stack(norm_actions) if norm_actions else np.zeros((0, 7)),
                chunk_id=np.array(chunk_id), wm_tokens=np.stack(wm_tok) if wm_tok else np.zeros((0, TOK_PER_FRAME + 1)),
                timing={k: np.array(v) for k, v in T.items()}, n_fallback=n_fallback, status=status,
                n_chunks=n_chunk, seconds=time.time() - t_start, retry_stats=st,
                gpu_peak_GB=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None)


def _sample_cfg(args):
    return dict(temperature=args.temperature, top_k=args.top_k) if args.sample_wm else None


PROTOCOL = _wmvp.P("$WMV_ROOT/protocol.yaml")


def _check_frozen(args):
    """once libero.worldvla.wm_prompt / instr are frozen in protocol.yaml, every imagination must use them (§16)."""
    import yaml
    fz = ((yaml.safe_load(open(PROTOCOL)) or {}).get("libero") or {}).get("worldvla") or {}
    for k in ("wm_prompt", "instr"):
        if k in fz and getattr(args, k) != fz[k]:
            raise SystemExit(f"protocol.yaml freezes libero.worldvla.{k} = {fz[k]!r}; got {getattr(args, k)!r}")


def cmd_imagine(args, rig=None):
    a = load_anchor(args.anchor)
    d = unit_dir(a["anchor_id"], args.seed)
    ff = np.load(os.path.join(d, "first_frame.npz"))
    meta = json.load(open(os.path.join(d, "first_frame.json")))
    _check_frozen(args)
    task = meta["task_description"]
    if args.instr == "name":                           # training-folder instruction variant (eval_design §16)
        _model_imports()
        from libero.libero import benchmark
        task = benchmark.get_benchmark_dict()[SUITE]().get_task(a["task_id"]).name.replace("_", " ")
    meta["task_description"] = task
    rig = rig or Rig(wm_prompt=args.wm_prompt)
    R = imagine(rig, ff["model_view"], task, args.seed, n_steps=args.steps,
                sample_wm=_sample_cfg(args))
    np.savez_compressed(os.path.join(d, "imagined.npz"), frames=R["frames"], env_actions=R["env_actions"],
                        norm_actions=R["norm_actions"], chunk_id=R["chunk_id"], wm_tokens=R["wm_tokens"],
                        **{f"t_{k}": v for k, v in R["timing"].items()})
    T = R["timing"]
    summ = dict(anchor_id=a["anchor_id"], seed=args.seed, task=meta["task_description"], status=R["status"],
                n_actions=int(len(R["env_actions"])), n_frames=int(len(R["frames"])), n_chunks=R["n_chunks"],
                n_struct_fallback=R["n_fallback"], seconds=R["seconds"], gpu_peak_GB=R["gpu_peak_GB"],
                generation_invalid=R["status"].startswith("generation_invalid"), **R["retry_stats"],
                load_model_s=rig.t_load_total, attn_impl=getattr(rig.model.config, "_attn_implementation", None),
                torch_version=rig.torch.__version__, wm_prompt=args.wm_prompt, instr=args.instr, sample_wm=_sample_cfg(args),
                s_per_step_mean=float(T["step"].mean()) if len(T["step"]) else None,
                vla_s_mean=float(T["vla"].mean()) if len(T["vla"]) else None,
                wm_generate_s_mean=float(T["wm_generate"].mean()) if len(T["wm_generate"]) else None,
                wm_decode_s_mean=float(T["wm_decode"].mean()) if len(T["wm_decode"]) else None,
                wm_tokenize_s_mean=float(T["wm_tokenize"].mean()) if len(T["wm_tokenize"]) else None,
                env_actions_sha256=hashlib.sha256(R["env_actions"].tobytes()).hexdigest(),
                frame_convention="256x256 RGB uint8, agentview rotated 180 deg (raw[::-1, ::-1]); frames[0] = real "
                                 "first frame, frames[k] = imagined after k actions")
    json.dump(summ, open(os.path.join(d, "imagined.json"), "w"), indent=1)
    try:
        import imageio
        imageio.mimwrite(os.path.join(d, "imagined.mp4"), R["frames"], fps=20)
    except Exception as e:
        print("video not written:", e)
    print("IMAGINE", json.dumps(summ, indent=1))
    return rig


def _alt_action(a_env, kind):
    a = np.asarray(a_env, dtype=np.float64).copy()
    if kind == "mirror":                     # opposite translation/rotation of at least 0.5 of the range, flip gripper
        b = a.copy()
        mag = np.maximum(np.abs(a[:6]), 0.5 * ACT_MAX[:6])
        b[:6] = -np.sign(np.where(a[:6] == 0, 1.0, a[:6])) * mag
        b[6] = -a[6] if a[6] != 0 else 1.0
        return np.clip(b, ACT_MIN, ACT_MAX)
    if kind == "zero":
        b = np.zeros(7); b[6] = a[6]; return b
    if kind == "gripper":
        b = a.copy(); b[6] = -a[6] if a[6] != 0 else 1.0; return b
    raise ValueError(kind)


def cmd_probe(args, rig=None):
    """(c) joint-consistency probe: same prefix (imagined frame t of the baseline rollout), WM-only roll of K steps with
    the baseline actions a_t..a_{t+K-1} (branch A) vs the same with a_t replaced (branch B). A vs baseline = noise floor
    (greedy decoding should reproduce the baseline exactly); B vs A = action effect on every subsequent frame."""
    a = load_anchor(args.anchor)
    d = unit_dir(a["anchor_id"], args.seed)
    I = np.load(os.path.join(d, "imagined.npz"))
    t, K = args.t, args.k
    assert t + K <= len(I["env_actions"]), "baseline rollout too short for this probe"
    rig = rig or Rig(wm_prompt=args.wm_prompt)
    torch = rig.torch
    torch.manual_seed(args.seed)
    acts = I["env_actions"][t:t + K].astype(np.float64)
    alt = _alt_action(acts[0], args.alt)
    branches = {}
    for name, seq in (("A_same", acts), ("B_swapped", np.concatenate([alt[None], acts[1:]], 0))):
        cur, F, Tk = _pil(I["frames"][t]), [], []
        for k in range(K):
            img, gen, _ = rig.wm(cur, seq[k], sample=_sample_cfg(args))
            if img is None:
                img, gen, _ = rig.wm(cur, seq[k], constrain=True, sample=_sample_cfg(args))
            F.append(np.asarray(img)); Tk.append(np.array(rig.frame_tokens(img)))
            cur = img
        branches[name] = (np.stack(F), np.stack(Tk))
    base_F = I["frames"][t + 1:t + 1 + K]
    base_T = np.stack([np.array(rig.frame_tokens(_pil(f))) for f in base_F])

    def cmp(F1, T1, F2, T2):
        mae = np.abs(F1.astype(float) - F2.astype(float)).mean(axis=(1, 2, 3))
        mse = ((F1.astype(float) - F2.astype(float)) ** 2).mean(axis=(1, 2, 3))
        psnr = np.where(mse > 0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)), np.inf)
        tokdiff = (T1 != T2).mean(axis=1)
        return dict(mae=mae.tolist(), psnr=[float(x) if np.isfinite(x) else None for x in psnr],
                    token_diff_frac=tokdiff.tolist())
    FA, TA = branches["A_same"]; FB, TB = branches["B_swapped"]
    floor = cmp(FA, TA, base_F, base_T)
    effect = cmp(FB, TB, FA, TA)
    fl = np.array(floor["mae"]); ef = np.array(effect["mae"])
    out = dict(anchor_id=a["anchor_id"], seed=args.seed, t=t, K=K, alt_kind=args.alt,
               a_t=acts[0].tolist(), a_t_alt=alt.tolist(), floor_A_vs_baseline=floor, effect_B_vs_A=effect,
               first_frame_changes=bool(ef[0] > max(5 * fl[0], 0.5)),
               all_subsequent_change=bool(np.all(ef > np.maximum(5 * fl, 0.5))),
               note="WM is Markov: a_t can affect frames t+2.. only through frame t+1")
    np.savez_compressed(os.path.join(d, f"probe_t{t}.npz"), A=FA, B=FB, base=base_F)
    json.dump(out, open(os.path.join(d, f"probe_t{t}.json"), "w"), indent=1)
    print("PROBE", json.dumps({k: out[k] for k in ("t", "K", "first_frame_changes", "all_subsequent_change")}),
          "effect mae", np.round(ef, 2).tolist(), "floor mae", np.round(fl, 2).tolist())
    return rig


def _psnr(x, y):
    mse = ((np.asarray(x, float) - np.asarray(y, float)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else None


def _score_wordings(rig, pairs):
    """teacher-forced one-step WM prediction for EVERY wording in WM_PROMPTS on the SAME (frame_t, a_t, frame_t+1) pairs."""
    out = dict(copy_last=[_psnr(f0, f1) for f0, _, f1 in pairs],
               vq_roundtrip=[_psnr(np.asarray(rig.decode(rig.frame_tokens(_pil(f1)))), f1) for _, _, f1 in pairs])
    keep = rig.wm_prompt
    for name, text in WM_PROMPTS.items():
        rig.wm_prompt = text
        ps, nf, t0 = [], 0, time.time()
        for f0, a, f1 in pairs:
            img, _, _ = rig.wm(_pil(f0), a)
            if img is None:
                nf += 1
                img, _, _ = rig.wm(_pil(f0), a, constrain=True)
            ps.append(_psnr(img, f1))
        out[f"wm_{name}"] = ps
        out[f"wm_{name}_n_struct_fallback"] = nf
        out[f"wm_{name}_seconds"] = round(time.time() - t0, 1)
        print(f"  wording {name}: done ({nf} structural fallbacks)", flush=True)
    rig.wm_prompt = keep
    mean = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
    keys = ["copy_last", "vq_roundtrip"] + [f"wm_{n}" for n in WM_PROMPTS]
    out["mean"] = {k: mean(out[k]) for k in keys}
    out["median"] = {k: float(np.median([x for x in out[k] if x is not None])) for k in keys if mean(out[k]) is not None}
    # paired differences (same pairs) vs the best mean, 95% bootstrap CI
    names = list(WM_PROMPTS)
    best = max(names, key=lambda n: -np.inf if out["mean"][f"wm_{n}"] is None else out["mean"][f"wm_{n}"])
    rs = np.random.RandomState(0)
    diffs = {}
    for n in names:
        if n == best:
            continue
        d = np.array([(x if x is not None else 99.0) - (y if y is not None else 99.0)
                      for x, y in zip(out[f"wm_{best}"], out[f"wm_{n}"])])
        bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        diffs[f"{best}_minus_{n}"] = dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         wins=int((d > 0).sum()), losses=int((d < 0).sum()), n=int(len(d)))
    out["best_by_mean_psnr"] = best
    out["paired_vs_best"] = diffs
    return out


def cmd_onestep(args, rig=None):
    """WM fidelity / prompt selection: teacher-forced one-step prediction from a REAL frame t with the real action a_t,
    scored (PSNR) against the real frame t+1, for EVERY wording in WM_PROMPTS on the same pairs.
    --source official (prompt selection, eval_design §16): transitions recorded by `official-eval` (closed-loop policy
        mode, no WM prompt involved -> neutral w.r.t. the wordings); n_onestep pairs drawn with ONESTEP_SEED.
    --source replay: open-loop replay of this unit's imagined actions (needs replay.npz / imagined.npz).
    Reference columns: copy-last-frame baseline and the VQ round-trip of the real target (upper bound)."""
    if args.source == "official":
        run = os.path.join(OUT, "official_eval", f"task{args.task_id}_n{args.n_episodes}", "transitions")
        pool = []
        for f in sorted(glob.glob(os.path.join(run, "ep*.npz"))):
            z = np.load(f)
            F, A, nd = z["frames"], z["actions"], int(z["n_dummy"])
            pool += [(f, t) for t in range(nd, len(A))]
        assert pool, f"no recorded transitions under {run} (run official-eval first)"
        rs = np.random.RandomState(ONESTEP_SEED)
        pick = sorted(rs.choice(len(pool), size=min(args.n_onestep, len(pool)), replace=False).tolist())
        cache, pairs, meta = {}, [], []
        for i in pick:
            f, t = pool[i]
            if f not in cache:
                cache[f] = np.load(f)
            z = cache[f]
            pairs.append((z["frames"][t], z["actions"][t], z["frames"][t + 1]))
            meta.append([os.path.basename(f), int(t)])
        d = os.path.join(OUT, "prompt_select"); os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f"onestep_official_task{args.task_id}.json")
        head = dict(source="official", task_id=args.task_id, pool_size=len(pool), seed=ONESTEP_SEED, pairs=meta,
                    wordings=WM_PROMPTS)
    else:
        a = load_anchor(args.anchor)
        d = unit_dir(a["anchor_id"], args.seed)
        real = np.load(os.path.join(d, "replay.npz"))["frames"]
        acts = np.load(os.path.join(d, "imagined.npz"))["env_actions"]
        n = min(len(acts), len(real) - 1)
        idx = np.unique(np.linspace(0, n - 1, args.n_onestep).astype(int))
        pairs = [(real[t], acts[t], real[t + 1]) for t in idx]
        dst = os.path.join(d, "onestep.json")
        head = dict(source="replay", anchor_id=a["anchor_id"], seed=args.seed, steps=idx.tolist(), wordings=WM_PROMPTS)
    rig = rig or Rig(wm_prompt=args.wm_prompt)
    out = dict(head, **_score_wordings(rig, pairs))
    json.dump(out, open(dst, "w"), indent=1)
    print("ONESTEP", json.dumps(dict(mean=out["mean"], best=out["best_by_mean_psnr"], paired=out["paired_vs_best"])))
    return rig


def cmd_model(args):
    """imagine + probe in one process (loads the 7B once)."""
    rig = cmd_imagine(args)
    if not args.no_probe:
        cmd_probe(args, rig)


# =====================================================================================================================
# (d) OFFICIAL EVAL WRAPPER (worldvla interpreter; LIBERO is installed in that env too: hf-libero 0.1.4 / mujoco 3.2.7)
# =====================================================================================================================
class _StopEval(BaseException):
    pass


def cmd_official_eval(args):
    """Run the repo's own LIBERO action-model eval (eval_solver_libero_256.py) unchanged, with the launch arguments of
    exps_256_all/eval_libero_7B_2_action_all_epochs_img_only_ck_5_1a2i_10.sh, restricted to ONE libero_10 task and
    n_episodes (default 50 = the repo's per-task count) by proxying the benchmark object. No repo file is edited:
    the relative paths the repo hard-codes ('../ckpts/chameleon/tokenizer/*', 'worldvla/ckpts/chameleon/base_model')
    are satisfied by a symlink farm under the run dir. Paper/README libero_10 (256) = 59.0% over 10 tasks x 50."""
    import runpy
    run = os.path.join(OUT, "official_eval", f"task{args.task_id}_n{args.n_episodes}" + ("" if args.instr == "language" else "_name"))
    # symlink farm on a local fs (the QNAP mount refuses symlinks, errno 95); outputs stay under `run`
    farm = os.path.join(os.path.expanduser("~/.cache/wm_verdict_worldvla_farm"), os.path.basename(run))
    cwd = os.path.join(farm, "cwd")
    os.makedirs(os.path.join(farm, "ckpts", "chameleon"), exist_ok=True)
    os.makedirs(os.path.join(cwd, "worldvla", "ckpts", "chameleon"), exist_ok=True)
    os.makedirs(os.path.join(run, "output"), exist_ok=True)   # configure_logger runs before the solver's mkdir (eval_solver:49-53)
    for link, target in ((os.path.join(farm, "ckpts", "chameleon", "tokenizer"), VQ_DIR),
                         (os.path.join(cwd, "worldvla", "ckpts", "chameleon", "base_model"), TOK_DIR)):
        if not os.path.islink(link):
            os.symlink(target, link)
    os.chdir(cwd)
    _model_imports()
    from libero.libero import benchmark
    real_get = benchmark.get_benchmark_dict
    tid, n_ep = args.task_id, args.n_episodes

    class InitProxy:
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            if i >= n_ep:
                raise _StopEval()
            return self.s[i]

    class SuiteProxy:
        def __init__(self, s): self.s = s; self.n_tasks = 1
        def get_task(self, i): return self.s.get_task(tid)
        def get_task_init_states(self, i): return InitProxy(self.s.get_task_init_states(tid))
        def __getattr__(self, k): return getattr(self.s, k)

    def patched():
        d = dict(real_get())
        d[SUITE] = lambda *a, **k: SuiteProxy(real_get()[SUITE](*a, **k))
        return d
    benchmark.get_benchmark_dict = patched
    import libero_util.libero_utils as lu
    results = []
    # --instr name: the training-data instruction variant (folder name = task.name incl. the scene prefix, "_" -> " ";
    # regenerate_libero_dataset_save_img_action.py:84, action_model_conv_generation.py:58). Default = repo eval (task.language).
    # Every env step is recorded (rotated 256 agentview after the step + the raw env action) for the WM prompt selection.
    tdir = os.path.join(run, "transitions"); os.makedirs(tdir, exist_ok=True)
    orig_env = lu.get_libero_env

    class Rec:
        def __init__(self, env): self.env, self.F, self.A = env, [], []
        def set_init_state(self, st):
            obs = self.env.set_init_state(st)
            self.F, self.A = [np.ascontiguousarray(lu.get_libero_image(obs, RES))], []
            return obs
        def step(self, act):
            obs, r, done, info = self.env.step(act)
            self.F.append(np.ascontiguousarray(lu.get_libero_image(obs, RES))); self.A.append(np.asarray(act, np.float64))
            return obs, r, done, info
        def __getattr__(self, k): return getattr(self.env, k)
    rec = {}

    def get_env(task, resolution=256):
        env, desc = orig_env(task, resolution=resolution)
        if args.instr == "name":
            desc = task.name.replace("_", " ")
        rec["env"] = Rec(env)
        return rec["env"], desc
    lu.get_libero_env = get_env
    orig_save = lu.save_rollout_video

    def save(rollout_dir, imgs, idx, success, task_description):
        results.append(dict(episode=int(idx), success=bool(success), steps=len(imgs)))
        R = rec["env"]
        if not args.no_record:
          np.savez_compressed(os.path.join(tdir, f"ep{int(idx):03d}.npz"), frames=np.stack(R.F), actions=np.stack(R.A),
                            n_dummy=10, success=bool(success))
        json.dump(dict(task_id=tid, task=task_description, instr=args.instr, episodes=results,
                       success_rate=float(np.mean([r["success"] for r in results]))),
                  open(os.path.join(run, "summary.json"), "w"), indent=1)
        return orig_save(rollout_dir, imgs, idx, success, task_description)
    lu.save_rollout_video = save
    sys.argv = ["eval_solver_libero_256.py", "--device", "0", "--task_suite_name", SUITE, "--his", "2h_1a_img_only",
                "--no_auto_resume", "--resume_path", CKPT, "--eval_only", "True", "--model_size", "7B",
                "--batch_size", "4", "--accum_iter", "1", "--epochs", "10", "--warmup_epochs", "0.01",
                "--lr", "5e-6", "--min_lr", "5e-6", "--wd", "0.1", "--clip_grad", "4",
                "--data_config_train", "unused", "--data_config_val_ind", "unused", "--data_config_val_ood", "unused",
                "--cache_ann_on_disk", "--num_workers", "8", "--output_dir", os.path.join(run, "output"),
                "--checkpointing", "--max_seq_len", "8192", "--unmask_image_logits", "--dropout", "0.05",
                "--z_loss_weight", "1e-5", "--ckpt_max_keep", "0"]
    t0 = time.time()
    try:
        runpy.run_path(os.path.join(REPO_WV, "eval_solver_libero_256.py"), run_name="__main__")
    except _StopEval:
        pass
    sr = float(np.mean([r["success"] for r in results])) if results else None
    out = dict(task_id=tid, instr=args.instr, n_episodes=len(results), success_rate=sr, seconds=time.time() - t0, episodes=results,
               reference="README Model Zoo LIBERO-Long 256: 59.0 (suite mean, 10 tasks x 50 episodes)")
    json.dump(out, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print("OFFICIAL_EVAL", json.dumps({k: out[k] for k in ("task_id", "instr", "n_episodes", "success_rate", "seconds")}))


# =====================================================================================================================
def cmd_mask_reset_check(args):
    """eval_design section 15: mask-reset equivalence. Our loop resets model.init_input_ids before every WM call. The
    official single-call path (Chameleon_utils.get_action_Chameleon_dis_awm_g_video, his_type "1a2i", = the frozen
    "helper" wording) called on a FRESH model (init_input_ids None, as at the first call of a process) must give the
    SAME tokens as our looped call, for every step of a short loop (K steps from the anchor's first frame, actions = the
    first VLA chunk). Informative extra: the official helper called back-to-back WITHOUT the reset. Mismatch -> stop."""
    _model_imports()
    from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_g_video
    a = load_anchor(args.anchor); d = unit_dir(a["anchor_id"], args.seed)
    ff = np.load(os.path.join(d, "first_frame.npz")); meta = json.load(open(os.path.join(d, "first_frame.json")))
    assert args.wm_prompt == "helper", "the official single-call path uses the helper wording"
    rig = Rig(wm_prompt="helper"); m = rig.model
    cap = {}
    orig = m.generate_img

    def gen_capture(input_ids, generation_config):
        out = orig(input_ids, generation_config); cap["toks"] = out[0].cpu().tolist(); return out
    m.generate_img = gen_capture
    chunk = rig.vla_checked(_pil(ff["model_view"]), [], meta["task_description"], 0,
                            dict(n_vla_calls=0, n_vla_malformed=0, n_vla_retries=0, first_vla_malformed_step=None,
                                 vla_malformed_steps=[], invalid_reason=None))
    K = min(args.k, len(chunk))
    cur = _pil(ff["model_view"]); rows = []; ours = []
    for k in range(K):
        a_env = unnorm_min_max(chunk[k])
        img, gen, _ = rig.wm(cur, a_env)                       # our loop: reset + generate
        m.init_input_ids = None
        get_action_Chameleon_dis_awm_g_video(m, "", rig.proc, [cur], [a_env], "1a2i")   # official, fresh model state
        off = cap["toks"]
        rows.append(dict(step=k, identical=(off == gen), n_ours=len(gen), n_official=len(off),
                         first_diff=next((i for i, (x, y) in enumerate(zip(off, gen)) if x != y), None)))
        ours.append((cur, a_env, gen))
        cur = img
    # informative: official helper back-to-back without reset on the same inputs
    m.init_input_ids = None; noreset = []
    for k, (c, a_env, g) in enumerate(ours):
        try:
            get_action_Chameleon_dis_awm_g_video(m, "", rig.proc, [c], [a_env], "1a2i")
            noreset.append(dict(step=k, identical_to_ours=(cap["toks"] == g), n=len(cap["toks"])))
        except Exception as e:
            noreset.append(dict(step=k, error=repr(e)[:120])); break
    m.generate_img = orig; m.init_input_ids = None
    out = dict(anchor_id=a["anchor_id"], K=K, all_identical=all(r["identical"] for r in rows), rows=rows,
               official_without_reset=noreset)
    json.dump(out, open(os.path.join(d, "mask_reset_check.json"), "w"), indent=1)
    print("MASK_RESET", json.dumps(dict(all_identical=out["all_identical"], rows=rows, without_reset=noreset)))


def cmd_real_vla(args):
    """Ruling 4 control: policy-mode malformation rate on REAL frames, same anchors (both tasks, bank order). The policy
    runs closed loop in the real simulator exactly as the official eval loop (eval_solver_libero_256.py:188-291: same env
    construction get_libero_env(256), env.seed(0), set_init_state(init_states[init_index]), 10 no-op steps, chunk
    queue), started from each anchor's init state; every VLA call goes through vla_checked (same malformation test and
    constrained retry as in imagination). Note: the official-eval env stack (LIBERO in the worldvla env, mujoco 3.2.7),
    so the start equals the anchor's init state, not its bit-exact LeRobot restore. Output: OUT/real_vla_control.json."""
    _model_imports()
    from PIL import Image
    from libero.libero import benchmark
    import libero_util.libero_utils as lu
    bank = json.load(open(BANK))["anchors"]
    rig = Rig(wm_prompt=args.wm_prompt)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    rows = []
    for a in bank:
        task = suite.get_task(a["task_id"]); inits = suite.get_task_init_states(a["task_id"])
        env, desc = lu.get_libero_env(task, resolution=RES)
        env.reset(); obs = env.set_init_state(inits[a["init_index"]])
        for _ in range(10):
            obs, *_ = env.step(lu.get_libero_dummy_action())
        st = dict(n_vla_calls=0, n_vla_malformed=0, n_vla_retries=0, first_vla_malformed_step=None,
                  vla_malformed_steps=[], invalid_reason=None)
        his, queue, done, t, status = [], [], False, 0, "ok"
        while t < H and not done:
            cur = Image.fromarray(lu.get_libero_image(obs, RES))
            if not queue:
                chunk = rig.vla_checked(cur, his[-1:], desc, t, st)
                if chunk is None:
                    status = f"generation_invalid@{t}"; break
                queue = list(chunk)
            obs, r, done, info = env.step(unnorm_min_max(queue.pop(0)).tolist())
            his.append(cur); t += 1
        env.close()
        rows.append(dict(anchor_id=a["anchor_id"], init_index=a["init_index"], steps=t, success=bool(done),
                         status=status, **st))
        print("REAL_VLA", json.dumps({k: rows[-1][k] for k in ("anchor_id", "steps", "success", "n_vla_calls",
                                                              "n_vla_malformed", "n_vla_retries")}), flush=True)
    tot = dict(n_vla_calls=sum(r["n_vla_calls"] for r in rows), n_vla_malformed=sum(r["n_vla_malformed"] for r in rows))
    tot["malformed_rate_per_call"] = tot["n_vla_malformed"] / max(1, tot["n_vla_calls"])
    json.dump(dict(control="policy mode on real frames, same anchors", rows=rows, total=tot),
              open(os.path.join(OUT, "real_vla_control.json"), "w"), indent=1)
    print("REAL_VLA_TOTAL", json.dumps(tot))


# =====================================================================================================================
def cmd_all(args):
    """full unit: env(first frame) -> model(imagine + probe) -> env(replay) -> model(one-step WM check vs real frames)."""
    me = os.path.abspath(__file__)
    common = ["--anchor", args.anchor, "--seed", str(args.seed)]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), PYTHONUNBUFFERED="1")
    steps = [[PY_ENV, me, "first-frame", *common, "--check_res"],
             [PY_MODEL, me, "model", *common, "--steps", str(args.steps), "--t", str(args.t), "--k", str(args.k),
              "--wm_prompt", args.wm_prompt, "--instr", args.instr, "--alt", args.alt] + (["--no_probe"] if args.no_probe else []),
             [PY_ENV, me, "replay", *common, "--check_res"]]
    if not args.no_onestep:
        steps.append([PY_MODEL, me, "onestep", *common, "--n_onestep", str(args.n_onestep), "--wm_prompt", args.wm_prompt])
    for c in steps:
        print(">>", " ".join(c), flush=True)
        subprocess.run(c, check=True, env=env, cwd=HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["first-frame", "imagine", "probe", "model", "replay", "onestep", "official-eval", "all",
                                   "real-vla", "mask-reset-check"])
    p.add_argument("--anchor", default="libero10_t5_a0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=H)
    p.add_argument("--wm_prompt", choices=list(WM_PROMPTS), default="helper")   # frozen 2026-10-15 (protocol libero.worldvla)
    p.add_argument("--sample_wm", action="store_true", help="sample WM tokens (default greedy = official)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--t", type=int, default=40, help="probe: step whose action is replaced")
    p.add_argument("--k", type=int, default=8, help="probe: WM steps after the swap")
    p.add_argument("--alt", choices=["mirror", "zero", "gripper"], default="mirror")
    p.add_argument("--no_probe", action="store_true")
    p.add_argument("--no_onestep", action="store_true")
    p.add_argument("--n_onestep", type=int, default=20)
    p.add_argument("--check_res", action="store_true", help="env: also compare physics with a bank-resolution env")
    p.add_argument("--task_id", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--instr", choices=["language", "name"], default="language",
                   help="official-eval: repo instruction (task.language) or training-folder variant with scene prefix (task.name)")
    p.add_argument("--source", choices=["official", "replay"], default="replay", help="onestep: frame source")
    p.add_argument("--no_record", action="store_true", help="official-eval: do not store per-step transitions")
    args = p.parse_args()
    {"first-frame": cmd_first_frame, "replay": cmd_replay, "imagine": cmd_imagine, "probe": cmd_probe,
     "model": cmd_model, "onestep": cmd_onestep, "official-eval": cmd_official_eval, "all": cmd_all,
     "real-vla": cmd_real_vla, "mask-reset-check": cmd_mask_reset_check}[args.cmd](args)


if __name__ == "__main__":
    main()
