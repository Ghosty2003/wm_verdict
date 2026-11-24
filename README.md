# WM-Verdict

Paired open-loop (imagination) / closed-loop (deployment) benchmark for world-action models on LIBERO-10 under
environment, physics and information perturbations (families A, P, K, E, G, N, O, F, B), with frozen anchors, frozen
success predicates and a process-integrity gate (`protocol/core/verify_run.py`).

## Layout

| Directory | Contents |
|---|---|
| `wmv_paths.py` | the path module: every machine path in the code is read from environment variables through it |
| `perturbations/` | perturbation implementations: `phys/` (E/G/N/K, `ActionStream`), `bfam/` (image corruptions; LIBERO-plus functions in a pinned venv), `ffam/` (single full-frame blackout), `init_state/` (A/P families, native LIBERO hooks) |
| `protocol/` | `protocol.yaml` (frozen protocol), `release_hashes.yaml`, `core/` (anchor banks, `verify_run.py`, archive minimum), `runners/` (LingBot-VA, OpenWAM-alpha, WorldVLA), `rerun/`, `checks/`, `banks/` (frozen anchors + derived BDDL), `assets/` |
| `predicates/` | success predicates, calibration and threshold selection, trajectory rule v2 (weights: release asset) |
| `analysis/` | closed-loop main table (incl. the loaders of the Cosmos Policy / VPP2 / UVA results), per-cell analyses, terminal-imagination optimism, results page, diagnostics |
| `analysis/judges/` | judging imagined frames: the per-frame rubrics (four-label and two-label), task reference images, the VLM judge runner, run instructions for an agent judge, the merged blind-package builder, and scoring / comparison of judges against the held-out key |
| `analysis/human_eval/` | human evaluation from the annotation-website export (rater accuracy on real frames, human verdicts on imagined frames, inter-rater agreement, agreement with the automated judges) |
| `analysis/paper_repro/` | scripts that regenerate the verdict-reliability table (with Wilson intervals), the plan-vs-verdict and plan-vs-deployment scatter statistics, and the shift-family / level overview; `run_all.sh` |
| `annotation_pipeline/` | builds the cards and page tables served by the annotation website |
| `docs/` | perturbation registry, evaluation design, alignment spec, exemptions, model cards, data formats, determinism notes |
| `scripts/` | `make_data_root.py`, `repro_check_openwam_t1.py`, `setup_openwam.sh`, `legacy_layout.tsv` |
| `env/`, `patches/`, `third_party/` | conda environments, our two patches to third-party files, fetch scripts (nothing vendored) |

## Install

Tested on Linux x86-64, CUDA driver >= 570 (cu128 wheels), RTX 5090 / 4090 / H100.

```bash
# 1) simulator + LingBot-VA env (python 3.12)
conda env create -f env/lerobot_libero.yml
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
git clone https://github.com/huggingface/lerobot third_party/lerobot
git -C third_party/lerobot checkout e624f3f7f8411ec3a02635d06e79373341e5ef35
conda run -n lerobot_libero pip install --no-deps -e third_party/lerobot
# optional, shared machines only (robosuite hard-codes /tmp/robosuite.log):
#   patch -d <site-packages> -p1 --ignore-whitespace < patches/robosuite-1.4.0_log_utils.patch

# 2) OpenWAM-alpha env (python 3.10)
conda env create -f env/openwam.yml
git clone https://github.com/OpenWAM-Official/OpenWAM third_party/OpenWAM
git -C third_party/OpenWAM checkout 1e103e1c61905c230dab6b0f99980370cf2f9e3c
conda run -n openwam pip install --no-deps -e third_party/OpenWAM
export OPENWAM_REPO=$PWD/third_party/OpenWAM
bash scripts/setup_openwam.sh                      # downloads Wan2.2-TI2V-5B + OpenWAM-Alpha-Sim-LIBERO
export WAN22_TI2V_5B_DIR=$OPENWAM_REPO/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
envsubst '$WAN22_TI2V_5B_DIR' < patches/openwam-1e103e1_wan22_model_path.patch | patch -d $OPENWAM_REPO -p1

# 3) B family: LIBERO-plus code pinned by commit + sha256, executed in its own venv
eval "$(bash third_party/fetch_libero_plus.sh | tail -1)"     # sets LIBERO_PLUS_ENV_WRAPPER
export WMV_VENV_ROOT=$HOME/bcheck_venvs
bash perturbations/bfam/healthcheck/mkvenv.sh      # numpy 1.22.4 / scikit-image 0.21.0

# 4) HF weights / caches: lerobot/lingbot_va_libero_long, robbyant/lingbot-va-base,
#    robbyant/lingbot-va-posttrain-libero-long, lerobot/libero-assets, lerobot/libero_10_image
#    integrity: env/weights_sha256.txt

# 5) predicate weights (GitHub Release asset "release_assets"): check with  (cd release_assets && sha256sum -c SHA256SUMS)
```

## Paths

All machine paths go through `wmv_paths.py`: `P("$WMV_ROOT/libero/out/...")` expands environment variables, and
`bootstrap()` puts this repository's code directories on `sys.path` (scripts find the repository root themselves, no
`PYTHONPATH` needed). A variable that is not set stays literally in the path (set `WMV_STRICT_PATHS=1` to fail
immediately instead).

| Variable | Meaning |
|---|---|
| `WMV_ROOT` | data root in the legacy layout (libero/out/..., protocol.yaml, banks); build it with scripts/make_data_root.py |
| `WMV_WEB_ROOT` | local clone of the annotation-website repository |
| `WMV_NAS` | shared storage root of the original runs (cloud pulls, node_sync); only for provenance scripts |
| `NAS` | shared storage mount point (provenance scripts only) |
| `LEROBOT_ROOT` | lerobot clone at e624f3f7 (only if it is not pip-installed) |
| `MMBENCH2_ROOT` | MMBench2 pilot directory (not shipped) |
| `DINO_WM_ROOT` | DINO-WM clone (not shipped) |
| `COSMOS_DATA_ROOT` | collaborators' Cosmos Policy / VPP2 audit data root |
| `COSMOS_DATA_HOST` | collaborator storage root (Cosmos / VPP2) |
| `UVA_DATA_ROOT` | collaborators' UVA validation data root |
| `UVA_DATA_HOST` | collaborator storage root (UVA) |
| `HOME` | home directory |
| `CONDA_ROOT` | conda installation root (envs lerobot_libero, openwam) |
| `WMV_VENV_ROOT` | directory holding venv_pinned (B-family corruption worker) |
| `SCRATCH` | scratch directory |
| `LIBERO_REPO` | a LIBERO source checkout (BDDL files) |
| `OPENWAM_REPO` | OpenWAM clone at 1e103e1 |
| `WORLDVLA_REPO` | WorldVLA clone (branch worldvla @ ebdc36dc5cfd) |
| `LIBERO_PLUS_ENV_WRAPPER` | LIBERO-plus env_wrapper.py @ 4976dc30 (third_party/fetch_libero_plus.sh) |
| `DROPBOX_REMOTE` | rclone remote of the data mirror (provenance scripts only) |
| `NODE_CLOUD` | directory name (placeholder) of the cloud node's pulled results |

`WMV_ROOT` is a **data root** in the layout the code addresses (`protocol.yaml`, `libero/out/bank/...`,
`libero/out/runs/<run>/...`, predicate weights at their protocol paths). It contains no code; build it with
symlinks (needs a filesystem that supports them):

```bash
python scripts/make_data_root.py --out /path/to/wmv_root --assets /path/to/release_assets \
    --data /path/to/run_data --subtree libero/out/runs/openwam_base
export WMV_ROOT=/path/to/wmv_root
```

Hashes: `protocol/protocol.yaml` keeps the sha256 of the original run archive (authoritative). Files whose release copy
differs (placeholders; path module) are listed in `protocol/release_hashes.yaml` with both hashes; `verify_run.py`
check 1 accepts a release-copy hash only when the protocol value equals the recorded original. Machine names (nodes) are
represented by placeholders only (`$NODE_A`...`$NODE_E`, `$NODE_CLOUD`, `$NODE`), in code, protocol and records alike.

## Reproduce one cell (OpenWAM-alpha, T1 = LIBERO-10 #3, no perturbation)

```bash
export MUJOCO_GL=egl WMV_ROOT=/path/to/wmv_root OPENWAM_REPO=/path/to/OpenWAM CONDA_ROOT=/path/to/conda
# a) run: 5 anchors x 10 imaginations; per unit imagination (13 chained calls) + truth replay x2 + closed loop x2 (GPU)
conda run -n openwam python protocol/runners/openwam/ow_runner.py --run base
# b) model-side verdicts with the frozen OpenWAM-view predicate
conda run -n openwam python predicates/openwam/ow_score.py --run base --tasks 3
# c) gate: no check may FAIL
conda run -n lerobot_libero python protocol/core/verify_run.py --run-dir $WMV_ROOT --protocol $WMV_ROOT/protocol.yaml --run openwam_base
# d) closed-loop main table snapshot
python analysis/main_table/gen_closed_main.py && python analysis/main_table/make_snapshot.py $(date +%Y%m%d)
# check an existing archive of this cell without a GPU (bit-exact replays + predicate re-scoring):
python scripts/repro_check_openwam_t1.py replay --units 0:50 --out out_check --workers 8 --no-gl
python scripts/repro_check_openwam_t1.py score  --units 0:50 --out out_check
```

## Reproduction check (release copy)

- **Bit-exact replay (CPU, no GPU)**: all 50 units of the cell (5 anchors x 10 imaginations, run `openwam_base`, `records_t3.jsonl`). For each unit a fresh environment is built from the frozen anchor and (a) the stored imagined actions (truth replay, 400 steps) and (b) the stored closed-loop actions (400 steps) are replayed. Per-step success identical: 50/50 (a), 50/50 (b); full-physics state after every step identical: 50/50 (a), 50/50 (b); first success step equal to the record: 50/50 (a, 30 successes), 50/50 (b, 50 successes). Host `$NODE_A` (node placeholder; CPU job, no GPU allocated; robosuite's offscreen renderer replaced by a blank-image stub because GPU-less nodes have no OpenGL, which does not touch physics), 84.5 s with 8 workers.
- **Predicate re-scoring (CPU)**: the release weights (`release_assets/predicates/openwam/t3/close.pt`, `bowl_out.pt`; sha256 equal to the predicate manifest: True) score the stored imagined canvases (101 frames per unit). Per-frame verdict sequence identical to the archived GPU scores: 50/50; model_verdict equal to the record: 50/50; model_first_success_step equal: 50/50. Probabilities are not bit-identical across CPU / GPU: max |dp| = 1.11e-03.
- **Integrity gate, run `openwam_base`**: the release `verify_run.py` (incl. the simulator replays of checks 2 / 4 / 8) gives overall PASS (archived report: PASS); check1 PASS, check2 PASS, check3 PASS, check4 PASS, check5 SKIP, check6 SKIP, check7 WARN, check8 SKIP, check9 PASS (new since the archived report), check10 SKIP (new since the archived report); status equal to the archived report for 8/8 archived checks.
- **Integrity gate, run `openwam_famA_obj`**: the release `verify_run.py` (incl. the simulator replays of checks 2 / 4 / 8) gives overall PASS (archived report: PASS); check1 PASS, check2 PASS, check3 PASS, check4 PASS, check5 SKIP, check6 WARN, check7 WARN, check8 PASS, check9 PASS, check10 SKIP (new since the archived report); status equal to the archived report for 9/9 archived checks; check 1 accepted the release-copy hash of `libero/out/bank/familyA/familyA_anchors.json` via `protocol/release_hashes.yaml`.
- **Code provenance**: every run used only this repository's code: no imported module and no `sys.path` entry points into the original project directory (OK; checked in the main process and, for the replays, in every worker). Data and predicate weights were read through `$WMV_ROOT`, a symlink view built by `scripts/make_data_root.py`.
- **Not done**: the optional GPU closed-loop rerun on the H100 machine was skipped: all four GPUs were fully occupied by production runs (99 % utilisation), and the run must not disturb them.
- Re-run: build the data root (section Paths), then `python scripts/repro_check_openwam_t1.py replay --units 0:50 --out DIR --workers 8 --no-gl`, `... score --units 0:50 --out DIR`, `... verify --out DIR --no-gl [--verify-run openwam_famA_obj]`; each exits 1 on any mismatch (on a GPU machine, drop `--no-gl`).

## Judging imagined frames

A verdict is a judgment, made from the imagined rollout alone, of whether the task is completed. It is scored against
the real outcome of the same run (same start, condition and seed). Labels: `looks_success`, `borderline` (the scene looks
complete but the image shows generation distortion, or the key part is hidden / exactly at the edge), `not_success`,
and `uncertain` (the image is too degraded to see the key objects). *Main* counts `looks_success` as an accepted verdict;
*wide* also counts `borderline`; `uncertain` is left out of denominators.

```bash
export WMV_ROOT=/path/to/wmv_root
# VLM judge (OpenRouter); the key is read from the environment only and never written anywhere
read -rsp 'OpenRouter key: ' OPENROUTER_API_KEY && export OPENROUTER_API_KEY
python analysis/judges/package/vlm_openrouter/run_vlm_judge.py --model qwen/qwen4-max --max-tokens 2000 --tag qwen4max \
    --items <items_blind.csv> --frames-dir <frames/>          # four-label rubric: package/prompt/judge_prompt.md
python analysis/judges/package_binary/vlm_openrouter/run_vlm_judge.py ...   # two-label (forced choice) rubric
unset OPENROUTER_API_KEY
# agent judge: give prompt/run_agent_judge_all.md to a coding agent working inside the unzipped blind package
python analysis/judges/build_merged_package.py               # blind package (shuffled neutral ids) + separate key
python analysis/judges/score_agent_judge.py <name> <labels.csv> <selfcheck.csv> <out_dir>
python analysis/judges/compare_judges.py                     # all judges side by side on shared frames
python analysis/human_eval/human_eval.py                     # needs an annotation-website export under $WMV_ROOT
```

Use `--max-tokens 2000` with reasoning VLMs: the runner's default (300) can cut the answer off before the JSON label.
Judges that never accept a task on imagined frames of real successes (true for every automated judge we ran on the
book-into-back-compartment task, while human raters accept most of them) are reported as a finding, not re-tuned.

## Verdict-reliability tables and figures

`analysis/paper_repro/run_all.sh` regenerates the verdict-reliability table (per condition group, with Wilson 95%
intervals, in a 106-level and a 113-level variant where the sham levels form their own row), the scatter statistics of
plan success vs. verdict acceptance and plan vs. deployment success, and the shift-family / level overview, from the
original open-loop records and the closed-loop main table under `$WMV_ROOT` (inputs and their sha256 are listed in
`inputs.md`). Figures are drawn with matplotlib when it is installed and fall back to raster output otherwise.

## Annotation website

The human-annotation website lives in its own repository: https://github.com/Ghosty2003/wm_verdict_web
(static GitHub Pages front end + Supabase). To deploy your own copy: create a Supabase project, run
`supabase/schema.sql`, and put the project URL and the **anon / publishable** key into `config.js` (safe to expose; access is enforced by row-level security). The `service_role` key is only ever read from the
`SUPABASE_SERVICE_KEY` environment variable by the admin scripts in `tools/` and must never be committed.
`annotation_pipeline/` in this repository produces the cards and page tables that the website serves
(`$WMV_WEB_ROOT` = your local clone of the website repository).

## Third-party code (not vendored)

| Component | Source | Commit / version | Our patch |
|---|---|---|---|
| LeRobot (LingBot-VA policy, LiberoEnv) | github.com/huggingface/lerobot | e624f3f7f8411ec3a02635d06e79373341e5ef35 | none |
| OpenWAM | github.com/OpenWAM-Official/OpenWAM | 1e103e1c61905c230dab6b0f99980370cf2f9e3c | `patches/openwam-1e103e1_wan22_model_path.patch` (weights path) |
| LIBERO-plus (B-family corruptions) | github.com/sylvestf/LIBERO-plus | 4976dc30028e805ff8094b55501d532c48fec182; `env_wrapper.py` sha256 e91d7b7b35cc…f39c3c | none (functions taken verbatim at run time) |
| LIBERO | PyPI `hf-libero` | 0.1.4 | none |
| robosuite / mujoco / bddl | PyPI | 1.4.0 / 3.8.1 / 1.0.1 | `patches/robosuite-1.4.0_log_utils.patch` (log file location, optional) |
| WorldVLA (appendix) | github.com/alibaba-damo-academy/WorldVLA, branch `worldvla` | ebdc36dc5cfd; HF weights 5952d61278d5 | none |
| Model weights | HF: lerobot/lingbot_va_libero_long, robbyant/lingbot-va-base, robbyant/lingbot-va-posttrain-libero-long; OpenWAM-Alpha-Sim-LIBERO, Wan2.2-TI2V-5B | sha256 in `env/weights_sha256.txt` | — |

Runtime wrappers we install on third-party objects (no file changes): a capture hook on the LingBot-VA policy's
`predict_action_chunk` (`protocol/runners/lingbot/lb_capture.py`), per-instance `LiberoEnv.step` recorders in several
runners, and wrappers around WorldVLA's official evaluation script (`protocol/runners/worldvla/worldvla_unit.py`).

## License

Code: Apache License 2.0 (`LICENSE`, `NOTICE`). The two files in `patches/` modify third-party files and follow the
licenses of the projects they patch (robosuite: MIT; OpenWAM: see its repository). `protocol/banks/**/bddl` is derived
from LIBERO task files (MIT). Predicate weights and run data are released separately.
