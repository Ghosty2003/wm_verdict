# Blind judge package — all imagined final frames (2026-11-24)

This package asks a judge (a person, a VLM, or a coding agent) one question per image:
**does this imagined frame show the robot task completed?**

## Contents

- `frames/` — 10,487 PNG images, named `v00001.png` … `v10487.png`. The ids are neutral and shuffled. They carry no
  information about the model, the condition, the source batch or the real outcome.
- `items_blind.csv` — one row per frame: `frame_id, task, task_title, task_criterion`. Five LIBERO-10 tasks, T0–T4.
- `prompt/judge_prompt.md` — the per-frame rubric with **four labels**: `looks_success`, `borderline`,
  `not_success`, `uncertain`. This is the rubric of the earlier judge runs; use it unless told otherwise.
- `prompt/judge_prompt_binary.md` — the same rubric with **two labels**, `success` / `not_success`, as a forced
  choice. Use it only if you are asked for binary labels.
- `prompt/refs/` — one reference image per task (real simulator frames showing the task completed), plus a
  success-controls sheet per task. Read the reference for a task before labeling its frames.
- `prompt/rubric_examples.md` — worked examples for the T0 rule.
- `prompt/run_agent_judge_all.md` — the run instructions for a coding agent that labels every frame.

## What every image is

Every image is the world model's **imagined** last frame: the final predicted camera frame of its last planning
call. It is not a real camera frame.

- Only the main (third-person) camera view is included, upright, at the model's native resolution.
- Image sizes differ between frames (128×128, 224×224, 320×256 and others). Size carries no information about the answer.
- Some frames come from image-corruption conditions and look blurred or foggy. Judge them the same way.

## What to hand back

A CSV `labels_<judge_name>.csv` with columns `frame_id,label,confidence,note`, one row per frame.

- `label` must be one of the allowed values of the rubric you used.
- `confidence` is a number from 0 to 1.
- Leave a frame out if you could not judge it. Do not guess on its behalf.

Do not compute any rates yourself. The organizers score the CSV against a held-out key, which is not part of this package.
