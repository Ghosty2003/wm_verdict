# Agent-judge run prompt — label every frame in the package (draft 2026-11-17, for review)

Give the text below to a coding agent that has file access to the unzipped blind package.
It wraps the per-frame rubric in `prompt/judge_prompt.md`; it does not change any label rule.

---

```
You are an image judge for a robot world-model benchmark. Your job: label EVERY frame listed in
items_blind.csv in this directory, following prompt/judge_prompt.md exactly, and write the labels to
labels_<your_judge_name>.csv. Work only from the images and the text files named here.

SETUP (do once, before labeling)
1. Read README.md and prompt/judge_prompt.md in full. The system prompt, the four labels, the two conventions
   (generation distortion -> borderline; T0 book invisible while the gripper is over/in the back compartment
   -> borderline) and the per-task rules in that file are the ONLY decision rules. Do not invent new ones.
2. For each task T0-T4, look at prompt/refs/<T>_reference_*.png and prompt/refs/<T>_success_controls.png so you
   know what success and typical near-misses look like. Re-open them whenever you are unsure.
3. Do NOT open, search for, or infer anything from: any _key/ directory or key.csv, file names other than the
   frame_id, file sizes, image resolution, timestamps, or the order of rows. Frames come from several models and
   both real successes and real failures; resolution and style carry no information.

LABELING
4. Process frames grouped by task (all T0, then T1, ...), in the row order of items_blind.csv within a task.
   Open every frame image yourself at full size (frames/<frame_id>.png). Small frames (128x128) may be upscaled
   for viewing; never judge from a thumbnail sheet.
5. For each frame decide exactly one label: looks_success | borderline | not_success | uncertain, a confidence
   in [0,1], and a short note naming what decided it (e.g. "drawer front shows white gap", "pot on stove, burner
   hidden"). Use uncertain only when the key objects truly cannot be made out.
6. Calibration self-check: after every 200 frames, re-label 10 frames you already labeled (pick them at random
   from that block, without looking at your earlier answer), record both answers in
   selfcheck_<your_judge_name>.csv (frame_id, first_label, second_label), and keep your FIRST label in the main
   file. If agreement in a block falls below 8/10, stop and re-read prompt/judge_prompt.md and the reference
   images before continuing, and say so in your final report.
7. Write results incrementally: append each labeled row to labels_<your_judge_name>.csv immediately
   (header: frame_id,label,confidence,note), so the run can resume. On restart, skip frame_ids already present.
   Never label a frame twice in the main file.

OUTPUT AND CHECKS
8. When done, verify: every frame_id in items_blind.csv appears exactly once in your CSV (list any you could not
   open instead of guessing); labels are only the four allowed values; confidence is numeric.
9. Final report (plain text): number of frames labeled per task and per label, frames skipped and why, the
   self-check agreement per block, and the 5 frames you found hardest per task with one line each.

Do not compute or report optimism rates yourself; the organizers score your CSV with score/score.py against a
held-out key.
```

---

Notes for whoever runs it:
- The full package has 6314 frames. For a first pass, give the agent a stratified subset (e.g. the first N rows of
  each task) by filtering `items_blind.csv` — the prompt works the same on any subset.
- Scoring (organizers only, with the key): `python3 score/score.py labels_<name>.csv _key/key.csv`.
