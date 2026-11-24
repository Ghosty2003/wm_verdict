# Rubric examples and reference images

Files in `refs/` (copies; the originals are unchanged):

| task | reference (real simulator frames) | success controls |
|---|---|---|
| T0 [t5] book → BACK compartment of caddy | `refs/T0_reference_t5.png` | `refs/T0_success_controls.png` |
| T1 [t3] bowl → bottom drawer, drawer closed | `refs/T1_reference_t3.png` | `refs/T1_success_controls.png` |
| T2 [t2] stove ON + moka pot on stove | `refs/T2_reference_t2.png` | `refs/T2_success_controls.png` |
| T3 [t4] white mug LEFT plate, yellow-and-white mug RIGHT plate | `refs/T3_reference_t4.png` | `refs/T3_success_controls.png` |
| T4 [t6] white mug ON plate, pudding RIGHT of plate | `refs/T4_reference_t6.png` | `refs/T4_success_controls.png` |

- **Reference images** are the same files the original agent judges and the annotation site use (`wm_verdict_web/cards/ref/`).
  They show *real* simulator frames: a start frame, a real success and real failures, each labelled in the image.
  Some show the model input layout (main view on the left, wrist camera on the right). The frames you judge are main view only.
- **Success-control sheets** (`success_controls_20261117`): the task reference on top. Below it, for each model, 2
  real-success runs, each shown as a pair: REAL last frame | IMAGINED terminal frame of the **same run**. Use them to
  see what a correct imagined success looks like in each model's rendering style, including the blur and low
  resolution you should *not* hold against a frame. The units on these sheets are not among the frames you judge.
- The original judges were told: "First Read the reference image(s) for your tasks … then the sheets", and
  "camera orientation matches the reference".

## What the labels looked like in practice (2026-11-14/15 agent judges)

These are the most frequent free-text notes the original agent judges attached, by task and label. They are
quoted as illustrations of how the rules were applied, not as extra rules.

**T0 (book → back compartment).**
borderline: "book tilted, partly in back compartment"; "book held flat atop back compartment"; "book not visible; gripper empty above caddy, maybe in back compartment".
not_success: "book tilted on/in right compartment"; "book still on table"; "book lying across right compartment top".
uncertain: "heavy blur", "fully blurred".
**Rule change since then.** The 2026-11-17 site criterion counts "book still held or hanging above the rim" and
"tilted half out" as NOT completed, where the old agent rule said borderline. It also makes "book invisible
anywhere while the gripper is over or inside the back compartment" an explicit borderline. For T0 frames, the old
agent labels and a judge using the new criterion can therefore disagree on the borderline/not_success boundary.
This does not change the main optimism rate (only looks_success counts), but it does change the wide rate.

**T1 (bowl → bottom drawer, closed).**
looks_success: "gripper pressed on drawer front, cabinet flush, no bowl visible".
borderline: "drawer looks shut, gap not discernible" (blur); "bowl-like texture at drawer front, drawer may be ajar"; "arm occludes drawer front, no bowl visible".
not_success: "drawer open, bowl in gripper"; "drawer open with bowl inside"; "drawer still pulled out".
**Since 2026-11-17:** drawer shut but the cabinet or gripper is visibly distorted (missing corner, warped front) → borderline.
This came from Jasmine's review of 17 OpenWAM T1 F160 frames: 13 of 17 were "looks closed but distorted", which
the agent had called looks_success.

**T2 (stove on + moka pot on stove).**
looks_success: "pot on lit burner"; "pot on stove, red glow under it".
borderline: "pot under gripper on stove, burner hidden"; "pot held at stove left edge, burner lit".
not_success: "pot tipped on table"; "pot tipped at stove edge"; "pot on stove, rings grey/off"; "stove off, pot on table".

**T3 (white mug LEFT plate, yellow-and-white mug RIGHT plate).**
looks_success: "both mugs partly out of frame but on plates"; "white mug cut off but on left plate".
borderline: "white mug tilted on left plate"; "white sliver" (only a sliver visible); "blurry, blobs near plates".
not_success: "white off plate"; "yellow off plate"; "white mug absent"; "white in gripper"; "left plate empty".

**T4 (white mug ON plate, pudding RIGHT of plate).**
looks_success: "pudding behind-right of plate"; "blurred but mug on plate, pudding right".
borderline: "mug on plate; pudding not visible"; "pudding at plate right edge"; "pudding behind plate, aligned".
not_success: "mug beside plate left"; "no white mug visible"; "mug tipped beside plate".

## Image-corruption frames (B family)

Frames from image-corruption runs (fog, gaussian blur, zoom blur) are often heavily degraded. The original judges
were told: "judge them the same way (use uncertain only if the key objects truly cannot be made out)". In practice
most borderline labels on these frames meant "corrupted, key object can't be confirmed". So for B-family rows the
analysis uses the **main** rate; the wide rate is reported but not used there.
