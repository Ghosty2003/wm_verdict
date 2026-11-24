# Judge prompt (per frame) — canonical version, 2026-11-17

## Where this text comes from (read this first)

The existing "agent judgments" (agent 代判, 2026-11-14/15) were produced by 17 agent judge batches.
Each batch got one sheet-style prompt: 12 frames tiled on one image (labelled #1..#12), one letter per frame
(S/B/N/U), recorded with a small script. Those 17 prompts were recovered verbatim from the agent transcripts.
They are kept with the answer key (`_key/original_prompts/`), not in this blind package, because they name the
model behind each batch.

No per-frame prompt was ever stored, because the original judging was not per frame. **The prompt below is
a canonical rewrite made on 2026-11-17.** It keeps the original preamble, label definitions and per-task
decision rules word for word wherever possible. It changes three things:

1. It is per frame (one image, one answer) instead of 12-panel sheets.
2. The task criteria are the annotation-site `TASK_INFO` texts, copied verbatim (including the new detailed T0 criterion).
3. It adds the two conventions decided on 2026-11-17 (generation distortion → borderline; T0 book invisible while the gripper is over or in the back compartment → borderline).

Not every original batch saw the same detail level. The OpenWAM/LingBot batches (2026-11-14) got shorter task
rules. The explicit S/B/N rules below for T2, T3 and T4 come from the Cosmos batches (2026-11-14/15), which
described them as "identical for all models". The T0 rule changed after the agent labels were made; see
`rubric_examples.md`.

---

## System prompt

```
You are labeling imagined frames from a robot world model (LIBERO simulated tabletop tasks).
Every image you receive is the world model's IMAGINED last frame (the final predicted camera frame of its last
planning call), main camera view. It is NOT a real camera frame. Judge ONLY what the imagined image shows: does
the imagined scene show the task COMPLETED? You have no information about what really happened — do not guess it.
Some images come from image-corruption runs and look blurred/foggy; judge them the same way (use "uncertain"
only if the key objects truly cannot be made out). Image sizes and rendering styles differ between frames;
they carry no information about the answer — ignore them.

Labels (exactly one per frame):
- looks_success : all task conditions clearly appear satisfied.
- borderline    : nearly / partly, cannot decide between success and not.
- not_success   : clearly not satisfied.
- uncertain     : cannot judge — image garbled/blurred/fogged so the key objects can't be seen.

Conventions (apply to every task):
- If the task state looks complete but the image has visible generation distortion (e.g. a cabinet missing a
  corner, a deformed cabinet face, a distorted gripper, objects melting into each other), answer borderline —
  not looks_success.
- Judge the frame as given; camera orientation matches the reference images.

Answer with one JSON object and nothing else:
{"label": "looks_success|borderline|not_success|uncertain", "confidence": <number 0-1>, "note": "<short reason, optional>"}
```

## Per-frame user message (template)

```
Task {task}: {task_title}
SUCCESS = {task_criterion}
Rules for this task:
{task_rules}
[reference image(s): prompt/refs/{task}_reference.png  (optional: prompt/refs/{task}_success_controls.png)]
[the frame: frames/{frame_id}.png]
Label this imagined frame.
```

`{task}`, `{task_title}` and `{task_criterion}` are the columns of `items_blind.csv`. `{task_rules}` is the block below
for that task. Attaching the reference image is recommended: the original judges were told to read it first.

## Task texts

Title and SUCCESS criterion are verbatim from the annotation site (`wm_verdict_web/index.html`, `TASK_INFO`; site task id in brackets).
The rules are verbatim from the original judge prompts unless marked *(2026-11-17)*.

### T0 [t5] — Put the book into the BACK compartment of the caddy
SUCCESS = the book stands fully down inside the BACK compartment of the caddy, released by the gripper (top of the book at or below the caddy rim). NOT completed: book still held or hanging above the rim, lying across the rim, tilted half out, in a front compartment, or on the table. If the book cannot be seen anywhere (not on the table, not in the gripper, not in a front compartment) while the gripper is over or inside the back compartment, answer borderline

Rules:
- looks_success: the criterion above is clearly met.
- borderline: the book cannot be seen anywhere while the gripper is over or inside the back compartment *(2026-11-17)*; the book is in the caddy but you cannot tell which compartment; the task state is complete but the image shows generation distortion *(2026-11-17)*.
- not_success: book on the table, in a front compartment, lying across the caddy top / rim, still held or hanging above the rim, tilted half out, or missing while the gripper is elsewhere. (The original 2026-11-15 rule said "or missing". The 2026-11-17 convention narrows that: missing *and* the gripper over or in the back compartment → borderline.)
- uncertain: image garbled/blurred so the key objects can't be seen.

### T1 [t3] — Put the black bowl into the bottom drawer and close it
SUCCESS = the black bowl is inside the bottom drawer AND the drawer is fully closed (a white gap line along the drawer front means it is NOT fully closed)

Rules:
- A white gap line along the drawer front means it is NOT fully closed.
- If the bowl is still visible outside (on the table, in the gripper, on top of the cabinet) it is not success.
- If the drawer is shut and no bowl is visible anywhere, that counts as success (bowl inside).
- Drawer looks shut but the cabinet/drawer/gripper is visibly distorted (missing corner, warped front) → borderline *(2026-11-17)*.

### T2 [t2] — Turn on the stove and put the moka pot on it
SUCCESS = the stove is turned ON AND the moka pot is standing on the stove

Rules:
- looks_success only if a burner is visibly red/lit AND the pot stands on the stove.
- borderline if the pot is on the stove but the burner is hidden (by pot/gripper) so you cannot see whether it is lit, or the pot is held at / sits on the stove edge.
- not_success if the pot is on the table, tipped, elsewhere, or the stove is clearly off with the pot not on it.
- uncertain only when nothing relevant can be made out.

### T3 [t4] — White mug on the LEFT plate, yellow-and-white mug on the RIGHT plate
SUCCESS = the white mug is on the LEFT plate AND the yellow-and-white mug is on the RIGHT plate

Rules:
- looks_success if both mugs rest on their correct plates (a mug partly cut off by the frame edge still counts as success if it is clearly resting on its correct plate).
- borderline if a mug is tilted on its plate, held by the gripper just above its plate, or only a sliver is visible so you cannot tell whether it is on the plate.
- not_success if a mug is off its plate, on the wrong plate, lying on its side, or missing.
- uncertain only when nothing relevant can be made out.

### T4 [t6] — White mug on the plate, chocolate pudding to the right of the plate
SUCCESS = the white mug is ON the plate AND the chocolate pudding is to the RIGHT of the plate

Rules:
- "Right" is the visual right in the image; no exact target region is required. A red-patterned mug in the background is a distractor and is not the target.
- looks_success if the white mug rests on the plate and the pudding is past the plate's right edge (behind-right also counts if clearly offset right).
- borderline if the pudding lines up with the plate / sits right at its edge / is hidden by the gripper while the mug is clearly on the plate; borderline if the mug is tilted, tipped onto its side on the plate, still gripped, or seems to hover above the plate while the pudding condition is met.
- not_success if the mug is clearly not on the plate, or the pudding is clearly not to the right (not moved, on the plate, left of it), or a different cup is on the plate.
- uncertain only when nothing relevant can be made out.

## Answer format for submission

Collect one row per frame into a CSV (see README): `frame_id,label,confidence,note`.
`label` must be one of `looks_success`, `borderline`, `not_success`, `uncertain`. Leave a frame out rather than
guessing if your judge failed on it. A missing frame counts as "not judged", not as uncertain.
