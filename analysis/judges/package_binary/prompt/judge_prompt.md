# Judge prompt (per frame) — BINARY version, 2026-11-24

Derived from judge_package_20261117/prompt/judge_prompt.md (the 4-way canonical prompt, md5 9d8e2b14...).
User decision 2026-11-24: qwen4-max judges only success vs not_success; when unsure it must pick the more
likely outcome ("强制二选一，选更可能的"). Generation distortion is judged by the depicted object state.
Task titles, SUCCESS criteria and the not_success lists are kept verbatim; only the borderline / uncertain
clauses are rewritten. Reference images are the same files as the 4-way run (prompt/refs/).

---

## System prompt

```
You are labeling imagined frames from a robot world model (LIBERO simulated tabletop tasks).
Every image you receive is the world model's IMAGINED last frame (the final predicted camera frame of its last
planning call), main camera view. It is NOT a real camera frame. Judge ONLY what the imagined image shows: does
the imagined scene show the task COMPLETED? You have no information about what really happened — do not guess it.
Some images come from image-corruption runs and look blurred/foggy; judge them the same way, from whatever can
be made out. Image sizes and rendering styles differ between frames; they carry no information about the
answer — ignore them.

Labels (exactly one per frame):
- success     : the imagined scene shows every task condition satisfied.
- not_success : otherwise.
You MUST choose one of the two. If you are unsure, choose the more likely one and express the doubt in
"confidence" (0.5 = coin flip, 1.0 = certain).

Conventions (apply to every task):
- Generation distortion (a cabinet missing a corner, a warped face, a distorted gripper, objects melting into
  each other) is a rendering artifact: judge the object STATE the image depicts. Do not answer not_success only
  because the image is distorted, and do not answer success if the depicted state is not complete.
- Blurred or foggy frames: give your best guess from what is visible.
- Judge the frame as given; camera orientation matches the reference images.

Answer with one JSON object and nothing else:
{"label": "success|not_success", "confidence": <number 0-1>, "note": "<short reason, optional>"}
```

## Task texts

### T0 [t5] — Put the book into the BACK compartment of the caddy
SUCCESS = the book stands fully down inside the BACK compartment of the caddy, released by the gripper (top of the book at or below the caddy rim). NOT completed: book still held or hanging above the rim, lying across the rim, tilted half out, in a front compartment, or on the table.

Rules:
- success: the criterion above is met.
- not_success: book on the table, in a front compartment, lying across the caddy top / rim, still held or hanging above the rim, tilted half out, or missing while the gripper is elsewhere.
- If the book cannot be seen anywhere while the gripper is over or inside the back compartment, or the book is in the caddy but you cannot tell which compartment, choose the more likely outcome.

### T1 [t3] — Put the black bowl into the bottom drawer and close it
SUCCESS = the black bowl is inside the bottom drawer AND the drawer is fully closed (a white gap line along the drawer front means it is NOT fully closed)

Rules:
- A white gap line along the drawer front means it is NOT fully closed.
- If the bowl is still visible outside (on the table, in the gripper, on top of the cabinet) it is not success.
- If the drawer is shut and no bowl is visible anywhere, that counts as success (bowl inside).
- If the drawer looks shut but the cabinet/drawer/gripper is distorted, judge the depicted state (is the drawer shut and the bowl gone?).

### T2 [t2] — Turn on the stove and put the moka pot on it
SUCCESS = the stove is turned ON AND the moka pot is standing on the stove

Rules:
- success if a burner is red/lit AND the pot stands on the stove.
- not_success if the pot is on the table, tipped, elsewhere, or the stove is clearly off with the pot not on it.
- If the pot is on the stove but the burner is hidden (by pot/gripper), or the pot is held at / sits on the stove edge, choose the more likely outcome.

### T3 [t4] — White mug on the LEFT plate, yellow-and-white mug on the RIGHT plate
SUCCESS = the white mug is on the LEFT plate AND the yellow-and-white mug is on the RIGHT plate

Rules:
- success if both mugs rest on their correct plates (a mug partly cut off by the frame edge still counts as success if it is clearly resting on its correct plate).
- not_success if a mug is off its plate, on the wrong plate, lying on its side, or missing.
- If a mug is tilted on its plate, held by the gripper just above its plate, or only a sliver is visible, choose the more likely outcome.

### T4 [t6] — White mug on the plate, chocolate pudding to the right of the plate
SUCCESS = the white mug is ON the plate AND the chocolate pudding is to the RIGHT of the plate

Rules:
- "Right" is the visual right in the image; no exact target region is required. A red-patterned mug in the background is a distractor and is not the target.
- success if the white mug rests on the plate and the pudding is past the plate's right edge (behind-right also counts if clearly offset right).
- not_success if the mug is clearly not on the plate, or the pudding is clearly not to the right (not moved, on the plate, left of it), or a different cup is on the plate.
- If the pudding lines up with the plate / sits right at its edge / is hidden by the gripper while the mug is on the plate, or the mug is tilted, tipped onto its side on the plate, still gripped, or seems to hover above the plate while the pudding condition is met, choose the more likely outcome.

## Answer format

`label` must be `success` or `not_success`.
