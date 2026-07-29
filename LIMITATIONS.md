# Known Limitations & Failure Modes

This is an honest accounting of where this pipeline is expected to be
unreliable, incomplete, or unvalidated — the kind of thing that should sit
next to the code, not just live in someone's head. Organized by component.

## Fire detection (Keras classifier)

- **No accuracy numbers on real footage.** `evaluate.py` exists to produce
  precision/recall against held-out labeled clips, but that requires the
  trained weights and a labeled test set neither of which ship with this
  repo. Until that's run, treat the classifier's real-world accuracy as
  unknown, not "presumably good."
- **Single-frame decisions, no temporal smoothing.** Each frame is
  classified independently; there's no mechanism to require, say, 3
  consecutive fire-flagged frames before treating a detection as reliable.
  A single misclassified frame (motion blur, compression artifact, odd
  lighting) is enough to flag a whole video as `fire_detected: true`, and
  conversely a single frame of real fire that happens to score just under
  `FIRE_CONFIDENCE_THRESHOLD` contributes nothing even if the frames around
  it are consistently well above it.
- **Small/distant fire.** Downsampling to `TARGET_FPS` and resizing every
  frame to 224×224 for the classifier (`FIRE_DETECTOR.predict_fire_batch`)
  shrinks small or far-away fire regions further, likely below whatever
  visual signal the model actually learned to key on.
- **Night / low-light footage** is a plausible failure mode for both the
  classifier (different visual signature) and the classical localizer
  (HSV thresholds tuned against daylight-ish footage) — untested here.

## Fire localization (classical HSV + hot-core check)

- **Skin-tone false positives — mitigated, not eliminated.** See
  `FIRE_DETECTOR.py`'s module comment: warm-lit skin can share fire's hue
  and saturation range. A "hot core" requirement (a small near-white,
  low-saturation, high-brightness sub-region) was added to discriminate
  real flame from skin, and `evaluate_fire_localizer_synthetic.py`
  quantifies the improvement on synthetic test images. But it's tuned by
  inspection, not against real labeled footage, and an overexposed hand
  under strong light could plausibly still produce a spurious "hot" pixel
  cluster and slip through.
- **No temporal flicker analysis.** Real fire flickers; a static
  fire-colored object (an orange traffic cone, a neon sign) doesn't. This
  pipeline never looks at consecutive frames together to use that signal —
  it's the more robust fix noted in `FIRE_DETECTOR.py` but requires
  multi-frame analysis this project doesn't currently do.
- **Bounding boxes are per-frame, not tracked.** There's no notion of "this
  is the same fire region as three frames ago" — every frame's boxes are
  independent, so a flickering flame can appear as several separate,
  inconsistently-sized boxes across a short span of frames rather than one
  tracked region.

## People detection (YOLO)

- **`PEOPLE_CONFIDENCE_THRESHOLD = 0.6` is a judgment call, not a tuned
  value.** It was picked to address a specific observed problem (false
  "person detected" on frames with no one in them) without a labeled
  validation set to optimize against. Raising it further would likely trade
  away real detections of small, distant, or partially-occluded people;
  lowering it reintroduces false positives. The right value depends on
  footage characteristics `evaluate.py` is meant to help establish.
- **Only runs on fire-flagged frames.** This is a deliberate compute
  optimization (see `MAIN.run_analysis`'s docstring), but it means a
  person's presence is only ever checked in relation to frames the fire
  classifier already flagged — if the fire classifier misses a frame with
  real fire in it, people detection on that exact frame never runs either,
  even if it would have correctly found someone nearby.
- **Custom weights vs. fallback have very different reliability.** If
  `best.pt` fails to load, `PEOPLE_DETECTOR.load_people_model` silently
  falls back to generic `yolov8n.pt` (trained on 80 COCO classes rather
  than the custom single-class dataset this project was built around). The
  pipeline keeps working, but its accuracy characteristics on this specific
  use case are untested in that fallback state.
- **Proximity is measured in raw pixels, not physical distance
  (`PROXIMITY_THRESHOLD_PX`).** A person 120px from a fire box means
  something completely different at different camera distances/zoom
  levels/resolutions. There's no depth estimation or camera-calibration
  step to normalize this.

## AI-generated detection

- **Only samples `AI_SAMPLE_SIZE` (10) frames per clip**, not the whole
  video — a video that's mostly real but has a short AI-generated
  insert (or vice versa) could easily be missed or misjudged depending on
  which frames happen to land in the sample.
- **Single third-party model, no ensemble or fallback.** If
  `umm-maybe/AI-image-detector` has blind spots for a particular generator
  (a newer diffusion model style it wasn't trained to recognize, for
  instance), there's no second opinion — `ai_check_available: false` only
  covers the "model failed to load" case, not "model loaded fine but is
  wrong."
- **`AI_GENERATED_THRESHOLD = 0.7` is untuned** against this project's
  actual use case; it's the value that happened to be set, not one
  validated against a labeled real-vs-AI-generated test set.

## Pipeline / system level

- **No end-to-end evaluation exists yet.** `evaluate.py` (added in this
  pass) is the infrastructure for measuring the fire and people detectors'
  real accuracy — see its docstring for exactly what's needed to run it and
  get real numbers, and `evaluate_fire_localizer_synthetic.py` for what's
  measurable right now without real footage or weights.
- **No confidence propagation into the final verdict.** The `verdict`
  string is built from boolean flags (`fire_detected`, `people_near_fire`,
  etc), not from the underlying confidence scores — a fire frame at 0.51
  confidence and one at 0.99 confidence produce identically-worded
  verdicts, even though the response does separately expose
  `max_fire_confidence` for a caller that wants to look.
- **Single global thresholds across all input footage.** Every constant in
  `config.py` is one fixed value applied to every video regardless of
  camera, lighting, or resolution differences between clips — there's no
  per-source calibration.
