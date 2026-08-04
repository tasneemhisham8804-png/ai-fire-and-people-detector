# Fire, People & AI-Generated Footage Detector

FastAPI backend + Streamlit frontend that analyzes a video for:
1. **Fire** — MobileNetV2-style classifier + HSV-based localization for bounding boxes
2. **People near fire** — YOLO detector, proximity check against fire boxes
3. **AI-generated footage** — HuggingFace ViT classifier (`umm-maybe/AI-image-detector`), sampled across `AI_SAMPLE_SIZE` frames per clip

## How it works

A small, evenly-spaced sample of frames (`AI_SAMPLE_SIZE`, decoded via
direct frame seeking, not a full extraction) is run through the
AI-generated-image classifier first and averaged into one clip-level
probability. **If that probability clears `AI_GENERATED_THRESHOLD`, fire
and people detection are skipped entirely** — the video is decoded and
downsampled to `TARGET_FPS` frames, and the fire/people pipeline below
runs, only when the clip *isn't* flagged as AI-generated (or the
AI-generated check itself is unavailable). This ordering means a clip
confidently identified as synthetic never pays for the two more expensive
models at all, rather than running them and discarding the result.

When fire/people detection does run: each extracted frame runs through the
fire classifier, producing a per-frame confidence score and (on flagged
frames) fire bounding boxes via classical HSV color thresholding. Frames
flagged as containing fire are then passed to the people detector, which
finds person bounding boxes and checks their distance to the fire boxes.
All of this is combined into a single JSON verdict — see `MAIN.py`'s module
docstring for the full request flow, and each detector module's docstring
for how its piece works.

Model inference (the Keras fire classifier, the YOLO people detector, and
the HuggingFace AI-generated pipeline) is each serialized behind its own
lock, since concurrent `/analyze` requests run in separate threadpool
workers and none of these underlying model objects are documented as safe
for concurrent calls from multiple threads.

## Setup

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place these two model paths in the project root (not committed to git —
matched by `*.keras` / `*.pt` in `.gitignore`):
- `Fire_DETECTOR.keras` — matches `config.FIRE_MODEL_PATH` exactly, including
  case. This is a Keras 3 model export: if it's a **folder**, that folder
  must contain `config.json` and `model.weights.h5` directly inside it
  (`FIRE_DETECTOR.load_fire_model` looks for those two files specifically,
  and falls back to treating it as a single legacy `.keras`/`.h5` file only
  if the path isn't a directory). If your export is an older TensorFlow
  SavedModel directory (`saved_model.pb` + a `variables/` folder) instead,
  it will **not** load with this code as-is — re-export with
  `model.save("Fire_DETECTOR.keras")` under Keras 3, or adjust
  `load_fire_model` to also handle the SavedModel layout.
- `best.pt` — the custom-trained YOLO weights. If this fails to load,
  `PEOPLE_DETECTOR.py` automatically falls back to the generic pretrained
  `yolov8n.pt` rather than failing startup — check `GET /health`'s
  `people_model_using_fallback` field to see whether that happened.

**Case matters on case-sensitive filesystems** (Linux, most cloud hosts,
WSL) even though it's invisible on Windows: the file/folder name must be
`Fire_DETECTOR.keras`, matching `config.FIRE_MODEL_PATH` exactly — not
`fire_detector.keras` or any other casing.

The AI-generated check uses `umm-maybe/AI-image-detector` from the
HuggingFace hub instead of a local weights file, so it needs internet
access the first time it runs (it caches locally after that). If it can't
reach the hub, `/analyze` still works — the response just comes back with
`ai_check_available: false` instead of a real AI-generated score. A failed
load is retried automatically after `config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS`
rather than being disabled for the rest of the process's lifetime.

## Running

Backend:
```bash
py -3.12 -m uvicorn MAIN:app --reload
```

Frontend:
```bash
py -3.12 -m streamlit run UI.py --server.fileWatcherType none
```

Start the backend first — the Streamlit UI's "system online" badge does a
live `GET /health` check against it, and `/analyze` calls will fail with a
"can't reach the backend" message in the UI if the backend isn't already
running.

## Project structure

All application code, tests, and evaluation scripts currently sit flat in
the project root (there is no `tests/` subfolder):

```
ai-fire-and-people-detector/
├── config.py                          # all thresholds/paths in one place
├── MAIN.py                            # FastAPI app, /analyze and /health endpoints
├── FIRE_DETECTOR.py                   # MobileNetV2 fire classifier + HSV localization
├── PEOPLE_DETECTOR.py                 # YOLO people detector + fire-proximity check
├── AI_DETECTOR.py                     # HuggingFace ViT real-vs-AI-generated classifier
├── UI.py                              # Streamlit frontend
├── conftest.py                        # stubs heavy ML deps so pure-logic tests run fast
├── test_fire_detector.py
├── test_people_detector.py
├── test_ai_detector.py
├── test_main.py
├── evaluate.py                        # real-accuracy evaluation harness (needs labeled clips)
├── evaluate_fire_localizer_synthetic.py
├── LIMITATIONS.md
├── requirements.txt
├── .gitignore
└── Fire_DETECTOR.keras / best.pt      # not committed — see Setup above
```

> **Note:** every `test_*.py` file inserts `Path(__file__).parent.parent`
> onto `sys.path`, which assumes the test file lives one level *below* the
> project root (e.g. inside a `tests/` folder). With the current flat
> layout, that resolves to the project root's *parent* directory instead —
> if `import config` or `import PEOPLE_DETECTOR` fails when you run
> `pytest`, this path mismatch is why. Either move the `test_*.py` files
> into an actual `tests/` subfolder to match what they assume, or change
> each file's `sys.path.insert` line to use `.parent` instead of
> `.parent.parent`.

## API

`POST /analyze` — multipart upload, field name `file`. Accepts `.mp4 .mov .avi .mkv`, max 100MB (see `MAX_VIDEO_DURATION_SECONDS` in `config.py` for the separate duration cap, and `MAX_EXTRACTED_FRAMES` for a hard backstop independent of that).

The AI-generated check runs *first*. If a clip is flagged as AI-generated,
fire and people detection are skipped entirely rather than run against
footage already known to be synthetic — `fire_check_skipped` /
`people_check_skipped` are `true` in that case, and `fire_detected` /
`people_detected` are `false` because they were never checked, not because
nothing was found.

Returns (a real, non-AI-generated clip with fire and people detected):
```json
{
  "fire_detected": true,
  "fire_check_skipped": false,
  "max_fire_confidence": 0.94,
  "fire_flagged_frames": [3, 4, 5],
  "fire_confidence_timeline": [0.1, 0.2, 0.94, ...],
  "people_detected": true,
  "people_check_skipped": false,
  "people_near_fire": true,
  "peak_people_count": 2,
  "closest_person_to_fire_px": 45.2,
  "ai_generated_probability": 0.03,
  "ai_check_available": true,
  "total_frames_analyzed": 42,
  "processing_time_seconds": 8.31,
  "verdict": "Real footage — fire detected with people in proximity"
}
```

Returns (a clip flagged as AI-generated — fire/people checks skipped):
```json
{
  "fire_detected": false,
  "fire_check_skipped": true,
  "max_fire_confidence": 0.0,
  "fire_flagged_frames": [],
  "fire_confidence_timeline": [],
  "people_detected": false,
  "people_check_skipped": true,
  "people_near_fire": false,
  "peak_people_count": 0,
  "closest_person_to_fire_px": null,
  "ai_generated_probability": 0.91,
  "ai_check_available": true,
  "total_frames_analyzed": 42,
  "processing_time_seconds": 1.94,
  "verdict": "AI-generated footage — fire/people detection skipped"
}
```

If the AI-generated model fails to load, `ai_check_available` is `false` and
`ai_generated_probability` is `null` — fire/people detection runs as normal
in that case (there's no AI-generated signal to skip on), and the verdict
will say "AI-generated check unavailable" instead of silently assuming the
footage is real.

`GET /health` — reports whether each model loaded successfully, including
`people_model_using_fallback` (true if `best.pt` failed to load and the
generic `yolov8n.pt` is being used instead).

`GET /` — basic liveness check; confirms the process is up and reports
`fire_model_loaded`.

## Running tests

```bash
py -3.12 -m pytest -v
```

Tests stub out `tensorflow`/`ultralytics`/`torch`/`transformers` via
`conftest.py` so they run in under a second without needing the actual
model weights. Coverage spans two layers:
- Pure-logic / classical-CV helpers (`_box_gap`, `_localize_fire_regions`),
  plus model-loading fallback and retry behavior — `test_fire_detector.py`,
  `test_people_detector.py`, `test_ai_detector.py`
- The HTTP orchestration layer (`/analyze`, `/health` — validation, error
  handling, rate limiting, API-key auth, verdict-assembly logic) —
  `test_main.py`, with all model inference calls monkeypatched out

See the note under **Project structure** above if these imports fail to
resolve when you run pytest.

Neither layer tells you how *accurate* the trained models are on real
footage — see Evaluation below for that.

## Evaluation

- `evaluate_fire_localizer_synthetic.py` — runs now, no trained weights
  needed. Quantifies the classical-CV fire localizer's precision/recall on
  synthetic test images, and specifically measures how much the hot-core
  skin-tone mitigation (see `FIRE_DETECTOR.py`) reduces false positives
  versus the earlier color-mask-only version.
- `evaluate.py` — measures the *trained* fire and people models' real
  accuracy against a labeled CSV of video clips. Needs `Fire_DETECTOR.keras`
  / `best.pt` present and a set of human-labeled clips (format documented in
  the script's docstring) — neither ships with this repo, so this hasn't
  been run against real footage yet. Run it once you have both:
  ```bash
  python evaluate.py --manifest labeled_clips.csv
  ```

## Known limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for an honest, component-by-component
list of failure modes and unvalidated assumptions (skin-tone edge cases,
untuned thresholds, no temporal smoothing, etc). Worth reading before
trusting this on footage very different from whatever it was informally
tested against.

## Config

`FIRE_BATCH_SIZE` in `config.py` controls how many frames get batched into
one `model.predict()` call. 16 is a reasonable starting point for a GTX
1650 — raise it if you have headroom, lower it if you hit CUDA OOM errors.
See `config.py` for explanations of every other tunable value.

## Optional API protection

Both are **off by default** — no header or setup needed for local/demo use.
Set the corresponding environment variable before starting the server to
turn either on:

| Env var | Effect |
|---|---|
| `FIRE_DETECTOR_API_KEY` | If set, every `/analyze` request must include a matching `X-API-Key` header, or it's rejected with `401`. |
| `FIRE_DETECTOR_RATE_LIMIT_PER_MINUTE` | If set to a number > 0, caps `/analyze` requests per client IP to that many per rolling 60s window; excess requests get `429`. |

The rate limiter is in-memory and per-process — it resets on restart and
doesn't share state across multiple server replicas. Fine for the
single-instance deployment this project targets; would need a shared store
(e.g. Redis) behind a load balancer with more than one backend process.

## Troubleshooting

- **`/analyze` always returns 503, or the UI never shows results below the
  uploader:** check `GET /health` first — this tells you which model
  failed to load. A `fire_model_loaded: false` almost always means
  `Fire_DETECTOR.keras` either isn't present at the path `config.py`
  expects, or its internal format doesn't match what `load_fire_model`
  expects (see **Setup** above). The Streamlit UI has no results to show
  and nothing to scroll to until a request actually succeeds — a missing
  scrollbar on a short page is usually a symptom of this, not a separate
  frontend bug.
- **Backend process won't start at all / crashes immediately:** both
  models load at import time in `MAIN.py`, before the server can accept
  any requests. `load_fire_model` catches its own errors and returns
  `None`, but check your terminal output for the actual exception either
  way — a broken `tensorflow`/`ultralytics`/`torch` install (e.g. CPU-only
  wheels installed instead of the CUDA build noted in `requirements.txt`)
  is a common cause and won't necessarily show up as a clean error message.
- **UI shows "Backend unreachable":** start the backend before the
  frontend, and confirm it's actually listening on
  `http://localhost:8000` (or set `DETECTOR_API_URL` if you've changed the
  port/host).
- **`pytest` fails to import `config` / `PEOPLE_DETECTOR` / etc.:** see the
  note under **Project structure** above about the `tests/`-folder
  assumption in each test file's `sys.path.insert` line.
