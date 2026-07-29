# Fire, People & AI-Generated Footage Detector

FastAPI backend + Streamlit frontend that analyzes a video for:
1. **Fire** — MobileNetV2-style classifier + HSV-based localization for bounding boxes
2. **People near fire** — YOLO detector, proximity check against fire boxes
3. **AI-generated footage** — HuggingFace ViT classifier (`umm-maybe/AI-image-detector`), sampled across `AI_SAMPLE_SIZE` frames per clip

## How it works

A video upload is decoded and downsampled to `TARGET_FPS` frames. Each
frame runs through the fire classifier, producing a per-frame confidence
score and (on flagged frames) fire bounding boxes via classical HSV color
thresholding. Frames flagged as containing fire are then passed to the
people detector, which finds person bounding boxes and checks their
distance to the fire boxes. Separately, a small evenly-spaced sample of
frames is run through the AI-generated-image classifier and averaged into
one probability. All of this is combined into a single JSON verdict — see
`MAIN.py`'s module docstring for the full request flow, and each detector
module's docstring for how its piece works.

## Setup

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place these two model files in the project root (not committed to git — see `.gitignore`):
- `fire_detector.keras`
- `best.pt`

The AI-generated check uses `umm-maybe/AI-image-detector` from the
HuggingFace hub instead of a local weights file, so it needs internet
access the first time it runs (it caches locally after that). If it can't
reach the hub, `/analyze` still works — the response just comes back with
`ai_check_available: false` instead of a real AI-generated score.

## Running

Backend:
```bash
py -3.12 -m uvicorn MAIN:app --reload
```

Frontend:
```bash
py -3.12 -m streamlit run UI.py --server.fileWatcherType none
```

## Project structure

```
TRAINING3/
├── config.py              # all thresholds/paths in one place
├── MAIN.py                # FastAPI app, /analyze and /health endpoints
├── FIRE_DETECTOR.py        # MobileNetV2 fire classifier + HSV localization
├── PEOPLE_DETECTOR.py      # YOLO people detector + fire-proximity check
├── AI_DETECTOR.py          # HuggingFace ViT real-vs-AI-generated classifier
├── UI.py                   # Streamlit frontend
├── requirements.txt
├── tests/
│   ├── conftest.py         # stubs heavy ML deps so pure-logic tests run fast
│   ├── test_fire_detector.py
│   └── test_people_detector.py
└── fire_detector.keras / best.pt   # not committed
```

## API

`POST /analyze` — multipart upload, field name `file`. Accepts `.mp4 .mov .avi .mkv`, max 100MB.

Returns:
```json
{
  "fire_detected": true,
  "max_fire_confidence": 0.94,
  "fire_flagged_frames": [3, 4, 5],
  "fire_confidence_timeline": [0.1, 0.2, 0.94, ...],
  "people_detected": true,
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

If the AI-generated model fails to load, `ai_check_available` is `false` and
`ai_generated_probability` is `null` — the verdict will say
"AI-generated check unavailable" instead of silently assuming the footage
is real.

`GET /health` — reports whether each model loaded successfully.

## Running tests

```bash
py -3.12 -m pytest -v
```

Tests stub out `tensorflow`/`ultralytics`/`torch`/`transformers` via
`conftest.py` so they run in under a second without needing the actual
model weights. Coverage spans two layers:
- Pure-logic / classical-CV helpers (`_box_gap`, `_localize_fire_regions`) —
  `test_fire_detector.py`, `test_people_detector.py`
- The HTTP orchestration layer (`/analyze`, `/health` — validation, error
  handling, verdict-assembly logic) — `test_main.py`, with all model
  inference calls monkeypatched out

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
