<div align="center">

# 🔥 Fire, People & AI-Generated Footage Detector

**A three-stage computer vision pipeline that analyzes uploaded video for fire, people near that fire, and whether the footage itself is AI-generated.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/Fire%20Model-TensorFlow%2FKeras-FF6F00.svg)](https://www.tensorflow.org/)
[![YOLO](https://img.shields.io/badge/People%20Model-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://docs.pytest.org/)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Model Weights](#model-weights)
   - [Running the App](#running-the-app)
6. [Project Structure](#project-structure)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Evaluation](#evaluation)
10. [Configuration](#configuration)
11. [Optional API Protection](#optional-api-protection)
12. [Known Limitations](#known-limitations)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Overview

Reviewing raw security or surveillance footage for fire hazards is slow, and it's becoming harder to trust that footage is even genuine. This project automates that first pass: upload a video clip, and the system reports back — in plain language, backed by structured data — whether it contains fire, whether anyone is standing near that fire, and whether the clip itself shows signs of being AI-generated rather than real camera footage.

It's built as two cooperating services:

| Component | Role |
|---|---|
| **Backend** (`MAIN.py`) | A FastAPI service that receives an uploaded video, runs it through the detection pipeline, and returns a structured JSON verdict. |
| **Frontend** (`UI.py`) | A Streamlit single-page app that lets a user upload a clip and view the results as an interactive dashboard — verdict card, confidence chart, and flagged frames. |

## Key Features

- 🔥 **Fire detection** — a trained MobileNetV2-style Keras classifier scores every sampled frame, with classical HSV color analysis localizing *where* the fire is once the classifier confirms it's present.
- 🧍 **People-near-fire proximity check** — a YOLOv8 detector finds people on fire-flagged frames and measures pixel distance to the nearest fire region.
- 🤖 **AI-generated footage detection** — a HuggingFace Vision Transformer classifier flags synthetic footage *before* the more expensive fire/people models run, skipping them entirely on clips already known to be fake.
- ⚡ **Cost-aware pipeline ordering** — the AI-generated check runs first, on a small sampled subset of frames, specifically so a flagged clip never pays for a full video decode or the fire/people models at all.
- 🛡️ **Defense-in-depth upload validation** — extension allow-listing, streamed size limits, video-duration caps, and file-signature ("magic byte") sniffing so a mislabeled or oversized file is rejected cheaply, before any real processing begins.
- 🔑 **Optional auth & rate limiting** — API-key and per-IP rate limiting are available and fully opt-in via environment variables, with zero effect on local/demo use.
- ✅ **Tested and documented** — a `pytest` suite covering both pure logic and full HTTP request/response behavior, plus a separate evaluation harness for measuring real-world model accuracy.

## How It Works

```
                         ┌────────────────────────┐
   Video upload  ─────▶  │   1. Upload Validation │  reject bad extension / oversize / spoofed content
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │  2. Duration Guard      │  reject clips over MAX_VIDEO_DURATION_SECONDS
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │ 3. Sample & Check for   │  small, evenly-spaced sample of frames
                         │    AI-Generated Content │  → HuggingFace ViT classifier
                         └───────────┬────────────┘
                          flagged ───┴─── not flagged
                             │                │
                             ▼                ▼
                    ┌─────────────┐  ┌──────────────────────┐
                    │ Skip fire / │  │ 4. Full Frame Extract │  downsample to TARGET_FPS
                    │ people —    │  └───────────┬──────────┘
                    │ stop here   │              ▼
                    └──────┬──────┘  ┌──────────────────────┐
                           │         │ 5. Fire Detection      │  Keras classifier + HSV localization
                           │         └───────────┬──────────┘
                           │                     ▼
                           │         ┌──────────────────────┐
                           │         │ 6. People-Proximity   │  YOLOv8, only on fire-flagged frames
                           │         └───────────┬──────────┘
                           └─────────────────────┤
                                                  ▼
                                     ┌──────────────────────┐
                                     │  7. Verdict Assembly  │  → JSON response
                                     └──────────────────────┘
```

Model inference (Keras, YOLO, and the HuggingFace pipeline) is each serialized behind its own lock, since concurrent `/analyze` requests run in separate threadpool workers and none of the underlying model objects are documented as safe for concurrent calls from multiple threads.

For the full request flow and design rationale behind each step, see `MAIN.py`'s module docstring and each detector module's own docstring — every non-obvious decision in this codebase is documented at the point it's made.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn (ASGI) |
| Frontend | Streamlit |
| Fire classification | TensorFlow / Keras 3 (MobileNetV2-style CNN) |
| Fire localization | OpenCV — HSV color thresholding + contour detection |
| People detection | Ultralytics YOLOv8 |
| AI-generated detection | HuggingFace Transformers — ViT image-classification pipeline (`umm-maybe/AI-image-detector`) |
| Testing | pytest, FastAPI `TestClient` (httpx) |

## Getting Started

### Prerequisites

- Python 3.12
- ~2 GB free disk space for model weights and dependencies
- (Optional but recommended) an NVIDIA GPU with CUDA 12.1 for reasonable inference speed — the project runs on CPU too, just slower

### Installation

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> `torch`/`torchvision` in `requirements.txt` are the generic CPU-capable PyPI wheels. If `torch.cuda.is_available()` returns `False` after installing and you have a CUDA-capable GPU, reinstall with:
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
> ```

### Model Weights

Two trained model files are required locally but are **not committed to version control** (see `.gitignore`, which matches `*.keras` / `*.pt`) since binary weights don't belong in plain git history:

| File | Notes |
|---|---|
| `Fire_DETECTOR.keras` | Must match `config.FIRE_MODEL_PATH` exactly, **including case** — this matters on case-sensitive filesystems (Linux, most cloud hosts, WSL) even though it's invisible on Windows. This is a Keras 3 export: if it's a *folder*, it must directly contain `config.json` and `model.weights.h5`. Older TensorFlow SavedModel exports (`saved_model.pb` + `variables/`) are **not** supported as-is — re-export under Keras 3, or extend `load_fire_model` to handle that layout. |
| `best.pt` | Custom-trained YOLOv8 weights. If loading fails, `PEOPLE_DETECTOR.py` automatically falls back to the generic pretrained `yolov8n.pt` rather than failing startup — check `GET /health`'s `people_model_using_fallback` field to see whether that happened. |

The AI-generated-content model instead downloads from the HuggingFace Hub on first run (and caches locally afterward), so it needs internet access once. If the Hub is unreachable, `/analyze` still works — the response comes back with `ai_check_available: false` instead of failing, and a failed load is retried automatically after `config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS`.

### Running the App

Start the **backend** first:
```bash
py -3.12 -m uvicorn MAIN:app --reload
```

Then the **frontend**, in a separate terminal:
```bash
py -3.12 -m streamlit run UI.py --server.fileWatcherType none
```

The Streamlit UI performs a live `GET /health` check against the backend and will show "Backend unreachable" if the backend isn't already running — start it first.

## Project Structure

All application code, tests, and evaluation scripts currently sit flat in the project root:

```
ai-fire-and-people-detector/
├── config.py                          # all thresholds/paths, in one place
├── MAIN.py                            # FastAPI app — /analyze and /health endpoints
├── FIRE_DETECTOR.py                   # Keras fire classifier + HSV localization
├── PEOPLE_DETECTOR.py                 # YOLOv8 people detector + fire-proximity check
├── AI_DETECTOR.py                     # HuggingFace ViT real-vs-AI-generated classifier
├── UI.py                              # Streamlit frontend
├── conftest.py                        # stubs heavy ML deps so tests run fast
├── test_fire_detector.py
├── test_people_detector.py
├── test_ai_detector.py
├── test_main.py
├── evaluate.py                        # real-accuracy evaluation harness (needs labeled clips)
├── evaluate_fire_localizer_synthetic.py
├── LIMITATIONS.md                     # honest, component-by-component failure-mode log
├── requirements.txt
├── .gitignore
└── Fire_DETECTOR.keras / best.pt      # not committed — see Model Weights above
```

> **Note on test imports:** every `test_*.py` file adds `Path(__file__).parent.parent` to `sys.path`, which assumes the test file lives one level *below* the project root (e.g. in a `tests/` subfolder). With the current flat layout, that path resolves one directory too high. If `import config` or a similar import fails when running `pytest`, either move the `test_*.py` files into an actual `tests/` subfolder, or change each file's `sys.path.insert` line to use `.parent` instead of `.parent.parent`.

## API Reference

### `POST /analyze`

Multipart upload, field name `file`. Accepts `.mp4`, `.mov`, `.avi`, `.mkv`, up to `MAX_UPLOAD_MB` (default 100 MB). A separate `MAX_VIDEO_DURATION_SECONDS` cap (default 300 s) and `MAX_EXTRACTED_FRAMES` hard backstop apply independently of file size.

The AI-generated check runs **first**. If a clip is flagged, fire and people detection are skipped entirely — `fire_check_skipped` / `people_check_skipped` are `true`, and `fire_detected` / `people_detected` are `false` because they were never evaluated, not because nothing was found.

<details>
<summary><strong>Example response — real footage, fire and people detected</strong></summary>

```json
{
  "fire_detected": true,
  "fire_check_skipped": false,
  "max_fire_confidence": 0.94,
  "fire_flagged_frames": [3, 4, 5],
  "fire_confidence_timeline": [0.1, 0.2, 0.94, "..."],
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
</details>

<details>
<summary><strong>Example response — flagged as AI-generated (fire/people skipped)</strong></summary>

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
</details>

If the AI-generated model fails to load, `ai_check_available` is `false` and `ai_generated_probability` is `null` — fire/people detection then runs as normal (there's no signal to skip on), and the verdict notes "AI-generated check unavailable" instead of silently assuming the footage is real.

### `GET /health`

Reports whether each model loaded successfully, including `people_model_using_fallback` — `true` if `best.pt` failed to load and the generic `yolov8n.pt` is being used instead.

### `GET /`

Basic liveness check; confirms the process is up and reports `fire_model_loaded`.

## Testing

```bash
py -3.12 -m pytest -v
```

`conftest.py` stubs out `tensorflow`, `ultralytics`, `torch`, and `transformers`, so the full suite runs in under a second without the actual model weights present. Coverage spans two layers:

- **Pure logic / classical CV** — box-proximity geometry, HSV fire localization, and model-loading fallback/retry behavior → `test_fire_detector.py`, `test_people_detector.py`, `test_ai_detector.py`
- **HTTP orchestration** — `/analyze` and `/health` request validation, error handling, rate limiting, API-key auth, and verdict-assembly logic, with all model inference monkeypatched out → `test_main.py`

See the note under [Project Structure](#project-structure) if imports fail when running `pytest`. Neither layer measures how *accurate* the trained models are on real footage — see [Evaluation](#evaluation) for that.

## Evaluation

| Script | Purpose | Requirements |
|---|---|---|
| `evaluate_fire_localizer_synthetic.py` | Measures the classical-CV fire localizer's precision/recall on synthetic test images, and quantifies how much the hot-core skin-tone mitigation reduces false positives. | None — runs immediately. |
| `evaluate.py` | Measures the *trained* fire and people models' real accuracy against a labeled CSV of video clips. | `Fire_DETECTOR.keras` / `best.pt`, plus a labeled clip manifest (format documented in the script's docstring). Neither ships with this repo. |

```bash
python evaluate.py --manifest labeled_clips.csv
```

## Configuration

Every tunable value — model paths, confidence thresholds, batch size, proximity threshold, upload/duration limits, target FPS, allowed extensions — lives in `config.py`, imported by every module (backend and frontend alike) so behavior can be retuned in one place without touching model code or risking the two sides drifting out of sync.

`FIRE_BATCH_SIZE` is worth calling out specifically: it controls how many frames get batched into one `model.predict()` call. 16 is a reasonable starting point for a 4 GB GPU (e.g. a GTX 1650) — raise it if you have headroom, lower it if you hit CUDA out-of-memory errors.

## Optional API Protection

Both are **off by default** — no header or setup needed for local/demo use. Set the corresponding environment variable before starting the server to enable either:

| Env var | Effect |
|---|---|
| `FIRE_DETECTOR_API_KEY` | If set, every `/analyze` request must include a matching `X-API-Key` header, or it's rejected with `401`. |
| `FIRE_DETECTOR_RATE_LIMIT_PER_MINUTE` | If set to a number > 0, caps `/analyze` requests per client IP to that many per rolling 60-second window; excess requests get `429`. |

The rate limiter is in-memory and per-process — it resets on restart and doesn't share state across multiple server replicas. This is sufficient for the single-instance deployment this project targets; a shared store (e.g. Redis) would be needed behind a load balancer fronting more than one backend process.

## Known Limitations

This project maintains an honest, component-by-component account of failure modes and unvalidated assumptions in [`LIMITATIONS.md`](LIMITATIONS.md) — skin-tone edge cases in fire localization, untuned thresholds, no temporal smoothing across frames, and more. Worth reading before trusting results on footage very different from whatever the system was informally tested against.

## Troubleshooting

<details>
<summary><strong>`/analyze` always returns 503, or the UI never shows results below the uploader</strong></summary>

Check `GET /health` first — it reports which model failed to load. `fire_model_loaded: false` almost always means `Fire_DETECTOR.keras` either isn't present at the path `config.py` expects, or its internal format doesn't match what `load_fire_model` expects (see [Model Weights](#model-weights)). The Streamlit UI has no results to show — and nothing to scroll to — until a request actually succeeds; an apparently missing scrollbar on a short page is usually a symptom of this, not a separate frontend bug.
</details>

<details>
<summary><strong>Backend process won't start at all / crashes immediately</strong></summary>

Both models load at import time in `MAIN.py`, before the server can accept any requests. `load_fire_model` catches its own errors and returns `None` rather than crashing, but check your terminal output for the underlying exception either way — a broken `tensorflow`/`ultralytics`/`torch` install (e.g. CPU-only wheels installed instead of the CUDA build noted in `requirements.txt`) is a common cause and won't always surface as a clean error message.
</details>

<details>
<summary><strong>UI shows "Backend unreachable"</strong></summary>

Start the backend before the frontend, and confirm it's actually listening on `http://localhost:8000` (or set `DETECTOR_API_URL` if you've changed the host/port).
</details>

<details>
<summary><strong>`pytest` fails to import `config` / `PEOPLE_DETECTOR` / etc.</strong></summary>

See the note under [Project Structure](#project-structure) about the `tests/`-folder assumption baked into each test file's `sys.path.insert` line.
</details>

## License

This project is provided for educational purposes. Add a license file appropriate to your institution's or instructor's requirements before distributing or open-sourcing it.
