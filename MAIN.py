"""
FastAPI backend for the fire / people / AI-generated footage detector.

This is the orchestration layer: it doesn't contain any ML logic itself,
it wires together the three detection modules (FIRE_DETECTOR,
PEOPLE_DETECTOR, AI_DETECTOR) around one HTTP endpoint.

Request flow for POST /analyze:
  1. Validate the upload's extension and enforce the size limit while
     streaming it to a temp file.
  2. Sniff the file's magic bytes to confirm it's really a video of the
     claimed type (not just correctly-named).
  3. extract_frames() decodes the video and samples it down to
     config.TARGET_FPS, saving each sampled frame as a JPEG.
  4. run_analysis() runs the three models over those frames and combines
     the results into one verdict — this is the part offloaded to a
     threadpool (see run_analysis's docstring for why).
  5. The JSON result (scores, boxes, timeline, verdict) is returned to the
     client — in this project's case, UI.py's Streamlit frontend.

Model loading happens once at process startup (module level, below), not
per-request — loading a Keras/YOLO model from disk takes real time, and
every request needs to hit the same warmed-up model in memory rather than
reloading it from scratch.
"""
import logging
import tempfile
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import config
from AI_DETECTOR import predict_ai_generated
from FIRE_DETECTOR import ModelLoadError, load_fire_model, predict_fire_batch
from PEOPLE_DETECTOR import load_people_model, predict_people_near_fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="Fire, People & AI Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo-only: relax for grading/local use, tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are loaded exactly once here, at import time, and the resulting
# objects are reused for the lifetime of the process across every request.
# fire_model/people_model being None (rather than raising) signals a failed
# load, which /health and /analyze both check for before doing any work.
fire_model = load_fire_model(config.FIRE_MODEL_PATH)
people_model = load_people_model(config.PEOPLE_MODEL_PATH)


def _sniff_video_content(video_path: Path) -> bool:
    """
    Confirms a saved upload's actual byte content matches a known video
    container format, as a check on top of (not instead of) the file
    extension check in analyze_video(). The extension alone is just
    metadata the client provided and can't be trusted — this looks at the
    first bytes of the file itself, which is much harder to spoof by
    accident (e.g. a plain text file simply renamed to .mp4).
    """
    try:
        with video_path.open("rb") as f:
            header = f.read(12)
    except OSError:
        return False

    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return True
    if header[4:8] == b"ftyp":  # covers .mp4 and .mov (ISO base media format)
        return True
    if header[:4] == b"\x1a\x45\xdf\xa3":  # Matroska/.mkv EBML header
        return True
    return False


def extract_frames(video_path: Path) -> list[Path]:
    """
    Decodes the video with OpenCV and writes out a downsampled subset of
    frames as JPEGs (in the same temp directory as the source video), one
    per config.TARGET_FPS "tick" of the original video's timeline.

    The source video's own FPS is read from its metadata and used to work
    out a stride (sample_rate) so a video runs through the pipeline at
    roughly config.TARGET_FPS regardless of its native frame rate — a
    60fps clip and a 24fps clip of the same real-world duration end up
    producing a similar number of analyzed frames. If the reported source
    FPS looks bogus (<=0 or absurdly high — corrupt metadata, an unusual
    container), 30fps is assumed as a safe default instead of dividing by a
    broken number.
    """
    frame_paths = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frame_paths

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or source_fps > 240:
        source_fps = 30.0
    sample_rate = max(1, int(round(source_fps / config.TARGET_FPS)))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frame_path = video_path.parent / f"frame_{saved_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_count += 1
        frame_count += 1

    cap.release()
    return frame_paths


def run_analysis(video_path: Path) -> dict:
    """
    Runs the full detection pipeline on one video and returns the combined
    result dict that becomes the /analyze response body.

    This is a plain sync function, called via run_in_threadpool from the
    async endpoint below, specifically because everything in it (frame
    decoding, Keras inference, YOLO inference, HF pipeline inference) is
    blocking, CPU/GPU-bound work. Running that directly inside an `async
    def` endpoint would block the whole event loop for the entire duration
    of one video's analysis — meaning the server couldn't accept or make
    progress on any other request, including a simple GET /health check,
    while one video was being processed. Offloading it to a threadpool
    keeps the event loop free.

    Stages, in order:
      - extract_frames(): decode + downsample the video (see above)
      - predict_fire_batch(): fire confidence + boxes per frame -> also
        builds the confidence timeline the UI charts
      - predict_people_near_fire(): only run on frames fire already
        flagged, since a "person near fire" check is meaningless on a frame
        with no fire in it
      - predict_ai_generated(): run on a small evenly-spaced sample of
        frames, then averaged into one clip-level AI-generated probability
      - a plain-English verdict string is assembled from all of the above
    """
    start = time.monotonic()

    frames = extract_frames(video_path)
    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")
    logger.info("Extracted %d frames from %s, starting inference", len(frames), video_path.name)

    # Fire detection + per-frame timeline (also powers the UI's confidence chart)
    fire_results = predict_fire_batch([str(f) for f in frames], fire_model)
    fire_scores = [r["fire_confidence"] for r in fire_results]
    fire_flagged = [idx for idx, s in enumerate(fire_scores) if s > config.FIRE_CONFIDENCE_THRESHOLD]
    max_fire_conf = round(max(fire_scores), 4) if fire_scores else 0.0
    fire_timeline = [round(s, 4) for s in fire_scores]

    # People detection + proximity, run only on fire-flagged frames — see
    # run_analysis docstring for why this is scoped down instead of running
    # on every frame.
    people_detected = False
    people_near_fire = False
    max_people_count = 0
    closest_proximity_px = None

    for idx in fire_flagged:
        fire_boxes = fire_results[idx].get("fire_boxes", [])
        result = predict_people_near_fire(str(frames[idx]), fire_boxes)
        if result["people_detected"]:
            people_detected = True
            max_people_count = max(max_people_count, result["people_count"])
        if result["people_near_fire"]:
            people_near_fire = True
        if result["closest_person_to_fire_px"] is not None:
            if closest_proximity_px is None or result["closest_person_to_fire_px"] < closest_proximity_px:
                closest_proximity_px = result["closest_person_to_fire_px"]

    # AI-generated detection: sampled, evenly spaced across the whole clip
    # (not just the first N frames), so the sample is representative of the
    # video as a whole rather than biased toward its opening seconds.
    ai_sample = frames[:: max(1, len(frames) // config.AI_SAMPLE_SIZE)][:config.AI_SAMPLE_SIZE]
    ai_scores = [predict_ai_generated(str(f)) for f in ai_sample]
    valid_ai_scores = [s for s in ai_scores if s is not None]
    ai_check_available = len(valid_ai_scores) > 0
    avg_ai_score = round(sum(valid_ai_scores) / len(valid_ai_scores), 4) if ai_check_available else None

    if not ai_check_available:
        logger.warning("AI-generated check unavailable for %s — ai_detector model not loaded", video_path.name)

    # Verdict: fire + people-proximity status is decided first (it's the
    # more "objective" physical read of the footage), then the AI-generated
    # check can override it entirely, since synthetic footage makes the
    # fire/people findings themselves untrustworthy regardless of what they
    # say.
    fire_detected = len(fire_flagged) > 0
    if fire_detected and people_near_fire:
        real_footage_verdict = "Real footage — fire detected with people in proximity"
    elif fire_detected and people_detected:
        real_footage_verdict = "Real footage — fire detected with people present"
    elif fire_detected:
        real_footage_verdict = "Real footage — fire detected, no people present"
    else:
        real_footage_verdict = "Real footage — no fire detected"

    if not ai_check_available:
        verdict = f"{real_footage_verdict} (AI-generated check unavailable)"
    elif avg_ai_score > config.AI_GENERATED_THRESHOLD:
        verdict = "AI-generated footage"
    else:
        verdict = real_footage_verdict

    elapsed = round(time.monotonic() - start, 2)
    logger.info("Analyzed %d frames in %.2fs — verdict: %s", len(frames), elapsed, verdict)

    return {
        "fire_detected": fire_detected,
        "max_fire_confidence": max_fire_conf,
        "fire_flagged_frames": fire_flagged,
        "fire_confidence_timeline": fire_timeline,
        "people_detected": people_detected,
        "people_near_fire": people_near_fire,
        "peak_people_count": max_people_count,
        "closest_person_to_fire_px": closest_proximity_px,
        "ai_generated_probability": avg_ai_score,
        "ai_check_available": ai_check_available,
        "total_frames_analyzed": len(frames),
        "processing_time_seconds": elapsed,
        "verdict": verdict,
    }


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts a multipart video upload (field name "file"), validates it, runs
    the full detection pipeline, and returns the combined JSON result
    described in run_analysis()'s docstring.

    Validation happens in layers, cheapest checks first, so an obviously bad
    request is rejected before any expensive work: extension check -> model-
    loaded check -> streamed size-limit check while saving -> magic-byte
    content check -> only then does the actual (slow) analysis run.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    if fire_model is None:
        raise HTTPException(
            status_code=503,
            detail="Fire detection model is not loaded on the server. Check server logs.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Using only the base filename (.name) — stripping any directory
        # components the client might have sent — keeps the saved file
        # confined to temp_path, since the path is built by joining
        # temp_path with a client-supplied string.
        video_path = temp_path / Path(file.filename).name

        # Written in chunks (not read into memory all at once) so an upload
        # can be rejected for exceeding MAX_UPLOAD_MB partway through,
        # without ever having buffered the whole oversized file in memory
        # first.
        bytes_written = 0
        max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
        with video_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File too large. Max is {config.MAX_UPLOAD_MB} MB.")
                buffer.write(chunk)

        if not _sniff_video_content(video_path):
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{ext}' doesn't match actual file content — is this really a video file?",
            )

        try:
            return await run_in_threadpool(run_analysis, video_path)
        except ModelLoadError as e:
            logger.error("Model error during analysis: %s", e)
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
def root():
    """Basic liveness endpoint — confirms the API process is up and reports whether the fire model loaded."""
    return {
        "status": "Fire & AI Detection API is running",
        "fire_model_loaded": fire_model is not None,
    }


@app.get("/health")
def health():
    """
    Reports per-model load status so the frontend (UI.py) can show an
    accurate "system online" indicator instead of a static badge that
    doesn't reflect whether the backend can actually serve a real request.
    """
    return {
        "fire_model_loaded": fire_model is not None,
        "people_model_loaded": people_model is not None,
    }
