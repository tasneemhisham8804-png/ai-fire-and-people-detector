"""
Central configuration for the fire / people / AI-generated detection pipeline.

This project runs three independent ML models over a video's sampled frames
and combines their outputs into a single verdict:

  1. Fire classifier (Keras/TensorFlow)  -> per-frame fire confidence
  2. People detector (YOLO / ultralytics) -> bounding boxes for people,
     only run on frames the fire classifier already flagged, then checked
     for proximity to the fire boxes
  3. AI-generated-image classifier (HuggingFace ViT) -> sampled across a
     handful of frames to estimate whether the footage itself is synthetic

All three modules (FIRE_DETECTOR.py, PEOPLE_DETECTOR.py, AI_DETECTOR.py) and
the orchestrator (MAIN.py) read their tunable knobs from this file instead
of hardcoding them, so behavior can be retuned without touching model code.

🎓 In plain English: think of this file as the project's single "settings
panel." Every number below that controls how strict/lenient the app is
(how confident the model has to be before it says "yes, that's fire," how
big a video is allowed to be, etc.) lives here in ONE place, instead of
being scattered as magic numbers across different files. That way, to
change a behavior you only ever have to edit it in one spot.
"""

# ── Model paths ──────────────────────────────────────────────────────────
FIRE_MODEL_PATH = "Fire_DETECTOR.keras"
PEOPLE_MODEL_PATH = "best.pt"
# Not currently used — AI_DETECTOR.py loads "umm-maybe/AI-image-detector"
# from the HuggingFace hub instead of a local weights file, after the
# EfficientNet-B0/CIFAKE model didn't generalize to full-resolution video
# frames. Left here in case that gets swapped back later.
AI_MODEL_PATH = "ai_detector.pth"

# ── Fire detection ───────────────────────────────────────────────────────
# The fire classifier outputs a single sigmoid confidence per frame; a frame
# only counts as "fire" if it clears this threshold. Tuned by hand rather
# than picked from a validation curve — 0.5 is the natural midpoint for a
# binary sigmoid output and is a reasonable starting point.
FIRE_CONFIDENCE_THRESHOLD = 0.5

# Minimum contour area (in pixels) for a color-matched blob to count as a
# real fire region rather than HSV noise (a single stray warm-colored pixel,
# JPEG compression artifacts, etc). Anything smaller is discarded before it
# ever becomes a bounding box.
MIN_FIRE_REGION_AREA = 150

# How many frames get stacked into a single model.predict() call. Batching
# trades a bit of memory for fewer, larger GPU calls, which is faster than
# calling predict() once per frame — but the GPU this was developed against
# (GTX 1650, 4GB VRAM) can't hold an unlimited batch, hence a fixed cap
# instead of just batching everything at once.
FIRE_BATCH_SIZE = 16

# Upper bound on how many fire bounding boxes get reported per frame, so a
# frame that's mostly fire-colored (e.g. an orange sunset) can't blow up the
# response payload or the proximity-check cost in PEOPLE_DETECTOR.py.
MAX_FIRE_BOXES = 5

# ── People detection ─────────────────────────────────────────────────────
# Two boxes (a person box and a fire box) count as "near" each other if the
# shortest gap between their edges is <= this many pixels. This is in raw
# frame pixels, not a physical distance — it implicitly assumes a roughly
# constant camera distance/resolution across input videos.
PROXIMITY_THRESHOLD_PX = 120

# ── AI-generated detection ───────────────────────────────────────────────
AI_IMG_SIZE = 224      # not used by the current HF pipeline (kept for reference / possible local-model fallback)
AI_GENERATED_IDX = 1   # not used by the current HF pipeline (1 = AI-generated per the original training notebook's label order)

# A clip is called "AI-generated" if its average AI-generated probability
# across the sampled frames exceeds this. Averaging over multiple frames
# (rather than trusting any single frame) smooths out per-frame noise from
# the classifier.
AI_GENERATED_THRESHOLD = 0.7

# Running the AI-generated classifier on every frame of a video would be
# slow for no real accuracy benefit (the visual "look" of AI-generated
# footage is fairly consistent frame to frame), so only this many frames,
# evenly spaced across the clip, are sampled instead.
AI_SAMPLE_SIZE = 10

# How long AI_DETECTOR.py's _get_model() waits before retrying to load the
# HuggingFace model after a failed attempt, instead of giving up for the
# rest of the process's lifetime. A transient failure (hub unreachable at
# startup, brief network hiccup) shouldn't permanently disable the
# AI-generated check — and now that the check gates whether fire/people
# detection runs at all (see MAIN.py's run_analysis), a stuck-off AI check
# means every video silently falls back to the old always-run-everything
# path with no way to recover without restarting the server.
AI_DETECTOR_RETRY_COOLDOWN_SECONDS = 600

# ── Video processing ─────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_MB = 100

# Upper bound on a video's *duration*, independent of MAX_UPLOAD_MB. A long,
# low-bitrate clip can be small in MB while still decoding into thousands of
# sampled frames at TARGET_FPS, which would blow up analysis time (and, via
# the people-detector loop over every fire-flagged frame, GPU cost) far
# beyond what the file-size limit alone would suggest. Checked against the
# video's own metadata (frame_count / fps) right after it's opened, before
# any frames are extracted.
MAX_VIDEO_DURATION_SECONDS = 300

# Hard backstop on the number of frames extract_frames() will ever write to
# disk, independent of the duration/FPS math above. MAX_VIDEO_DURATION_SECONDS
# relies on the video's *reported* metadata (CAP_PROP_FRAME_COUNT) being
# accurate — some corrupt or unusual containers report 0 or garbage there, in
# which case that check silently no-ops. This cap doesn't trust metadata at
# all: it just stops the decode loop once it's written this many frames,
# regardless of what the video claims about itself. Set generously above the
# expected worst case (MAX_VIDEO_DURATION_SECONDS * TARGET_FPS = 900) so it
# never triggers on a legitimate video, only on ones metadata already failed
# to catch.
MAX_EXTRACTED_FRAMES = 1200

# Frames-per-second the video is *downsampled* to before running any model.
# Running inference at the source video's native FPS (often 24-60) would be
# far more compute than the analysis needs — fire/people/AI-ness don't
# change frame-to-frame fast enough to require that density of sampling.
TARGET_FPS = 3

# Magic-byte signatures used for a lightweight content sniff on top of the
# extension check — the extension on an uploaded filename is just a string
# the client sends and can't be trusted on its own (e.g. a .txt renamed to
# .mp4 would otherwise sail through). Checking the first few bytes of the
# actual file content against known container-format signatures catches
# that class of mismatch before the file is ever handed to OpenCV.
#
# Each entry is (byte_offset, magic_bytes, valid_extensions) — the file is
# recognized as a known video container if `magic_bytes` is found at
# `byte_offset`, and considered a *match for the claimed extension* only if
# that claimed extension is in `valid_extensions`. The ISO base media
# 'ftyp' box (offset 4) is shared by both .mp4 and .mov, so both extensions
# are accepted for that one signature; RIFF/AVI and Matroska/EBML each map
# to exactly one extension. Read directly by MAIN.py's
# _sniff_video_content(), which checks the claimed extension against this
# table rather than just confirming "this is some kind of video" — a .mp4
# upload that's actually a renamed .avi is a content/extension mismatch
# worth rejecting, not a false alarm.
VIDEO_MAGIC_BYTES = [
    (0, b"RIFF", {".avi"}),           # full check also confirms bytes 8-11 == b"AVI "
    (0, b"\x1a\x45\xdf\xa3", {".mkv"}),
    (4, b"ftyp", {".mp4", ".mov"}),
]

# ── API protection ───────────────────────────────────────────────────────
# /analyze is the expensive endpoint (up to three model inferences per
# request) and CORS is wide open (see MAIN.py) — fine for local development
# or a graded demo, but worth having *some* protection available before
# this is ever exposed beyond localhost.
#
# Both are opt-in and disabled by default (None / 0) so the existing
# no-auth, unlimited local-demo behavior is unchanged unless explicitly
# configured — set via environment variables rather than hardcoded here so
# a real deployment can turn them on without editing source.
import os

# If set, /analyze requires a matching `X-API-Key` header on every request.
API_KEY = os.environ.get("FIRE_DETECTOR_API_KEY")  # None = auth disabled

# Max /analyze requests allowed per client IP per rolling 60-second window.
# 0 or unset = rate limiting disabled.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("FIRE_DETECTOR_RATE_LIMIT_PER_MINUTE", "0"))
