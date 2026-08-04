"""
People detection + fire-proximity module.

Uses a YOLO object detector (via ultralytics) to find people in a frame,
then computes how close the nearest detected person is to any detected fire
region. This module is only ever called on frames FIRE_DETECTOR.py has
already flagged as containing fire — running person detection on every
frame regardless of fire content would be wasted compute, since the
"person near fire" signal is meaningless without fire present in the first
place.
"""
import logging
import threading

from ultralytics import YOLO

import config

logger = logging.getLogger("people_detector")

_model = None
_weights_path = config.PEOPLE_MODEL_PATH
# True once load_people_model() has fallen back to the generic yolov8n.pt
# because the custom weights failed to load. Tracked separately from
# `_model is None` (which is never true after a load attempt, by design —
# see load_people_model) so callers like MAIN.py's /health can tell "a
# model is loaded" apart from "the *intended* model is loaded".
_using_fallback = False
# Serializes calls into the shared YOLO model object — see the docstring on
# predict_people_near_fire for why.
_inference_lock = threading.Lock()

# Both the custom-trained weights and the yolov8n.pt fallback report a class
# literally named "person" (index 0 in standard COCO; the custom model was
# trained on a single-class human-detection dataset). Filtering by class
# *name* rather than hardcoding index 0 means this still works correctly
# regardless of which weights end up loaded (see load_people_model's
# fallback below), instead of silently assuming index 0 always means the
# same thing across different models.
PERSON_CLASS_NAMES = {"person", "human"}


def load_people_model(weights_path: str = config.PEOPLE_MODEL_PATH):
    """
    Loads the people-detection YOLO model, with a graceful fallback.

    If the custom-trained weights (config.PEOPLE_MODEL_PATH, e.g. "best.pt")
    fail to load — missing file, corrupted weights, version mismatch — this
    falls back to ultralytics' pretrained yolov8n.pt so the pipeline can
    still detect people (using the generic 80-class COCO model) rather than
    the whole /analyze endpoint failing outright.

    This is intentionally idempotent per-weights-path: calling it twice with
    the SAME path is a no-op(ish) reload, but calling it again with a
    DIFFERENT path swaps the active model. MAIN.py should only ever call
    this once, for one weights file, at process startup — the model is
    meant to be loaded a single time and reused across all requests, not
    reloaded per-request.

    Whether the fallback was actually used is recorded in the module-level
    `_using_fallback` flag (read via is_using_fallback_weights()) rather
    than only being logged — a silently-degraded person detector matters
    enough here (this is a safety-adjacent tool) that it should be
    queryable at runtime, not just discoverable by reading server logs.
    """
    global _model, _weights_path, _using_fallback
    _weights_path = weights_path
    try:
        _model = YOLO(weights_path)
        _using_fallback = False
        logger.info("People model loaded successfully from %s", weights_path)
    except Exception as e:
        logger.error("People model '%s' failed to load, falling back to yolov8n.pt: %s", weights_path, e)
        _model = YOLO("yolov8n.pt")
        _using_fallback = True
    return _model


def is_using_fallback_weights() -> bool:
    """
    True if the active model is the generic yolov8n.pt fallback rather than
    the project's custom-trained weights — i.e. the custom weights failed
    to load. Exposed so callers (MAIN.py's /health) can surface this to
    anyone judging the pipeline's real-world accuracy: "a person model is
    loaded" and "the *trained* person model is loaded" are different claims,
    and only this flag distinguishes them.
    """
    return _using_fallback


def _box_gap(box_a: list[float], box_b: list[float]) -> float:
    """
    Shortest Euclidean distance between the edges of two axis-aligned boxes,
    given as [x1, y1, x2, y2].

    For each axis, the gap is the positive separation between the boxes on
    that axis (0 if they overlap or touch on that axis). Combining the two
    axis gaps with the Pythagorean theorem gives the true shortest distance
    between the box edges, including the diagonal case where the boxes are
    offset on both axes at once (not just directly left/right or
    above/below each other).　Overlapping boxes are treated as distance 0
    ("touching"), not a negative "how far they overlap" value.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def predict_people_near_fire(frame_path: str, fire_boxes: list[dict]) -> dict:
    """
    Runs YOLO on one frame and reports whether any detected person is within
    config.PROXIMITY_THRESHOLD_PX of any of the given fire bounding boxes.

    fire_boxes comes from FIRE_DETECTOR._localize_fire_regions and is
    typically empty only in the edge case where the classifier was confident
    enough to flag the frame but the HSV localizer didn't find a matching
    color region — in that case no proximity check is possible and
    people_near_fire is reported as False even if people are present.

    The actual YOLO call is wrapped in a lock: MAIN.py's run_analysis calls
    this once per fire-flagged frame from a single thread, but multiple
    concurrent /analyze requests each run in their own threadpool worker
    thread and would otherwise call into this same shared `_model` object
    at once. Ultralytics' YOLO wraps a PyTorch model and isn't documented
    as safe for concurrent inference calls from multiple threads on one
    instance — serializing here trades a little throughput under
    concurrent load for not needing a per-request model instance (which
    would cost real memory, especially on the project's target 4GB-VRAM
    GPU) or having to independently verify thread-safety.
    """
    if _model is None:
        load_people_model(_weights_path)

    with _inference_lock:
        results = _model(frame_path, verbose=False)
    people_boxes = []

    for result in results:
        # Filtering by class name (not just taking every detected box) is
        # what makes the yolov8n.pt fallback above safe to use: that model
        # is trained on the full 80-class COCO set, so without this filter
        # every car, dog, chair, etc it detects would get reported as a
        # "person," corrupting the proximity check. With the custom
        # single-class weights this filter is a no-op, since every
        # detection is already a person.
        names = getattr(result, "names", {})
        for i, cls_tensor in enumerate(result.boxes.cls):
            cls_id = int(cls_tensor.item())
            cls_name = names.get(cls_id, "")
            if cls_name and cls_name.lower() not in PERSON_CLASS_NAMES:
                continue
            box = result.boxes.xyxy[i].tolist()
            confidence = result.boxes.conf[i].item()
            people_boxes.append({"box": box, "confidence": round(confidence, 4)})

    people_near_fire = False
    closest_gap = None

    if fire_boxes and people_boxes:
        # Every person box is checked against every fire box, and the
        # single smallest gap across all those pairs is what determines
        # "near fire" — this correctly handles multiple people and/or
        # multiple fire regions in one frame, since only the closest
        # pairing actually matters for the proximity verdict.
        gaps = [
            _box_gap(p["box"], f["box"])
            for p in people_boxes
            for f in fire_boxes
        ]
        closest_gap = round(min(gaps), 1)
        people_near_fire = closest_gap <= config.PROXIMITY_THRESHOLD_PX

    return {
        "people_detected": len(people_boxes) > 0,
        "people_near_fire": people_near_fire,
        "people_count": len(people_boxes),
        "closest_person_to_fire_px": closest_gap,
        "people_boxes": people_boxes,
    }
