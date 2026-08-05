"""
Fire detection module.

Two separate techniques are combined here, each doing a different job:

  1. A trained Keras/TensorFlow image classifier (MobileNetV2-style) decides,
     per frame, "does this frame contain fire?" and outputs a confidence
     score. This is the accuracy-critical part — it's learned from data and
     generalizes across lighting, fire size, camera angle, etc.

  2. A classical, non-learned computer-vision routine (HSV color
     thresholding + contour detection) finds *where* in a fire-flagged frame
     the fire actually is, producing bounding boxes. This is deliberately
     not a second neural network — a color-threshold approach is cheap,
     needs no training data, and is "good enough" for localization once the
     classifier has already confirmed fire is present. It's only run on
     frames the classifier flagged, so its lower precision on its own
     (it can't tell fire from other orange/red objects) doesn't matter much
     in context.

predict_fire_batch() is the main entry point MAIN.py calls: given a list of
frame file paths and a loaded model, it returns, per frame, a confidence
score plus any localized fire bounding boxes.
"""
import os
import json
import logging
import threading

import cv2
import numpy as np
import tensorflow as tf

import config

logger = logging.getLogger("main.fire_detector")

# Serializes calls into the shared Keras model object across concurrent
# /analyze requests — see predict_fire_batch's docstring for why.
_inference_lock = threading.Lock()

class ModelLoadError(Exception):
    """Custom exception raised when a model fails to load properly."""
    pass

def load_fire_model(model_path: str):
    """
    🎓 In plain English: this function loads our trained fire-recognizing
    "brain" (a neural network) from disk into memory, one time, so that
    every video we analyze later can reuse it instead of reloading it from
    scratch on every single request.

    Loads the fire detector.

    Keras 3's default save format is a directory containing config.json
    (the model architecture) and model.weights.h5 (just the numeric
    weights) rather than a single flat file. To load that format, the
    architecture is rebuilt directly from the JSON spec first, and the
    weight values are then loaded onto that freshly-built architecture --
    this two-step process is necessary because model.weights.h5 alone has
    no way to describe *what* architecture the weights belong to.

    A single flat-file model (the older Keras/TF format) is also supported
    via the plain load_model() fallback, so this function works with either
    a Keras-3-style directory or a legacy .keras/.h5 model file.
    """
    try:
        if os.path.isdir(model_path):
            config_file = os.path.join(model_path, "config.json")
            weights_file = os.path.join(model_path, "model.weights.h5")
            
            if not os.path.exists(config_file) or not os.path.exists(weights_file):
                raise FileNotFoundError(f"Missing config.json or model.weights.h5 inside {model_path}")
            
            logger.info("Rebuilding architecture from configuration: %s", config_file)
            
            with open(config_file, "r") as f:
                raw_config = json.load(f)
            
            # Keras 3's saved config.json wraps the actual layer graph one
            # level deeper, under a "model_config" key, alongside other
            # top-level save metadata (Keras version, etc). model_from_json
            # expects just the layer graph, so that inner value is what
            # actually gets passed to it when present.
            if "model_config" in raw_config:
                clean_json = json.dumps(raw_config["model_config"])
            else:
                clean_json = json.dumps(raw_config)

            model = tf.keras.models.model_from_json(clean_json)
            
            logger.info("Loading explicit matching weights onto architecture: %s", weights_file)
            model.load_weights(weights_file)
            
        else:
            # Legacy path: a single .keras/.h5 file with architecture and
            # weights bundled together, loadable in one call.
            model = tf.keras.models.load_model(model_path)
            
        logger.info("Fire model loaded successfully with a perfect configuration match!")
        return model

    except Exception as e:
        logger.error("Fire model failed to load from %s: %s", model_path, e)
        return None


# ── Fire localization (classical CV, no model needed) ───────────────────────
# Fire pixels tend to cluster in a fairly narrow orange/red/yellow hue band
# with high saturation and brightness in HSV color space, which is what
# these bounds capture. HSV is used instead of RGB/BGR here specifically
# because it separates hue (color) from brightness — a color-based rule like
# this is far more robust to a flame's brightness varying across a frame
# than a similar rule written directly against BGR channel values would be.
FIRE_HSV_LOWER = np.array([0, 100, 100], dtype=np.uint8)
FIRE_HSV_UPPER = np.array([35, 255, 255], dtype=np.uint8)
_MORPH_KERNEL = np.ones((5, 5), np.uint8)

# ── Skin-tone false-positive mitigation ──────────────────────────────────
# Known limitation: warm-lit skin frequently falls inside FIRE_HSV_LOWER/
# UPPER too — a hand or face can share fire's hue and, under warm lighting,
# a similar saturation. Hue+saturation bounds alone can't reliably tell the
# two apart, because both are legitimately warm-hued and can both be highly
# saturated.
#
# The signal that DOES reliably separate them: real flame is almost never a
# uniform color. Somewhere in a genuine fire blob there's an overexposed,
# near-white "hot core" — very high brightness (V) combined with LOW
# saturation, since color washes out toward white at a flame's hottest
# point. Skin, even bright/saturated skin, essentially never has a
# low-saturation, high-brightness sub-region the way flame does — a skin
# pixel that bright is just... bright skin, not desaturated-toward-white.
# A candidate blob is only kept as "fire" if some minimum fraction of its
# pixels meet that hot-core profile.
#
# This is a mitigation, not a full fix — it reduces but doesn't eliminate
# skin-tone false positives (e.g. an overexposed hand under a bright light
# could still slip through), and it's tuned by inspection rather than
# against a labeled dataset. See LIMITATIONS.md for the more robust
# approaches (temporal flicker analysis across frames, a small trained
# classifier) that would be needed to close this gap properly.
FIRE_CORE_V_THRESHOLD = 200      # brightness a pixel needs to count toward a "hot core"
FIRE_CORE_MAX_SATURATION = 80    # ...and it must ALSO be this desaturated (near-white) to count
FIRE_MIN_CORE_RATIO = 0.03       # min fraction of a blob's pixels that must be "hot core" pixels


def _localize_fire_regions(frame: np.ndarray, max_boxes: int = None) -> list:
    """
    🎓 In plain English: once the AI model has already said "yes, there's
    fire in this frame," this function's job is just to draw a box around
    WHERE the fire is. It does this the old-fashioned way (no AI) — it
    looks for pixels that are the right fire-like color (orange/red/yellow)
    and groups nearby fire-colored pixels together into blobs, then filters
    out fake ones (like skin) using the "hot core" trick explained below.

    Finds fire-colored blobs in a BGR frame and returns their bounding boxes.

    Pipeline: HSV color mask -> morphological cleanup -> contour detection
    -> area filter -> hot-core check (see module comment above) -> bounding
    box + sort by size.

    Returns a list of {"box": [x1, y1, x2, y2], "area": float}, largest area
    first, capped at `max_boxes` (defaults to config.MAX_FIRE_BOXES).
    """
    if max_boxes is None:
        max_boxes = config.MAX_FIRE_BOXES

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)

    # Opening (erode then dilate) first clears out single-pixel/salt noise
    # that's too small for the kernel to survive; closing (dilate then
    # erode) afterward fills small gaps inside a real fire blob so a single
    # flame doesn't get fragmented into several smaller contours.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hot-core mask: pixels bright enough AND desaturated enough to be part
    # of an overexposed flame highlight. Computed once, directly from the
    # frame's HSV (independent of `mask` above), then checked against each
    # candidate blob individually below.
    core_mask = cv2.inRange(
        hsv,
        np.array([0, 0, FIRE_CORE_V_THRESHOLD], dtype=np.uint8),
        np.array([179, FIRE_CORE_MAX_SATURATION, 255], dtype=np.uint8),
    )

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_FIRE_REGION_AREA:
            continue

        # Filled version of this contour's footprint, used to check what
        # fraction of the blob's interior — including any small internal
        # gaps the base fire-color mask didn't cover, like a genuine
        # near-white core — qualifies as a hot core.
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, thickness=cv2.FILLED)
        blob_pixel_count = cv2.countNonZero(blob_mask)
        core_pixel_count = cv2.countNonZero(cv2.bitwise_and(core_mask, blob_mask))
        core_ratio = core_pixel_count / blob_pixel_count if blob_pixel_count else 0.0

        if core_ratio < FIRE_MIN_CORE_RATIO:
            # Fire-colored, but with no overexposed "hot" sub-region
            # anywhere in it — much more consistent with skin or another
            # warm-colored object than with real flame.
            continue

        x, y, w, h = cv2.boundingRect(c)
        boxes.append({"box": [float(x), float(y), float(x + w), float(y + h)], "area": float(area)})

    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes[:max_boxes]


def predict_fire_batch(frame_paths: list, model) -> list:
    """
    🎓 In plain English: this is the main "check these video frames for
    fire" function. It hands the model a stack of frame images at once
    (a "batch") instead of one at a time, because that's faster. For every
    frame, it gets back a confidence score from 0 to 1 (how sure the model
    is that fire is present) — and for the frames the model flags as
    fiery, it also calls _localize_fire_regions() above to draw boxes
    around exactly where the fire is in that frame.

    Runs the fire classifier over all frames and localizes fire regions on
    any frame that comes back above the confidence threshold.

    Frames are grouped into batches of config.FIRE_BATCH_SIZE and passed to
    model.predict() together rather than one at a time — a single call over
    N stacked images is meaningfully faster on a GPU than N separate calls,
    since it amortizes the per-call overhead across the whole batch.

    Localization (_localize_fire_regions) only runs on frames whose
    confidence clears the threshold, since there's no point running contour
    detection on a clean frame just to throw away an empty result.

    Order matters here — MAIN.py lines results[i] up with frames[i] to build
    the confidence timeline the UI charts — so a pre-sized `results` list
    indexed by original position is used instead of appending, so a failed
    frame partway through a batch can't shift everything after it out of
    order.

    The actual model.predict() call is wrapped in a lock: multiple
    concurrent /analyze requests each run in their own threadpool worker
    thread (see MAIN.py) and would otherwise call into this same shared
    Keras model object at once. TensorFlow/Keras models aren't documented
    as safe for concurrent predict() calls on one instance from multiple
    threads — serializing here trades a little throughput under concurrent
    load for not needing a per-request model instance or independently
    verifying thread-safety. Only the predict() call itself is inside the
    lock; image loading/decoding above and box localization below stay
    outside it so they can still happen concurrently across requests.
    """
    if model is None:
        raise ModelLoadError("Inference failed: Fire model layout not initialized.")

    results = [None] * len(frame_paths)
    batch_size = max(1, config.FIRE_BATCH_SIZE)

    for batch_start in range(0, len(frame_paths), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(frame_paths)))

        images = []
        good_indices = []
        for idx in batch_indices:
            try:
                img = tf.keras.utils.load_img(frame_paths[idx], target_size=(224, 224))
                images.append(tf.keras.utils.img_to_array(img) / 255.0)  # Normalize to [0, 1] to match training preprocessing
                good_indices.append(idx)
            except Exception as e:
                logger.warning("Failed to load frame %s: %s", frame_paths[idx], e)
                results[idx] = {"fire_confidence": 0.0, "fire_boxes": []}

        if not images:
            continue

        try:
            with _inference_lock:
                predictions = model.predict(np.stack(images, axis=0), verbose=0)
        except Exception as e:
            logger.warning("Batch prediction failed for %d frame(s): %s", len(images), e)
            for idx in good_indices:
                results[idx] = {"fire_confidence": 0.0, "fire_boxes": []}
            continue

        for idx, prediction in zip(good_indices, predictions):
            confidence = round(float(prediction[0]), 4)
            fire_boxes = []
            if confidence > config.FIRE_CONFIDENCE_THRESHOLD:
                frame_img = cv2.imread(frame_paths[idx])
                if frame_img is not None:
                    fire_boxes = _localize_fire_regions(frame_img)
            results[idx] = {"fire_confidence": confidence, "fire_boxes": fire_boxes}

    return results
