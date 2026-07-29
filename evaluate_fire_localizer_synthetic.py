"""
Quantitative evaluation of the fire localizer's hot-core discriminator,
comparing it against the pre-fix (color-mask-only) behavior on a synthetic
labeled test set.

WHY SYNTHETIC IMAGES: _localize_fire_regions is classical CV — no trained
model, no weights file — so it's the one detector in this project that CAN
be meaningfully evaluated right now, without real labeled video footage.
The fire classifier (Keras) and people detector (YOLO) can't be evaluated
this way; see evaluate.py for the harness meant for those, which needs real
model weights and labeled clips this environment doesn't have.

This is a real, run-now precision/recall comparison — not fabricated
numbers — but it's still a synthetic proxy for the true question ("does
this correctly separate real fire from skin in actual video"), since the
synthetic negatives are a simplified stand-in for the range of ways skin
tone actually appears on camera (mixed lighting, JPEG compression, motion
blur, partial shadow, etc). Treat the numbers below as evidence the
discriminator does what it's designed to do, not as a claim about
real-world accuracy.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Reuses the same heavy-dependency stubs as conftest.py (tensorflow isn't
# actually used by anything this script calls — only FIRE_DETECTOR.py's
# module-level `import tensorflow as tf` needs to succeed).
import conftest  # noqa: F401  (import side effect: registers the stubs)

import config
from FIRE_DETECTOR import _localize_fire_regions, FIRE_HSV_LOWER, FIRE_HSV_UPPER, _MORPH_KERNEL

random.seed(0)


def _blank_frame(w=300, h=300):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (80, 30, 10)  # bluish background, outside fire HSV range
    return frame


def _legacy_localize(frame, max_boxes=None):
    """
    Reproduces the pre-fix localizer (color mask + area filter only, no
    hot-core check) so its false-positive rate can be measured side by side
    with the current version, on the same test images.
    """
    if max_boxes is None:
        max_boxes = config.MAX_FIRE_BOXES
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_FIRE_REGION_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append({"box": [float(x), float(y), float(x + w), float(y + h)], "area": float(area)})
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes[:max_boxes]


def _make_fire_frame():
    """Positive example: fire-colored patch with a near-white hot core, like real flame."""
    frame = _blank_frame()
    size = random.randint(25, 60)
    x = random.randint(10, 300 - size - 10)
    y = random.randint(10, 300 - size - 10)
    hue = random.randint(0, 20)
    frame[y:y + size, x:x + size] = _hsv_to_bgr(hue, random.randint(200, 255), random.randint(200, 255))
    core = max(2, size // 4)
    cx, cy = x + size // 2, y + size // 2
    frame[cy - core // 2: cy + core // 2, cx - core // 2: cx + core // 2] = (245, 245, 255)
    return frame


def _make_skin_frame():
    """
    Negative example: warm, saturated, fire-hue-band patch with NO hot core
    — a simplified stand-in for skin under warm lighting, the false
    positive this whole check exists to catch.
    """
    frame = _blank_frame()
    size = random.randint(25, 60)
    x = random.randint(10, 300 - size - 10)
    y = random.randint(10, 300 - size - 10)
    hue = random.randint(0, 20)
    frame[y:y + size, x:x + size] = _hsv_to_bgr(hue, random.randint(120, 200), random.randint(150, 230))
    return frame


def _make_empty_frame():
    """Negative example: no fire-colored content at all."""
    return _blank_frame()


def _hsv_to_bgr(h, s, v):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0])


def _evaluate(localize_fn, n_fire=40, n_skin=40, n_empty=20):
    tp = fp = tn = fn = 0

    for _ in range(n_fire):
        boxes = localize_fn(_make_fire_frame())
        if boxes:
            tp += 1
        else:
            fn += 1

    for _ in range(n_skin):
        boxes = localize_fn(_make_skin_frame())
        if boxes:
            fp += 1
        else:
            tn += 1

    for _ in range(n_empty):
        boxes = localize_fn(_make_empty_frame())
        if boxes:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    false_positive_rate_on_skin = None
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


if __name__ == "__main__":
    print("Synthetic fire-localizer evaluation")
    print("  positives : fire-colored patch + hot core (n=40)")
    print("  negatives : skin-like patch, no hot core (n=40) + plain background (n=20)")
    print()

    print("=== Current localizer (with hot-core discriminator) ===")
    current = _evaluate(_localize_fire_regions)
    for k, v in current.items():
        print(f"  {k}: {v}")

    print()
    print("=== Legacy localizer (color mask + area filter only) ===")
    legacy = _evaluate(_legacy_localize)
    for k, v in legacy.items():
        print(f"  {k}: {v}")

    print()
    print(f"Skin-patch false positives — legacy: {legacy['fp']}/40, current: {current['fp']}/40 "
          "(lower is better; these are the false 'fire near person' triggers this check targets)")
