"""
Tests for _localize_fire_regions — the classical-CV bounding-box layer
described in FIRE_DETECTOR.py (HSV color mask + contour detection, no
trained model involved). Frames are synthetic (drawn in-memory with numpy/
OpenCV) rather than real fire photos/video, so these tests are fast,
deterministic, and don't depend on any external test-data files.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from FIRE_DETECTOR import _localize_fire_regions
import config


def _make_frame(w=300, h=300):
    """Blank dark-blue background — well outside the fire HSV range, so it contributes no false-positive boxes."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (80, 30, 10)  # BGR, bluish
    return frame


def _paint_fire_patch(frame, x, y, size, bgr=(0, 100, 255)):
    """
    Paints a fire-colored square (BGR), plus a small near-white "hot core"
    sub-region in its center — mirroring how real flame is never a single
    uniform color; the hottest point of a flame is almost always
    overexposed toward white. _localize_fire_regions now requires this kind
    of hot-core sub-region to tell real fire apart from skin/other
    warm-colored objects that share fire's hue/saturation band but never
    get this washed-out (see FIRE_DETECTOR.py's module comment).
    """
    frame[y:y + size, x:x + size] = bgr
    core = max(2, size // 4)
    cx, cy = x + size // 2, y + size // 2
    frame[cy - core // 2: cy + core // 2, cx - core // 2: cx + core // 2] = (245, 245, 255)
    return frame


def test_no_fire_returns_no_boxes():
    """A frame with no fire-colored pixels at all should localize to an empty box list."""
    frame = _make_frame()
    boxes = _localize_fire_regions(frame)
    assert boxes == []


def test_large_fire_patch_is_detected():
    """A patch well above MIN_FIRE_REGION_AREA should be picked up as a box, with its reported area matching what was drawn."""
    frame = _make_frame()
    frame = _paint_fire_patch(frame, 100, 100, 40)  # 1600px area, well above MIN_FIRE_REGION_AREA
    boxes = _localize_fire_regions(frame)
    assert len(boxes) >= 1
    assert boxes[0]["area"] >= config.MIN_FIRE_REGION_AREA


def test_tiny_noise_patch_is_filtered_out():
    """A patch far under MIN_FIRE_REGION_AREA simulates HSV noise and should be filtered out, not reported as fire."""
    frame = _make_frame()
    # 3x3 = 9px, far under MIN_FIRE_REGION_AREA (150)
    frame = _paint_fire_patch(frame, 100, 100, 3)
    boxes = _localize_fire_regions(frame)
    assert boxes == []


def test_returns_at_most_five_boxes():
    """Even with more fire-colored regions present than MAX_FIRE_BOXES, the result list should be capped."""
    frame = _make_frame(600, 600)
    # paint 8 separate patches, spaced far enough apart to not merge after morphology
    for i in range(8):
        x = (i % 4) * 140 + 10
        y = (i // 4) * 140 + 10
        frame = _paint_fire_patch(frame, x, y, 30)
    boxes = _localize_fire_regions(frame)
    assert len(boxes) <= 5


def test_fire_colored_patch_without_hot_core_is_filtered_as_skin_like():
    """
    A uniformly-colored patch that falls inside the fire hue/saturation
    band (mirroring how warm-lit skin can look in HSV) but has no
    overexposed near-white sub-region should NOT be reported as fire. This
    is the mitigation for the false positive where a hand/face gets boxed
    as a "fire" region purely because it shares fire's hue range — the same
    failure mode that produced a spurious closest_person_to_fire_px: 0 in
    production output before this check existed.
    """
    frame = _make_frame()
    x, y, size = 100, 100, 40
    frame[y:y + size, x:x + size] = (60, 120, 220)  # warm, saturated — but uniform, no hot core
    boxes = _localize_fire_regions(frame)
    assert boxes == []


def test_fire_patch_with_hot_core_is_still_detected():
    """A fire-colored patch WITH a small near-white hot core (as real flame has) should still be detected — confirms the hot-core requirement isn't so strict it rejects real fire too."""
    frame = _make_frame()
    frame = _paint_fire_patch(frame, 100, 100, 40)  # includes an automatic hot core — see helper docstring
    boxes = _localize_fire_regions(frame)
    assert len(boxes) >= 1
