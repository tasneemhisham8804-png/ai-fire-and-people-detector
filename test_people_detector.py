"""
Tests for _box_gap — the pure geometry function behind "is a person near
fire" (see PEOPLE_DETECTOR.py). Doesn't touch the YOLO model at all, so
these run fast with no GPU or model weights needed — conftest.py's stubs
just need to let PEOPLE_DETECTOR.py import cleanly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PEOPLE_DETECTOR import _box_gap


def test_overlapping_boxes_have_zero_gap():
    """Two boxes that overlap should report a gap of 0, not a negative 'overlap amount'."""
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    assert _box_gap(box_a, box_b) == 0


def test_touching_boxes_have_zero_gap():
    """Boxes that share an edge with no space between them are treated the same as overlapping: gap 0."""
    box_a = [0, 0, 100, 100]
    box_b = [100, 0, 200, 100]
    assert _box_gap(box_a, box_b) == 0


def test_horizontally_separated_boxes():
    """Simple axis-aligned case: gap should equal the plain horizontal distance between the boxes' edges."""
    box_a = [0, 0, 100, 100]
    box_b = [150, 0, 250, 100]
    assert _box_gap(box_a, box_b) == 50


def test_diagonally_separated_boxes():
    """Boxes offset on both axes at once should combine via the Pythagorean theorem, not just one axis."""
    box_a = [0, 0, 100, 100]
    box_b = [130, 140, 200, 200]
    # dx = 130-100 = 30, dy = 140-100 = 40 -> hypot(30, 40) = 50
    assert _box_gap(box_a, box_b) == 50


def test_proximity_threshold_boundary():
    """Sanity check that a gap exactly equal to PROXIMITY_THRESHOLD_PX is computed correctly at the boundary the app actually uses for its 'near fire' decision."""
    from config import PROXIMITY_THRESHOLD_PX
    box_a = [0, 0, 100, 100]
    box_b = [100 + PROXIMITY_THRESHOLD_PX, 0, 300, 100]
    gap = _box_gap(box_a, box_b)
    assert gap == PROXIMITY_THRESHOLD_PX
