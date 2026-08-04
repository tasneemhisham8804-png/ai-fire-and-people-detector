"""
Integration-level tests for MAIN.py's HTTP layer (FastAPI TestClient).

These don't exercise real inference at all — the Keras/YOLO/HF models are
never actually loaded (conftest.py stubs their packages) and the prediction
functions are monkeypatched with fakes. What's under test here is everything
AROUND the models: request validation (extension, magic-byte content
sniffing, upload size limits), the "model not loaded" guard, error handling,
and that a successful request correctly assembles fire + people + AI results
into the response shape the frontend depends on (including which verdict
branch gets picked for a given combination of inputs). The pure-logic pieces
of the models themselves (_box_gap, _localize_fire_regions) have their own
dedicated tests in test_people_detector.py / test_fire_detector.py.
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import MAIN
import config


@pytest.fixture
def client():
    return TestClient(MAIN.app)


# Minimal byte headers that pass _sniff_video_content's magic-byte checks
# without needing an actual playable video file — only the first 12-16 bytes
# are inspected, so the rest of the "video" content is irrelevant filler.
VALID_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00" + b"\x00" * 20
NOT_A_VIDEO = b"just a plain text file pretending to be a video" * 5
# A real AVI file's byte signature, uploaded under a .mp4 filename — this is
# the extension/content *mismatch* case (as opposed to NOT_A_VIDEO, which
# isn't a recognized video signature at all).
AVI_HEADER_WRONG_EXTENSION = b"RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00" + b"\x00" * 20


def test_rejects_disallowed_extension(client):
    """A .txt upload should never reach model/content checks — extension is validated first."""
    resp = client.post(
        "/analyze",
        files={"file": ("clip.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_rejects_when_fire_model_not_loaded(client, monkeypatch):
    """Simulates the real startup failure mode (missing/corrupt model file) — should fail loudly with 503, not proceed silently."""
    monkeypatch.setattr(MAIN, "fire_model", None)
    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )
    assert resp.status_code == 503


def test_rejects_content_that_doesnt_match_extension(client, monkeypatch):
    """Extension says .mp4, but the actual bytes don't match any known video container signature — should be caught before frame extraction is ever attempted."""
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(NOT_A_VIDEO), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "doesn't match actual file content" in resp.json()["detail"]


def test_rejects_content_extension_mismatch(client, monkeypatch):
    """
    Extension says .mp4, but the actual bytes are a valid AVI signature —
    this should be rejected as a mismatch, not waved through just because
    the bytes belong to *some* recognized video container. Distinct from
    test_rejects_content_that_doesnt_match_extension, which covers content
    that isn't a recognized video signature at all.
    """
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(AVI_HEADER_WRONG_EXTENSION), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "doesn't match actual file content" in resp.json()["detail"]


def test_accepts_avi_content_with_avi_extension(client, monkeypatch):
    """The same AVI byte signature should pass when the claimed extension actually is .avi."""
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "extract_frames", lambda path: [Path("/tmp/frame_0.jpg")])
    monkeypatch.setattr(MAIN, "predict_fire_batch", lambda paths, model: [{"fire_confidence": 0.0, "fire_boxes": []}])
    monkeypatch.setattr(MAIN, "predict_people_near_fire", lambda *a, **k: {
        "people_detected": False, "people_near_fire": False, "people_count": 0,
        "closest_person_to_fire_px": None, "people_boxes": [],
    })
    monkeypatch.setattr(MAIN, "predict_ai_generated", lambda frame_path: 0.0)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.avi", io.BytesIO(AVI_HEADER_WRONG_EXTENSION), "video/x-msvideo")},
    )
    assert resp.status_code == 200


def test_rejects_video_exceeding_max_duration(client, monkeypatch):
    """
    A video whose metadata reports a duration beyond MAX_VIDEO_DURATION_SECONDS
    should be rejected before frame extraction proceeds, independent of
    whether it also happens to be under the file-size limit.
    """
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(config, "MAX_VIDEO_DURATION_SECONDS", 10)

    # 3000 frames / 30 fps = 100s, well past the 10s cap set above.
    fake_props = {MAIN.cv2.CAP_PROP_FPS: 30.0, MAIN.cv2.CAP_PROP_FRAME_COUNT: 3000.0}
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.get.side_effect = lambda prop: fake_props.get(prop, 0.0)
    monkeypatch.setattr(MAIN.cv2, "VideoCapture", lambda path: fake_cap)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "too long" in resp.json()["detail"].lower()


def test_rejects_oversized_upload(client, monkeypatch):
    """Upload size is enforced while streaming to disk, not after the fact — verified with the limit shrunk so the test doesn't need to push 100MB through."""
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)  # ~1KB
    big_body = VALID_MP4_HEADER + (b"\x00" * 5000)
    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(big_body), "video/mp4")},
    )
    assert resp.status_code == 413


def test_successful_analysis_assembles_expected_response(client, monkeypatch):
    """
    End-to-end through the HTTP layer with every model call faked out.
    Confirms run_analysis correctly wires fire + people + AI results
    together into the documented response shape, and that the verdict logic
    picks the "fire + people near fire" branch for inputs that should
    produce it.
    """
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "people_model", MagicMock())

    fake_frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(3)]
    monkeypatch.setattr(MAIN, "extract_frames", lambda path: fake_frames)

    # Frame 1 is "on fire"; the other two are not.
    monkeypatch.setattr(
        MAIN,
        "predict_fire_batch",
        lambda paths, model: [
            {"fire_confidence": 0.1, "fire_boxes": []},
            {"fire_confidence": 0.92, "fire_boxes": [{"box": [10, 10, 50, 50], "area": 1600.0}]},
            {"fire_confidence": 0.05, "fire_boxes": []},
        ],
    )

    monkeypatch.setattr(
        MAIN,
        "predict_people_near_fire",
        lambda frame_path, fire_boxes: {
            "people_detected": True,
            "people_near_fire": True,
            "people_count": 2,
            "closest_person_to_fire_px": 15.0,
            "people_boxes": [],
        },
    )

    monkeypatch.setattr(MAIN, "predict_ai_generated", lambda frame_path: 0.02)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["fire_detected"] is True
    assert body["fire_check_skipped"] is False
    assert body["fire_flagged_frames"] == [1]
    assert body["people_detected"] is True
    assert body["people_check_skipped"] is False
    assert body["people_near_fire"] is True
    assert body["ai_check_available"] is True
    assert body["ai_generated_probability"] == 0.02
    assert body["verdict"] == "Real footage — fire detected with people in proximity"
    assert body["total_frames_analyzed"] == 3


def test_ai_generated_verdict_skips_fire_and_people_detection(client, monkeypatch):
    """
    If the AI-generated score clears the threshold, fire/people detection
    must be skipped entirely — not just overridden in the final verdict
    text. predict_fire_batch/predict_people_near_fire are wired as spies
    here specifically to assert they're never called, since running them
    against footage already known to be synthetic would be wasted compute
    for findings that would only need to be discarded anyway.
    """
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "extract_frames", lambda path: [Path("/tmp/frame_0.jpg")])

    fire_spy = MagicMock(return_value=[{"fire_confidence": 0.99, "fire_boxes": []}])
    people_spy = MagicMock(return_value={
        "people_detected": True, "people_near_fire": True, "people_count": 3,
        "closest_person_to_fire_px": 5.0, "people_boxes": [],
    })
    monkeypatch.setattr(MAIN, "predict_fire_batch", fire_spy)
    monkeypatch.setattr(MAIN, "predict_people_near_fire", people_spy)
    monkeypatch.setattr(MAIN, "predict_ai_generated", lambda frame_path: 0.95)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "AI-generated footage — fire/people detection skipped"
    assert body["fire_check_skipped"] is True
    assert body["people_check_skipped"] is True
    assert body["fire_detected"] is False
    assert body["people_detected"] is False
    assert body["fire_flagged_frames"] == []
    assert body["fire_confidence_timeline"] == []
    fire_spy.assert_not_called()
    people_spy.assert_not_called()


def test_ai_check_unavailable_is_reported_not_hidden(client, monkeypatch):
    """
    When every sampled frame's AI check returns None (model unavailable),
    the response must say so explicitly rather than silently defaulting to
    a bare 'real footage' claim — and fire/people detection must still run
    as normal, since an unavailable AI check gives no signal to skip on.
    """
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "extract_frames", lambda path: [Path("/tmp/frame_0.jpg")])
    monkeypatch.setattr(MAIN, "predict_fire_batch", lambda paths, model: [{"fire_confidence": 0.0, "fire_boxes": []}])
    monkeypatch.setattr(MAIN, "predict_people_near_fire", lambda *a, **k: {
        "people_detected": False, "people_near_fire": False, "people_count": 0,
        "closest_person_to_fire_px": None, "people_boxes": [],
    })
    monkeypatch.setattr(MAIN, "predict_ai_generated", lambda frame_path: None)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ai_check_available"] is False
    assert body["ai_generated_probability"] is None
    assert body["fire_check_skipped"] is False
    assert body["people_check_skipped"] is False
    assert "AI-generated check unavailable" in body["verdict"]


def test_unexpected_inference_error_returns_clean_500(client, monkeypatch):
    """An unexpected exception mid-inference (corrupt frame, model quirk, etc) should come back as a clean generic 500, not a raw traceback leaked to the client."""
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "extract_frames", lambda path: [Path("/tmp/frame_0.jpg")])

    def _boom(paths, model):
        raise RuntimeError("simulated decode failure")

    monkeypatch.setattr(MAIN, "predict_fire_batch", _boom)

    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", io.BytesIO(VALID_MP4_HEADER), "video/mp4")},
    )
    assert resp.status_code == 500
    assert "unexpectedly" in resp.json()["detail"].lower()


def test_health_endpoint_reports_model_status(client, monkeypatch):
    """/health should reflect the actual current load state of each model, not a hardcoded/static value."""
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "people_model", None)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fire_model_loaded"] is True
    assert body["people_model_loaded"] is False


def test_health_endpoint_reports_people_fallback_state(client, monkeypatch):
    """
    /health must distinguish "a people model is loaded" from "the *trained*
    people model is loaded" — people_model_loaded alone can't tell these
    apart, since a failed custom-weights load silently falls back to
    yolov8n.pt rather than leaving people_model as None (see
    PEOPLE_DETECTOR.load_people_model).
    """
    import PEOPLE_DETECTOR
    monkeypatch.setattr(MAIN, "fire_model", MagicMock())
    monkeypatch.setattr(MAIN, "people_model", MagicMock())
    monkeypatch.setattr(PEOPLE_DETECTOR, "_using_fallback", True)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["people_model_loaded"] is True
    assert body["people_model_using_fallback"] is True
