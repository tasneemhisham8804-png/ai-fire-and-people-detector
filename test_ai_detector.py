"""
Tests for AI_DETECTOR.py's model-loading/retry logic and its label
normalization in predict_ai_generated. Previously this module had no
dedicated tests at all — its behavior was only exercised indirectly through
test_main.py's monkeypatches, which never touched the real _get_model()/
predict_ai_generated() code paths. These tests use conftest.py's stubbed
`transformers` module directly, monkeypatching AI_DETECTOR.pipeline itself
(the module-level name AI_DETECTOR.py imported) to control what "loading
the model" does, without needing network access or the real HuggingFace
model.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import AI_DETECTOR
import config


def _reset_module_state(monkeypatch):
    """
    AI_DETECTOR.py tracks load state in module-level globals (_detector,
    _last_load_attempt), which persist across tests unless explicitly
    reset. Every test below starts from a clean slate via this helper
    rather than relying on test execution order.
    """
    monkeypatch.setattr(AI_DETECTOR, "_detector", None)
    monkeypatch.setattr(AI_DETECTOR, "_last_load_attempt", None)


def test_successful_load_is_cached_and_not_reloaded(monkeypatch):
    """Once _get_model() successfully loads a model, a second call must return the SAME cached object rather than calling pipeline() again."""
    _reset_module_state(monkeypatch)
    call_count = {"n": 0}

    def _fake_pipeline(*args, **kwargs):
        call_count["n"] += 1
        return object()

    monkeypatch.setattr(AI_DETECTOR, "pipeline", _fake_pipeline)

    model1 = AI_DETECTOR._get_model()
    model2 = AI_DETECTOR._get_model()

    assert model1 is not None
    assert model1 is model2
    assert call_count["n"] == 1


def test_failed_load_returns_none(monkeypatch):
    """A pipeline() call that raises should leave _get_model() returning None, not propagate the exception up to callers."""
    _reset_module_state(monkeypatch)

    def _broken_pipeline(*args, **kwargs):
        raise RuntimeError("simulated: HF hub unreachable")

    monkeypatch.setattr(AI_DETECTOR, "pipeline", _broken_pipeline)

    assert AI_DETECTOR._get_model() is None


def test_failed_load_is_not_retried_within_cooldown(monkeypatch):
    """
    Immediately after a failed load, a second call within
    config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS must NOT attempt to call
    pipeline() again — retrying on every single call would make every
    frame's prediction slow whenever the HF hub is unreachable.
    """
    _reset_module_state(monkeypatch)
    call_count = {"n": 0}

    def _broken_pipeline(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(AI_DETECTOR, "pipeline", _broken_pipeline)
    monkeypatch.setattr(config, "AI_DETECTOR_RETRY_COOLDOWN_SECONDS", 600)

    AI_DETECTOR._get_model()
    AI_DETECTOR._get_model()
    AI_DETECTOR._get_model()

    assert call_count["n"] == 1  # only the first call actually tried to load


def test_failed_load_is_retried_after_cooldown_elapses(monkeypatch):
    """
    This is the actual fix over the old one-shot `_load_attempted` flag:
    once the cooldown window has passed, the NEXT call must attempt to
    load again rather than being permanently stuck returning None for the
    rest of the process's lifetime.
    """
    _reset_module_state(monkeypatch)
    call_count = {"n": 0}

    def _flaky_pipeline(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated: transient failure on first attempt")
        return object()  # succeeds on retry

    monkeypatch.setattr(AI_DETECTOR, "pipeline", _flaky_pipeline)
    monkeypatch.setattr(config, "AI_DETECTOR_RETRY_COOLDOWN_SECONDS", 0.05)

    assert AI_DETECTOR._get_model() is None  # first attempt fails
    time.sleep(0.06)  # let the (short, test-only) cooldown elapse
    model = AI_DETECTOR._get_model()  # should retry now

    assert model is not None
    assert call_count["n"] == 2


def test_predict_normalizes_artificial_label(monkeypatch):
    """When the pipeline's top label is 'artificial', its score IS the P(artificial) returned directly."""
    _reset_module_state(monkeypatch)
    monkeypatch.setattr(AI_DETECTOR, "_detector", lambda path: [{"label": "artificial", "score": 0.87}])

    result = AI_DETECTOR.predict_ai_generated("fake/path.jpg")
    assert result == 0.87


def test_predict_normalizes_human_label(monkeypatch):
    """When the pipeline's top label is 'human', P(artificial) is the COMPLEMENT of that score (1 - score), not the raw score itself."""
    _reset_module_state(monkeypatch)
    monkeypatch.setattr(AI_DETECTOR, "_detector", lambda path: [{"label": "human", "score": 0.9}])

    result = AI_DETECTOR.predict_ai_generated("fake/path.jpg")
    assert result == 0.1


def test_predict_returns_none_when_model_unavailable(monkeypatch):
    """If the model failed to load, predict_ai_generated must return None (not 0.0 or raise) — None specifically means 'no signal,' which callers (MAIN.py) must be able to tell apart from a confident 'not AI-generated' score."""
    _reset_module_state(monkeypatch)
    monkeypatch.setattr(AI_DETECTOR, "pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    result = AI_DETECTOR.predict_ai_generated("fake/path.jpg")
    assert result is None


def test_predict_returns_none_on_inference_error(monkeypatch):
    """An exception raised DURING inference (not during loading) should also come back as None, not propagate and crash the caller's request."""
    _reset_module_state(monkeypatch)

    def _broken_call(path):
        raise RuntimeError("simulated: corrupt frame")

    monkeypatch.setattr(AI_DETECTOR, "_detector", _broken_call)

    result = AI_DETECTOR.predict_ai_generated("fake/path.jpg")
    assert result is None
