"""
AI-generated footage detection module.

Wraps a pretrained HuggingFace image-classification pipeline
("umm-maybe/AI-image-detector", a ViT-based real-vs-AI-generated
classifier) rather than a custom-trained local model. An earlier local
EfficientNet-B0/CIFAKE-trained model didn't generalize well from its
training set (small, low-resolution generated-image benchmarks) to full-
resolution real-world video frames, so this project uses a model trained on
a broader mix of generated-image sources instead.

"""
import logging
import threading
import time

from transformers import pipeline
import config
logger = logging.getLogger("ai_detector")

_detector = None
_last_load_attempt = None  # monotonic timestamp of the most recent load attempt, success or failure
# Serializes calls into the HF pipeline itself (see predict_ai_generated) —
# not the load, which is already effectively single-shot via the cooldown
# check below.
_inference_lock = threading.Lock()

def _get_model():
    """
    Lazily loads the HuggingFace pipeline on first use and caches it in a
    module-level global, rather than loading it at import time or on every
    call. Loading downloads/initializes the model from the HuggingFace hub,
    which is slow and needs network access on first run; doing this lazily
    avoids paying that cost until the model is actually needed, and doing
    it once means later frames/requests reuse the already-loaded model
    instead of reloading it each time.

    A failed load is retried at most once per config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS
    (tracked via _last_load_attempt), rather than either retrying on every
    single call (which would make every frame's prediction slow whenever the
    HF hub is unreachable) or never retrying again for the rest of the
    process's lifetime (the previous behavior — a one-shot `_load_attempted`
    flag meant a transient failure at startup disabled the AI-generated
    check permanently, with no recovery short of restarting the server).
    This matters more now than it used to: the AI-generated check gates
    whether fire/people detection even runs (see MAIN.py's run_analysis), so
    a stuck-off check silently reverts every subsequent video to the more
    expensive always-run-everything path.
    """
    global _detector, _last_load_attempt
    if _detector is not None:
        return _detector

    now = time.monotonic()
    if _last_load_attempt is not None and (now - _last_load_attempt) < config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS:
        return None

    _last_load_attempt = now
    try:
        _detector = pipeline("image-classification",
                            model="umm-maybe/AI-image-detector")
        logger.info("HuggingFace AI detector loaded")
    except Exception as e:
        logger.error("AI detector failed to load, will retry in %ds: %s",
                      config.AI_DETECTOR_RETRY_COOLDOWN_SECONDS, e)
        _detector = None
    return _detector

def predict_ai_generated(frame_path: str) -> float | None:
    """
    Returns the probability that a single frame is AI-generated, as a float
    in [0, 1], or None if the model isn't available (failed to load — e.g.
    no network on first run, HF hub unreachable).

    The underlying pipeline's output label is either 'artificial' or
    'human' with an associated confidence for whichever label it picked;
    this normalizes both cases to "P(artificial)" so callers always get a
    single consistent probability regardless of which label the model
    happened to return.

    Callers (MAIN.py) must treat None as "check unavailable" and not as a
    score of 0.0 — those mean very different things (no information vs.
    "confidently real").

    The actual pipeline call is wrapped in a lock: run_analysis() calls
    this once per sampled frame from a single thread, but multiple
    concurrent /analyze requests each land in their own threadpool worker
    thread (see MAIN.py) and would otherwise all call into this same
    shared pipeline object at once. Most inference runtimes tolerate that
    fine, but HF pipelines wrapping a PyTorch model aren't guaranteed
    thread-safe for concurrent forward passes on shared internal state —
    serializing calls here trades a little throughput under concurrent load
    for not having to verify that guarantee.
    """
    model = _get_model()
    if model is None:
        return None
    try:
        with _inference_lock:
            result = model(frame_path)[0]
        if result['label'] == 'artificial':
            return round(result['score'], 4)
        else:
            return round(1 - result['score'], 4)
    except Exception as e:
        logger.error("AI prediction failed: %s", e)
        return None
