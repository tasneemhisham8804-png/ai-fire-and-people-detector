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

from transformers import pipeline
import config
logger = logging.getLogger("ai_detector")

_detector = None
_load_attempted = False

def _get_model():
    """
    Lazily loads the HuggingFace pipeline on first use and caches it in a
    module-level global, rather than loading it at import time or on every
    call. Loading downloads/initializes the model from the HuggingFace hub,
    which is slow and needs network access on first run ,doing this lazily
    doesn't pay that cost, and
    doing it once means later frames/requests reuse the already-loaded model
    instead of reloading it each time.

    _load_attempted (separate from _detector being None) ensures a failed
    load is only attempted once per process, not retried on every single
    frame/request if the HF hub is unreachable, retrying the slow download
    on every request would make every subsequent frame's prediction slow
    for no benefit, since the failure reason isn't going to change mid-run.
    """
    global _detector, _load_attempted
    if _load_attempted:
        return _detector
    _load_attempted = True
    try:
        _detector = pipeline("image-classification",
                            model="umm-maybe/AI-image-detector")
        logger.info("HuggingFace AI detector loaded")
    except Exception as e:
        logger.error("AI detector failed to load: %s", e)
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
    """
    model = _get_model()
    if model is None:
        return None
    try:
        result = model(frame_path)[0]
        if result['label'] == 'artificial':
            return round(result['score'], 4)
        else:
            return round(1 - result['score'], 4)
    except Exception as e:
        logger.error("AI prediction failed: %s", e)
        return None
