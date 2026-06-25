"""Model loading and prediction.

Wraps the HuggingFace `distilbert-base-uncased-finetuned-sst-2-english`
sentiment pipeline. CPU only — no GPU is used or required.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Module-level handle to the loaded pipeline. Populated by load_model().
_pipeline = None


def load_model():
    """Load the sentiment-analysis pipeline onto CPU and cache it module-side.

    Returns the loaded pipeline. Safe to call more than once — subsequent
    calls return the already-loaded instance.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    # Imported lazily so importing this module (e.g. in tests that monkeypatch
    # predict) does not require torch/transformers to be installed.
    import torch
    from transformers import pipeline

    # Pin torch's intra-op thread pool. By default torch grabs one thread per
    # host core, which is disastrous under a Kubernetes CPU limit: every replica
    # spawns ~Ncore threads, they all contend for the same node cores, and CFS
    # throttling turns that contention into multi-second latency and failed
    # health probes. Capping threads to the pod's CPU budget keeps inference
    # fast and predictable. Honors TORCH_NUM_THREADS if set (default 1).
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    torch.set_num_threads(n_threads)
    logger.info("torch intra-op threads set to %d", n_threads)

    logger.info("Loading model %s on CPU...", MODEL_NAME)
    _pipeline = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=-1,  # CPU
    )
    logger.info("Model %s loaded.", MODEL_NAME)
    return _pipeline


def is_loaded() -> bool:
    """Whether the model has been loaded."""
    return _pipeline is not None


def predict(texts: list[str]) -> list[dict]:
    """Run sentiment classification on a list of texts.

    Returns a list of dicts shaped like ``{"label": str, "score": float}``,
    one per input text and in the same order.
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    raw = _pipeline(texts)
    # The HF pipeline returns a list of {"label", "score"} dicts. Normalise
    # so callers always get plain floats.
    return [{"label": r["label"], "score": float(r["score"])} for r in raw]
