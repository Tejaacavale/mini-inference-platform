"""FastAPI application for the mini inference platform.

Endpoints:
  POST /v1/predict        — classify a single text (goes through the batcher)
  POST /v1/predict/batch  — classify a list of texts (direct batch inference)
  GET  /healthz           — liveness: always 200
  GET  /readyz            — readiness: 200 only once the model is loaded
  GET  /metrics           — Prometheus exposition
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics, model
from app.batcher import Batcher
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and start the batcher at startup; tear down at shutdown."""
    logger.info("Startup: loading model...")
    model.load_model()
    app.state.batcher = Batcher()
    app.state.batcher.start()
    logger.info("Startup complete.")
    yield
    logger.info("Shutdown: stopping batcher...")
    await app.state.batcher.stop()


app = FastAPI(title="mini-inference-platform", lifespan=lifespan)


@app.post("/v1/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest) -> PredictResponse:
    """Classify a single text. Submitted to the batcher and awaited."""
    endpoint = "/v1/predict"
    metrics.inflight_requests.inc()
    start = time.perf_counter()
    status = "200"
    try:
        result = await app.state.batcher.submit(req.text)
        return PredictResponse(label=result["label"], score=result["score"])
    except Exception:
        status = "500"
        raise
    finally:
        metrics.inflight_requests.dec()
        metrics.request_latency_seconds.labels(endpoint=endpoint).observe(
            time.perf_counter() - start
        )
        metrics.requests_total.labels(endpoint=endpoint, status=status).inc()


@app.post("/v1/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a list of texts directly in one model call."""
    endpoint = "/v1/predict/batch"
    metrics.inflight_requests.inc()
    start = time.perf_counter()
    status = "200"
    try:
        import asyncio

        results = await asyncio.to_thread(model.predict, req.texts)
        metrics.batch_size.observe(len(req.texts))
        return BatchPredictResponse(
            results=[PredictResponse(label=r["label"], score=r["score"]) for r in results]
        )
    except Exception:
        status = "500"
        raise
    finally:
        metrics.inflight_requests.dec()
        metrics.request_latency_seconds.labels(endpoint=endpoint).observe(
            time.perf_counter() - start
        )
        metrics.requests_total.labels(endpoint=endpoint, status=status).inc()


@app.get("/healthz")
async def healthz() -> Response:
    """Liveness probe — always 200 while the process is up."""
    return Response(status_code=200, content="ok")


@app.get("/readyz")
async def readyz() -> Response:
    """Readiness probe — 200 only once the model is loaded, else 503."""
    if model.is_loaded():
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="model not loaded")


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
