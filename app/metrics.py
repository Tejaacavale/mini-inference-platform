"""Prometheus metrics for the inference API."""
from prometheus_client import Counter, Gauge, Histogram

# Total requests, partitioned by endpoint and resulting HTTP status.
requests_total = Counter(
    "requests_total",
    "Total number of requests received.",
    ["endpoint", "status"],
)

# End-to-end request latency in seconds, per endpoint.
request_latency_seconds = Histogram(
    "request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
)

# Size of each batch handed to model.predict() by the batcher.
batch_size = Histogram(
    "batch_size",
    "Number of items in each model inference batch.",
    buckets=(1, 2, 4, 8, 16, 24, 32),
)

# Number of requests currently being processed (in flight).
inflight_requests = Gauge(
    "inflight_requests",
    "Number of in-flight requests currently being processed.",
)
