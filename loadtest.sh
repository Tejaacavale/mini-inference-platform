#!/usr/bin/env bash
#
# Load test the /v1/predict endpoint with `hey`.
#
# Sends 5000 requests at concurrency 50. Run this while watching the Grafana
# dashboard: you should see batch_size climb (the batcher packs concurrent
# requests together) while p99 latency stays bounded.
#
# Usage:
#   ./loadtest.sh [URL]
# Default URL assumes `kubectl port-forward svc/inference 8000:80`.

set -euo pipefail

URL="${1:-http://localhost:8000/v1/predict}"

echo "Load testing ${URL} with 5000 requests @ concurrency 50..."

hey -n 5000 -c 50 \
  -m POST \
  -T "application/json" \
  -d '{"text":"this movie was absolutely fantastic and I loved every minute"}' \
  "${URL}"
