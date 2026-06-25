# ---- Stage 1: build deps into a venv ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Create an isolated virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# CPU-only torch wheels keep the image small (no CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install -r requirements.txt

# Pre-download the model into the venv stage's HF cache so the runtime image
# ships with weights baked in (no network needed at container start).
COPY app/ ./app/
RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)"

# ---- Stage 2: slim runtime ----
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1

# Non-root user.
RUN useradd --create-home --uid 1000 appuser

# Copy the prepared venv and the pre-downloaded model cache.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

WORKDIR /app
COPY app/ ./app/

RUN chown -R appuser:appuser /home/appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
