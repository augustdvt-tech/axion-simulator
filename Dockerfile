# =============================================================================
# Axion AI — API server image
# =============================================================================
#
# Multi-stage so the runtime image doesn't carry build toolchains.
# Non-root user, embedded healthcheck, deterministic uvicorn entrypoint.
#
# Build:
#   docker build -t axion-api:latest .
#
# Run standalone:
#   docker run --rm -p 8000:8000 axion-api:latest
#
# In docker compose: see docker-compose.yml — the `axion-api` service
# wires AXION_DB_URL, MLFLOW_TRACKING_URI and waits for timescaledb.
# =============================================================================

# ---- builder stage ---------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# psycopg2-binary still wants build-essential headers to satisfy some wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install into a venv so we can copy the entire site-packages out cleanly
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


# ---- runtime stage ---------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# libpq for psycopg2 runtime, curl for the HEALTHCHECK probe.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -r axion && useradd -r -g axion -u 1000 axion

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . /app

# Drop to non-root for everything below.
RUN chown -R axion:axion /app
USER axion

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
