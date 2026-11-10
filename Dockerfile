# syntax=docker/dockerfile:1
# Container image for the swing-setup screener.
#
# The service reads secrets from the environment (.env) and its settings from
# config.toml. Neither is baked into the image: mount them at runtime and point
# SCREENER_CONFIG at the mounted TOML (see the docker-* targets in the Makefile).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/New_York

# libgomp1     -> OpenMP runtime required by xgboost / scikit-learn
# tzdata       -> zoneinfo needs IANA data for America/New_York (16:10 ET scan)
# ca-certs     -> yfinance HTTPS calls
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the project sources first so the editable install sees the packages.
# The runtime state (setup_bundle_v1.pkl, signal_log.jsonl, decision_params.json)
# is cwd-relative, so it lives in /data, not here.
COPY pyproject.toml __init__.py __main__.py screener.py ./
COPY agent ./agent
COPY train ./train
COPY config ./config
COPY notifications ./notifications
COPY backtest ./backtest
RUN pip install -e .

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown -R app:app /data
USER app

# Keep the model cache and logs on a volume; override with -v <dir>:/data.
VOLUME ["/data"]

# Defaults to the config baked into the image. The Makefile mounts the host
# config/config.toml at /config/config.toml and overrides this variable.
ENV SCREENER_CONFIG=/app/config/config.toml

WORKDIR /data

ENTRYPOINT ["python", "/app/__main__.py"]
