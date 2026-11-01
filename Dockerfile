# syntax=docker/dockerfile:1.7

# ---- builder: resolve and install dependencies into a virtualenv -------------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build

# CPU-only PyTorch: ~2 GB smaller than the default CUDA wheel and enough for serving.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui]"

# ---- runtime: slim image, non-root user ----------------------------------------------------
FROM python:3.11-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app
COPY --chown=app:app configs ./configs
COPY --chown=app:app data/sample ./data/sample
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app .streamlit ./.streamlit
RUN mkdir -p artifacts && chown app:app artifacts

USER app
EXPOSE 8000 8501

# The first start downloads the models, hence the generous start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "nq_qa.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
