# ---- Stage 1: Build (download model) ----
FROM python:3.11-slim AS builder
WORKDIR /autotagger

ENV HF_HOME=/autotagger/models

RUN pip install --no-cache-dir huggingface-hub && \
    python - <<'EOF'
import huggingface_hub
repo = "SmilingWolf/wd-eva02-large-tagger-v3"
huggingface_hub.hf_hub_download(repo, "model.onnx")
huggingface_hub.hf_hub_download(repo, "selected_tags.csv")
EOF

# ---- Stage 2: Runtime ----
FROM python:3.11-slim
WORKDIR /autotagger

RUN apt-get update && \
    apt-get install -y --no-install-recommends tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH=/autotagger:$PATH \
  HF_HOME=/autotagger/models \
  MODEL_PATH=SmilingWolf/wd-eva02-large-tagger-v3

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --no-install-project

COPY --from=builder /autotagger/models /autotagger/models
COPY . .

EXPOSE 5000
ENTRYPOINT ["tini", "--", "uv", "run"]
CMD ["gunicorn"]
