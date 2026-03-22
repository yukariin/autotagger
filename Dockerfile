FROM openvino/ubuntu22_runtime:2024.1.0

USER root
WORKDIR /autotagger

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH=/autotagger:$PATH \
  # Tell huggingface_hub where to store downloaded models
  HF_HOME=/autotagger/models \
  MODEL_PATH=SmilingWolf/wd-eva02-large-tagger-v3

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    tini \
    python3-pip && \
  apt-get clean && rm -rf /var/lib/apt/lists/* && \
  pip install "poetry==1.8.5"

COPY pyproject.toml poetry.lock* ./
RUN \
  python -m poetry config virtualenvs.create false && \
  python -m poetry install --only main --no-interaction --no-ansi && \
  rm -rf /root/.cache/pypoetry

# Pre-download the ONNX model and tag vocabulary at build time so the
# container starts instantly without a network fetch at runtime.
RUN python - <<'EOF'
import huggingface_hub
repo = "SmilingWolf/wd-eva02-large-tagger-v3"
huggingface_hub.hf_hub_download(repo, "model.onnx")
huggingface_hub.hf_hub_download(repo, "selected_tags.csv")
EOF

COPY . .

EXPOSE 5000
ENTRYPOINT ["tini", "--", "poetry", "run"]
CMD ["gunicorn"]
