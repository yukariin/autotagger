# ---- Stage 1: Build (download + convert to FP16) ----
FROM python:3.11-slim AS builder
WORKDIR /autotagger

ENV HF_HOME=/autotagger/models

RUN pip install --no-cache-dir huggingface-hub onnx && \
    python - <<'EOF'
import huggingface_hub
from onnxruntime.transformers import float16
from pathlib import Path

repo = "SmilingWolf/wd-eva02-large-tagger-v3"
onnx_path = huggingface_hub.hf_hub_download(repo, "model.onnx")
huggingface_hub.hf_hub_download(repo, "selected_tags.csv")

fp16_path = Path(onnx_path).parent / "model_fp16.onnx"
print(f"Converting {onnx_path} -> {fp16_path}")
model_fp16 = float16.convert_float_to_float16(onnx_path)
import onnx
onnx.save(model_fp16, str(fp16_path))
print(f"Done. FP16 model: {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")
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
