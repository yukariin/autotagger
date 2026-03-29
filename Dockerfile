# ---- Stage 1: Build (download + quantize) ----
FROM python:3.11-slim AS builder
WORKDIR /autotagger

ENV HF_HOME=/autotagger/models

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --no-install-project --extra quantize

# Download model and produce INT8 quantized copy.
RUN .venv/bin/python - <<'EOF'
import huggingface_hub
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

repo = "SmilingWolf/wd-eva02-large-tagger-v3"
onnx_path = huggingface_hub.hf_hub_download(repo, "model.onnx")
huggingface_hub.hf_hub_download(repo, "selected_tags.csv")

int8_path = Path(onnx_path).parent / "model_int8.onnx"
print(f"Quantizing {onnx_path} -> {int8_path}")
quantize_dynamic(
    onnx_path,
    str(int8_path),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gather"],
)
print(f"Done. INT8 model: {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
EOF

# Strip FP32 model when building int8-only image.
ARG INT8_ONLY=false
RUN if [ "$INT8_ONLY" = "true" ]; then \
      find "$HF_HOME" -name model.onnx -not -name model_int8.onnx -delete; \
    fi

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
