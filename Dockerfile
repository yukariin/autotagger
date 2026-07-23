# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.11.31
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-build
FROM --platform=$TARGETPLATFORM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-target


# Model conversion runs natively on the build host. OpenVINO IR is portable,
# so Apple Silicon can prepare the model without emulating amd64.
FROM --platform=$BUILDPLATFORM python:3.12-slim-bookworm AS model-builder

WORKDIR /build

ENV \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=from=uv-build,source=/uv,target=/usr/local/bin/uv \
  uv sync --frozen --no-dev --no-install-project

ARG MODEL_REPO=itterative/convnextv2_huge.dbv4-full-onnx
ARG MODEL_REVISION=b266f136697bf0f6c37dbf818c47e550cf7e8f6a

COPY scripts/prepare_model.py /tmp/prepare_model.py
RUN --mount=type=secret,id=HF_TOKEN,required=false \
  HF_TOKEN_PATH=/run/secrets/HF_TOKEN \
  .venv/bin/python /tmp/prepare_model.py \
    --repo "${MODEL_REPO}" \
    --revision "${MODEL_REVISION}" \
    --output /opt/autotagger/model


FROM --platform=$TARGETPLATFORM ubuntu:24.04 AS runtime

WORKDIR /opt/autotagger/app

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  UV_LINK_MODE=copy \
  PATH="/opt/autotagger/app/.venv/bin:/opt/autotagger/app:${PATH}"

# The OpenVINO wheel contains the GPU plugin. These system packages provide
# the OpenCL loader and Intel compute runtime required to reach /dev/dri.
RUN \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    ocl-icd-libopencl1 \
    python3 \
    python3-venv \
    tini && \
  rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=from=uv-target,source=/uv,target=/usr/local/bin/uv \
  uv sync --python /usr/bin/python3 --frozen --no-dev --no-install-project && \
  rm -rf /root/.cache /root/.local

RUN \
  groupadd --gid 10001 autotagger && \
  useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin autotagger

ENV \
  AUTOTAGGER_DEVICE=AUTO \
  AUTOTAGGER_OPENVINO_CACHE_DIR=/tmp/autotagger-openvino-cache \
  AUTOTAGGER_PERFORMANCE_HINT=LATENCY \
  HF_HOME=/tmp/autotagger-huggingface \
  MODEL_PATH=/opt/autotagger/model

COPY --from=model-builder --chown=10001:10001 /opt/autotagger/model /opt/autotagger/model
COPY --chown=10001:10001 . .

USER 10001:10001

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
CMD ["gunicorn"]
