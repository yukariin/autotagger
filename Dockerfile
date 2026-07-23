# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.11.31
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-build
FROM --platform=$TARGETPLATFORM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-target


# Ubuntu 24.04's stock compute runtime predates Meteor Lake support through the
# Xe kernel driver. Pin Intel's production-tested runtime and verify every
# package while BuildKit downloads it.
FROM --platform=$TARGETPLATFORM scratch AS intel-gpu-runtime

ADD --checksum=sha256:0ce1e715ec5bf30b1304b42acfbda34cc514a0f616dd48c6a31973c4075d8d09 \
  https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb \
  /packages/intel-igc-core-2_2.34.4+21428_amd64.deb
ADD --checksum=sha256:09a71e8b6ad432ed0511b09fa4f580c1b44534a60ccb0545212507bf67dc0d1b \
  https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb \
  /packages/intel-igc-opencl-2_2.34.4+21428_amd64.deb
ADD --checksum=sha256:f36cb8a6899353c61cc7261b650f287f4a652acadaad859103bdfc51f93b6e8a \
  https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-ocloc_26.18.38308.1-0_amd64.deb \
  /packages/intel-ocloc_26.18.38308.1-0_amd64.deb
ADD --checksum=sha256:b2d0c924e56b3f9e5837774d68b0c67461b8633035d93ca18b1a8e3e5ead15fa \
  https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd_26.18.38308.1-0_amd64.deb \
  /packages/intel-opencl-icd_26.18.38308.1-0_amd64.deb
ADD --checksum=sha256:6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54 \
  https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb \
  /packages/libigdgmm12_22.10.0_amd64.deb
ADD --checksum=sha256:12b8254e6d3415c32cee9cd13943030b991d91212445c79fe1cc27176a72eca4 \
  https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb \
  /packages/libze-intel-gpu1_26.18.38308.1-0_amd64.deb


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

# The OpenVINO wheel contains the GPU plugin. Intel's compute runtime provides
# both OpenCL and Level Zero backends for the DRA-injected /dev/dri device.
COPY --from=intel-gpu-runtime /packages /tmp/intel-gpu
RUN \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
    libze1 \
    ocl-icd-libopencl1 \
    python3 \
    python3-venv \
    tini && \
  apt-get install -y --no-install-recommends /tmp/intel-gpu/*.deb && \
  rm -rf /tmp/intel-gpu && \
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
  AUTOTAGGER_INFERENCE_PRECISION=f16 \
  AUTOTAGGER_FP16_GUARD_BOUND=65000 \
  AUTOTAGGER_FP16_SATURATION_GUARD=true \
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
