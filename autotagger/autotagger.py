import csv
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image, ImageOps


DEFAULT_MODEL_REPO = "itterative/convnextv2_huge.dbv4-full-onnx"
MODEL_FILENAMES = ("model.xml", "model.onnx")
MODEL_DATA_FILENAME = "model.onnx_data"
LABEL_FILENAME = "selected_tags.csv"
CONFIG_FILENAME = "config.json"

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
def _resolve_local_model(directory: Path, explicit_model: Path | None = None):
    model_path = explicit_model
    if model_path is None:
        model_path = next(
            (directory / name for name in MODEL_FILENAMES if (directory / name).is_file()),
            None,
        )

    if model_path is None:
        expected = " or ".join(MODEL_FILENAMES)
        raise FileNotFoundError(f"No {expected} found in {directory}")

    labels_path = directory / LABEL_FILENAME
    config_path = directory / CONFIG_FILENAME
    for path in (labels_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    if model_path.suffix == ".onnx":
        data_path = directory / MODEL_DATA_FILENAME
        if not data_path.is_file():
            raise FileNotFoundError(f"Required ONNX external data file not found: {data_path}")

    return model_path, labels_path, config_path


def resolve_model_files(model_path, revision=None):
    """Resolve a local model path or download a Hugging Face model snapshot."""
    candidate = Path(model_path)
    if candidate.is_file():
        return _resolve_local_model(candidate.parent, candidate)
    if candidate.is_dir():
        return _resolve_local_model(candidate)

    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=str(model_path),
            revision=revision,
            allow_patterns=[
                *MODEL_FILENAMES,
                MODEL_DATA_FILENAME,
                LABEL_FILENAME,
                CONFIG_FILENAME,
            ],
        )
    )
    return _resolve_local_model(snapshot_path)


def _api_tag_name(name: str, category: int) -> str:
    if category != 9:
        return name
    return f"rating:{name}"


def _load_tag_names(labels_path: Path) -> list[str]:
    with labels_path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [_api_tag_name(row["name"], int(row["category"])) for row in rows]


def _load_target_size(config_path: Path) -> int:
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    input_size = config.get("test_input_size") or config.get("input_size")
    if (
        not isinstance(input_size, list)
        or len(input_size) != 3
        or input_size[1] != input_size[2]
    ):
        raise ValueError(f"Unsupported model input size in {config_path}: {input_size!r}")
    return int(input_size[1])


class Autotagger:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_REPO,
        *,
        device=None,
        model_revision=None,
        openvino_cache_dir=None,
        performance_hint=None,
        inference_precision=None,
    ):
        """Load the ConvNeXt V2 DBV4 tagger with OpenVINO.

        ``model_path`` may be a Hugging Face repo ID, a directory containing
        the model files, or a direct path to ``model.xml``/``model.onnx``.
        """
        model_revision = model_revision or os.getenv("MODEL_REVISION")
        model_path, labels_path, config_path = resolve_model_files(
            model_path, revision=model_revision
        )

        self.tag_names = _load_tag_names(labels_path)
        self.target_size = _load_target_size(config_path)

        core = ov.Core()
        model = core.read_model(model_path)
        input_name = model.input(0).get_any_name()
        model.reshape({input_name: [-1, 3, self.target_size, self.target_size]})

        output_shape = model.output(0).get_partial_shape()
        if output_shape.rank.get_length() != 2:
            raise ValueError(f"Unsupported model output shape: {output_shape}")
        if output_shape[1].is_static and output_shape[1].get_length() != len(
            self.tag_names
        ):
            raise ValueError(
                f"Model emits {output_shape[1].get_length()} scores, "
                f"but {labels_path} contains {len(self.tag_names)} tags"
            )

        device = device or os.getenv("AUTOTAGGER_DEVICE", "AUTO")
        openvino_cache_dir = openvino_cache_dir or os.getenv(
            "AUTOTAGGER_OPENVINO_CACHE_DIR", "/tmp/autotagger-openvino-cache"
        )
        performance_hint = performance_hint or os.getenv(
            "AUTOTAGGER_PERFORMANCE_HINT", "LATENCY"
        )
        # OpenVINO's GPU plugin defaults to FP16 execution. ConvNeXt V2 Huge
        # produces NaN logits in that mode on Meteor Lake, while FP32 execution
        # is stable and still uses the compressed FP16 weights from the IR.
        inference_precision = inference_precision or os.getenv(
            "AUTOTAGGER_INFERENCE_PRECISION", "f32"
        )
        compile_options = {
            "CACHE_DIR": openvino_cache_dir,
            "PERFORMANCE_HINT": performance_hint,
            "INFERENCE_PRECISION_HINT": inference_precision,
        }

        self._compiled_model = core.compile_model(model, device, compile_options)
        self._input_name = self._compiled_model.input(0).get_any_name()
        self._infer_request = self._compiled_model.create_infer_request()
        self._infer_lock = threading.Lock()

        try:
            execution_devices = self._compiled_model.get_property("EXECUTION_DEVICES")
            self.execution_devices = (
                (execution_devices,)
                if isinstance(execution_devices, str)
                else tuple(execution_devices)
            )
        except RuntimeError:
            self.execution_devices = (device,)
        logging.info(
            "Loaded %s on OpenVINO device(s): %s",
            model_path,
            ", ".join(self.execution_devices),
        )

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """Apply the model's official white-pad and ImageNet normalization."""
        image = ImageOps.exif_transpose(image)

        # Composite transparency onto the white background used in training.
        rgba = image.convert("RGBA")
        flattened = Image.new("RGBA", rgba.size, "white")
        flattened.alpha_composite(rgba)
        image = flattened.convert("RGB")

        width, height = image.size
        scale = min(self.target_size / width, self.target_size / height)
        resized_size = (round(width * scale), round(height * scale))
        if image.size != resized_size:
            image = image.resize(resized_size, Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.target_size, self.target_size), "white")
        offset = (
            (self.target_size - image.width) // 2,
            (self.target_size - image.height) // 2,
        )
        canvas.paste(image, offset)

        pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))

    def _predict_batch(self, images) -> np.ndarray:
        batch = np.stack([self._prepare_image(image) for image in images])
        # A single InferRequest is not thread-safe. Copy its output before
        # releasing the lock because the next request reuses the same buffer.
        with self._infer_lock:
            self._infer_request.infer({self._input_name: batch})
            logits = self._infer_request.get_output_tensor(0).data.copy()

        if not np.isfinite(logits).all():
            raise RuntimeError(
                "Model returned non-finite logits. "
                "Use AUTOTAGGER_INFERENCE_PRECISION=f32 on Intel GPU."
            )

        # This ONNX conversion exposes raw classifier logits.
        clipped = np.clip(logits, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _format_scores(self, scores, threshold, limit):
        indices = np.flatnonzero(scores >= threshold)
        if not len(indices) or limit == 0:
            return {}

        if limit is not None and len(indices) > limit:
            selected = np.argpartition(scores[indices], -limit)[-limit:]
            indices = indices[selected]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return {self.tag_names[index]: float(scores[index]) for index in indices}

    def predict(self, images, threshold=0.01, limit=50, bs=64):
        """Yield a score-sorted ``{tag: score}`` mapping for each image."""
        images = list(images)
        if not images:
            return
        if bs <= 0:
            raise ValueError("bs must be greater than zero")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        for batch_start in range(0, len(images), bs):
            batch_images = images[batch_start : batch_start + bs]
            predictions = self._predict_batch(batch_images)
            for scores in predictions:
                yield self._format_scores(scores, threshold, limit)
