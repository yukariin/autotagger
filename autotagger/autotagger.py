import numpy as np
import pandas as pd
import onnxruntime as rt

from pathlib import Path
from PIL import Image

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


class Autotagger:
    def __init__(self, model_path="SmilingWolf/wd-eva02-large-tagger-v3"):
        """Load the WD EVA02-Large Tagger v3 ONNX model.

        `model_path` may be:
          - A HuggingFace repo ID (e.g. "SmilingWolf/wd-eva02-large-tagger-v3")
          - A local directory that contains model.onnx and selected_tags.csv
          - A direct path to the model.onnx file (selected_tags.csv must sit
            alongside it in the same directory)
        """
        model_path = Path(model_path)

        # Resolve actual file paths
        if model_path.is_file():
            # Direct path to the .onnx file
            onnx_path = model_path
            csv_path = model_path.parent / LABEL_FILENAME
        elif model_path.is_dir():
            # Local directory containing the model files
            onnx_path = model_path / MODEL_FILENAME
            csv_path = model_path / LABEL_FILENAME
        else:
            # Treat as a HuggingFace repo ID and download via huggingface_hub
            import huggingface_hub
            repo_id = str(model_path)
            csv_path = huggingface_hub.hf_hub_download(repo_id, LABEL_FILENAME)
            onnx_path = huggingface_hub.hf_hub_download(repo_id, MODEL_FILENAME)

        tags_df = pd.read_csv(csv_path)
        self.tag_names = [
            f"rating:{name}" if category == 9 else name
            for name, category in zip(tags_df["name"], tags_df["category"])
        ]

        # Load ONNX model
        self.model = rt.InferenceSession(str(onnx_path))
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.target_size = height  # 448 for EVA02-Large v3

        self._input_name = self.model.get_inputs()[0].name
        self._output_name = self.model.get_outputs()[0].name

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL image into the NHWC BGR float32 array the model expects."""
        # Flatten RGBA onto white background
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")

        # Pad to square with white
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        # Resize to model input size
        if max_dim != self.target_size:
            padded = padded.resize((self.target_size, self.target_size), Image.BICUBIC)

        # HWC RGB → HWC BGR float32
        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB → BGR
        return arr  # shape: (H, W, 3)

    def predict(self, images, threshold=0.01, limit=50, bs=64):
        """Yield a {tag: score} dict for each image in `images`.

        Parameters
        ----------
        images : iterable of PIL.Image.Image
        threshold : float
            Minimum confidence score to include a tag.
        limit : int
            Maximum number of tags to return per image.
        bs : int
            Batch size for ONNX inference.
        """
        images = list(images)
        if not images:
            return

        # Process in batches
        for batch_start in range(0, len(images), bs):
            batch_imgs = images[batch_start : batch_start + bs]
            batch_arr = np.stack([self._prepare_image(img) for img in batch_imgs])
            # shape: (N, H, W, 3) — NHWC as the model expects

            preds = self.model.run(
                [self._output_name], {self._input_name: batch_arr}
            )[0]  # shape: (N, num_tags)

            for scores in preds:
                # Pair every tag name with its score
                all_tags = list(zip(self.tag_names, scores.astype(float)))

                # Filter by threshold, sort descending, cap at limit
                filtered = [
                    (name, score)
                    for name, score in all_tags
                    if score >= threshold
                ]
                filtered.sort(key=lambda x: x[1], reverse=True)
                filtered = filtered[:limit]

                yield dict(filtered)
