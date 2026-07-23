#!/usr/bin/env python3

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import openvino as ov
from huggingface_hub import HfApi, snapshot_download


FILES = ("model.onnx", "model.onnx_data", "selected_tags.csv", "config.json")
EXPECTED_TAGS_SHA256 = (
    "c08f1359510e09355728d2a61e7a29e38ace8533b0f1018b18da493db723fecd"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the DBV4 ONNX model and save an FP16 OpenVINO IR."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    import hashlib

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="autotagger-model-") as temp_dir:
        snapshot = Path(
            snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                local_dir=temp_dir,
                allow_patterns=list(FILES),
            )
        )

        with (snapshot / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
        input_size = config.get("test_input_size") or config["input_size"]

        tags_digest = hashlib.sha256(
            (snapshot / "selected_tags.csv").read_bytes()
        ).hexdigest()
        if tags_digest != EXPECTED_TAGS_SHA256:
            raise RuntimeError(
                f"Unexpected selected_tags.csv digest: {tags_digest}"
            )

        model = ov.convert_model(snapshot / "model.onnx")
        model.reshape(
            {
                model.input(0).get_any_name(): [
                    -1,
                    int(input_size[0]),
                    int(input_size[1]),
                    int(input_size[2]),
                ]
            }
        )
        ov.save_model(model, args.output / "model.xml", compress_to_fp16=True)

        for filename in ("selected_tags.csv", "config.json"):
            shutil.copy2(snapshot / filename, args.output / filename)

    resolved_revision = HfApi().model_info(
        args.repo, revision=args.revision
    ).sha
    provenance = {
        "conversion": "OpenVINO FP16 IR (compressed FP16 weights)",
        "source_repo": args.repo,
        "source_revision": resolved_revision,
        "upstream_model": "animetimm/convnextv2_huge.dbv4-full",
    }
    with (args.output / "model-provenance.json").open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
