from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from autotagger.autotagger import Autotagger, _api_tag_name


def bare_tagger(*, size=2, tags=None):
    tagger = Autotagger.__new__(Autotagger)
    tagger.target_size = size
    tagger.tag_names = tags or []
    return tagger


@pytest.mark.parametrize(
    ("name", "category", "expected"),
    [
        ("general", 9, "rating:g"),
        ("sensitive", 9, "rating:s"),
        ("questionable", 9, "rating:q"),
        ("explicit", 9, "rating:e"),
        ("1girl", 0, "1girl"),
    ],
)
def test_api_tag_name_preserves_rating_convention(name, category, expected):
    assert _api_tag_name(name, category) == expected


def test_prepare_image_uses_nchw_imagenet_normalization_and_white_padding():
    tagger = bare_tagger()
    image = Image.new("RGB", (2, 1), "red")

    result = tagger._prepare_image(image)

    assert result.shape == (3, 2, 2)
    assert result.dtype == np.float32
    expected_red = (np.asarray([1.0, 0.0, 0.0]) - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    expected_white = (np.ones(3) - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    np.testing.assert_allclose(result[:, 0, 0], expected_red, rtol=1e-5)
    np.testing.assert_allclose(result[:, 1, 0], expected_white, rtol=1e-5)


def test_prepare_image_composites_transparency_onto_white():
    tagger = bare_tagger(size=1)
    image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    result = tagger._prepare_image(image)

    expected_white = (np.ones(3) - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    np.testing.assert_allclose(result[:, 0, 0], expected_white, rtol=1e-5)


def test_predict_applies_sigmoid_threshold_sort_and_limit():
    tagger = bare_tagger(tags=["low", "middle", "high"])
    tagger._input_name = "pixel_values"
    tagger._infer_lock = __import__("threading").Lock()
    tagger._prepare_image = lambda image: image

    class FakeInferRequest:
        def infer(self, inputs):
            assert inputs["pixel_values"].shape == (1, 3, 2, 2)

        def get_output_tensor(self, index):
            assert index == 0
            return SimpleNamespace(
                data=np.asarray([[-3.0, 0.0, 3.0]], dtype=np.float32)
            )

    tagger._infer_request = FakeInferRequest()
    image = np.zeros((3, 2, 2), dtype=np.float32)

    result = list(tagger.predict([image], threshold=0.1, limit=2, bs=1))

    assert list(result[0]) == ["high", "middle"]
    assert result[0]["high"] == pytest.approx(0.95257413)
    assert result[0]["middle"] == pytest.approx(0.5)


def test_predict_rejects_invalid_batch_size():
    tagger = bare_tagger()
    with pytest.raises(ValueError, match="bs"):
        list(tagger.predict([object()], bs=0))
