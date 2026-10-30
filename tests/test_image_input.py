"""In-memory image input: a PIL.Image.Image or a numpy.ndarray goes through the pipeline exactly like the same
image encoded as PNG.

No Tesseract needed: the OCR calls are recorded with the faked ocr of pipeline_recording.py.
"""
import io
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from mrzmini import imageproc as ip
from mrzmini.image import Loader, read_mrz
from pipeline_recording import (IMAGES, SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_62_B, SCORE_81, VALID, png,
                                run_read_mrz, to_luma)


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    monkeypatch.delenv('MRZMINI_TESSERACT_LANG', raising=False)


def no_image_open():
    return mock.patch('mrzmini.imageproc.imread_gray', side_effect=AssertionError('in-memory image decoded again'))


# ---------------------------------------------------------------------------
# Same OCR calls and result as the PNG
# ---------------------------------------------------------------------------
# (image of test_pipeline_parity.py, scripted OCR texts)
RGB_CASES = [
    pytest.param('card_td3_rgb', [VALID], id='valid'),
    pytest.param('card_td3_rgb', [SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_81, SCORE_62_B], id='all_variants'),
    pytest.param('a4_card_480', ['', VALID], id='then_wider'),
    pytest.param('a4_card_320', [VALID], id='wider_only'),
]
GRAY_CASES = [
    pytest.param('card_td3_gray', [VALID], id='valid'),
    pytest.param('card_td3_gray', [], id='then_wider'),
]


@pytest.mark.parametrize('image, texts', RGB_CASES)
def test_rgb_in_memory_matches_png(image, texts):
    arr = IMAGES[image]()
    assert arr.ndim == 3 and arr.dtype == np.uint8
    expected = run_read_mrz(png(arr), texts, None)
    assert expected['ocr_calls']

    with no_image_open():
        from_pil = run_read_mrz(Image.fromarray(arr), texts, None)
        from_array = run_read_mrz(arr, texts, None)

    # Fingerprints of the images tesseract reads, method and to_dict() included
    assert from_pil == expected
    assert from_array == expected


@pytest.mark.parametrize('image, texts', GRAY_CASES)
def test_gray_in_memory_matches_png(image, texts):
    arr = IMAGES[image]()
    assert arr.ndim == 2 and arr.dtype == np.uint8
    expected = run_read_mrz(png(arr), texts, None)

    with no_image_open():
        from_pil = run_read_mrz(Image.fromarray(arr), texts, None)
        from_array = run_read_mrz(arr, texts, None)

    assert from_pil == expected
    assert from_array == expected


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def rgb_sample():
    return IMAGES['card_td3_rgb']()[:120, :160]


def encode_png(im):
    """The PIL image as PNG, mode kept (palette included)."""
    out = io.BytesIO()
    im.save(out, format='PNG')
    return out.getvalue()


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'P', 'L', 'I;16'])
def test_pil_modes_match_png(mode):
    rgb = rgb_sample()
    if mode == 'RGBA':
        im = Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 128, np.uint8)]))
    elif mode == 'I;16':
        im = Image.fromarray((rgb[..., 0].astype(np.uint16) * 257))
    else:
        im = Image.fromarray(rgb).convert(mode)
    assert im.mode == mode

    expected = Loader(encode_png(im))()
    with no_image_open():
        got = Loader(im)()

    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    if mode in ('L', 'I;16'):
        assert got.dtype == np.asarray(im).dtype  # kept as is
    else:
        assert got.dtype == np.float64  # luma


@pytest.mark.parametrize('layout', [
    pytest.param(np.asfortranarray, id='fortran'),
    pytest.param(lambda a: np.moveaxis(np.ascontiguousarray(np.moveaxis(a, -1, 0)), 0, -1), id='chw_as_hwc'),
])
def test_rgb_array_layout(layout):
    rgb = IMAGES['a4_card_480']()
    arr = layout(rgb)
    assert not arr.flags.c_contiguous

    # Exactly the luma of the PNG, not only within rounding
    np.testing.assert_array_equal(Loader(arr)(), Loader(png(rgb))())


def test_rgba_array_ignores_alpha():
    rgb = rgb_sample()
    rgba = np.dstack([rgb, np.zeros(rgb.shape[:2], np.uint8)])

    got = Loader(rgba)()

    np.testing.assert_array_equal(got, Loader(rgb)())
    np.testing.assert_array_equal(got, ip.to_gray(Image.fromarray(rgba)))


def test_bool_array_matches_png():
    arr = to_luma(IMAGES['card_td3_rgb']()) > 128
    expected = run_read_mrz(png(arr), [VALID], None)
    assert expected['boxes_250'] > 0

    with no_image_open():
        assert run_read_mrz(arr, [VALID], None) == expected
        assert run_read_mrz(Image.fromarray(arr), [VALID], None) == expected


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int32, np.float32, np.float64])
def test_2d_array_keeps_dtype(dtype):
    arr = np.zeros((20, 30), dtype)

    got = Loader(arr)()

    assert got.dtype == dtype
    np.testing.assert_array_equal(got, arr)


# ---------------------------------------------------------------------------
# Unsupported input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('arr', [
    pytest.param(np.zeros((20, 30, 3), np.float64), id='float_rgb'),
    pytest.param(np.zeros((20, 30, 2), np.uint8), id='two_channels'),
    pytest.param(np.zeros((20, 30, 3), np.uint16), id='uint16_rgb'),
    pytest.param(np.zeros((2, 20, 30, 3), np.uint8), id='4d'),
])
def test_unsupported_array_shape_raises(arr):
    with mock.patch('mrzmini.image.ocr', side_effect=AssertionError('OCR called')):
        with pytest.raises(TypeError, match='3 or 4 channels'):
            read_mrz(arr)


@pytest.mark.parametrize('arr', [
    pytest.param(np.zeros((20, 30), object), id='object'),
    pytest.param(np.zeros((20, 30), complex), id='complex'),
    pytest.param(np.zeros((20, 30), 'm8[s]'), id='timedelta'),
    pytest.param(np.zeros((0, 0), np.uint8), id='empty'),
])
def test_unsupported_array_values_raise(arr):
    with pytest.raises(TypeError, match='non-empty array of numbers'):
        read_mrz(arr)


@pytest.mark.parametrize('file', [None, 42, [[0, 1], [1, 0]], bytearray(b'\x89PNG')],
                         ids=lambda f: type(f).__name__)
def test_unsupported_type_raises(file):
    with pytest.raises(TypeError, match='PIL.Image.Image or a numpy.ndarray'):
        read_mrz(file)
