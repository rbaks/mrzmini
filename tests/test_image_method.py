"""Non-regression tests for the ``KeyError: 'method'`` bug (spec cases 18 and 19).

No image and no Tesseract needed: the pipeline and OCR are faked.
"""
from unittest import mock

import numpy as np

from mrzmini.image import BoxToMRZ, TryOtherMaxWidth
from mrzmini.text import MRZ

# Two lines of 36 characters without any '<' (triggers the early exit of BoxToMRZ).
NO_FILLER_TEXT = 'A' * 36 + '\n' + '1' * 36


class FakePipeline(object):
    """Exposes only what TryOtherMaxWidth uses."""

    def __init__(self, mrz):
        self._data = {
            'img_binary': np.zeros((10, 10)),
            'img': np.zeros((10, 10)),
            'mrz': mrz,
        }

    def __getitem__(self, key):
        return self._data[key]

    def replace_component(self, name, component):
        pass


class FakeBox(object):
    def extract_from_image(self, img, scale):
        return np.zeros((20, 200))


def test_try_other_max_width_without_method():
    # MRZ straight from from_ocr: no aux['method'], as returned by the early exit before the fix.
    rerun_mrz = MRZ.from_ocr(NO_FILLER_TEXT)
    assert 'method' not in rerun_mrz.aux

    result = TryOtherMaxWidth()(None, FakePipeline(rerun_mrz))

    assert result is rerun_mrz
    assert result.aux['method'] == 'direct|max_width(1000)'


def test_box_to_mrz_early_exit_sets_method():
    with mock.patch('mrzmini.image.ocr', return_value=NO_FILLER_TEXT):
        roi, text, mrz = BoxToMRZ()(FakeBox(), np.zeros((100, 100)), np.zeros((50, 50)), 0.5)

    assert text == NO_FILLER_TEXT
    assert mrz.aux['method'] == 'direct'
