"""Shared by the pipeline tests: synthetic images, scripted OCR texts, and a recorder of the OCR calls read_mrz makes.

No Tesseract needed: the fake runs the real ocr() with subprocess.run faked, which returns scripted texts by
call rank without reading the image.
Images are drawn with numpy only (block glyphs, no font), so they do not depend on the Pillow or FreeType version.
All MRZ texts are fictitious (ICAO Doc 9303 specimen, synthetic CNI_FR of test_cni_fr.py).
"""
import functools
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import mrzmini.ocr
from mrzmini.image import MRZPipeline, read_mrz, read_mrz_detailed

# ICAO Doc 9303 TD3 specimen (Anna Maria Eriksson, Utopia)
TD3_L1 = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<'
TD3_L2 = 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'
# Synthetic CNI_FR of test_cni_fr.py
CNI_L1 = 'IDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123'
CNI_L2 = '1507069012348JEAN<<PIERRE<<8503127M2'


def td3(line2):
    return TD3_L1 + '\n' + line2


VALID = td3(TD3_L2)  # score 100
# Invalid reads: wrong check digits. Equal scores test the strict '>' between variants.
SCORE_25 = td3('L898902C37UTO7408123F1204150ZE184226B<<<<<10')
SCORE_44_A = td3('L898902C37UTO7408122F1204150ZE184226B<<<<<10')
SCORE_44_B = td3('L898902C36UTO7408123F1204150ZE184226B<<<<<10')
SCORE_62_A = td3('L898902C37UTO7408122F1204159ZE184226B<<<<<10')
SCORE_62_B = td3('L898902C36UTO7408123F1204159ZE184226B<<<<<10')
SCORE_81 = td3('L898902C36UTO7408122F1204159ZE184226B<<<<<11')
# An upside-down MRZ as tesseract reads it: '<' becomes '>', so BoxToMRZ flips the ROI.
UPSIDE_DOWN = TD3_L2[::-1].replace('<', '>') + '\n' + TD3_L1[::-1].replace('<', '>')
# No '<' at all: early exit of BoxToMRZ, with a positive score (5).
NO_FILLER = 'A' * 44 + '\n' + '1' * 44
CNI_FR = CNI_L1 + '\n' + CNI_L2
# Administrative code changed: number and date of birth checks of line 2 valid, composite not (score 71)
CNI_FR_LINE2 = CNI_L1[:-1] + '4' + '\n' + CNI_L2


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------
def glyph(h, w, char):
    """A deterministic block glyph: a few strokes of a 7-segment-like grid, picked from the character code."""
    g = np.zeros((h, w), bool)
    sw = max(1, w // 5)
    if char == '<':
        g[h // 2 - sw // 2:h // 2 - sw // 2 + sw, 1:w - 1] = True
        return g
    strokes = [
        (slice(0, sw), slice(0, w)),
        (slice(h // 2 - sw // 2, h // 2 - sw // 2 + sw), slice(0, w)),
        (slice(h - sw, h), slice(0, w)),
        (slice(0, h), slice(0, sw)),
        (slice(0, h), slice(w - sw, w)),
        (slice(0, h), slice(w // 2 - sw // 2, w // 2 - sw // 2 + sw)),
    ]
    bits = (ord(char) * 2654435761) >> 7
    chosen = [s for i, s in enumerate(strokes) if bits >> i & 1]
    for s in chosen if len(chosen) >= 2 else strokes[3:5]:
        g[s] = True
    return g


def draw_card(lines, w):
    """An ID-1 card (RGB uint8, 1.585 ratio) with a photo, a few text bars and a 2-line MRZ at the bottom."""
    h = int(w / 1.585)
    card = np.empty((h, w, 3), np.uint8)
    card[:] = (226, 229, 233)
    card[h * 7 // 100:h // 2, w * 5 // 100:w * 27 // 100] = (150, 140, 135)
    for k in range(5):
        y = h * 12 // 100 + k * h * 7 // 100
        card[y:y + max(2, h // 50), w * 31 // 100:w * (78 - 5 * k) // 100] = (90, 90, 100)

    n = len(lines[0])
    cell_w = (w - w * 10 // 100) // n
    cell_h = int(cell_w * 1.4)
    pad = max(1, cell_w // 6)
    left = (w - n * cell_w) // 2
    top = h - h * 7 // 100 - 2 * cell_h - cell_h // 2
    for r, line in enumerate(lines):
        y = top + r * (cell_h + cell_h // 2)
        for c, char in enumerate(line):
            x = left + c * cell_w + pad
            card[y:y + cell_h, x:x + cell_w - 2 * pad][glyph(cell_h, cell_w - 2 * pad, char)] = (30, 30, 40)
    return card


def on_a4(card):
    """The card at the centre of a white A4 page at 150 dpi."""
    page = np.full((1754, 1240, 3), 255, np.uint8)
    h, w = card.shape[:2]
    y, x = (page.shape[0] - h) // 2, (page.shape[1] - w) // 2
    page[y:y + h, x:x + w] = card
    return page


def to_luma(rgb):
    """RGB to 8-bit gray with the ITU-R 601 integer formula, so that no Pillow conversion is involved."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return ((r * 299 + g * 587 + b * 114 + 500) // 1000).astype(np.uint8)


def png(arr):
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format='PNG')
    return out.getvalue()


IMAGES = {
    'card_td3_rgb': lambda: draw_card([TD3_L1, TD3_L2], 640),
    # Mode L: read as uint8, so img.mean() > 0.95 always holds in TryOtherMaxWidth
    'card_td3_gray': lambda: to_luma(draw_card([TD3_L1, TD3_L2], 640)),
    'card_cni_fr_rgb': lambda: draw_card([CNI_L1, CNI_L2], 640),
    # np.rot90 turns counterclockwise, like PIL's Image.rotate(90)
    'card_td3_rotated_90': lambda: np.ascontiguousarray(np.rot90(draw_card([TD3_L1, TD3_L2], 640))),
    # MRZ found on the 250 px thumbnail and again at 1000 px
    'a4_card_480': lambda: on_a4(draw_card([TD3_L1, TD3_L2], 480)),
    # MRZ lines too thin on the 250 px thumbnail: only found at 1000 px
    'a4_card_320': lambda: on_a4(draw_card([TD3_L1, TD3_L2], 320)),
    'blank_page': lambda: np.full((404, 640, 3), 255, np.uint8),
}


@functools.cache
def image_png(name):
    return png(IMAGES[name]())


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class ScriptedOCR(object):
    """Stands for mrzmini.image.ocr: records each call, then runs the real ocr() with tesseract faked,
    which outputs texts[rank], then ''."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def __call__(self, img, *args, **kwargs):
        call = {
            'shape': list(img.shape),
            'dtype': str(img.dtype),
            'positional_args': len(args),
            'keywords': sorted(kwargs),
            'extra_cmdline_params': kwargs.get('extra_cmdline_params', args[1] if len(args) > 1 else None),
            'tesseract': [],
        }
        self.calls.append(call)
        rank = len(self.calls) - 1
        text = self.texts[rank] if rank < len(self.texts) else ''
        with mock.patch('mrzmini.ocr.subprocess.run', functools.partial(fake_tesseract, call['tesseract'], text)):
            return mrzmini.ocr.ocr(img, *args, **kwargs)


def fake_tesseract(runs, text, cmd, **kwargs):
    """Stands for subprocess.run in mrzmini.ocr: records the command line (temporary paths reduced to their
    names) and the image tesseract would read, and writes text as its output."""
    # Checked here rather than recorded: other keywords (such as capturing stderr) do not change what tesseract does
    assert kwargs.get('timeout') is None, 'tesseract must get no time limit by default'
    run = {'args': [os.path.basename(a) for a in cmd[1:3]] + list(cmd[3:])}
    if len(cmd) >= 3 and os.path.exists(cmd[1]):
        with Image.open(cmd[1]) as im:
            u8 = np.ascontiguousarray(np.asarray(im))
        # Fingerprint the uint8 image tesseract reads rather than the floats ocr() gets, which may differ in the last bits
        run['uint8_sha256'] = hashlib.sha256(u8.tobytes()).hexdigest()
        run['uint8_sum'] = int(u8.sum(dtype=np.int64))
        Path(cmd[2] + '.txt').write_text(text, encoding='utf-8')
    runs.append(run)
    return subprocess.CompletedProcess(cmd, 0)


def thumbnail_boxes(data):
    """Number of boxes found on the 250 px thumbnail (no OCR involved)."""
    with mock.patch('mrzmini.image.ocr', side_effect=AssertionError('OCR called while locating boxes')):
        return len(MRZPipeline(data)['boxes'])


def run_read_mrz(data, texts, extra, **kwargs):
    """Records read_mrz(data); kwargs (such as lang) are passed on. Not for lang='auto': the fake tesseract
    does not answer --list-langs."""
    def read():
        return read_mrz(data, **kwargs) if extra is None else read_mrz(data, False, extra, **kwargs)
    return record_read(read, data, texts)


def run_read_mrz_detailed(data, texts, extra, **kwargs):
    """Records read_mrz_detailed(data), with extra (if not None) and kwargs as keywords. Returns the record, made
    from result.mrz as for run_read_mrz, and the MRZResult."""
    if extra is not None:
        kwargs['extra_cmdline_params'] = extra
    results = []

    def read():
        results.append(read_mrz_detailed(data, **kwargs))
        return results[0].mrz
    return record_read(read, data, texts), results[0]


def record_read(read, data, texts):
    """Records the OCR calls of read(), which reads data through one MRZPipeline and returns an MRZ or None."""
    fake = ScriptedOCR(texts)
    pipelines = []
    init = MRZPipeline.__init__

    # Keeps a handle on the pipeline read() builds, to inspect its final state. Replacing the class itself
    # would break MRZPipeline.__init__, which calls super() through the module global.
    def kept_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        pipelines.append(self)

    with mock.patch('mrzmini.image.ocr', fake), mock.patch.object(MRZPipeline, '__init__', kept_init):
        mrz = read()

    assert len(pipelines) == 1
    p = pipelines[0]
    record = {
        'boxes_250': thumbnail_boxes(data),
        # After TryOtherMaxWidth, the scaler is the 1000 px one and the boxes are those found with it
        'final_max_width': p.components['scaler'].max_width,
        # None if the image was not even loaded (deadline already passed)
        'final_boxes': len(p.data['boxes']) if 'boxes' in p.data else None,
        'ocr_calls': fake.calls,
        'method': None if mrz is None else mrz.aux.get('method'),
        'result': None if mrz is None else mrz.to_dict(),
    }
    return json.loads(json.dumps(record))
