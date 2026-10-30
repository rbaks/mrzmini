"""Regenerate the reference with `MRZMINI_UPDATE_GOLDEN=1 uv run --group dev pytest tests/test_pipeline_parity.py` (whole file), only for an intended behaviour change.

Pipeline parity with mrzmini 0.2.0: read_mrz runs on synthetic images with mrzmini.image.ocr faked, and the
sequence of OCR calls (arguments, tesseract command line, fingerprint of the image tesseract reads) and the
result are compared to pipeline_parity_golden.json.
With default parameters, later versions must make exactly the same calls and return the same result.

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
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import mrzmini.ocr
from mrzmini.image import MRZPipeline, read_mrz

GOLDEN_PATH = Path(__file__).with_name('pipeline_parity_golden.json')
UPDATE_GOLDEN = os.environ.get('MRZMINI_UPDATE_GOLDEN') == '1'

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
# Scenarios
# ---------------------------------------------------------------------------
# zone: 'thumbnail' (boxes on the 250 px thumbnail), 'wide' (boxes only at 1000 px) or None (no box at all).
# extra: extra_cmdline_params, passed positionally as existing callers may do; None to leave the default.
SCENARIOS = [
    pytest.param('card_td3_rgb', [VALID], None, 'thumbnail', 1, 'direct', id='valid_first_call'),
    pytest.param('card_td3_gray', [VALID], None, 'thumbnail', 1, 'direct', id='valid_first_call_gray'),
    # A uint8 image always counts as mostly white (img.mean() > 0.95 on 0-255): TryOtherMaxWidth reruns,
    # and finds no box since the page is narrower than 1000 px
    pytest.param('card_td3_gray', [], None, 'thumbnail', 1, None, id='gray_no_positive_score_then_wider'),
    pytest.param('card_td3_rgb', [VALID], '-l mrz', 'thumbnail', 1, 'direct', id='extra_cmdline_params'),
    # direct 44, rescaled(3) 44 (not strictly better), rescaled(1) 62, black_tophat 81, black_tophat(rescaled(3)) 62
    pytest.param('card_td3_rgb', [SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_81, SCORE_62_B], None,
                 'thumbnail', 5, 'black_tophat', id='invalid_all_variants'),
    pytest.param('card_td3_rgb', [UPSIDE_DOWN, VALID], None, 'thumbnail', 2, 'direct', id='flipped'),
    # flip, then direct 25, rescaled(3) 62, rescaled(1) 44, black_tophat 62 (not strictly better), its rescale 81
    pytest.param('card_td3_rgb', [UPSIDE_DOWN, SCORE_25, SCORE_62_A, SCORE_44_A, SCORE_62_B, SCORE_81], None,
                 'thumbnail', 6, 'black_tophat(rescaled(3))', id='flipped_then_invalid'),
    pytest.param('card_td3_rgb', [NO_FILLER], None, 'thumbnail', 1, 'direct', id='no_filler_early_exit'),
    pytest.param('card_cni_fr_rgb', [CNI_FR], None, 'thumbnail', 1, 'direct', id='cni_fr'),
    # No positive score on a mostly white page: TryOtherMaxWidth reruns at 1000 px
    pytest.param('a4_card_480', ['', VALID], None, 'thumbnail', 2, 'direct|max_width(1000)',
                 id='no_positive_score_then_wider'),
    pytest.param('a4_card_320', [VALID], None, 'wide', 1, 'direct|max_width(1000)', id='a4_card_wider_only'),
    # No positive score on a page that is not mostly white: no TryOtherMaxWidth, no result
    pytest.param('card_td3_rotated_90', [], None, 'thumbnail', 2, None, id='rotated_90'),
    # Two boxes, 5 calls each. FindFirstValidMRZ keeps the best score, and the last box on a tie (stable sort).
    pytest.param('card_td3_rotated_90', [SCORE_62_A, '', '', '', '', SCORE_44_A], None, 'thumbnail', 10, 'direct',
                 id='two_boxes_best_score'),
    pytest.param('card_td3_rotated_90', [SCORE_44_A, '', '', '', '', SCORE_44_B], None, 'thumbnail', 10, 'direct',
                 id='two_boxes_tie_takes_last'),
    pytest.param('blank_page', [], None, None, 0, None, id='blank_page'),
]
SCENARIO_IDS = [p.id for p in SCENARIOS]


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


def run_read_mrz(data, texts, extra):
    fake = ScriptedOCR(texts)
    pipelines = []
    init = MRZPipeline.__init__

    # Keeps a handle on the pipeline read_mrz builds, to inspect its final state. Replacing the class itself
    # would break MRZPipeline.__init__, which calls super() through the module global.
    def kept_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        pipelines.append(self)

    with mock.patch('mrzmini.image.ocr', fake), mock.patch.object(MRZPipeline, '__init__', kept_init):
        mrz = read_mrz(data) if extra is None else read_mrz(data, False, extra)

    assert len(pipelines) == 1
    p = pipelines[0]
    record = {
        'boxes_250': thumbnail_boxes(data),
        # After TryOtherMaxWidth, the scaler is the 1000 px one and the boxes are those found with it
        'final_max_width': p.components['scaler'].max_width,
        'final_boxes': len(p.data['boxes']),
        'ocr_calls': fake.calls,
        'method': None if mrz is None else mrz.aux.get('method'),
        'result': None if mrz is None else mrz.to_dict(),
    }
    return json.loads(json.dumps(record))


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def git(*args):
    try:
        out = subprocess.run(['git', *args], cwd=GOLDEN_PATH.parent, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def source_commit():
    """HEAD, flagged when mrzmini/ has uncommitted changes (the reference would not match that commit)."""
    head = git('rev-parse', 'HEAD') or 'unknown'
    return head + (' (mrzmini/ modified)' if git('status', '--porcelain', '--', '../mrzmini') else '')


@pytest.fixture(scope='module')
def golden():
    """The reference scenarios; in update mode, collects the records and writes them once all scenarios passed."""
    if not UPDATE_GOLDEN:
        if not GOLDEN_PATH.exists():
            pytest.fail('%s is missing: regenerate it (see the top of this file)' % GOLDEN_PATH.name)
        yield json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))['scenarios']
        return

    recorded = {}
    yield recorded
    missing = [k for k in SCENARIO_IDS if k not in recorded]
    if missing:
        # A partial run (-k) or a failing scenario would mix records of different origins
        raise RuntimeError('%s not written, scenarios not recorded: %s' % (GOLDEN_PATH.name, ', '.join(missing)))
    out = {
        'comment': 'Reference of tests/test_pipeline_parity.py. Regenerate only for an intended behaviour change.',
        'generated_from': source_commit(),
        'mrzmini_version': metadata.version('mrzmini'),
        # For information only: the fingerprints are meant not to depend on these
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': Image.__version__},
        'scenarios': {k: recorded[k] for k in SCENARIO_IDS},
    }
    GOLDEN_PATH.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    # The parity contract is about default parameters: no tesseract language set from the environment
    monkeypatch.delenv('MRZMINI_TESSERACT_LANG', raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', SCENARIOS)
def test_pipeline_parity(request, golden, image, texts, extra, zone, n_calls, method):
    record = run_read_mrz(image_png(image), texts, extra)

    # The scenario does what it claims, whatever the reference says
    if zone == 'thumbnail':
        assert record['boxes_250'] > 0
    elif zone == 'wide':
        assert record['boxes_250'] == 0
        assert record['final_max_width'] == 1000 and record['final_boxes'] > 0
    else:
        assert record['boxes_250'] == 0 and record['final_boxes'] == 0
    assert len(record['ocr_calls']) == n_calls
    assert record['method'] == method
    # Checked here too so that a regeneration cannot record them: lang and timeout must never be passed by default
    assert all(call['keywords'] == ['extra_cmdline_params'] for call in record['ocr_calls'])
    # One tesseract run per call, and no other one (such as a --list-langs probe)
    assert all(len(call['tesseract']) == 1 for call in record['ocr_calls'])

    name = request.node.callspec.id
    if UPDATE_GOLDEN:
        golden[name] = record
        return
    assert name in golden, 'no reference for %s: regenerate it (see the top of this file)' % name
    expected = golden[name]
    assert record['ocr_calls'] == expected['ocr_calls']
    assert record == expected


def test_reference_has_every_scenario(golden):
    if UPDATE_GOLDEN:
        pytest.skip('reference being regenerated')
    # A stale entry would mean a scenario was removed or renamed without regenerating
    assert sorted(golden) == sorted(SCENARIO_IDS)
