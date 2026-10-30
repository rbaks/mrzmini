"""Regenerate the reference with `MRZMINI_UPDATE_GOLDEN=1 uv run --group dev pytest tests/test_pipeline_parity.py` (whole file), only for an intended behaviour change.

Pipeline parity with mrzmini 0.2.0: read_mrz, and read_mrz_detailed without options, run on synthetic images with
mrzmini.image.ocr faked, and the sequence of OCR calls (arguments, tesseract command line, fingerprint of the image
tesseract reads) and the result are compared to pipeline_parity_golden.json.
With default parameters, later versions must make exactly the same calls and return the same result.

The images, texts and recorder are in pipeline_recording.py (no Tesseract needed).
"""
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline_recording import (CNI_FR, NO_FILLER, SCORE_25, SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_62_B,
                                SCORE_81, UPSIDE_DOWN, VALID, image_png, run_read_mrz, run_read_mrz_detailed)

GOLDEN_PATH = Path(__file__).with_name('pipeline_parity_golden.json')
UPDATE_GOLDEN = os.environ.get('MRZMINI_UPDATE_GOLDEN') == '1'


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


@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', SCENARIOS)
def test_detailed_parity(request, golden, image, texts, extra, zone, n_calls, method):
    if UPDATE_GOLDEN:
        pytest.skip('reference being regenerated')
    record, result = run_read_mrz_detailed(image_png(image), texts, extra)

    # Same OCR calls, pipeline state and result as read_mrz in 0.2.0
    expected = golden[request.node.callspec.id]
    assert record['ocr_calls'] == expected['ocr_calls']
    assert record == expected
    assert record['result'] == run_read_mrz(image_png(image), texts, extra)['result']

    assert result.ocr_calls == n_calls
    assert result.budget_exhausted is False
    assert result.elapsed_s >= 0
    if method is None:
        assert result.mrz is None and result.view is None and result.accepted is False
    else:
        assert result.view == ((0, 1000) if method.endswith('|max_width(1000)') else (0, 250))
        assert result.accepted is result.mrz.valid


def test_reference_has_every_scenario(golden):
    if UPDATE_GOLDEN:
        pytest.skip('reference being regenerated')
    # A stale entry would mean a scenario was removed or renamed without regenerating
    assert sorted(golden) == sorted(SCENARIO_IDS)
