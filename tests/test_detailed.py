"""read_mrz_detailed: MRZResult, accept, max_ocr_calls and deadline.

No Tesseract needed: mrzmini.image.ocr is faked, except in test_deadline_kills_tesseract, which runs a shell script
standing for tesseract. Parity with read_mrz without options is tested in test_pipeline_parity.py.
"""
import json
import os
import subprocess
import threading
import time
from unittest import mock

import numpy as np
import pytest

import mrzmini
from mrzmini.image import MRZResult, read_mrz_detailed
from mrzmini.ocr import ocr
from pipeline_recording import (CNI_FR_LINE2, IMAGES, SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_62_B, SCORE_81,
                                VALID, image_png, run_read_mrz, run_read_mrz_detailed)
from test_pipeline_parity import GOLDEN_PATH, SCENARIOS

ALL_VARIANTS = [SCORE_44_A, SCORE_44_B, SCORE_62_A, SCORE_81, SCORE_62_B]


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    monkeypatch.delenv('MRZMINI_TESSERACT_LANG', raising=False)


def cni_fr_line2(mrz):
    """greenloc's rule: a CNI_FR of FRA whose two check digits of line 2 are valid."""
    return mrz.mrz_type == 'CNI_FR' and mrz.country == 'FRA' and mrz.valid_number and mrz.valid_date_of_birth


def test_exports():
    assert {'read_mrz_detailed', 'MRZResult'} <= set(mrzmini.__all__)
    assert mrzmini.read_mrz_detailed is read_mrz_detailed


def test_result_is_frozen():
    result = read_mrz_detailed(image_png('blank_page'))
    assert result == MRZResult(None, False, None, 0, result.elapsed_s, False)
    with pytest.raises(AttributeError):
        result.mrz = None


# ---------------------------------------------------------------------------
# accept
# ---------------------------------------------------------------------------
def test_accept_stops_at_first_variant():
    data = image_png('card_cni_fr_rgb')
    record, result = run_read_mrz_detailed(data, [CNI_FR_LINE2], None, accept=cni_fr_line2)

    assert len(record['ocr_calls']) == 1
    assert result.ocr_calls == 1
    assert result.accepted is True and result.mrz.valid is False
    assert (result.mrz.mrz_type, result.mrz.valid_score) == ('CNI_FR', 71)
    assert record['method'] == 'direct' and result.view == (0, 250)
    assert result.budget_exhausted is False

    # Without accept: all the variants, as read_mrz
    record, result = run_read_mrz_detailed(data, [CNI_FR_LINE2], None)
    assert record == run_read_mrz(data, [CNI_FR_LINE2], None)
    assert len(record['ocr_calls']) == 5
    assert result.accepted is False and result.mrz.valid_score == 71


def test_accept_stops_between_boxes():
    # Box 1 reads a valid TD3, which the rule rejects: all its variants, then box 2, accepted at its first call
    data = image_png('card_td3_rotated_90')
    texts = [VALID, '', '', '', '', CNI_FR_LINE2]
    record, result = run_read_mrz_detailed(data, texts, None, accept=cni_fr_line2)

    assert result.ocr_calls == len(record['ocr_calls']) == 6
    assert result.accepted is True and result.mrz.mrz_type == 'CNI_FR'

    # Without accept: the valid TD3 stops the read at its first call
    record, result = run_read_mrz_detailed(data, texts, None)
    assert record == run_read_mrz(data, texts, None)
    assert result.ocr_calls == 1 and result.accepted is True and result.mrz.mrz_type == 'TD3'


def test_accept_replaces_valid():
    # A predicate never satisfied: all the variants of a valid read, which is still returned as the best one
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), [VALID], None, accept=lambda mrz: False)

    assert result.ocr_calls == 5
    assert result.mrz.valid is True and result.accepted is False
    assert record['method'] == 'direct'


def test_accept_stops_before_black_tophat_rescale():
    # The black-tophat read (4th call) is accepted: its rescale (5th call, a better-scoring TD3) is not tried
    texts = [SCORE_44_A, '', '', CNI_FR_LINE2, VALID]
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), texts, None, accept=cni_fr_line2)

    assert result.ocr_calls == 4
    assert result.accepted is True and result.mrz.mrz_type == 'CNI_FR'
    assert record['method'] == 'black_tophat'

    # Without accept, 0.2.0 tries the rescale even after a valid black-tophat read
    texts = [SCORE_44_A, '', '', VALID, '']
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), texts, None)
    assert record == run_read_mrz(image_png('card_td3_rgb'), texts, None)
    assert result.ocr_calls == 5 and record['method'] == 'black_tophat'


def test_accept_with_tryothermaxwidth():
    # No positive score at 250 px: TryOtherMaxWidth still reruns, and accept applies there too
    record, result = run_read_mrz_detailed(image_png('a4_card_480'), ['', CNI_FR_LINE2], None, accept=cni_fr_line2)

    assert result.ocr_calls == 2
    assert result.accepted is True and result.view == (0, 1000)
    assert record['method'] == 'direct|max_width(1000)'


# ---------------------------------------------------------------------------
# max_ocr_calls
# ---------------------------------------------------------------------------
def golden_scenarios():
    return json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))['scenarios']


@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', SCENARIOS)
def test_max_ocr_calls_prefix(request, image, texts, extra, zone, n_calls, method):
    expected = golden_scenarios()[request.node.callspec.id]
    for max_calls in range(n_calls + 2):
        record, result = run_read_mrz_detailed(image_png(image), texts, extra, max_ocr_calls=max_calls)

        # The first calls of the unbounded read, and no other
        assert record['ocr_calls'] == expected['ocr_calls'][:max_calls]
        assert result.ocr_calls == min(max_calls, n_calls)
        if max_calls < n_calls:
            assert result.budget_exhausted is True
        elif max_calls > n_calls:
            # Never reached: the same read as without bound
            assert result.budget_exhausted is False
            assert record == expected
        # max_calls == n_calls: exhausted only if a TryOtherMaxWidth rerun (needing no OCR) was skipped
        elif result.budget_exhausted:
            assert expected['final_max_width'] == 1000 and record['final_max_width'] == 250


def test_one_call():
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None, max_ocr_calls=1)

    (call,) = record['ocr_calls']
    assert len(call['tesseract']) == 1
    assert result.budget_exhausted is True
    assert result.mrz.valid_score == 44 and record['method'] == 'direct'

    # Nothing left to do after the first call
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), [VALID], None, max_ocr_calls=1)
    assert result.ocr_calls == 1 and result.budget_exhausted is False and result.accepted is True


@pytest.mark.parametrize('max_calls, score, method', [
    (2, 44, 'direct'),  # rescaled(3) not strictly better
    (3, 62, 'rescaled(1)'),
    (4, 81, 'black_tophat'),  # stopped between black-tophat and its rescale
])
def test_box_keeps_best_variant(max_calls, score, method):
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None, max_ocr_calls=max_calls)

    assert result.ocr_calls == max_calls and result.budget_exhausted is True
    assert result.mrz.valid_score == score and record['method'] == method


def test_best_box_so_far():
    # Box 1 reads 62 in its 5 calls, box 2 is not read
    texts = [SCORE_62_A, '', '', '', '', VALID]
    record, result = run_read_mrz_detailed(image_png('card_td3_rotated_90'), texts, None, max_ocr_calls=5)

    assert result.ocr_calls == 5 and result.budget_exhausted is True
    assert result.mrz.valid_score == 62 and result.view == (0, 250)


def test_skips_tryothermaxwidth():
    record, result = run_read_mrz_detailed(image_png('a4_card_480'), ['', VALID], None, max_ocr_calls=1)

    assert result.ocr_calls == 1 and result.budget_exhausted is True
    assert result.mrz is None and result.view is None
    assert record['final_max_width'] == 250


def test_zero_calls():
    with mock.patch('mrzmini.image.ocr', side_effect=AssertionError('OCR called')):
        result = read_mrz_detailed(image_png('card_td3_rgb'), max_ocr_calls=0)
    assert result.ocr_calls == 0 and result.mrz is None and result.budget_exhausted is True


@pytest.mark.parametrize('max_calls, error', [(-1, ValueError), (1.5, TypeError)])
def test_invalid_max_ocr_calls(max_calls, error):
    with pytest.raises(error):
        read_mrz_detailed(image_png('blank_page'), max_ocr_calls=max_calls)


def test_budgets_not_shared_between_threads():
    # Both reads are in their first OCR call at the same time; one is bounded, the other is not
    barrier = threading.Barrier(2, timeout=10)
    first_call = threading.local()

    def fake_ocr(img, **kwargs):
        if not getattr(first_call, 'done', False):
            first_call.done = True
            barrier.wait()
        return SCORE_44_A

    results = {}

    def read(name, **kwargs):
        results[name] = read_mrz_detailed(image_png('card_td3_rgb'), **kwargs)

    with mock.patch('mrzmini.image.ocr', fake_ocr):
        threads = [threading.Thread(target=read, args=('bounded',), kwargs={'max_ocr_calls': 1}),
                   threading.Thread(target=read, args=('free',))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert (results['bounded'].ocr_calls, results['bounded'].budget_exhausted) == (1, True)
    assert (results['free'].ocr_calls, results['free'].budget_exhausted) == (5, False)


# ---------------------------------------------------------------------------
# deadline, on a fake clock
# ---------------------------------------------------------------------------
class Clock(object):
    """Stands for the time module in mrzmini.image."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class TimedOCR(object):
    """Stands for mrzmini.image.ocr: returns texts[rank] (then '') after `duration` seconds of the clock, and records
    the keyword arguments of each call."""

    def __init__(self, clock, texts, duration=1.0):
        self.clock = clock
        self.texts = texts
        self.duration = duration
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)
        self.clock.now += self.duration
        rank = len(self.calls) - 1
        return self.texts[rank] if rank < len(self.texts) else ''


def read_on_clock(image, texts, deadline_in, **kwargs):
    clock = Clock()
    fake = TimedOCR(clock, texts)
    with mock.patch('mrzmini.image.time', clock), mock.patch('mrzmini.image.ocr', fake):
        result = read_mrz_detailed(image_png(image), deadline=clock.now + deadline_in, **kwargs)
    return result, fake.calls


def test_deadline_between_variants():
    result, calls = read_on_clock('card_td3_rgb', ALL_VARIANTS, 3)

    # Each call gets the time left as timeout
    assert [c['timeout'] for c in calls] == [3, 2, 1]
    assert result.ocr_calls == 3 and result.budget_exhausted is True
    assert result.mrz.valid_score == 62 and result.mrz.aux['method'] == 'rescaled(1)'
    assert result.elapsed_s == 3


def test_deadline_between_boxes():
    result, calls = read_on_clock('card_td3_rotated_90', [SCORE_62_A, '', '', '', '', VALID], 5)

    assert result.ocr_calls == 5 and result.budget_exhausted is True
    assert result.mrz.valid_score == 62


def test_deadline_before_tryothermaxwidth():
    result, calls = read_on_clock('a4_card_480', ['', VALID], 1)

    assert result.ocr_calls == 1 and result.budget_exhausted is True and result.mrz is None


def test_deadline_not_reached():
    result, calls = read_on_clock('card_td3_rgb', [VALID], 10)

    assert [c['timeout'] for c in calls] == [10]
    assert result.ocr_calls == 1 and result.budget_exhausted is False and result.accepted is True


def test_deadline_reached_by_last_call():
    # The read is complete, but its last call ended at the deadline: it may have been cut
    result, calls = read_on_clock('card_td3_rgb', [VALID], 1)

    assert result.ocr_calls == 1 and result.budget_exhausted is True and result.accepted is True


def test_no_timeout_without_deadline():
    with mock.patch('mrzmini.image.ocr', return_value=SCORE_44_A) as fake:
        result = read_mrz_detailed(image_png('card_td3_rgb'), max_ocr_calls=3, lang='mrz')
    assert result.ocr_calls == 3
    assert all(c.kwargs.keys() == {'extra_cmdline_params', 'lang'} for c in fake.call_args_list)


def test_deadline_passed_loads_nothing():
    with mock.patch('mrzmini.imageproc.imread_gray', side_effect=AssertionError('image loaded')), \
            mock.patch('mrzmini.image.ocr', side_effect=AssertionError('OCR called')):
        result = read_mrz_detailed(image_png('card_td3_rgb'), deadline=time.monotonic())
    assert result.mrz is None and result.ocr_calls == 0 and result.budget_exhausted is True


# ---------------------------------------------------------------------------
# a PDF without an embedded JPEG image
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('options', [{}, {'views': mrzmini.ROTATED_VIEWS}, {'pass1': False, 'views': mrzmini.ROTATED_VIEWS}],
                         ids=['pass1', 'both_passes', 'pass2'])
def test_pdf_without_jpeg(tmp_path, options):
    pdf = tmp_path / 'no_jpeg.pdf'
    pdf.write_bytes(b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n'
                    b'trailer<</Root 1 0 R>>\n%%EOF\n')
    with mock.patch('mrzmini.image.ocr', side_effect=AssertionError('OCR called')):
        result = read_mrz_detailed(str(pdf), keep_candidates=True, **options)
    assert result.mrz is None and result.view is None and result.accepted is False
    assert result.ocr_calls == 0 and result.budget_exhausted is False and result.candidates == ()


# ---------------------------------------------------------------------------
# deadline, with a tesseract that does not answer
# ---------------------------------------------------------------------------
def test_ocr_timeout_reads_empty(monkeypatch):
    runs = []

    def no_answer(cmd, **kwargs):
        runs.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])

    monkeypatch.setattr('mrzmini.ocr.subprocess.run', no_answer)
    assert ocr(np.full((20, 200), 255, np.uint8), timeout=0.5) == ''
    assert runs == [{'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'check': True, 'timeout': 0.5}]



@pytest.mark.skipif(os.name != 'posix', reason='needs /bin/sh')
def test_deadline_kills_tesseract(tmp_path, monkeypatch):
    pid_file = tmp_path / 'pid'
    script = tmp_path / 'tesseract'
    # exec: sleep takes the PID of the process subprocess.run started, so killing it stops the sleep
    script.write_text('#!/bin/sh\necho $$ > "%s"\nexec sleep 60\n' % pid_file)
    script.chmod(0o755)
    monkeypatch.setattr('mrzmini.ocr.TESSERACT_CMD', str(script))
    image = IMAGES['card_td3_rgb']()

    # Ample time to reach the first OCR call, even on a slow CI runner
    deadline = time.monotonic() + 3
    result = read_mrz_detailed(image, deadline=deadline)
    returned = time.monotonic()

    assert result.ocr_calls == 1
    assert result.mrz is None and result.budget_exhausted is True
    assert deadline <= returned <= deadline + 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
