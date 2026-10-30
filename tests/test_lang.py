"""Choice of the tesseract model: lang argument, MRZMINI_TESSERACT_LANG and lang='auto'.

No Tesseract needed: subprocess.run is faked in mrzmini.ocr, --list-langs included.
"""
import logging
import re
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import mrzmini.ocr
from mrzmini.image import BoxToMRZ, MRZPipeline, read_mrz
from mrzmini.ocr import LANG_ENV, ocr
from pipeline_recording import VALID, image_png, run_read_mrz

MRZ_CONFIG = ['--psm', '6',
              '-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><',
              '-c', 'load_system_dawg=F',
              '-c', 'load_freq_dawg=F']
ROI = np.full((20, 200), 255, np.uint8)


class FakeTesseract(object):
    """Stands for subprocess.run in mrzmini.ocr. `--list-langs` lists `models` on `stream`, or raises
    `list_error`; an OCR run records its command line, then writes an empty output or raises `run_error`."""

    def __init__(self, models=('eng', 'osd'), stream='stdout', list_error=None, run_error=False):
        self.models = models
        self.stream = stream
        self.list_error = list_error
        self.run_error = run_error
        self.list_calls = 0
        self.runs = []

    def __call__(self, cmd, **kwargs):
        if cmd[1:] == ['--list-langs']:
            self.list_calls += 1
            if self.list_error is not None:
                raise self.list_error
            out = 'List of available languages in "/fake/tessdata/" (%d):\n%s\n' % (
                len(self.models), '\n'.join(self.models))
            streams = {'stdout': '', 'stderr': ''}
            streams[self.stream] = out
            return subprocess.CompletedProcess(cmd, 0, **streams)
        self.runs.append((cmd, kwargs))
        if self.run_error:
            raise subprocess.CalledProcessError(1, cmd)
        Path(cmd[2] + '.txt').write_text('', encoding='utf-8')
        return subprocess.CompletedProcess(cmd, 0)

    def args(self, i=-1):
        """Command line of an OCR run after the input file and output base."""
        return self.runs[i][0][3:]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv(LANG_ENV, raising=False)
    monkeypatch.delenv('TESSDATA_PREFIX', raising=False)
    mrzmini.ocr._clear_lang_cache()
    yield
    mrzmini.ocr._clear_lang_cache()


@pytest.fixture
def tesseract(monkeypatch):
    fake = FakeTesseract()
    monkeypatch.setattr('mrzmini.ocr.subprocess.run', fake)
    return fake


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.WARNING, logger='mrzmini.ocr')
    return caplog


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('lang, mrz_mode, extra, expected', [
    pytest.param(None, True, '', MRZ_CONFIG + ['txt'], id='none'),
    pytest.param('mrz', True, '', ['-l', 'mrz'] + MRZ_CONFIG + ['txt'], id='mrz'),
    pytest.param('mrz+eng', True, '', ['-l', 'mrz+eng'] + MRZ_CONFIG + ['txt'], id='mrz+eng'),
    pytest.param('mrz', True, '--oem 1', ['-l', 'mrz'] + MRZ_CONFIG + ['--oem', '1', 'txt'], id='mrz_extra'),
    pytest.param(None, False, '', ['txt'], id='none_no_mrz_mode'),
    pytest.param('mrz', False, '--oem 1', ['-l', 'mrz', '--oem', '1', 'txt'], id='mrz_no_mrz_mode'),
])
def test_command_line(tesseract, lang, mrz_mode, extra, expected):
    ocr(ROI, mrz_mode, extra, lang=lang)

    assert tesseract.list_calls == 0
    (cmd, kwargs), = tesseract.runs
    assert cmd[0] == mrzmini.ocr.TESSERACT_CMD
    assert [Path(a).name for a in cmd[1:3]] == ['roi.bmp', 'out']
    assert cmd[3:] == expected
    # Same call as 0.2.0: stderr discarded, no time limit
    assert kwargs == {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'check': True}


def test_empty_image_runs_nothing(tesseract):
    assert ocr(np.zeros((20, 0)), lang='auto') == ''
    assert tesseract.list_calls == 0 and tesseract.runs == []


@pytest.mark.parametrize('lang', ['', '  '])
def test_empty_lang_is_none(tesseract, lang):
    ocr(ROI, lang=lang)
    ocr(ROI, extra_cmdline_params='-l eng', lang=lang)  # no conflict

    assert tesseract.args(0) == MRZ_CONFIG + ['txt']
    assert tesseract.args(1) == MRZ_CONFIG + ['-l', 'eng', 'txt']
    with mock.patch('mrzmini.image.ocr', return_value='') as fake:
        read_mrz(image_png('card_td3_rgb'), lang=lang)
    assert all(c.kwargs.keys() == {'extra_cmdline_params'} for c in fake.call_args_list)


def test_lang_stripped(tesseract):
    ocr(ROI, lang=' mrz+eng ')
    assert tesseract.args()[:2] == ['-l', 'mrz+eng']


@pytest.mark.parametrize('lang', ['auto+eng', 'eng+auto'])
def test_auto_not_combined(tesseract, monkeypatch, lang):
    with pytest.raises(ValueError, match="'auto' cannot be combined"):
        BoxToMRZ(lang=lang)
    monkeypatch.setenv(LANG_ENV, lang)
    with pytest.raises(ValueError, match="'auto' cannot be combined"):
        ocr(ROI)
    assert tesseract.runs == [] and tesseract.list_calls == 0


# ---------------------------------------------------------------------------
# Environment variable
# ---------------------------------------------------------------------------
def test_env_sets_lang(tesseract, monkeypatch):
    monkeypatch.setenv(LANG_ENV, 'mrz')
    ocr(ROI)
    ocr(ROI, mrz_mode=False)

    assert tesseract.args(0) == ['-l', 'mrz'] + MRZ_CONFIG + ['txt']
    assert tesseract.args(1) == ['-l', 'mrz', 'txt']


def test_env_read_at_each_call(tesseract, monkeypatch):
    ocr(ROI)
    monkeypatch.setenv(LANG_ENV, 'mrz')
    ocr(ROI)
    monkeypatch.setenv(LANG_ENV, '')
    ocr(ROI)

    assert [tesseract.args(i)[:2] for i in range(3)] == [MRZ_CONFIG[:2], ['-l', 'mrz'], MRZ_CONFIG[:2]]


def test_argument_overrides_env(tesseract, monkeypatch):
    monkeypatch.setenv(LANG_ENV, 'mrz')
    ocr(ROI, lang='eng')

    assert tesseract.args()[:2] == ['-l', 'eng']


# ---------------------------------------------------------------------------
# lang='auto'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_auto_with_mrz(tesseract, warnings, stream):
    tesseract.models = ('eng', 'mrz', 'osd')
    tesseract.stream = stream  # tesseract 3 lists on stderr

    ocr(ROI, lang='auto')

    assert tesseract.list_calls == 1
    assert tesseract.args()[:2] == ['-l', 'mrz']
    assert not warnings.records


def test_auto_without_mrz(tesseract, warnings):
    ocr(ROI, lang='auto')

    assert tesseract.args() == MRZ_CONFIG + ['txt']
    assert "no 'mrz' model" in warnings.text


@pytest.mark.parametrize('error', [FileNotFoundError('tesseract'), subprocess.CalledProcessError(1, 'tesseract')],
                         ids=lambda e: type(e).__name__)
def test_auto_list_langs_fails(tesseract, warnings, error):
    tesseract.list_error = error

    ocr(ROI, lang='auto')

    assert tesseract.args() == MRZ_CONFIG + ['txt']
    assert '--list-langs` failed' in warnings.text


def test_auto_from_env(tesseract, monkeypatch):
    tesseract.models = ('mrz',)
    monkeypatch.setenv(LANG_ENV, 'auto')

    ocr(ROI)

    assert tesseract.args()[:2] == ['-l', 'mrz']


def test_auto_cache(tesseract, warnings, monkeypatch):
    for _ in range(3):
        ocr(ROI, lang='auto')
    assert tesseract.list_calls == 1
    assert len(warnings.records) == 1  # warned once, not at each call

    monkeypatch.setenv('TESSDATA_PREFIX', '/other/tessdata')
    tesseract.models = ('eng', 'mrz')
    ocr(ROI, lang='auto')
    ocr(ROI, lang='auto')
    assert tesseract.list_calls == 2
    assert tesseract.args()[:2] == ['-l', 'mrz']

    # TESSERACT_CMD is part of the key, read at call time
    monkeypatch.setattr('mrzmini.ocr.TESSERACT_CMD', '/other/tesseract')
    ocr(ROI, lang='auto')
    assert tesseract.list_calls == 3
    assert tesseract.runs[-1][0][0] == '/other/tesseract'

    mrzmini.ocr._clear_lang_cache()
    ocr(ROI, lang='auto')
    assert tesseract.list_calls == 4


def test_auto_one_probe_per_page(tesseract, warnings):
    # Two boxes, and no '<' in the (empty) OCR output: one call per box
    read_mrz(image_png('card_td3_rotated_90'), lang='auto')

    assert len(tesseract.runs) == 2
    assert tesseract.list_calls == 1
    assert len(warnings.records) == 1


# ---------------------------------------------------------------------------
# Missing model
# ---------------------------------------------------------------------------
# With 'a+b', tesseract fails only if no model loads (5.5.3), so 'mrz+fra' with neither installed
@pytest.mark.parametrize('lang, missing', [('mrz', 'mrz'), ('mrz+fra', 'mrz+fra')])
def test_missing_model(tesseract, lang, missing):
    tesseract.run_error = True

    with pytest.raises(ValueError, match=r"model %s not installed .*installed: eng, osd" % re.escape(missing)) as exc:
        ocr(ROI, lang=lang)

    assert isinstance(exc.value.__cause__, subprocess.CalledProcessError)
    assert tesseract.list_calls == 1
    # The tesseract run itself is unchanged: stderr still discarded
    assert tesseract.runs[0][1]['stderr'] == subprocess.DEVNULL


def test_missing_model_probes_again(tesseract):
    tesseract.models = ('eng', 'mrz')
    ocr(ROI, lang='auto')
    tesseract.models = ('eng',)  # mrz removed since the cached probe
    tesseract.run_error = True

    with pytest.raises(ValueError, match='model mrz not installed'):
        ocr(ROI, lang='auto')
    assert tesseract.list_calls == 2


def test_other_error_reraised(tesseract):
    tesseract.models = ('eng', 'mrz')
    tesseract.run_error = True

    with pytest.raises(subprocess.CalledProcessError) as exc:
        ocr(ROI, lang='mrz')
    assert exc.value.cmd == tesseract.runs[0][0]


def test_error_reraised_with_tessdata_dir(tesseract):
    # --list-langs would list TESSDATA_PREFIX, not this directory
    tesseract.run_error = True

    with pytest.raises(subprocess.CalledProcessError):
        ocr(ROI, extra_cmdline_params='--tessdata-dir /other/tessdata', lang='mrz')
    assert tesseract.list_calls == 0


def test_error_reraised_when_list_langs_fails(tesseract):
    tesseract.run_error = True
    tesseract.list_error = FileNotFoundError('tesseract')

    with pytest.raises(subprocess.CalledProcessError):
        ocr(ROI, lang='mrz')


def test_error_without_lang_does_not_probe(tesseract):
    tesseract.run_error = True

    with pytest.raises(subprocess.CalledProcessError):
        ocr(ROI)
    assert tesseract.list_calls == 0


# ---------------------------------------------------------------------------
# Conflict with -l in extra_cmdline_params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('lang', ['mrz', 'auto'])
def test_conflict_with_argument(tesseract, lang):
    with pytest.raises(ValueError, match='conflicts with -l'):
        ocr(ROI, extra_cmdline_params='--oem 1 -l eng', lang=lang)
    assert tesseract.runs == [] and tesseract.list_calls == 0


def test_conflict_raised_at_construction(tesseract):
    with pytest.raises(ValueError, match='conflicts with -l'):
        BoxToMRZ(extra_cmdline_params='-l eng', lang='mrz')
    with pytest.raises(ValueError, match='conflicts with -l'):
        MRZPipeline(image_png('blank_page'), '-l eng', lang='mrz')
    # Even when the page has no box, hence no OCR call
    with pytest.raises(ValueError, match='conflicts with -l'):
        read_mrz(image_png('blank_page'), False, '-l eng', lang='mrz')


def test_no_conflict_without_l_token(tesseract):
    # Only a '-l' token counts
    ocr(ROI, extra_cmdline_params='-lmrz --oem 1', lang='eng')
    assert tesseract.args()[:2] == ['-l', 'eng']


def test_env_ignored_when_extra_sets_l(tesseract, warnings, monkeypatch):
    monkeypatch.setenv(LANG_ENV, 'mrz')

    ocr(ROI, extra_cmdline_params='-l eng')
    ocr(ROI, extra_cmdline_params='-l eng')

    assert tesseract.args() == MRZ_CONFIG + ['-l', 'eng', 'txt']
    assert tesseract.list_calls == 0
    assert len(warnings.records) == 1
    assert 'ignored' in warnings.text


# ---------------------------------------------------------------------------
# Through read_mrz
# ---------------------------------------------------------------------------
def without_lang(record):
    """The record of run_read_mrz as it would be without lang='mrz'."""
    for call in record['ocr_calls']:
        call['keywords'].remove('lang')
        for run in call['tesseract']:
            assert run['args'][2:4] == ['-l', 'mrz']
            del run['args'][2:4]
    return record


@pytest.mark.parametrize('image, texts, extra, method', [
    pytest.param('card_td3_rgb', [VALID], None, 'direct', id='thumbnail'),
    pytest.param('card_td3_rgb', [VALID], '--oem 1', 'direct', id='extra_positional'),
    # TryOtherMaxWidth reruns the pipeline with another scaler: lang must still apply
    pytest.param('a4_card_480', ['', VALID], None, 'direct|max_width(1000)', id='then_wider'),
    pytest.param('a4_card_320', [VALID], None, 'direct|max_width(1000)', id='wider_only'),
])
def test_lang_reaches_ocr(image, texts, extra, method):
    record = run_read_mrz(image_png(image), texts, extra, lang='mrz')

    assert record['method'] == method
    assert record['ocr_calls']
    for call in record['ocr_calls']:
        assert call['keywords'] == ['extra_cmdline_params', 'lang']
        assert call['tesseract'][0]['args'][2:4] == ['-l', 'mrz']
    # Otherwise the same calls and result as without lang
    assert without_lang(record) == run_read_mrz(image_png(image), texts, extra)


def test_default_passes_no_lang():
    with mock.patch('mrzmini.image.ocr', return_value='') as fake:
        read_mrz(image_png('card_td3_rgb'))
    assert fake.call_args_list
    assert all(c.kwargs.keys() == {'extra_cmdline_params'} for c in fake.call_args_list)
