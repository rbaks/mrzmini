'''
mrzmini.ocr: run Tesseract on an image region.

Drop-in replacement for PassportEye's ``passporteye/util/ocr.py``. Instead of
``pytesseract`` + ``imageio`` this writes a temporary BMP with Pillow and calls
the ``tesseract`` binary directly via :mod:`subprocess`. The image-to-uint8
conversion and the exact tesseract command-line configuration are reproduced so
the OCR output matches upstream.

Requires the external ``tesseract`` binary on PATH (or via the ``TESSERACT_CMD``
environment variable). The tesseract model can be chosen with ``lang`` or the
``MRZMINI_TESSERACT_LANG`` environment variable (see :func:`ocr`).

Author of original: Konstantin Tretyakov. License: MIT.
'''
import logging
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

TESSERACT_CMD = os.environ.get('TESSERACT_CMD', 'tesseract')
LANG_ENV = 'MRZMINI_TESSERACT_LANG'

log = logging.getLogger(__name__)

# Models listed by `tesseract --list-langs` (None if it failed), per (TESSERACT_CMD, TESSDATA_PREFIX)
_installed_models = {}
# Warnings already logged, each logged once per process
_warned = set()


def _clear_lang_cache():
    """Forgets the --list-langs results and the warnings already logged (for tests)."""
    _installed_models.clear()
    _warned.clear()


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _list_models(refresh=False):
    """Returns (cache key, models listed by `tesseract --list-langs` or None if it fails)."""
    key = (TESSERACT_CMD, os.environ.get('TESSDATA_PREFIX'))
    if refresh or key not in _installed_models:
        try:
            out = subprocess.run([key[0], '--list-langs'], capture_output=True, text=True, errors='replace',
                                 check=True)
        except (OSError, subprocess.CalledProcessError):
            models = None
        else:
            # Tesseract 3 printed the list on stderr. A model name has no space, unlike the header or a warning.
            lines = (line.strip() for line in (out.stdout + '\n' + out.stderr).splitlines())
            models = [line for line in lines
                      if line and not line.startswith('List of available languages') and ' ' not in line]
        _installed_models[key] = models
    return key, _installed_models[key]


def _sets_lang(extra_cmdline_params):
    return '-l' in (extra_cmdline_params or '').split()


def _normalize_lang(lang):
    """Strips ``lang``, '' meaning None; ValueError for 'auto' within a combination."""
    lang = (lang or '').strip() or None
    if lang is not None and lang != 'auto' and 'auto' in lang.split('+'):
        raise ValueError("lang=%r: 'auto' cannot be combined with other models" % lang)
    return lang


def _check_lang(lang, extra_cmdline_params):
    """Returns ``lang`` normalized; ValueError if it is given together with ``-l`` in ``extra_cmdline_params``."""
    lang = _normalize_lang(lang)
    if lang is not None and _sets_lang(extra_cmdline_params):
        raise ValueError('lang=%r conflicts with -l in extra_cmdline_params=%r: set the model only once'
                         % (lang, extra_cmdline_params))
    return lang


def _resolve_lang(lang, extra_cmdline_params):
    """The model to pass with -l, or None for no -l (see ocr)."""
    if lang is None:
        lang = _normalize_lang(os.environ.get(LANG_ENV))
        if lang is not None and _sets_lang(extra_cmdline_params):
            _warn_once(('env', lang, extra_cmdline_params), '%s=%r ignored: extra_cmdline_params=%r already sets -l',
                       LANG_ENV, lang, extra_cmdline_params)
            return None
    if lang != 'auto':
        return lang
    key, models = _list_models()
    if models is not None and 'mrz' in models:
        return 'mrz'
    if models is None:
        _warn_once(('auto', key), "lang='auto': `%s --list-langs` failed (TESSDATA_PREFIX=%r), running without -l",
                   *key)
    else:
        _warn_once(('auto', key), "lang='auto': no 'mrz' model for %s (TESSDATA_PREFIX=%r), running without -l",
                   *key)
    return None


def _raise_if_missing(lang, err):
    """After tesseract failed with -l lang: ValueError if --list-langs (run again) lacks a model of lang."""
    key, models = _list_models(refresh=True)
    if models is None:
        return
    missing = [m for m in lang.split('+') if m not in models]
    if missing:
        raise ValueError('tesseract model %s not installed for %s (TESSDATA_PREFIX=%r); installed: %s'
                         % ('+'.join(missing), key[0], key[1], ', '.join(models) or 'none')) from err


def ocr(img, mrz_mode=True, extra_cmdline_params='', *, lang=None):
    """Runs Tesseract on a given image and returns the recognised text.

    :param mrz_mode: when True (default) tesseract is configured to recognise
        MRZs (single uniform block, restricted character whitelist, dictionaries
        disabled). When False, only ``extra_cmdline_params`` are passed.
    :param extra_cmdline_params: extra parameters appended to the tesseract
        command line (e.g. ``"--oem 0"`` to use the legacy engine).
    :param lang: the tesseract model, passed as ``-l`` (also when ``mrz_mode``
        is False): a model name such as ``"mrz"`` or ``"mrz+eng"``, or
        ``"auto"`` for ``"mrz"`` if ``tesseract --list-langs`` lists it, else no
        ``-l`` (with a logged warning). Surrounding spaces are stripped. When
        None or empty, the ``MRZMINI_TESSERACT_LANG`` environment variable is
        read at each call; if it is unset too, there is no ``-l`` and tesseract
        uses its default model (``eng``), as in PassportEye. A ``lang`` argument
        together with ``-l`` in ``extra_cmdline_params`` raises ValueError; the
        environment variable is ignored, with a warning, when
        ``extra_cmdline_params`` has ``-l``.
        If tesseract fails and a model is missing from ``--list-langs``, raises
        ValueError listing the installed models (not checked when
        ``extra_cmdline_params`` has ``--tessdata-dir``). With ``"a+b"``, tesseract
        fails only if no model loads: it silently reads with those it found.
        ``"auto"`` lists the models of ``TESSDATA_PREFIX``, not of a
        ``--tessdata-dir`` in ``extra_cmdline_params``, and does not fall back
        if a listed ``mrz`` model cannot be loaded: tesseract then fails.
    """
    lang = _check_lang(lang, extra_cmdline_params)
    if img is None or img.shape[-1] == 0:  # Issue #34
        return ''
    lang = _resolve_lang(lang, extra_cmdline_params)

    # Prevent a lossy-conversion warning and match upstream's exact rounding.
    if str(img.dtype).startswith('float') and np.nanmin(img) >= 0 and np.nanmax(img) <= 1:
        img = img.astype(np.float64) * (np.power(2.0, 8) - 1) + 0.499999999
        img = img.astype(np.uint8)
    else:
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix='mrzmini_') as tmpdir:
        input_file_name = os.path.join(tmpdir, 'roi.bmp')
        output_file_base = os.path.join(tmpdir, 'out')
        output_file_name = output_file_base + '.txt'

        Image.fromarray(img).save(input_file_name)

        if mrz_mode:
            # NB: Tesseract 4.0+ may ignore tessedit_char_whitelist with the LSTM engine.
            config = ["--psm", "6",
                      "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><",
                      "-c", "load_system_dawg=F",
                      "-c", "load_freq_dawg=F"]
            if extra_cmdline_params:
                config += extra_cmdline_params.split()
        else:
            config = extra_cmdline_params.split() if extra_cmdline_params else []

        cmd = [TESSERACT_CMD, input_file_name, output_file_base]
        if lang is not None:
            cmd += ["-l", lang]
        cmd += config + ["txt"]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as err:
            # --list-langs reads TESSDATA_PREFIX: it cannot tell for another --tessdata-dir
            if lang is not None and '--tessdata-dir' not in (extra_cmdline_params or '').split():
                _raise_if_missing(lang, err)
            raise

        with open(output_file_name, encoding='utf-8') as f:
            return f.read().strip()
