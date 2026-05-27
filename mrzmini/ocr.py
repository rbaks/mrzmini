'''
mrzmini.ocr: run Tesseract on an image region.

Drop-in replacement for PassportEye's ``passporteye/util/ocr.py``. Instead of
``pytesseract`` + ``imageio`` this writes a temporary BMP with Pillow and calls
the ``tesseract`` binary directly via :mod:`subprocess`. The image-to-uint8
conversion and the exact tesseract command-line configuration are reproduced so
the OCR output matches upstream.

Requires the external ``tesseract`` binary on PATH (or via the ``TESSERACT_CMD``
environment variable).

Author of original: Konstantin Tretyakov. License: MIT.
'''
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

TESSERACT_CMD = os.environ.get('TESSERACT_CMD', 'tesseract')


def ocr(img, mrz_mode=True, extra_cmdline_params=''):
    """Runs Tesseract on a given image and returns the recognised text.

    :param mrz_mode: when True (default) tesseract is configured to recognise
        MRZs (single uniform block, restricted character whitelist, dictionaries
        disabled). When False, only ``extra_cmdline_params`` are passed.
    :param extra_cmdline_params: extra parameters appended to the tesseract
        command line (e.g. ``"--oem 0"`` to use the legacy engine).
    """
    if img is None or img.shape[-1] == 0:  # Issue #34
        return ''

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

        cmd = [TESSERACT_CMD, input_file_name, output_file_base] + config + ["txt"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        with open(output_file_name, encoding='utf-8') as f:
            return f.read().strip()
