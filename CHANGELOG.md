# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `read_mrz` and `MRZPipeline` accept an image already in memory: a
  `PIL.Image.Image`, converted like a decoded file (without `Image.open`), or
  a `numpy.ndarray`, 2D as grayscale with its dtype kept (a boolean array is
  read like a 1-bit image), or 3D `uint8` with 3 or 4 channels as RGB(A). The OCR calls and the result are the same as with
  the image saved as PNG. The conversion is `mrzmini.imageproc.to_gray`, which
  `imread_gray` now calls.
- Keyword argument `lang` on `read_mrz`, `MRZPipeline`, `FindFirstValidMRZ`,
  `BoxToMRZ` and `mrzmini.ocr.ocr`, to choose the tesseract model (passed as
  `-l`): a model name such as `'mrz'` or `'mrz+eng'`, or `'auto'` for `mrz`
  when `tesseract --list-langs` lists it, else no `-l` and a logged warning
  (the list is cached per process, per `TESSERACT_CMD` and `TESSDATA_PREFIX`).
  Without `lang`, the `MRZMINI_TESSERACT_LANG` environment variable is read at
  each call; this changes the default for the whole process and intentionally
  breaks parity with PassportEye. If it is unset too, the tesseract command
  line is the same as in 0.2.0. `lang` together with `-l` in
  `extra_cmdline_params` raises `ValueError` (when the pipeline is built); the
  environment variable is then ignored with a warning instead. When tesseract
  fails, a model missing from `--list-langs` raises `ValueError` with the
  installed models; with `'a+b'`, tesseract fails only if no model loads.

### Changed

- `Loader` raises `TypeError` for an unsupported input type, instead of
  returning `None` and failing later in the pipeline. This includes
  `pathlib.Path` and `bytearray`, which already failed in 0.2.0.

## [0.2.0] - 2026-10-29

### Added

- New `mrz_type` `CNI_FR`: the MRZ of the French national ID card issued before
  2021 (2 lines × 36 characters, non-ICAO). Detected, sliced and validated
  (number, date of birth and composite check digits) in `MRZ.from_ocr`, hence
  in `read_mrz`. `MRZ(lines)` called directly is unchanged.
- New attributes for `CNI_FR`: `administrative_code` (6 characters) and
  `issue_date` (`YYMM`, first 4 characters of the card number).
  `expiration_date` is `None` for this type, and `valid_check_digits` is
  `[valid_number, valid_date_of_birth, valid_composite]`.
- New `valid_format` attribute for `CNI_FR` (not emitted by `to_dict()`):
  `True` only if both raw OCR lines (after the `IDFRA` prefix is normalised)
  contain only `A-Z`, `a-z`, `0-9`, `<`, every position of the corrected
  lines matches its class in the template (a department, in the card number
  or the first 3 characters of the administrative code, accepts digits, and
  `A`/`B` only right after a `2` within the same department for Corsica; the
  office part of the administrative code accepts only digits and `<`; `<` is
  accepted only in the administrative code), and `country == 'FRA'`. It
  catches some errors mod-10 check digits miss (e.g. `MARTIN>`, or a letter
  left in the card number), but not
  same-value substitutions, including those created by the character
  corrector, which runs first (`<` read as `4` in a name becomes `A`). When
  `False`, `valid_score` is capped at 99, so `valid` is `False`.
- A pytest test suite (`tests/`, run with `uv run --group dev pytest`, which
  also runs the `text.py` doctests) and a CI workflow (`tests.yml`).

### Fixed

- `KeyError: 'method'` raised by `TryOtherMaxWidth` when the retried pipeline
  returned an MRZ from the `BoxToMRZ` early exit (OCR text without any `<`).
  The early exit now sets `aux['method'] = 'direct'`, and `TryOtherMaxWidth`
  reads it defensively. Side effect: `to_dict()` now emits `method='direct'`
  for such MRZ.

### Changed

- Any invalid read whose raw OCR lines are a `CNI_FR` candidate can now be
  returned as `CNI_FR`, with a different `valid_score`, whatever its previous
  type: usually `TD2`, also `MRVB`, rarely `TD3`/`MRVA` with non-ASCII
  characters (the standard type is guessed on cleaned lines, where `upper()`
  can lengthen a character, `ß` → `SS`). Candidates are 2 raw lines of 34–38
  characters whose line 1 is within one character of `IDFRA` (possibly after
  one stray leading character), or whose line 2 has the shape of a CNI line 2,
  whatever line 1 is. Valid TD1/TD2/TD3/MRVA/MRVB reads and `MRZ(lines)` are
  unchanged.
- Old French ID cards read correctly now have `valid=True` (previously a
  `TD2` with `valid=False`).

## [0.1.0] - 2026-05-27

### Added

- Initial release: minimal-footprint reimplementation of PassportEye's MRZ
  detection and reading pipeline (numpy + Pillow + the `tesseract` binary).

[Unreleased]: https://github.com/rbaks/mrzmini/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rbaks/mrzmini/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rbaks/mrzmini/releases/tag/v0.1.0
