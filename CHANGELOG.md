# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
