# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2026-10-31

### Fixed

- The MRZ zone of a box tilted counterclockwise is read from the same pixels
  as in 0.3.1 again. 0.3.2 took such a box the other way round (+180°): the
  zone was upright but resampled differently, and with tesseract's default
  model reads got worse (on 473 identity documents read fully valid by
  0.3.1, 51 were no longer valid, none gained). The box now keeps its angle;
  `_fixup_boxes` marks it `upside_down` and its ROI is turned back pixel for
  pixel (`roi[::-1, ::-1]`), which is what the `>>` check of 0.3.1 did. The
  zone stays upright without relying on that check, so the `mrz` model keeps
  most of the 0.3.2 gain. `pipeline['rois']` is upright too. With the default
  options, the parity scenarios give the results and the zone sizes of 0.3.1.
  On the 36 test files of PassportEye (tesseract 5.5.3), the boxes and the
  reads are again those of PassportEye.

## [0.3.2] - 2026-10-30

### Changed

- TD1 and TD2: a document number longer than 9 characters is now read whole
  (ICAO 9303: the first 9 characters fill the number field, a `<` takes the
  place of the check digit, and the rest of the number, then its check digit
  and a `<`, open the optional data). `number` gives the full number (for
  example `092150918ZY4` instead of `092150918`), `check_number` its check
  digit, and `valid_number`, hence `valid`, can be `True`; before, they were
  always `False` on such documents. The check digit is computed on the field
  as written, the `<` included. This applies only when the check digit read is
  `<`, at least 2 characters follow it and the check digit of the long number
  is right; otherwise `number` and `valid_number` are unchanged. `optional1`
  stays the raw text of the line. A `0` read as `<` on a document whose
  optional data is filled passes the composite check, and the long number
  check 1 time in 10.

### Fixed

- The MRZ zone is extracted upright whatever the direction of its tilt. A box
  whose text rises to the right by more than 0.57° came out of
  `RotatedBox.from_points` near -90° and was extracted turned by about 180°;
  it was then read only if tesseract read the `<` upside down as `>` (with the
  `mrz` model, it does not). `MRZBoxLocator._fixup_boxes` now takes a box
  whose angle is below -45° the other way round (+180°, the same pixels).
  Side effect: an upside-down document is no longer turned upright by chance
  one time in two; it relies on the `>>` check of `BoxToMRZ` or on the views
  of pass 2. With the default options, the zone extracted in 8 of the parity
  scenarios is 1 pixel larger; the OCR calls and the results are unchanged.
  On the 36 test files of PassportEye (tesseract 5.5.3), 3 reads change: one
  is better (`24_pass-egy`), one valid read is lost (`100_visa-polx`, score
  100 → 97), one has the same score.

## [0.3.1] - 2026-10-30

### Changed

- Shorter README: only the essentials. The details stay in the docstrings.

## [0.3.0] - 2026-10-30

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
- `read_mrz_detailed(file, *, extra_cmdline_params='', lang=None, accept=None,
  max_ocr_calls=None, deadline=None)` returns an `MRZResult`, a frozen
  dataclass, also when no MRZ is read (a PDF without an embedded JPEG image
  included, where `read_mrz` fails): `mrz` (what `read_mrz` returns),
  `accepted`, `view` (`(0, 250)`, `(0, 1000)` after `TryOtherMaxWidth`, or
  `None`), `ocr_calls`, `elapsed_s` and `budget_exhausted`. `accept`, a
  predicate `MRZ -> bool`, replaces `mrz.valid` as the condition that stops
  the search between the variants of a zone and between zones. Without
  options, the OCR calls and `mrz` are those of `read_mrz`. `accept` is also
  a keyword argument of `MRZPipeline`, `FindFirstValidMRZ` and `BoxToMRZ`.
- OCR budget and deadline: `max_ocr_calls` bounds the calls to
  `mrzmini.image.ocr` of a read, and `deadline` (on the `time.monotonic()`
  clock) stops it. They are checked before each OCR call, each variant of a
  zone, each zone and the `TryOtherMaxWidth` rerun; a passed deadline
  prevents even loading the image. When a bound is reached, the read
  returns the best MRZ read so far with `budget_exhausted=True`, without
  exception. `max_ocr_calls < 0` raises `ValueError`. Steps without OCR are
  not interrupted (overrun measured in the README). The bounds are keyword
  arguments of `MRZPipeline`, which keeps them in `MRZPipeline.budget`, a new
  `OCRBudget`, also a keyword argument of `FindFirstValidMRZ`, `BoxToMRZ` and
  `TryOtherMaxWidth`. `mrzmini.ocr.ocr` has a keyword argument `timeout`:
  tesseract is killed when it expires, and the result is `''`. It is passed
  only with a deadline.
- Pass 2 on rotated or wider views: `read_mrz_detailed(..., pass1=True,
  views=())`. When pass 1 gives no MRZ that satisfies `accept`, each view
  `(rotation, width)` (counterclockwise rotation of 0, 90, 180 or 270
  degrees, applied with `np.rot90`, and thumbnail width) is searched for
  zones without OCR. A view without any zone whose image is mostly white
  (the `TryOtherMaxWidth` condition) adds the same rotation at 1000 px, once
  per rotation; views pass 1 covered are skipped. The zones are then read by
  decreasing score (elongation and relative area on the thumbnail,
  whatever the angle; on equal scores, the order of `views`), each in the
  full-resolution image of its view, until a read satisfies `accept`. A read
  of pass 2 that is not accepted never replaces the MRZ of pass 1.
  `MRZResult.view` gives the view read. `pass1=False` runs pass 2 only
  (`ValueError` without views); an invalid view raises `ValueError`. The
  budget and deadline cover both passes, and are also checked before each
  view. `mrzmini.ROTATED_VIEWS = ((270, 250), (90, 250), (90, 500), (270,
  500))`. With `views=()`, the OCR calls and the result are unchanged.
- Ranking and candidates: `read_mrz_detailed(..., rank_key=None,
  keep_candidates=False)`, also keyword arguments of `MRZPipeline`,
  `FindFirstValidMRZ` and `BoxToMRZ`. `rank_key`, a function `MRZ -> key`,
  replaces `valid_score` wherever reads are compared: between the variants
  of a zone (strictly greater key; the first one on a tie) and between the
  zones (those with a positive `valid_score`; the last one on a tie). With
  `rank_key`, the `CNI_FR` reading that `MRZ.from_ocr` computed but did not
  return competes as a read of its own. `accept` still stops the search,
  tested on the best read so far; with `rank_key`, a valid black-tophat read
  thus skips the enlarged black-tophat variant (one OCR call fewer).
  `keep_candidates=True` fills `MRZResult.candidates`, a new field (`()` by
  default), with every read made in both passes, kept or not, also when
  `mrz` is `None`: one per OCR call, plus the `CNI_FR` reading that
  `from_ocr` discarded (even without `rank_key`), but neither the text read
  upside down before flipping the zone nor the empty read of a zone whose
  first OCR call the deadline skipped. Best first by
  `rank_key` (or `valid_score`), in reading order on a tie, without any
  image, each with `aux['method']` (now also for the variants not kept),
  `aux['raw_text']` and `aux['view']`. It changes neither `mrz` (but for
  `aux['view']`), `accepted`, `view` nor the OCR calls. Without both, the
  OCR calls and the result are unchanged.
- `CNI_FR` API: `detection_mode` (`'prefix'`, `'shifted'` or `'line2'`),
  `surname_truncated` and `names_truncated` (the field is full, so its last
  name may be cut; `None` unless in `'prefix'` mode with two lines of 36
  characters). They are `None` on the other types, `MRZ(lines)` included,
  and not in `to_dict()`, which is unchanged. `mrzmini.looks_like_cni_fr(lines)`
  runs the shape test of `MRZ.from_ocr` on lines already split and returns
  the detection mode, or `None`; a string raises `TypeError`.

### Changed

- `Loader` raises `TypeError` for an unsupported input type, instead of
  returning `None` and failing later in the pipeline. This includes
  `pathlib.Path` and `bytearray`, which already failed in 0.2.0.
- Requires Pillow 10.4 or later (was 9). Pillow 9 reads a 16-bit grayscale
  PNG as 32-bit integers, so the same image in memory and as PNG differed.

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

[Unreleased]: https://github.com/rbaks/mrzmini/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/rbaks/mrzmini/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/rbaks/mrzmini/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/rbaks/mrzmini/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rbaks/mrzmini/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rbaks/mrzmini/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rbaks/mrzmini/releases/tag/v0.1.0
