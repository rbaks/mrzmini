# mrzmini

A minimal-footprint, drop-in reimplementation of [PassportEye](https://github.com/konstantint/PassportEye)'s
MRZ (machine-readable zone) detection and reading pipeline.

PassportEye does one small, precise job — find and read the MRZ on an ID document —
but drags in a heavy dependency stack to do it: `numpy`, `scipy`, `scikit-image`,
`scikit-learn`, `matplotlib`, `imageio`, `pdfminer`, `pytesseract`.

`mrzmini` reproduces the **exact same algorithm and output** for valid ICAO
reads (plus one addition, the [old French ID card](#old-french-id-card-cni_fr),
which can also change the type of some invalid reads, and one bug fix, see
[Parity](#parity-with-passporteye)) using only:

- **numpy**
- **Pillow**
- the external **`tesseract`** binary (called via `subprocess`; required only for OCR)

Everything `skimage` / `sklearn` / `scipy` / `imageio` / `pdfminer` / `pytesseract`
did is reimplemented on numpy in [`mrzmini/imageproc.py`](mrzmini/imageproc.py),
[`mrzmini/geometry.py`](mrzmini/geometry.py), [`mrzmini/ocr.py`](mrzmini/ocr.py)
and [`mrzmini/pdf.py`](mrzmini/pdf.py). The MRZ text parser
([`text.py`](mrzmini/text.py)) and the pipeline engine ([`pipeline.py`](mrzmini/pipeline.py))
are pure-Python and copied from upstream (MIT), unchanged except for the
`CNI_FR` addition in `text.py`.

## Install

```bash
pip install mrzmini
```

`mrzmini` needs the external **`tesseract`** OCR binary on `PATH` for the OCR step
(detection alone does not need it):

```bash
apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract           # macOS
```

Override the binary location with the `TESSERACT_CMD` environment variable.

## Usage

```python
from mrzmini import read_mrz

mrz = read_mrz('passport.jpg')        # path, bytes, file-like, or a .pdf
print(mrz)                            # None if nothing was found
print(mrz.to_dict())
```

The public surface mirrors PassportEye: `read_mrz(file, save_roi=False,
extra_cmdline_params='')` returns an `MRZ` object with the same fields
(`mrz_type`, `valid`, `valid_score`, `number`, `names`, `surname`,
`date_of_birth`, `expiration_date`, `nationality`, `sex`, check digits, …).

Command-line demo:

```bash
mrzmini passport.jpg                  # installed console script
python -m mrzmini passport.jpg        # or via the module
```

## Old French ID card (`CNI_FR`)

In addition to the ICAO formats (TD1/TD2/TD3/MRVA/MRVB), `mrzmini` reads the
MRZ of the French national ID card (CNI) issued before 2021, reported as
`mrz_type == 'CNI_FR'`. This is a France-specific, non-ICAO format of
**2 lines × 36 characters**:

```
IDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123     ID, FRA, surname (25), administrative code (6)
1507069012348JEAN<<PIERRE<<8503127M2     number (12) + check, given names (14),
                                         date of birth + check, sex, composite check
```

Attributes specific to this type:

- `administrative_code`: 6 characters (department + office), may contain `<`.
- `issue_date`: `YYMM` of issue, the first 4 characters of the card number.
- `expiration_date` is `None`: this MRZ carries no expiry date.
- `nationality` is always `'FRA'`; `type` and `country` are read from line 1
  and can differ from `'ID'` / `'FRA'` on a misread.
- `valid_format`: format check (see below); not emitted by `to_dict()`.

**`valid_check_digits` has a different meaning for this type:** it holds 3
items, `[valid_number, valid_date_of_birth, valid_composite]`. Index 2 is the
composite check, **not** the expiration date as for the other types. Generic
code should read the named attributes (`valid_number`, `valid_date_of_birth`,
`valid_composite`), not the indices.

`to_dict()` emits, in this order:

- `mrz_type`, `valid_score`, `raw_text` (if present in `aux`);
- `type`, `country`, `number`, `issue_date`, `date_of_birth`, `nationality`, `sex`, `names`, `surname`, `administrative_code`;
- `check_number`, `check_date_of_birth`, `check_composite`;
- `valid_number`, `valid_date_of_birth`, `valid_composite`;
- `method` (if present in `aux`).

It never emits `expiration_date`, `check_expiration_date`,
`valid_expiration_date`, `optional1`, `optional2` or `personal_number` for
`CNI_FR`.

**Format check.** Check digits cannot see a character outside the MRZ
alphabet (`MRZCheckDigit` weighs it −1000 × weight: at a weight-1 position of
the 71-character composite the sum usually stays positive and it acts as `<`,
so `MARTIN>` passes, while at a weight-3 or weight-7 position, or in a short
field, the sum goes negative and the check fails), nor a character of the
wrong class whose value differs by a multiple of 10 (a letter left in the card
number). `valid_format` is `True` only if both raw OCR lines (after the
`IDFRA` prefix is normalised) contain only `A-Z`, `a-z`, `0-9` and `<`, every
position of the corrected lines matches its class in the `CNI_FR` template
(letter, digit, `<` where allowed), and `country == 'FRA'`. A department (the
card number's, line 2 positions 5–7, and the department part of the
administrative code, line 1 positions 31–33) accepts digits, and `A` or `B`
only right after a `2` within the same department (Corsica `2A`/`2B`). The
office part of the administrative code (line 1 positions 34–36) accepts only
digits and `<`; `<` is accepted only in the administrative code, never in the
card number. When `valid_format` is `False`, `valid_score` is capped at 99, so
`valid` is `False`.

**No line repair, by design.** `mrzmini` never repairs a line's length or
content; only the constant `IDFRA` prefix is normalised. A line that is not
exactly 36 characters long makes the MRZ invalid (`valid_line_lengths` is
`False`; the composite check is always `False` when line 1 has the wrong
length): line 1 is protected only by a mod-10 composite check, so any repair
would accept a wrong name or code about 1 time in 10.

**Limitation: mod-10 check digits, checked after correction.** Not repairing
removes the false positives that repairs would create, not all of them. As for
the ICAO formats, a misread whose value differs by a multiple of 10 still
passes every check, notably `<` read as `K` (`MARTINK`): `A`, `K`, `U` (10,
20, 30) and `0` are all worth `<` modulo 10. And since the character corrector
runs before the check digits, it can turn a misread into such a same-value
substitution: `<` read as `C`/`D`/`O`/`Q` in the administrative code becomes
`0` (the specimen then gives `administrative_code='0<<<<<'`), and `<` read as
`4` in a name becomes `A` (`MARTIN` → `MARTINA`). These reads come back valid.
Two misreads can also cancel each other out, as with any check digit.

The `CNI_FR` path runs inside `MRZ.from_ocr` (and thus `read_mrz`), on the raw
OCR text. Constructing `MRZ(lines)` directly keeps the 0.1.0 behavior and never
yields `CNI_FR`.

## Requirements

- Python ≥ 3.12, `numpy`, `pillow` (installed via `uv sync`).
- The `tesseract` OCR binary on `PATH` (e.g. `apt install tesseract-ocr`).
  Override its location with the `TESSERACT_CMD` environment variable.

## How it works

The pipeline is identical to PassportEye's (a lazy DAG of components), apart
from two deliberate differences, the `CNI_FR` path and the `KeyError: 'method'`
fix (see [Parity](#parity-with-passporteye)):

| Step | What it does | Upstream dependency replaced |
|------|--------------|------------------------------|
| `Loader` | read image → grayscale (color → float64 luma, gray → uint8) | `skimage.io` / `imageio`; `pdfminer` for PDFs |
| `Scaler` | downscale so width ≤ 250 (anti-aliased) | `skimage.transform.rescale` (→ scipy) |
| `BooneTransform` | `threshold_otsu(closing(\|sobel_v(black_tophat(img))\|))` → binary | `skimage.morphology` + `skimage.filters` |
| `MRZBoxLocator` | marching-squares contours → `RotatedBox` (PCA) → merge parallel boxes | `skimage.measure` + `sklearn.PCA` |
| `extract_from_image` | un-rotate + crop the ROI | `skimage.transform.rotate` |
| `ocr` | run Tesseract on the ROI | `pytesseract` |
| `MRZ.from_ocr` | clean up + parse + checksum-validate | (pure Python) |

## Parity with PassportEye

[`parity_check.py`](parity_check.py) compares `mrzmini` against the real
PassportEye stage by stage. Across the **entire PassportEye test corpus
(36 images: TD1/TD2/TD3/MRVA/MRVB, scores 0–100, JPG/PNG/PDF)**:

- `img_binary`: **0 pixel disagreements** on every image
- detected MRZ **boxes: identical** on every image
- `read_mrz(...).to_dict()`: **identical** on every image, apart from the
  deliberate differences below

Parity holds for valid ICAO reads. Two differences are deliberate:

- The `CNI_FR` type is an `mrzmini`-only addition: PassportEye does not know
  it. Any invalid read whose raw OCR lines are a `CNI_FR` candidate (2 lines
  of 34–38 characters, with line 1 within one character of `IDFRA`, possibly
  after one stray leading character, **or** line 2 shaped like a CNI line 2,
  whatever line 1 is) may be returned as `CNI_FR`, with a different
  `valid_score`, whatever type PassportEye returns: usually `TD2`, also
  `MRVB`, rarely `TD3`/`MRVA` with non-ASCII characters (the standard type is
  guessed on cleaned lines, and `upper()` can lengthen a character, `ß` →
  `SS`).
- The 0.2.0 `KeyError: 'method'` fix: where PassportEye raises, or emits no
  `method`, `mrzmini` returns `method='direct'`.

Valid ICAO reads never change, and neither does `MRZ(lines)`.

```bash
uv run --group parity python parity_check.py            # scans testdata/
uv run --group parity python parity_check.py some.jpg   # specific files
```

Internally every reimplemented primitive matches its scikit-image / scipy /
scikit-learn counterpart to floating-point precision (bilinear resize and
marching-squares contours are bit-exact; the bicubic OCR-retry resize matches
scipy to ~1e-14, and the PCA box geometry to ~1e-12).

## Running tests

```bash
uv run --group dev pytest
```

## Notes / limitations

- PDF support is best-effort: like upstream it extracts the first embedded JFIF
  JPEG (`\xff\xd8\xff\xe0`); other embedded image encodings are not handled.
- Tesseract is an external binary, not a Python library, so it remains a
  requirement for the OCR step (detection alone does not need it).

## License

The reused PassportEye algorithm and copied modules are MIT
(© Konstantin Tretyakov). Reimplemented primitives follow scikit-image /
scipy behavior (BSD).
