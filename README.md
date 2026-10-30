# mrzmini

mrzmini finds and reads the MRZ (machine-readable zone) of ID documents.
It uses the same algorithm as [PassportEye](https://github.com/konstantint/PassportEye),
but it has only two Python dependencies: numpy and Pillow.

## Install

```bash
pip install mrzmini
```

mrzmini needs Python 3.12 or later.

For the OCR step, install the `tesseract` binary on `PATH`:

```bash
apt install tesseract-ocr   # Debian/Ubuntu
brew install tesseract      # macOS
```

To use a different binary, set the `TESSERACT_CMD` environment variable.

## Usage

```python
from mrzmini import read_mrz

mrz = read_mrz('passport.jpg')  # path, bytes, file object or PDF
print(mrz.to_dict())            # mrz is None if no MRZ is found
```

`read_mrz(file, save_roi=False, extra_cmdline_params='')` has the same
interface and the same `MRZ` fields as in PassportEye. mrzmini adds the
keyword argument `lang` (see below).

Command line:

```bash
mrzmini passport.jpg
```

### Images in memory

`read_mrz` and `MRZPipeline` also take an image already in memory, so a
rendered page does not need to be encoded as PNG first:

```python
mrz = read_mrz(pil_image)  # PIL.Image.Image
mrz = read_mrz(array)      # numpy.ndarray: 2D gray, or uint8 RGB/RGBA
```

The result is the same as with the same image saved as PNG. A 2D array keeps
its dtype, like a grayscale file; a boolean array is read like a 1-bit image.
Any other input type raises `TypeError`.

### Tesseract model

By default, tesseract runs without `-l`, so with its default model (`eng`), as
in PassportEye. `lang` chooses the model:

```python
mrz = read_mrz('passport.jpg', lang='mrz')   # or 'eng', 'mrz+eng'
mrz = read_mrz('passport.jpg', lang='auto')  # 'mrz' if installed, else the default
```

Tesseract does not ship `mrz`: put an `mrz.traineddata` in its tessdata
directory (`TESSDATA_PREFIX`). With `'auto'`, mrzmini checks
`tesseract --list-langs` once per tesseract binary and `TESSDATA_PREFIX`, and
logs a warning if `mrz` is missing. `'auto'` does not see a `--tessdata-dir`
passed in `extra_cmdline_params`. If a listed `mrz` cannot be loaded (for
example, built for another tesseract version), the reads fail.

If tesseract fails because the model is missing, `read_mrz` raises
`ValueError` with the installed models, unless `extra_cmdline_params` contains
`--tessdata-dir`. With `'mrz+eng'`, tesseract fails only
if no model loads: if `mrz` is missing, it reads with `eng` only, silently.

Without `lang`, mrzmini reads the `MRZMINI_TESSERACT_LANG` environment
variable, for example `MRZMINI_TESSERACT_LANG=auto`. **It changes every read
of the process, and thus intentionally breaks parity with PassportEye.** It is
ignored, with a warning, if `extra_cmdline_params` contains `-l`. Passing both
`lang` and `-l` raises `ValueError`.

### Detailed result, OCR budget and deadline

`read_mrz_detailed` reads like `read_mrz`, but always returns an `MRZResult`,
also when no MRZ is found. All its arguments after `file` are keywords:

```python
from mrzmini import read_mrz_detailed

result = read_mrz_detailed(page, lang='auto')
result.mrz               # what read_mrz returns: an MRZ, or None
result.accepted          # accept(mrz), or mrz.valid without accept
result.view              # (0, 250), or (0, 1000) if found by the rerun on a wider thumbnail (see views below)
result.ocr_calls         # OCR calls made by this read
result.elapsed_s         # duration of the call, in seconds
result.budget_exhausted  # True if max_ocr_calls or deadline stopped the read
```

Without options, it makes the same OCR calls as `read_mrz`, and `result.mrz`
is the same.

- `accept`: a function `MRZ -> bool` that replaces `mrz.valid` as the
  condition that stops the search. When the best read so far satisfies it,
  mrzmini tries no further variant of the zone and no further zone. The best
  read is still chosen by `valid_score` (or `rank_key`, see
  [Candidates and ranking](#candidates-and-ranking)): an accepted read that
  does not rank strictly higher than the best one is not kept.
- `max_ocr_calls`: maximum number of OCR calls for the whole read. `0` runs
  no OCR.
- `deadline`: an instant on the `time.monotonic()` clock. No step starts
  after it. Each tesseract run gets the time left as timeout: tesseract is
  killed then, and the call counts as an empty read. If the deadline has
  already passed, the image is not loaded.

When a bound is reached, `read_mrz_detailed` raises no exception: it returns
the best read so far, with `budget_exhausted=True`.

The bounds apply to the whole call. To never cut the normal read, make it
without bounds, then bound only a second call with a shared deadline (see
the example in [Rotated and wider views](#rotated-and-wider-views)).

Steps without OCR (loading, scaling, zone detection, zone extraction,
black-tophat) cannot be interrupted, so a read can return after its deadline.
With a tesseract that never answers, on a synthetic 2000 px page (Apple M5
Pro), the maximum overrun was:

- 0.18 s when the zone is found on the 250 px thumbnail: extracting the zone
  rotates the full image;
- 0.58 s when the deadline falls just after the read checks it, before the
  rerun on a 1000 px thumbnail (scaling and detection).

A black-tophat on a 1850 px wide zone took 0.15 s more. Expect several times
these values on a slower CPU.

With `lang='auto'`, the first OCR call of the process also runs
`tesseract --list-langs` (about 10 ms), without time limit.

### Rotated and wider views

The normal read (pass 1) looks for the MRZ zone once, on a 250 px wide
thumbnail of the page as it is, and again at 1000 px if the page is mostly
white. It misses a page turned by a quarter, or a small card on a large
scan. `views` adds a second pass, which looks for the zone on other views
before any OCR:

```python
from mrzmini import ROTATED_VIEWS, read_mrz_detailed

result = read_mrz_detailed(page, views=ROTATED_VIEWS)
result.view  # the view the MRZ was read in, e.g. (270, 250)
```

A view is a pair `(rotation, width)`: the page turned counterclockwise by
`rotation` degrees (0, 90, 180 or 270, like `Image.rotate`), on a thumbnail
`width` pixels wide (never wider than the page). `ROTATED_VIEWS` is
`((270, 250), (90, 250), (90, 500), (270, 500))`.

Pass 2 runs only if pass 1 gives no accepted MRZ:

1. It looks for zones on each view, in order, without OCR. A view without
   any zone whose image is mostly white adds the same rotation at 1000 px at
   the end, once per rotation. The views pass 1 covered are skipped.
2. It scores each zone on its thumbnail: elongated and large zones first,
   whatever their angle (a turned view often finds the MRZ of an upright
   page as a vertical zone). On equal scores, the order of `views` decides.
3. It reads the zones in that order, each in the full-resolution page turned
   as its view, and stops at the first read that `accept` accepts
   (`mrz.valid` by default).

A read of pass 2 that is not accepted is not returned: `result.mrz` stays
the MRZ of pass 1, or `None`.

`pass1=False` runs pass 2 only, so you can run pass 1 on all your pages
first. For example, accept a valid MRZ or a French ID card valid on line 2
in the normal read, without bounds; then, if nothing is accepted, try the
views with a deadline, accepting only a valid MRZ:

```python
import time

def cni_fr_line2(mrz):
    return mrz.mrz_type == 'CNI_FR' and mrz.country == 'FRA' and mrz.valid_number and mrz.valid_date_of_birth

deadline = time.monotonic() + 15
result = read_mrz_detailed(page, lang='auto', accept=lambda m: m.valid or cni_fr_line2(m))
if not result.accepted:
    result = read_mrz_detailed(page, lang='auto', pass1=False, views=ROTATED_VIEWS,
                               accept=lambda m: m.valid, deadline=deadline)
```

`max_ocr_calls` and `deadline` cover both passes, and `ocr_calls` counts
both. No view starts after the deadline. The zones are read only once all
views are searched, so a deadline that passes during the search stops the
read before any zone of pass 2 is read. Searching a 1000 px view of a
2000 px page took 0.57 s (Apple M5 Pro).

As in pass 1, an 8-bit grayscale image (mode `L`, or a 2D `uint8` array)
always counts as mostly white. Each view without any zone then adds its
1000 px view: on a 2000 px card, this added 1.3 s to the search.

### Candidates and ranking

Each zone is read in up to 5 variants (direct, enlarged twice, black-tophat,
black-tophat enlarged), and the page may have several zones.
mrzmini keeps the variant with the highest `valid_score`, then the zone
with the highest `valid_score`. A French ID card valid on line 2 (score 71)
can thus lose to a misread TD2 with 3 valid check digits (score 77).

`rank_key` replaces `valid_score` in these choices: a function that maps an
`MRZ` to a comparable key, the highest key winning. Ties are broken as
before: the first variant of a zone, the last zone. Only zones with a
`valid_score` above 0 compete, as before.

```python
def partial_rank(mrz):
    # A French ID card valid on line 2 first, then a read with a valid check digit, then valid_score
    return cni_fr_line2(mrz), any(getattr(mrz, 'valid_check_digits', None) or []), mrz.valid_score

result = read_mrz_detailed(page, lang='auto', accept=lambda m: m.valid or cni_fr_line2(m), rank_key=partial_rank)
```

`rank_key` is called on every read, also an empty one
(`mrz_type` is then `None`, and the check digit attributes are missing).

`MRZ.from_ocr` may read a text both as a standard MRZ and as `CNI_FR`, and
return the standard reading (its score is higher, or no check digit of line
2 is valid). With `rank_key`, that `CNI_FR` reading competes as a read of
its own, so that `rank_key` can prefer it. Thus `rank_key=lambda m:
m.valid_score` may give another result than no `rank_key`.

`accept` still stops the search, whatever `rank_key`, but it tests the best
read so far: a key that ranks accepted reads first keeps them. With
`rank_key`, a valid black-tophat read thus stops the zone: mrzmini skips the
enlarged black-tophat variant that it tries otherwise, as PassportEye does,
which saves one OCR call. `accept` and `rank_key` never see `aux['view']`,
which is set once the result is known.

`keep_candidates=True` fills `result.candidates` with every read made, one
per OCR call, kept or not, in both passes, also when `result.mrz` is `None`:

```python
result = read_mrz_detailed(page, views=ROTATED_VIEWS, keep_candidates=True)
for mrz in result.candidates:  # best first, by rank_key or valid_score
    mrz.aux['method'], mrz.aux['view'], mrz.aux['raw_text']
```

- The candidates are sorted from best to worst; on a tie, in reading order.
  `result.mrz`, if any, is one of them.
- A zone read upside down (`>>`) is flipped and read again: the text read
  upside down is not a candidate, the flipped read is. If the deadline skips
  the first OCR call of a zone, the empty read is not a candidate either.
- `aux['method']` is the variant, for example `'black_tophat(rescaled(3))'`,
  also for a read that was not kept. Only `result.mrz` has the
  `'|max_width(1000)'` suffix, as before.
- `aux['view']` is the view of the read, as `result.view`: `(0, 250)`,
  `(0, 1000)` for the rerun on a wider thumbnail, or a view of pass 2.
- The `CNI_FR` reading that `MRZ.from_ocr` did not return is a candidate
  too, even without `rank_key`.
- No zone image is kept.

`keep_candidates` changes neither `result.mrz` (but for `aux['view']`) nor
`accepted`, `view` or the OCR calls. Without `rank_key` and
`keep_candidates`, the OCR calls and the result are those of `read_mrz`.

## Supported formats

- ICAO: TD1, TD2, TD3, MRVA, MRVB.
- `CNI_FR`: French national ID card issued before 2021 (2 lines of 36
  characters). PassportEye does not support this format.

### `CNI_FR` fields

- `administrative_code`: 6 characters (department and office).
- `issue_date`: `YYMM`, the first 4 characters of the card number.
- `expiration_date` is `None`. This MRZ has no expiry date.
- `valid_format` is `True` if each character has the correct class for its
  position and `country` is `'FRA'`. If it is `False`, `valid` is `False`.
- `valid_check_digits` is `[number, date_of_birth, composite]`.
  **Index 2 is the composite check, not the expiry date.** Use the named
  attributes `valid_number`, `valid_date_of_birth` and `valid_composite`.
- `to_dict()` does not give the expiry, optional or personal number fields.
- `detection_mode` tells how the raw lines were recognised: `'prefix'`
  (line 1 starts with `IDFRA`, one character off at most), `'shifted'` (the
  same after one stray character) or `'line2'` (only line 2 has the
  `CNI_FR` shape; line 1 may be anything).
- `surname_truncated` is `True` if the surname field (25 characters) is full,
  so the last name may be cut. `names_truncated` does the same for the given
  names field (14 characters). Both are `None` unless the positions are
  reliable: `'prefix'` mode with two lines of 36 characters.
- These three attributes are `None` on the other types, also for
  `MRZ(lines)`. Like `valid_format`, they are not in `to_dict()`.

mrzmini does not repair the lines. If a line does not have 36 characters,
the MRZ is not valid.

The check digits use modulo 10. Thus, some misreads pass all checks, for
example `<` read as `K`.

Only `read_mrz` and `MRZ.from_ocr` find `CNI_FR`. `MRZ(lines)` does not.

`looks_like_cni_fr(lines)` runs the shape test of `MRZ.from_ocr` on lines
you split yourself, and returns the detection mode, or `None`:

```python
from mrzmini import looks_like_cni_fr

lines = [ln.replace(' ', '') for ln in text.splitlines() if ln.strip()]
looks_like_cni_fr(lines)  # 'prefix', 'shifted', 'line2' or None
```

It does not split the text: `MRZ.from_ocr` drops the lines shorter than 20
characters, which you may want to keep. To test only line 1 (`IDFRA`,
maybe after one stray character), ignore `'line2'`.

## How it works

The pipeline is the same as in PassportEye:

| Step | What it does | Upstream dependency replaced |
|------|--------------|------------------------------|
| `Loader` | read image → grayscale (color → float64 luma, gray → uint8) | `skimage.io` / `imageio`; `pdfminer` for PDFs |
| `Scaler` | downscale so width ≤ 250 (anti-aliased) | `skimage.transform.rescale` (→ scipy) |
| `BooneTransform` | `threshold_otsu(closing(\|sobel_v(black_tophat(img))\|))` → binary | `skimage.morphology` + `skimage.filters` |
| `MRZBoxLocator` | marching-squares contours → `RotatedBox` (PCA) → merge parallel boxes | `skimage.measure` + `sklearn.PCA` |
| `extract_from_image` | un-rotate + crop the ROI | `skimage.transform.rotate` |
| `ocr` | run Tesseract on the ROI | `pytesseract` |
| `MRZ.from_ocr` | clean up + parse + checksum-validate | (pure Python) |

## Differences from PassportEye

[`parity_check.py`](parity_check.py) compares mrzmini with PassportEye on the
36 PassportEye test images. The results are identical, with two exceptions:

- mrzmini can give an invalid read as `CNI_FR`, with a different `valid_score`.
- If PassportEye raises `KeyError: 'method'` or gives no `method`, mrzmini
  gives `method='direct'`.

Valid ICAO reads are always identical.

```bash
uv run --group parity python parity_check.py
```

## Tests

```bash
uv run --group dev pytest
```

## Limitations

- In a PDF, mrzmini reads only the first embedded JFIF JPEG image.

## License

MIT. The PassportEye algorithm and the copied modules are © Konstantin
Tretyakov (MIT). The reimplemented primitives follow scikit-image and scipy
(BSD).
