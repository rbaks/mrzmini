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

mrzmini does not repair the lines. If a line does not have 36 characters,
the MRZ is not valid.

The check digits use modulo 10. Thus, some misreads pass all checks, for
example `<` read as `K`.

Only `read_mrz` and `MRZ.from_ocr` find `CNI_FR`. `MRZ(lines)` does not.

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
