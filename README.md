# mrzmini

A minimal-footprint, drop-in reimplementation of [PassportEye](https://github.com/konstantint/PassportEye)'s
MRZ (machine-readable zone) detection and reading pipeline.

PassportEye does one small, precise job — find and read the MRZ on an ID document —
but drags in a heavy dependency stack to do it: `numpy`, `scipy`, `scikit-image`,
`scikit-learn`, `matplotlib`, `imageio`, `pdfminer`, `pytesseract`.

`mrzmini` reproduces the **exact same algorithm and output** using only:

- **numpy**
- **Pillow**
- the external **`tesseract`** binary (called via `subprocess`; required only for OCR)

Everything `skimage` / `sklearn` / `scipy` / `imageio` / `pdfminer` / `pytesseract`
did is reimplemented on numpy in [`mrzmini/imageproc.py`](mrzmini/imageproc.py),
[`mrzmini/geometry.py`](mrzmini/geometry.py), [`mrzmini/ocr.py`](mrzmini/ocr.py)
and [`mrzmini/pdf.py`](mrzmini/pdf.py). The MRZ text parser
([`text.py`](mrzmini/text.py)) and the pipeline engine ([`pipeline.py`](mrzmini/pipeline.py))
are pure-Python and copied verbatim from upstream (MIT).

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

## Requirements

- Python ≥ 3.12, `numpy`, `pillow` (installed via `uv sync`).
- The `tesseract` OCR binary on `PATH` (e.g. `apt install tesseract-ocr`).
  Override its location with the `TESSERACT_CMD` environment variable.

## How it works

The pipeline is identical to PassportEye's (a lazy DAG of components):

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
- `read_mrz(...).to_dict()`: **identical** on every image

```bash
uv run --group parity python parity_check.py            # scans testdata/
uv run --group parity python parity_check.py some.jpg   # specific files
```

Internally every reimplemented primitive matches its scikit-image / scipy /
scikit-learn counterpart to floating-point precision (bilinear resize and
marching-squares contours are bit-exact; the bicubic OCR-retry resize matches
scipy to ~1e-14, and the PCA box geometry to ~1e-12).

## Notes / limitations

- PDF support is best-effort: like upstream it extracts the first embedded JFIF
  JPEG (`\xff\xd8\xff\xe0`); other embedded image encodings are not handled.
- Tesseract is an external binary, not a Python library, so it remains a
  requirement for the OCR step (detection alone does not need it).

## License

The reused PassportEye algorithm and copied modules are MIT
(© Konstantin Tretyakov). Reimplemented primitives follow scikit-image /
scipy behavior (BSD).
