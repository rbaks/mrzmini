# mrzmini

mrzmini finds and reads the MRZ (machine-readable zone) of ID documents.
It uses the algorithm of [PassportEye](https://github.com/konstantint/PassportEye).
It has only two Python dependencies: numpy and Pillow.

## Install

```bash
pip install mrzmini
```

You must have Python 3.12 or later.

Install the `tesseract` binary on `PATH`:

```bash
apt install tesseract-ocr   # Debian/Ubuntu
brew install tesseract      # macOS
```

To use a different binary, set `TESSERACT_CMD`.

## Usage

```python
from mrzmini import read_mrz

mrz = read_mrz('passport.jpg')  # path, bytes, file object, PDF, PIL image or numpy array
print(mrz.to_dict())            # mrz is None if there is no MRZ
```

```bash
mrzmini passport.jpg
```

`read_mrz` has the same interface and the same `MRZ` fields as in PassportEye.
mrzmini adds the `lang` argument and images in memory.

### Tesseract model

By default, tesseract uses its default model (`eng`). To change the model, use `lang`:

```python
read_mrz('passport.jpg', lang='mrz')   # or 'eng', 'mrz+eng'
read_mrz('passport.jpg', lang='auto')  # 'mrz' if installed, else the default
```

Tesseract does not supply `mrz`. Put an `mrz.traineddata` file in the tessdata directory.

The environment variable `MRZMINI_TESSERACT_LANG` sets `lang` for all reads.
**Then the results can be different from PassportEye.**

### Detailed result

`read_mrz_detailed` always returns an `MRZResult`, also if there is no MRZ:

```python
from mrzmini import ROTATED_VIEWS, read_mrz_detailed

result = read_mrz_detailed(page, lang='auto')
result.mrz               # the best read: an MRZ or None
result.accepted          # accept(mrz), or mrz.valid without accept
result.view              # the view of the read, for example (0, 250)
result.ocr_calls         # number of OCR calls
result.elapsed_s         # duration, in seconds
result.budget_exhausted  # True if a limit stopped the read
```

All the arguments after `file` are keywords. Without options, the result is the same as `read_mrz`.

| Argument | Effect |
|----------|----------|
| `accept` | A function `MRZ -> bool`. It replaces `mrz.valid` as the stop condition. |
| `max_ocr_calls` | The maximum number of OCR calls. |
| `deadline` | A `time.monotonic()` instant. No step starts after it. |
| `views` | More views to examine if the normal read finds no accepted MRZ. Use `ROTATED_VIEWS` for pages turned by 90°. |
| `pass1` | `False` skips the normal read. Use it with `views`. |
| `rank_key` | A function `MRZ -> key`. It replaces `valid_score` to select the best read. |
| `keep_candidates` | `True` keeps all the reads in `result.candidates`, best first. |

When a limit stops the read, there is no exception. The result is the best read before the limit.

The steps without OCR cannot stop. Thus a read can end some tenths of a second after `deadline`.

## Supported formats

- ICAO: TD1, TD2, TD3, MRVA, MRVB.
- `CNI_FR`: French national ID card issued before 2021 (2 lines of 36 characters).
  PassportEye does not read this format.

### `CNI_FR`

- `expiration_date` is `None`. This MRZ has no expiry date.
- `valid_check_digits` is `[number, date_of_birth, composite]`.
  **Index 2 is not the expiry date.** Use `valid_number`, `valid_date_of_birth` and `valid_composite`.
- `issue_date` is `YYMM`. `administrative_code` has 6 characters.
- `detection_mode` is `'prefix'`, `'shifted'` or `'line2'`.
- `surname_truncated` and `names_truncated` are `True` if the field is full. Then the name can be cut.
  They are `None` if the positions are not reliable (not `'prefix'` mode with 2 lines of 36 characters).

mrzmini does not repair the lines. A line that does not have 36 characters is not valid.

Only `read_mrz` and `MRZ.from_ocr` find `CNI_FR`. `MRZ(lines)` does not.
To test lines that you split yourself, use `looks_like_cni_fr(lines)`.

## Differences from PassportEye

[`parity_check.py`](parity_check.py) compares mrzmini with PassportEye on the test images of PassportEye.
There are four differences:

- mrzmini reads document numbers longer than 9 characters (TD1, TD2).
  `number` is the full number, and the MRZ can be valid.
  PassportEye gives the first 9 characters, and the MRZ is never valid.
- mrzmini extracts the MRZ zone upright, whatever the direction of its tilt.
  PassportEye extracts it upside down when the text rises to the right.
  Then it can read the zone only if tesseract reads `<` upside down as `>`
  (with the `mrz` model, it does not).
  mrzmini turns such a zone back pixel for pixel before reading it.
  With tesseract's default model, it reads the same pixels as PassportEye once PassportEye has turned the zone back.
  With tesseract 5.5.3, the 36 test files give the same boxes and the same reads.
- mrzmini can read an invalid MRZ as `CNI_FR`, with a different `valid_score`.
- When PassportEye gives no `method`, mrzmini gives `method='direct'`.

```bash
uv run --group parity python parity_check.py
```

## Tests

```bash
uv run --group dev pytest
```

## Limitations

- In a PDF, mrzmini reads only the first embedded JFIF JPEG image.
- The check digits use modulo 10. Some misreads pass all the checks, for example `<` read as `K`.

## License

MIT. The PassportEye algorithm and the copied modules are © Konstantin Tretyakov (MIT).
The reimplemented primitives follow scikit-image and scipy (BSD).
