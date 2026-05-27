'''
mrzmini.imageproc: minimal numpy/Pillow reimplementations of the handful of
scikit-image / scipy primitives that PassportEye's MRZ pipeline relies on.

Every function here is a drop-in replacement for the corresponding
``skimage`` call used by ``image.py`` / ``geometry.py``, reproducing the
behaviour of scikit-image 0.26 closely enough to match its output.

The replaced upstream calls are noted in each docstring. Where a border /
interpolation convention matters it is spelled out, because scipy/skimage's
defaults were verified empirically against the installed library:

  * skimage ``imread(as_gray=True)`` keeps grayscale images as uint8 and only
    converts *colour* images to float64 (via the 0.2125/0.7154/0.0721 luma).
  * grayscale morphology (erosion/dilation) defaults to scipy ``mode='reflect'``
    which equals ``numpy.pad(mode='symmetric')``.
  * ``filters.sobel_v`` is ``ndi.convolve(VSOBEL/4, mode='reflect')`` with no
    border zeroing (that behaviour was removed in modern skimage).
  * ``transform.resize`` anti-aliases with a Gaussian of
    ``sigma = max(0, (in/out - 1)/2)`` then resamples with
    ``ndi.zoom(order=1, grid_mode=True, mode='grid-constant')`` and clips.

Original algorithm/code: Konstantin Tretyakov (PassportEye) and the
scikit-image developers (MIT / BSD).
'''
import io as _io

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# Luma coefficients used by skimage.color.rgb2gray (ITU-R BT.709-ish).
_GRAY_COEFFS = np.array([0.2125, 0.7154, 0.0721], dtype=np.float64)


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------
def img_as_float(image):
    """Mirror of ``skimage.util.img_as_float`` for the dtypes we encounter.

    Floats are returned unchanged (preserving range), unsigned integers are
    scaled into ``[0, 1]`` and booleans become ``0.0``/``1.0``.
    """
    image = np.asarray(image)
    if image.dtype == bool:
        return image.astype(np.float64)
    if np.issubdtype(image.dtype, np.floating):
        return np.asarray(image, dtype=np.float64)
    if np.issubdtype(image.dtype, np.unsignedinteger):
        return image.astype(np.float64) / float(np.iinfo(image.dtype).max)
    if np.issubdtype(image.dtype, np.integer):
        # Signed integers: scale by the positive max (matches skimage for our use).
        return image.astype(np.float64) / float(np.iinfo(image.dtype).max)
    raise ValueError("Unsupported image dtype: %s" % image.dtype)


# ---------------------------------------------------------------------------
# Loading  (replaces skimage.io.imread(file, as_gray=True))
# ---------------------------------------------------------------------------
def imread_gray(source):
    """Read ``source`` into a grayscale array, matching skimage's as_gray=True.

    * Colour images -> ``float64`` in ``[0, 1]`` via the rgb2gray luma.
    * Already-grayscale images -> returned with their native (e.g. uint8) dtype,
      exactly as skimage's ``imread(as_gray=True)`` does.

    ``source`` may be a path string, a ``bytes`` object, or a binary file-like.
    """
    if isinstance(source, (bytes, bytearray)):
        fobj = _io.BytesIO(bytes(source))
    else:
        fobj = source
    with Image.open(fobj) as im:
        im.load()
        mode = im.mode
        if mode in ('RGB', 'RGBA'):
            arr = np.asarray(im)
        elif mode == 'L':
            return np.asarray(im)  # keep uint8, like skimage
        elif mode in ('I', 'I;16', 'F'):
            return np.asarray(im)
        else:
            # Palette, LA, CMYK, etc.: normalise to RGB first, like imageio would.
            arr = np.asarray(im.convert('RGB'))

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = img_as_float(arr[..., :3])
        return rgb @ _GRAY_COEFFS
    return arr


# ---------------------------------------------------------------------------
# Rescale  (replaces skimage.transform.rescale / resize, anti_aliasing=True)
# ---------------------------------------------------------------------------
def _gaussian_kernel1d(sigma, truncate=4.0):
    """scipy.ndimage's 1-D Gaussian (order 0): exp(-x^2/2sigma^2), normalised."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k, radius


def _correlate1d_constant(arr, kernel, axis, radius):
    """1-D correlation with zero (constant) padding along ``axis``."""
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode='constant', constant_values=0.0)
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    sl = [slice(None)] * arr.ndim
    for i, w in enumerate(kernel):
        sl[axis] = slice(i, i + n)
        out += w * padded[tuple(sl)]
    return out


def _gaussian_filter(image, sigmas):
    """Separable Gaussian, sequential per axis, constant(0) borders.

    Matches ``scipy.ndimage.gaussian_filter(image, sigmas, mode='grid-constant',
    cval=0)`` used by skimage resize (for FIR filtering grid-constant == constant).
    """
    out = np.array(image, dtype=np.float64)
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        kernel, radius = _gaussian_kernel1d(sigma)
        out = _correlate1d_constant(out, kernel, axis, radius)
    return out


def _zoom_axis_bilinear(arr, axis, n_out):
    """Linear (order=1) resample of one axis, scipy ``grid_mode=True`` mapping.

    Output index ``o`` samples input coordinate ``(o + 0.5) / zoom - 0.5`` with
    ``zoom = n_out / n_in``; out-of-range samples use 0 (grid-constant, cval=0).
    """
    n_in = arr.shape[axis]
    zoom = n_out / n_in
    o = np.arange(n_out, dtype=np.float64)
    coord = (o + 0.5) / zoom - 0.5
    i0 = np.floor(coord).astype(np.int64)
    frac = coord - i0
    i1 = i0 + 1

    def gather(idx):
        valid = (idx >= 0) & (idx < n_in)
        clipped = np.clip(idx, 0, n_in - 1)
        vals = np.take(arr, clipped, axis=axis)
        shape = [1] * arr.ndim
        shape[axis] = len(idx)
        return vals * valid.reshape(shape)

    wshape = [1] * arr.ndim
    wshape[axis] = n_out
    w = frac.reshape(wshape)
    return gather(i0) * (1.0 - w) + gather(i1) * w


# Single pole of the cubic (order-3) B-spline.
_CUBIC_POLE = np.sqrt(3.0) - 2.0


def _spline_prefilter_axis(coeff, axis):
    """Cubic B-spline prefilter along ``axis`` (Unser whole-sample mirror init).

    Matches ``scipy.ndimage.spline_filter1d(order=3, mode='grid-constant')`` to
    machine precision (scipy uses the same mirror boundary for the order-3
    prefilter under constant/grid-constant/mirror modes).
    """
    c = np.moveaxis(np.asarray(coeff, dtype=np.float64), axis, -1).copy()
    n = c.shape[-1]
    if n == 1:
        return np.moveaxis(c, -1, axis)
    z = _CUBIC_POLE
    c *= (1.0 - z) * (1.0 - 1.0 / z)

    # Initial causal coefficient (exact mirror boundary, Unser 1991).
    zn = z
    iz = 1.0 / z
    z2n = z ** (n - 1)
    s = c[..., 0] + z2n * c[..., n - 1]
    z2n = z2n * z2n * iz
    for k in range(1, n - 1):
        s = s + (zn + z2n) * c[..., k]
        zn *= z
        z2n *= iz
    c[..., 0] = s / (1.0 - z ** (2 * n - 2))
    # Causal recursion.
    for i in range(1, n):
        c[..., i] += z * c[..., i - 1]
    # Initial anti-causal coefficient (mirror boundary).
    c[..., n - 1] = (z / (z * z - 1.0)) * (z * c[..., n - 2] + c[..., n - 1])
    # Anti-causal recursion.
    for i in range(n - 2, -1, -1):
        c[..., i] = z * (c[..., i + 1] - c[..., i])
    return np.moveaxis(c, -1, axis)


def _cubic_bspline(t):
    """Cubic B-spline kernel B3(t)."""
    t = np.abs(t)
    out = np.zeros_like(t)
    m1 = t < 1.0
    m2 = (t >= 1.0) & (t < 2.0)
    out[m1] = 2.0 / 3.0 - t[m1] ** 2 + 0.5 * t[m1] ** 3
    out[m2] = ((2.0 - t[m2]) ** 3) / 6.0
    return out


# Zero-padding margin used to emulate the infinite grid-constant (cval=0)
# extension. |pole| ~ 0.27 so its influence decays geometrically; 16 samples
# drives the residual vs scipy below 1e-14.
_CUBIC_PAD = 16


def _zoom_axis_cubic(coeff, axis, n_out, pad):
    """Cubic (order=3) resample of one axis with scipy ``grid_mode=True`` mapping.

    ``coeff`` is the B-spline coefficient array of the input *zero-padded by*
    ``pad`` on this axis; the original (unpadded) length is therefore
    ``coeff.shape[axis] - 2*pad``. Output index ``o`` samples padded coordinate
    ``(o + 0.5)/zoom - 0.5 + pad``. Taps outside the padded array use 0.
    """
    n_pad = coeff.shape[axis]
    n_in = n_pad - 2 * pad
    zoom = n_out / n_in
    o = np.arange(n_out, dtype=np.float64)
    x = (o + 0.5) / zoom - 0.5 + pad
    base = np.floor(x).astype(np.int64)
    out = np.zeros(
        [n_out if a == axis else coeff.shape[a] for a in range(coeff.ndim)],
        dtype=np.float64,
    )
    wshape = [1] * coeff.ndim
    wshape[axis] = n_out
    for tap in (-1, 0, 1, 2):
        idx = base + tap
        w = _cubic_bspline(x - idx)
        valid = (idx >= 0) & (idx < n_pad)
        vals = np.take(coeff, np.clip(idx, 0, n_pad - 1), axis=axis)
        out += vals * np.where(valid.reshape(wshape), w.reshape(wshape), 0.0)
    return out


def _zoom_spline(image, output_shape, order=1):
    """Separable resample of every axis (scipy ``zoom`` with ``grid_mode=True``).

    Order 1 (bilinear) is the only order used by the detection pipeline; order 3
    (bicubic) is used only on the OCR "try larger image" retry path. Both match
    ``scipy.ndimage.zoom(..., mode='grid-constant', cval=0, grid_mode=True)`` to
    ~1e-14 on the test corpus.
    """
    if order == 1:
        out = image
        for axis, n_out in enumerate(output_shape):
            out = _zoom_axis_bilinear(out, axis, int(n_out))
        return out
    if order == 3:
        pad = _CUBIC_PAD
        coeff = np.pad(np.asarray(image, dtype=np.float64), pad, mode='constant')
        for axis in range(coeff.ndim):
            coeff = _spline_prefilter_axis(coeff, axis)
        for axis, n_out in enumerate(output_shape):
            coeff = _zoom_axis_cubic(coeff, axis, int(n_out), pad)
        return coeff
    raise ValueError("Unsupported interpolation order: %r" % order)


def _clip_warp_output(input_image, output_image, cval=0.0):
    """Port of skimage's _clip_warp_output for mode='constant'."""
    min_val = float(np.min(input_image))
    max_val = float(np.max(input_image))
    if not (min_val <= cval <= max_val) and (
        float(np.min(output_image)) <= cval <= float(np.max(output_image))
    ):
        min_val = min(min_val, cval)
        max_val = max(max_val, cval)
    np.clip(output_image, min_val, max_val, out=output_image)


def resize(image, output_shape, order=1):
    """Replaces ``skimage.transform.resize(mode='constant', cval=0)``.

    ``order`` selects the spline interpolation order (1 = bilinear, the only
    order used by the detection pipeline; 3 = bicubic, used only on the OCR
    "try larger image" retry path). See :func:`_zoom_spline`.
    """
    image = np.asarray(image)
    input_shape = image.shape
    output_shape = tuple(int(s) for s in output_shape)
    anti_aliasing = (image.dtype != bool) and any(
        o < i for o, i in zip(output_shape, input_shape)
    )
    factors = np.divide(input_shape, output_shape, dtype=np.float64)

    img = img_as_float(image)  # convert_to_float(preserve_range=False)
    if anti_aliasing:
        sigma = np.maximum(0.0, (factors - 1.0) / 2.0)
        filtered = _gaussian_filter(img, sigma)
    else:
        filtered = img
    out = _zoom_spline(filtered, output_shape, order=order)
    _clip_warp_output(img, out, cval=0.0)
    return out


def rescale(image, scale, order=1):
    """Replaces ``skimage.transform.rescale(image, scale, mode='constant',
    channel_axis=None, anti_aliasing=True)``."""
    orig_shape = np.asarray(image.shape, dtype=np.float64)
    output_shape = np.maximum(np.round(scale * orig_shape), 1).astype(np.int64)
    return resize(image, tuple(output_shape), order=order)


# ---------------------------------------------------------------------------
# Morphology  (replaces skimage.morphology square/disk/closing/black_tophat)
# ---------------------------------------------------------------------------
def square(width):
    """Replaces ``skimage.morphology.square``."""
    return np.ones((width, width), dtype=np.uint8)


def disk(radius):
    """Replaces ``skimage.morphology.disk`` (default behaviour)."""
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.uint8)


def _morph(image, footprint, want_max):
    """Grayscale max (dilation) / min (erosion) over ``footprint``.

    Border handling is scipy ``mode='reflect'`` == ``numpy.pad('symmetric')``,
    matching skimage's grayscale erosion/dilation default. Footprint cells that
    are 0 are excluded by filling them with -inf (max) / +inf (min). For the
    symmetric footprints used here (square, disk) the footprint mirroring that
    skimage applies to ``dilation`` is a no-op.
    """
    fp = np.asarray(footprint).astype(bool)
    fh, fw = fp.shape
    rh, rw = fh // 2, fw // 2
    padded = np.pad(image.astype(np.float64), ((rh, rh), (rw, rw)), mode='symmetric')
    windows = sliding_window_view(padded, (fh, fw))  # (H, W, fh, fw)
    if np.all(fp):
        return (windows.max(axis=(-2, -1)) if want_max
                else windows.min(axis=(-2, -1)))
    fill = -np.inf if want_max else np.inf
    masked = np.where(fp, windows, fill)
    return masked.max(axis=(-2, -1)) if want_max else masked.min(axis=(-2, -1))


def dilation(image, footprint):
    return _morph(image, footprint, want_max=True)


def erosion(image, footprint):
    return _morph(image, footprint, want_max=False)


def closing(image, footprint):
    """Replaces ``skimage.morphology.closing`` = erosion(dilation(image))."""
    return erosion(dilation(image, footprint), footprint)


def black_tophat(image, footprint):
    """Replaces ``skimage.morphology.black_tophat`` = closing(image) - image."""
    return closing(image, footprint) - image


# ---------------------------------------------------------------------------
# Edge filter  (replaces skimage.filters.sobel_v)
# ---------------------------------------------------------------------------
_VSOBEL_WEIGHTS = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=np.float64) / 4.0


def _convolve2d_reflect(img, kernel):
    """2-D convolution with scipy ``mode='reflect'`` (== numpy 'symmetric')."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='symmetric')
    flipped = kernel[::-1, ::-1]  # convolution = correlation with flipped kernel
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += flipped[i, j] * padded[i:i + h, j:j + w]
    return out


def sobel_v(image):
    """Replaces ``skimage.filters.sobel_v`` (no border masking in skimage >=0.20)."""
    return _convolve2d_reflect(img_as_float(image), _VSOBEL_WEIGHTS)


# ---------------------------------------------------------------------------
# Otsu threshold  (replaces skimage.filters.threshold_otsu, nbins=256)
# ---------------------------------------------------------------------------
def threshold_otsu(image, nbins=256):
    """Replaces ``skimage.filters.threshold_otsu``."""
    image = np.asarray(image).ravel()
    lo, hi = float(image.min()), float(image.max())
    if lo == hi:
        return lo
    counts, edges = np.histogram(image, bins=nbins, range=(lo, hi))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    counts = counts.astype(np.float64)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    return bin_centers[idx]


# ---------------------------------------------------------------------------
# Marching squares  (replaces skimage.measure.find_contours)
# ---------------------------------------------------------------------------
def _get_fraction(from_value, to_value, level):
    if to_value == from_value:
        return 0.0
    return (level - from_value) / (to_value - from_value)


def _get_contour_segments(array, level, vertex_connect_high):
    """Pure-Python port of skimage's Cython ``_get_contour_segments`` (no mask).

    Returns a list of ``(from_point, to_point)`` pairs where each point is a
    ``(row, col)`` tuple of Python floats. Shared-edge points across adjacent
    cells are computed by the identical formula, so they compare equal as dict
    keys in :func:`_assemble_contours` (matching the Cython behaviour).
    """
    segments = []
    rows, cols = array.shape
    for r0 in range(rows - 1):
        r1 = r0 + 1
        for c0 in range(cols - 1):
            c1 = c0 + 1
            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]
            if ul != ul or ur != ur or ll != ll or lr != lr:  # any NaN
                continue

            square_case = 0
            if ul > level:
                square_case += 1
            if ur > level:
                square_case += 2
            if ll > level:
                square_case += 4
            if lr > level:
                square_case += 8

            if square_case == 0 or square_case == 15:
                continue

            top = (float(r0), c0 + _get_fraction(ul, ur, level))
            bottom = (float(r1), c0 + _get_fraction(ll, lr, level))
            left = (r0 + _get_fraction(ul, ll, level), float(c0))
            right = (r0 + _get_fraction(ur, lr, level), float(c1))

            if square_case == 1:
                segments.append((top, left))
            elif square_case == 2:
                segments.append((right, top))
            elif square_case == 3:
                segments.append((right, left))
            elif square_case == 4:
                segments.append((left, bottom))
            elif square_case == 5:
                segments.append((top, bottom))
            elif square_case == 6:
                if vertex_connect_high:
                    segments.append((left, top))
                    segments.append((right, bottom))
                else:
                    segments.append((right, top))
                    segments.append((left, bottom))
            elif square_case == 7:
                segments.append((right, bottom))
            elif square_case == 8:
                segments.append((bottom, right))
            elif square_case == 9:
                if vertex_connect_high:
                    segments.append((top, right))
                    segments.append((bottom, left))
                else:
                    segments.append((top, left))
                    segments.append((bottom, right))
            elif square_case == 10:
                segments.append((bottom, top))
            elif square_case == 11:
                segments.append((bottom, left))
            elif square_case == 12:
                segments.append((left, right))
            elif square_case == 13:
                segments.append((top, right))
            elif square_case == 14:
                segments.append((left, top))
    return segments


def _assemble_contours(segments):
    """Verbatim port of skimage.measure._find_contours._assemble_contours."""
    from collections import deque
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            else:
                if tail_num > head_num:
                    head.extend(tail)
                    contours.pop(tail_num, None)
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:
                    tail.extendleft(reversed(head))
                    starts.pop(head[0], None)
                    contours.pop(head_num, None)
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]


def find_contours(image, level, fully_connected='low', positive_orientation='low'):
    """Replaces ``skimage.measure.find_contours`` for the 2-D, no-mask case."""
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")
    segments = _get_contour_segments(
        np.asarray(image, dtype=np.float64), float(level), fully_connected == 'high'
    )
    contours = _assemble_contours(segments)
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]
    return contours


# ---------------------------------------------------------------------------
# Affine transforms  (replaces skimage.transform.SimilarityTransform / rotate)
# ---------------------------------------------------------------------------
class SimilarityTransform(object):
    """Minimal 2-D similarity transform (scale * rotation + translation).

    Reproduces the subset of ``skimage.transform.SimilarityTransform`` used by
    ``geometry.RotatedBox`` and :func:`rotate`. Coordinates are ``(x, y)`` =
    ``(col, row)``, matching skimage. Composition ``a + b`` yields a transform
    whose ``__call__`` applies ``a`` first then ``b`` (params = b.params @ a.params).
    """

    def __init__(self, matrix=None, scale=None, rotation=None, translation=None):
        if matrix is not None:
            self.params = np.asarray(matrix, dtype=np.float64).copy()
            return
        if scale is None:
            scale = 1.0
        if rotation is None:
            rotation = 0.0
        if translation is None:
            translation = (0.0, 0.0)
        c, s = np.cos(rotation), np.sin(rotation)
        self.params = np.array([
            [scale * c, -scale * s, translation[0]],
            [scale * s, scale * c, translation[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def __add__(self, other):
        # (self + other)(x) == other(self(x))
        return SimilarityTransform(matrix=other.params @ self.params)

    def _apply(self, matrix, coords):
        coords = np.asarray(coords, dtype=np.float64)
        single = coords.ndim == 1
        pts = np.atleast_2d(coords)
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        dst = homog @ matrix.T
        dst = dst[:, :2] / dst[:, 2:3]
        return dst[0] if single else dst

    def __call__(self, coords):
        return self._apply(self.params, coords)

    def inverse(self, coords):
        return self._apply(np.linalg.inv(self.params), coords)


def _warp_affine(image, matrix, output_shape, cval=0.0):
    """Bilinear (order=1) affine warp with constant(0) borders.

    ``matrix`` maps OUTPUT homogeneous coords ``[col, row, 1]`` to INPUT coords,
    exactly as skimage's fast affine ``warp`` uses ``inverse_map.params``.
    """
    img = img_as_float(image)
    out_rows, out_cols = int(output_shape[0]), int(output_shape[1])
    in_rows, in_cols = img.shape

    cc, rr = np.meshgrid(np.arange(out_cols), np.arange(out_rows))
    ones = np.ones_like(cc, dtype=np.float64)
    out_xy1 = np.stack([cc.ravel().astype(np.float64),
                        rr.ravel().astype(np.float64),
                        ones.ravel()], axis=0)  # (3, N)
    src = matrix @ out_xy1  # (3, N)
    src_x = src[0] / src[2]
    src_y = src[1] / src[2]

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = src_x - x0
    wy = src_y - y0

    def sample(yy, xx):
        valid = (xx >= 0) & (xx < in_cols) & (yy >= 0) & (yy < in_rows)
        xc = np.clip(xx, 0, in_cols - 1)
        yc = np.clip(yy, 0, in_rows - 1)
        vals = img[yc, xc]
        return np.where(valid, vals, cval)

    top = sample(y0, x0) * (1 - wx) + sample(y0, x1) * wx
    bot = sample(y1, x0) * (1 - wx) + sample(y1, x1) * wx
    out = top * (1 - wy) + bot * wy
    out = out.reshape(out_rows, out_cols)
    _clip_warp_output(img, out, cval=cval)
    return out


def rotate(image, angle, center, resize=True, cval=0.0):
    """Replaces ``skimage.transform.rotate(..., resize=True, mode='constant')``.

    ``angle`` in degrees, counter-clockwise. ``center`` is ``(x, y)`` = (col, row),
    matching skimage's convention.
    """
    rows, cols = image.shape[0], image.shape[1]
    center = np.asarray(center, dtype=np.float64)
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))
        tform4 = SimilarityTransform(translation=(minc, minr))
        tform = tform4 + tform
    else:
        output_shape = (rows, cols)

    tform.params[2] = (0, 0, 1)
    return _warp_affine(image, tform.params, output_shape, cval=cval)
