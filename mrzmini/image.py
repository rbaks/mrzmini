'''
mrzmini.image: the MRZ extraction pipeline.

Adapted from PassportEye's ``passporteye/mrz/image.py``. The pipeline structure
and parameters are reproduced exactly; only the underlying image-processing
calls are routed through :mod:`mrzmini.imageproc` (numpy/Pillow) instead of
``skimage`` / ``imageio``, and PDF loading goes through :mod:`mrzmini.pdf`
instead of ``pdfminer``. One deliberate difference: aux['method'] is always set,
so TryOtherMaxWidth no longer raises KeyError: 'method' (an early exit of
BoxToMRZ now reports 'direct').

Author of original: Konstantin Tretyakov. License: MIT.
'''
import io
import operator
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import imageproc as ip
from .pdf import extract_first_jpeg_in_pdf
from .pipeline import Pipeline
from .geometry import RotatedBox
from .ocr import ocr, _check_lang
from .text import MRZ


class OCRBudget(object):
    """Bounds the OCR calls of one read: at most `max_calls` calls, and no step started after `deadline`, an
    instant on the time.monotonic() clock. None: no bound. ValueError if max_calls is negative.

    `calls` counts the calls to mrzmini.image.ocr; `exhausted` is set when a bound stopped the read. Not
    thread-safe: one budget per read, never shared between concurrent reads.
    """

    def __init__(self, max_calls=None, deadline=None):
        if max_calls is not None:
            max_calls = operator.index(max_calls)
            if max_calls < 0:
                raise ValueError('max_ocr_calls=%d: expected 0 or more' % max_calls)
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.exhausted = False

    def time_left(self):
        """Seconds left before the deadline (negative once passed), None without deadline."""
        return None if self.deadline is None else self.deadline - time.monotonic()

    def spent(self):
        """True, and `exhausted` set, if no further step may start: max_calls calls made or deadline passed."""
        if self.max_calls is not None and self.calls >= self.max_calls:
            self.exhausted = True
        elif self.deadline is not None and self.time_left() <= 0:
            self.exhausted = True
        return self.exhausted


class Loader(object):
    """Loads `file` to `img`.

    `file` may be a path (str, PDF included), bytes, a binary stream, a PIL.Image.Image (converted like a
    decoded file, without Image.open) or a numpy.ndarray: 2D as grayscale with its dtype kept (bool converted
    like a 1-bit image), 3D uint8 with 3 or 4 channels as RGB(A). Any other type raises TypeError.
    """

    __depends__ = []
    __provides__ = ['img']

    def __init__(self, file, as_gray=True, pdf_aware=True):
        self.file = file
        self.as_gray = as_gray
        self.pdf_aware = pdf_aware

    def _imread(self, file):
        """Proxy to mrzmini.imageproc.imread_gray (replaces skimage.io.imread)."""
        return ip.imread_gray(file)

    def __call__(self):
        if isinstance(self.file, str):
            if self.pdf_aware and self.file.lower().endswith('.pdf'):
                with open(self.file, 'rb') as f:
                    img_data = extract_first_jpeg_in_pdf(f)
                if img_data is None:
                    return None
                return self._imread(img_data)
            else:
                return self._imread(self.file)
        elif isinstance(self.file, (bytes, io.IOBase)):
            return self._imread(self.file)
        elif isinstance(self.file, Image.Image):
            return ip.to_gray(self.file)
        elif isinstance(self.file, np.ndarray):
            return self._from_array(self.file)
        raise TypeError('unsupported image type %s: expected a path (str), bytes, a binary stream, '
                        'a PIL.Image.Image or a numpy.ndarray' % type(self.file).__name__)

    def _from_array(self, arr):
        if arr.size == 0 or arr.dtype.kind not in 'biuf':
            raise TypeError('unsupported array of shape %s and dtype %s: expected a non-empty array of numbers'
                            % (arr.shape, arr.dtype))
        if arr.ndim == 2 and arr.dtype == bool:
            # Like a 1-bit image file (mode '1'), converted to luma: kept as bool, no box would be found
            return ip.to_gray(Image.fromarray(arr))
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[-1] in (3, 4):
            # The luma product rounds differently on another memory layout (e.g. a CHW buffer seen as HWC)
            return ip.rgb_to_gray(np.ascontiguousarray(arr))
        raise TypeError('unsupported array of shape %s and dtype %s: expected 2D (grayscale) '
                        'or 3D uint8 with 3 or 4 channels (RGB or RGBA)' % (arr.shape, arr.dtype))


class Scaler(object):
    """Scales `image` down to `img_scaled` so that its width is at most 250."""

    __depends__ = ['img']
    __provides__ = ['img_small', 'scale_factor']

    def __init__(self, max_width=250):
        self.max_width = max_width

    def __call__(self, img):
        scale_factor = self.max_width / float(img.shape[1])
        if scale_factor <= 1:
            img_small = ip.rescale(img, scale_factor)
        else:
            scale_factor = 1.0
            img_small = img
        return img_small, scale_factor


class BooneTransform(object):
    """Processes `img_small` according to Hans Boone's method
    (http://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)
    Outputs a `img_binary` - a result of threshold_otsu(closing(sobel(black_tophat(img_small)))"""

    __depends__ = ['img_small']
    __provides__ = ['img_binary']

    def __init__(self, square_size=5):
        self.square_size = square_size

    def __call__(self, img_small):
        m = ip.square(self.square_size)
        img_th = ip.black_tophat(img_small, m)
        img_sob = abs(ip.sobel_v(img_th))
        img_closed = ip.closing(img_sob, m)
        threshold = ip.threshold_otsu(img_closed)
        return img_closed > threshold


class MRZBoxLocator(object):
    """Extracts putative MRZs as RotatedBox instances from the contours of `img_binary`"""

    __depends__ = ['img_binary']
    __provides__ = ['boxes']

    def __init__(self, max_boxes=4, min_points_in_contour=50, min_area=500, min_box_aspect=5, angle_tol=0.1,
                 lineskip_tol=1.5, box_type='bb'):
        self.max_boxes = max_boxes
        self.min_points_in_contour = min_points_in_contour
        self.min_area = min_area
        self.min_box_aspect = min_box_aspect
        self.angle_tol = angle_tol
        self.lineskip_tol = lineskip_tol
        self.box_type = box_type

    def __call__(self, img_binary):
        cs = ip.find_contours(img_binary, 0.5)

        # Collect contours into RotatedBoxes
        results = []
        for c in cs:
            # Now examine the bounding box. If it is too small, we ignore the contour
            ll, ur = np.min(c, 0), np.max(c, 0)
            wh = ur - ll
            if wh[0] * wh[1] < self.min_area:
                continue

            # Finally, construct the rotatedbox. If its aspect ratio is too small, we ignore it
            rb = RotatedBox.from_points(c, self.box_type)
            if rb.height == 0 or rb.width / rb.height < self.min_box_aspect:
                continue

            # All tests fine, add to the list
            results.append(rb)

        # Next sort and leave only max_boxes largest boxes by area
        results.sort(key=lambda x: -x.area)
        return self._fixup_boxes(self._merge_boxes(results[0:self.max_boxes]))

    def _are_aligned_angles(self, b1, b2):
        "Are two boxes aligned according to their angle?"
        return abs(b1 - b2) <= self.angle_tol or abs(np.pi - abs(b1 - b2)) <= self.angle_tol

    def _are_nearby_parallel_boxes(self, b1, b2):
        "Are two boxes nearby, parallel, and similar in width?"
        if not self._are_aligned_angles(b1.angle, b2.angle):
            return False
        # Otherwise pick the smaller angle and see whether the two boxes are close according to the "up" direction wrt that angle
        angle = min(b1.angle, b2.angle)
        return abs(np.dot(b1.center - b2.center, [-np.sin(angle), np.cos(angle)])) < self.lineskip_tol * (
            b1.height + b2.height) and (b1.width > 0) and (b2.width > 0) and (0.5 < b1.width / b2.width < 2.0)

    def _merge_any_two_boxes(self, box_list):
        """Given a list of boxes, finds two nearby parallel ones and merges them. Returns false if none found."""
        n = len(box_list)
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_nearby_parallel_boxes(box_list[i], box_list[j]):
                    # Remove the two boxes from the list, add a new one
                    a, b = box_list[i], box_list[j]
                    merged_points = np.vstack([a.points, b.points])
                    merged_box = RotatedBox.from_points(merged_points, self.box_type)
                    if merged_box.width / merged_box.height >= self.min_box_aspect:
                        box_list.remove(a)
                        box_list.remove(b)
                        box_list.append(merged_box)
                        return True
        return False

    def _merge_boxes(self, box_list):
        """Mergest nearby parallel boxes in the given list."""
        while self._merge_any_two_boxes(box_list):
            pass
        return box_list

    def _fixup_boxes(self, box_list):
        """If the box's angle is np.pi/2 +- 0.01, we shall round it to np.pi/2.

        This way image extraction is fast and introduces no distortions.
        and this may be more important than being perfectly straight
        similar for 0 angle.
        The box_list is fixed in-place (but also returned).
        """
        for box in box_list:
            if abs(abs(box.angle) - np.pi / 2) <= 0.01:
                box.angle = np.pi / 2
            if abs(box.angle) <= 0.01:
                box.angle = 0.0
        return box_list


class ExtractAllBoxes(object):
    """Extract all the images from the boxes, for external OCR processing"""

    __provides__ = ['rois']
    __depends__ = ['boxes', 'img', 'scale_factor']

    def __call__(self, boxes, img, scale_factor):
        return [box.extract_from_image(img, 1.0 / scale_factor) for box in boxes]


class FindFirstValidMRZ(object):
    """Iterates over boxes found by MRZBoxLocator, passes them to BoxToMRZ, finds the first valid MRZ
    or the best-scoring MRZ"""

    __provides__ = ['box_idx', 'roi', 'text', 'mrz']
    __depends__ = ['boxes', 'img', 'img_small', 'scale_factor', '__data__']

    def __init__(self, use_original_image=True, extra_cmdline_params='', *, lang=None, accept=None, budget=None):
        """
        :param lang: the tesseract model, see BoxToMRZ
        :param accept: stops at the first box whose MRZ satisfies it, see BoxToMRZ
        :param budget: an OCRBudget, shared with BoxToMRZ and checked before each box. Once spent, the best
            MRZ read so far is returned.
        """
        self.budget = budget if budget is not None else OCRBudget()
        self.box_to_mrz = BoxToMRZ(use_original_image, extra_cmdline_params=extra_cmdline_params, lang=lang,
                                   accept=accept, budget=self.budget)

    def __call__(self, boxes, img, img_small, scale_factor, data):
        mrzs = []
        data['__debug__mrz'] = []
        for i, b in enumerate(boxes):
            if self.budget.spent():
                break
            roi, text, mrz = self.box_to_mrz(b, img, img_small, scale_factor)
            data['__debug__mrz'].append((roi, text, mrz))
            if self.box_to_mrz.accepts(mrz):
                return i, roi, text, mrz
            elif mrz.valid_score > 0:
                mrzs.append((i, roi, text, mrz))
        if not mrzs:
            return None, None, None, None
        else:
            mrzs.sort(key=lambda x: x[3].valid_score)
            return mrzs[-1]


class BoxToMRZ(object):
    """Extracts ROI from the image, corresponding to a box found by MRZBoxLocator, does OCR and MRZ parsing on this region."""

    __provides__ = ['roi', 'text', 'mrz']
    __depends__ = ['box', 'img', 'img_small', 'scale_factor']

    def __init__(self, use_original_image=True, extra_cmdline_params='', *, lang=None, accept=None, budget=None):
        """
        :param use_original_image: when True, the ROI is extracted from img, otherwise from img_small
        :param lang: the tesseract model passed to ocr() (a name such as 'mrz', or 'auto'); None leaves the
            choice to ocr(), i.e. the MRZMINI_TESSERACT_LANG environment variable, else tesseract's default.
            Stripped, '' meaning None. ValueError if extra_cmdline_params also has -l.
        :param accept: a predicate MRZ -> bool that stops the search: no further variant is tried once the
            current MRZ satisfies it. None means mrz.valid, as in 0.2.0, which tries the black-tophat rescale
            even after a valid black-tophat read. The best variant is still chosen by valid_score.
        :param budget: an OCRBudget, checked before each variant and each OCR call. Once spent, the best
            variant read so far is returned. With a deadline, ocr() gets the time left as timeout.
        """
        self.lang = _check_lang(lang, extra_cmdline_params)
        self.use_original_image = use_original_image
        self.extra_cmdline_params = extra_cmdline_params
        self.accept = accept
        self.budget = budget if budget is not None else OCRBudget()

    def accepts(self, mrz):
        return bool(self.accept(mrz)) if self.accept is not None else mrz.valid

    def _ocr(self, img):
        if self.budget.spent():
            return ''
        self.budget.calls += 1
        # lang and timeout are passed only when set, so that ocr() gets exactly the arguments of 0.2.0 by default
        kwargs = {}
        if self.lang is not None:
            kwargs['lang'] = self.lang
        timeout = self.budget.time_left()
        if timeout is not None:
            kwargs['timeout'] = timeout
        text = ocr(img, extra_cmdline_params=self.extra_cmdline_params, **kwargs)
        if timeout is not None and self.budget.time_left() <= 0:
            # Tesseract killed at the deadline, or done after it: the read is out of time either way
            self.budget.exhausted = True
        return text

    def __call__(self, box, img, img_small, scale_factor):
        img = img if self.use_original_image else img_small
        scale = 1.0 / scale_factor if self.use_original_image else 1.0
        roi = box.extract_from_image(img, scale)
        text = self._ocr(roi)

        if ('>>' in text or ('>' in text and '<' not in text)) and not self.budget.spent():
            # Most probably we need to reverse the ROI
            roi = roi[::-1, ::-1]
            text = self._ocr(roi)

        if '<' not in text:
            # Assume this is unrecoverable and stop here (TODO: this may be premature, although it saves time on useless stuff)
            mrz = MRZ.from_ocr(text)
            mrz.aux['method'] = 'direct'
            return roi, text, mrz

        mrz = MRZ.from_ocr(text)
        mrz.aux['method'] = 'direct'

        # Now try improving the result via hacks, while the budget lasts
        if not self.accepts(mrz):
            text, mrz = self._try_larger_image(roi, text, mrz)

        # Sometimes the filter used for enlargement is important!
        if not self.accepts(mrz):
            text, mrz = self._try_larger_image(roi, text, mrz, 1)

        if not self.accepts(mrz) and not self.budget.spent():
            text, mrz = self._try_black_tophat(roi, text, mrz)

        return roi, text, mrz

    def _try_larger_image(self, roi, cur_text, cur_mrz, filter_order=3):
        """Attempts to improve the OCR result by scaling the image. If the new mrz is better, returns it, otherwise returns
        the old mrz."""
        if roi.shape[1] <= 700 and not self.budget.spent():
            scale_by = int(1050.0 / roi.shape[1] + 0.5)
            roi_lg = ip.rescale(roi, scale_by, order=filter_order)
            new_text = self._ocr(roi_lg)
            new_mrz = MRZ.from_ocr(new_text)
            new_mrz.aux['method'] = 'rescaled(%d)' % filter_order
            if new_mrz.valid_score > cur_mrz.valid_score:
                cur_mrz = new_mrz
                cur_text = new_text
        return cur_text, cur_mrz

    def _try_black_tophat(self, roi, cur_text, cur_mrz):
        roi_b = ip.black_tophat(roi, ip.disk(5))
        # There are some examples where this line basically hangs for an undetermined amount of time.
        new_text = self._ocr(roi_b)
        new_mrz = MRZ.from_ocr(new_text)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux['method'] = 'black_tophat'
            cur_text, cur_mrz = new_text, new_mrz

        # 0.2.0 tries the rescale even after a valid black-tophat read: kept without accept, for parity
        if self.accept is None or not self.accepts(cur_mrz):
            new_text, new_mrz = self._try_larger_image(roi_b, cur_text, cur_mrz)
            if new_mrz.valid_score > cur_mrz.valid_score:
                new_mrz.aux['method'] = 'black_tophat(rescaled(3))'
                cur_text, cur_mrz = new_text, new_mrz

        return cur_text, cur_mrz


class TryOtherMaxWidth(object):
    """
    If mrz was not found so far in the current pipeline,
    changes the max_width parameter of the scaler to 1000 and reruns the pipeline again.
    """

    __provides__ = ['mrz_final']
    __depends__ = ['mrz', '__pipeline__']

    def __init__(self, other_max_width=1000, *, budget=None):
        """
        :param budget: an OCRBudget, checked before rerunning: once spent, no rerun.
        """
        self.other_max_width = other_max_width
        self.budget = budget if budget is not None else OCRBudget()

    def __call__(self, mrz, __pipeline__):
        # We'll only try this if we see that img_binary.mean() is very small or img.mean() is very large (i.e. image is mostly white).
        if mrz is None and (__pipeline__['img_binary'].mean() < 0.01 or __pipeline__['img'].mean() > 0.95) \
                and not self.budget.spent():
            __pipeline__.replace_component('scaler', Scaler(self.other_max_width))
            new_mrz = __pipeline__['mrz']
            if new_mrz is not None:
                new_mrz.aux['method'] = new_mrz.aux.get('method', 'direct') + '|max_width(%d)' % self.other_max_width
            mrz = new_mrz
        return mrz


class MRZPipeline(Pipeline):
    """This is the "currently best-performing" pipeline for parsing MRZ from a given image file."""

    def __init__(self, file, extra_cmdline_params='', *, lang=None, accept=None, max_ocr_calls=None, deadline=None):
        """
        :param file: see read_mrz
        :param lang: the tesseract model, see read_mrz. If given, ValueError here if extra_cmdline_params
            also has -l.
        :param accept, max_ocr_calls, deadline: see read_mrz_detailed. The bounds apply to the whole pipeline,
            TryOtherMaxWidth included; self.budget (an OCRBudget) counts the OCR calls.
        """
        super(MRZPipeline, self).__init__()
        self.version = '1.0'  # In principle we might have different pipelines in use, so possible backward compatibility is an issue
        self.file = file
        self.budget = OCRBudget(max_ocr_calls, deadline)
        self.add_component('loader', Loader(file))
        self.add_component('scaler', Scaler())
        self.add_component('boone', BooneTransform())
        self.add_component('box_locator', MRZBoxLocator())
        self.add_component('mrz', FindFirstValidMRZ(extra_cmdline_params=extra_cmdline_params, lang=lang,
                                                    accept=accept, budget=self.budget))
        self.add_component('other_max_width', TryOtherMaxWidth(budget=self.budget))

        # Step used by extract_mrz_rois (not even invoked by the standard result method)
        self.add_component('extractor', ExtractAllBoxes())

    @property
    def result(self):
        return self['mrz_final']


def read_mrz(file, save_roi=False, extra_cmdline_params='', *, lang=None):
    """The main interface function to this module, encapsulating the recognition pipeline.
       Given an image filename, runs MRZPipeline on it, returning the parsed MRZ object.

    :param file: A filename (image or PDF), bytes or a binary stream to read the file data from, or an image
        already in memory: a PIL.Image.Image, or a numpy.ndarray (2D grayscale, or 3D uint8 RGB/RGBA).
        Any other type raises TypeError.
    :param save_roi: when this is True, the .aux['roi'] field will contain the Region of Interest where the MRZ was parsed from.
    :param extra_cmdline_params: extra parameters to the OCR.
    :param lang: the tesseract model: a name such as 'mrz' or 'mrz+eng', or 'auto' for 'mrz' when installed.
        None (default) reads the MRZMINI_TESSERACT_LANG environment variable at each OCR call; if unset,
        tesseract runs without -l, as in PassportEye. ValueError if extra_cmdline_params also has -l, or if
        tesseract fails and the model is not installed. See mrzmini.ocr.ocr.
    """
    p = MRZPipeline(file, extra_cmdline_params, lang=lang)
    mrz = p.result
    if mrz is not None and save_roi:
        mrz.aux['roi'] = p['roi']
    return mrz


@dataclass(frozen=True)
class MRZResult(object):
    """The outcome of read_mrz_detailed, also when no MRZ was read."""

    #: The MRZ read, as read_mrz returns it without accept and bounds; None if no read has a positive score
    mrz: MRZ | None
    #: accept(mrz), or mrz.valid without accept; False if mrz is None
    accepted: bool
    #: (rotation, thumbnail width) the MRZ was found with: (0, 250), or (0, 1000) after TryOtherMaxWidth;
    #: None if mrz is None
    view: tuple[int, int] | None
    #: Calls to mrzmini.image.ocr made by this read, cut ones included
    ocr_calls: int
    #: Duration of the whole call in seconds, on the time.monotonic() clock
    elapsed_s: float
    #: True if max_ocr_calls or deadline stopped the read before its end, or the deadline passed during an OCR call
    budget_exhausted: bool


def read_mrz_detailed(file, *, extra_cmdline_params='', lang=None, accept=None, max_ocr_calls=None, deadline=None):
    """Like read_mrz, but returns an MRZResult, with the OCR calls counted and optionally bounded.

    :param file, extra_cmdline_params, lang: see read_mrz.
    :param accept: a predicate MRZ -> bool, replacing mrz.valid as the condition that stops the search: no further
        variant of a box and no further box once the best read so far satisfies it. Otherwise the pipeline is
        unchanged (best variant by valid_score, so an accepted read that does not score strictly higher is not
        kept; best box with a positive valid_score; TryOtherMaxWidth when there is none).
    :param max_ocr_calls: maximum number of OCR calls for the whole read; ValueError if negative. 0 runs no OCR.
    :param deadline: an instant on the time.monotonic() clock. No step starts after it, and each tesseract run gets
        the time left as timeout (killed then, read as empty). Steps without OCR (loading, scaling, box detection,
        ROI extraction, black-tophat) are not interrupted. If the deadline has already passed, the file is not even
        loaded.

    When a bound is reached, the read stops without exception and returns the best MRZ read so far, with
    budget_exhausted=True. With the default parameters, the OCR calls and .mrz are those of read_mrz.
    """
    start = time.monotonic()
    p = MRZPipeline(file, extra_cmdline_params, lang=lang, accept=accept, max_ocr_calls=max_ocr_calls,
                    deadline=deadline)
    if deadline is not None and p.budget.time_left() <= 0:
        # Already out of time: the file is not even loaded
        p.budget.exhausted = True
        mrz = None
    else:
        mrz = p.result
    return MRZResult(
        mrz=mrz,
        accepted=mrz is not None and p.components['mrz'].box_to_mrz.accepts(mrz),
        view=None if mrz is None else (0, p.components['scaler'].max_width),
        ocr_calls=p.budget.calls,
        elapsed_s=time.monotonic() - start,
        budget_exhausted=p.budget.exhausted,
    )
