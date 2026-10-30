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
            # from_points gives an angle in (-pi/2, pi/2]: a box near -pi/2 would be
            # extracted turned by ~180 degrees, so take the same box the other way round.
            if box.angle < -np.pi / 4:
                box.angle += np.pi
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

    def __init__(self, use_original_image=True, extra_cmdline_params='', *, lang=None, accept=None, budget=None,
                 rank_key=None, keep_candidates=False):
        """
        :param lang: the tesseract model, see BoxToMRZ
        :param accept: stops at the first box whose MRZ satisfies it, see BoxToMRZ
        :param budget: an OCRBudget, shared with BoxToMRZ and checked before each box. Once spent, the best
            MRZ read so far is returned.
        :param rank_key, keep_candidates: see BoxToMRZ. rank_key also picks the best box.
        """
        self.budget = budget if budget is not None else OCRBudget()
        self.box_to_mrz = BoxToMRZ(use_original_image, extra_cmdline_params=extra_cmdline_params, lang=lang,
                                   accept=accept, budget=self.budget, rank_key=rank_key,
                                   keep_candidates=keep_candidates)

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
            mrzs.sort(key=lambda x: self.box_to_mrz.rank(x[3]))
            return mrzs[-1]


class BoxToMRZ(object):
    """Extracts ROI from the image, corresponding to a box found by MRZBoxLocator, does OCR and MRZ parsing on this region."""

    __provides__ = ['roi', 'text', 'mrz']
    __depends__ = ['box', 'img', 'img_small', 'scale_factor']

    def __init__(self, use_original_image=True, extra_cmdline_params='', *, lang=None, accept=None, budget=None,
                 rank_key=None, keep_candidates=False):
        """
        :param use_original_image: when True, the ROI is extracted from img, otherwise from img_small
        :param lang: the tesseract model passed to ocr() (a name such as 'mrz', or 'auto'); None leaves the
            choice to ocr(), i.e. the MRZMINI_TESSERACT_LANG environment variable, else tesseract's default.
            Stripped, '' meaning None. ValueError if extra_cmdline_params also has -l.
        :param accept: a predicate MRZ -> bool that stops the search: no further variant is tried once the
            current MRZ satisfies it. None means mrz.valid, as in 0.2.0, which tries the black-tophat rescale
            even after a valid black-tophat read (without rank_key only). The best variant is still chosen by rank.
        :param budget: an OCRBudget, checked before each variant and each OCR call. Once spent, the best
            variant read so far is returned. With a deadline, ocr() gets the time left as timeout.
        :param rank_key: a function MRZ -> comparable key that picks the best variant (strictly greater key; on a
            tie, the first one). None means valid_score, as in 0.2.0. Given, the CNI_FR reading MRZ.from_ocr
            computed but did not return competes as a read of its own.
        :param keep_candidates: keep every read in self.candidates, in reading order, the CNI_FR reading
            MRZ.from_ocr did not return included: one per OCR call (two with that CNI_FR reading), none for a call
            the budget skipped. Only observes: the reads chosen are the same. The list grows over all the calls of
            this object: use one per read.
        """
        self.lang = _check_lang(lang, extra_cmdline_params)
        self.use_original_image = use_original_image
        self.extra_cmdline_params = extra_cmdline_params
        self.accept = accept
        self.budget = budget if budget is not None else OCRBudget()
        self.rank_key = rank_key
        self.candidates = [] if keep_candidates else None

    def accepts(self, mrz):
        return bool(self.accept(mrz)) if self.accept is not None else mrz.valid

    def rank(self, mrz):
        return self.rank_key(mrz) if self.rank_key is not None else mrz.valid_score

    def _better(self, cur_text, cur_mrz, new_text, method):
        """Parses new_text, read by the variant `method`, and returns the best of (cur_text, cur_mrz) and its reads:
        a read replaces the current one only if its rank is strictly higher. cur_mrz None: the first read wins.

        new_text None: the budget skipped the OCR call, nothing was read. The current read is kept; without one, ''
        is parsed as in 0.2.0, but not kept as a candidate."""
        read = new_text is not None
        if not read:
            if cur_mrz is not None:
                # '' could only replace it with a rank_key, and was never read
                return cur_text, cur_mrz
            new_text = ''
        if self.rank_key is None and self.candidates is None:
            # As in 0.2.0: from_ocr alone
            reads = [MRZ.from_ocr(new_text)]
        else:
            reads = MRZ._from_ocr_reads(new_text)  # pylint: disable=protected-access
        for mrz in reads:
            mrz.aux['method'] = method
        if self.candidates is not None and read:
            self.candidates.extend(reads)
        if self.rank_key is None:
            # The CNI_FR reading from_ocr did not return is then only a candidate
            reads = reads[:1]
        for i, mrz in enumerate(reads):
            if cur_mrz is None or self.rank(mrz) > self.rank(cur_mrz):
                cur_text, cur_mrz = new_text, mrz
                if i + 1 < len(reads) and self.accepts(cur_mrz):
                    # The best read so far is accepted: the CNI_FR reading after it may not replace it
                    break
        return cur_text, cur_mrz

    def _ocr(self, img):
        """The OCR text of img, or None if the budget is spent (no call made)."""
        if self.budget.spent():
            return None
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

        if text is not None and ('>>' in text or ('>' in text and '<' not in text)) and not self.budget.spent():
            # Most probably we need to reverse the ROI
            roi = roi[::-1, ::-1]
            text = self._ocr(roi)

        text, mrz = self._better(None, None, text, 'direct')
        if '<' not in text:
            # Assume this is unrecoverable and stop here (TODO: this may be premature, although it saves time on useless stuff)
            return roi, text, mrz

        # Now try improving the result via hacks, while the budget lasts
        if not self.accepts(mrz):
            text, mrz = self._try_larger_image(roi, text, mrz)

        # Sometimes the filter used for enlargement is important!
        if not self.accepts(mrz):
            text, mrz = self._try_larger_image(roi, text, mrz, 1)

        if not self.accepts(mrz) and not self.budget.spent():
            text, mrz = self._try_black_tophat(roi, text, mrz)

        return roi, text, mrz

    def _try_larger_image(self, roi, cur_text, cur_mrz, filter_order=3, method=None):
        """Attempts to improve the OCR result by scaling the image. If the new mrz is better, returns it, otherwise returns
        the old mrz. method: of the new read, 'rescaled(<filter_order>)' by default."""
        if roi.shape[1] <= 700 and not self.budget.spent():
            scale_by = int(1050.0 / roi.shape[1] + 0.5)
            roi_lg = ip.rescale(roi, scale_by, order=filter_order)
            new_text = self._ocr(roi_lg)
            cur_text, cur_mrz = self._better(cur_text, cur_mrz, new_text, method or 'rescaled(%d)' % filter_order)
        return cur_text, cur_mrz

    def _try_black_tophat(self, roi, cur_text, cur_mrz):
        roi_b = ip.black_tophat(roi, ip.disk(5))
        # There are some examples where this line basically hangs for an undetermined amount of time.
        new_text = self._ocr(roi_b)
        cur_text, cur_mrz = self._better(cur_text, cur_mrz, new_text, 'black_tophat')

        # 0.2.0 tries the rescale even after a valid black-tophat read: kept without accept and rank_key, for parity.
        # With rank_key, its read could replace the valid one.
        if (self.accept is None and self.rank_key is None) or not self.accepts(cur_mrz):
            cur_text, cur_mrz = self._try_larger_image(roi_b, cur_text, cur_mrz, method='black_tophat(rescaled(3))')

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

    def __init__(self, file, extra_cmdline_params='', *, lang=None, accept=None, max_ocr_calls=None, deadline=None,
                 rank_key=None, keep_candidates=False):
        """
        :param file: see read_mrz
        :param lang: the tesseract model, see read_mrz. If given, ValueError here if extra_cmdline_params
            also has -l.
        :param accept, max_ocr_calls, deadline: see read_mrz_detailed. The bounds apply to the whole pipeline,
            TryOtherMaxWidth included; self.budget (an OCRBudget) counts the OCR calls.
        :param rank_key, keep_candidates: see read_mrz_detailed. The reads are kept, in reading order and without
            aux['view'], in self.components['mrz'].box_to_mrz.candidates.
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
                                                    accept=accept, budget=self.budget, rank_key=rank_key,
                                                    keep_candidates=keep_candidates))
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


#: A preset of four views for pages turned by a quarter, in order: see read_mrz_detailed
ROTATED_VIEWS = ((270, 250), (90, 250), (90, 500), (270, 500))


@dataclass(frozen=True)
class MRZResult(object):
    """The outcome of read_mrz_detailed, also when no MRZ was read."""

    #: The MRZ read: in pass 1, as read_mrz returns it without accept and bounds; an accepted read of pass 2
    #: otherwise. None if neither gave one (pass 1 not run or without a read of positive score, and pass 2 without
    #: an accepted read).
    mrz: MRZ | None
    #: accept(mrz), or mrz.valid without accept; False if mrz is None
    accepted: bool
    #: (rotation, thumbnail width) of the view the MRZ was read in: (0, 250) in pass 1, or (0, 1000) after
    #: TryOtherMaxWidth; in pass 2, the view of its box, e.g. (90, 1000) for a view added because (90, 250) was
    #: mostly white. None if mrz is None
    view: tuple[int, int] | None
    #: Calls to mrzmini.image.ocr made by this read, cut ones included
    ocr_calls: int
    #: Duration of the whole call in seconds, on the time.monotonic() clock
    elapsed_s: float
    #: True if max_ocr_calls or deadline stopped the read before its end, or the deadline passed during an OCR call
    budget_exhausted: bool
    #: With keep_candidates, every read made, from the best to the worst by rank_key (valid_score without it), in
    #: reading order on a tie; mrz is one of them, unless accept accepted the empty read of a zone whose first OCR
    #: call the deadline skipped. () otherwise
    candidates: tuple[MRZ, ...] = ()


def read_mrz_detailed(file, *, extra_cmdline_params='', lang=None, pass1=True, views=(), accept=None,
                      max_ocr_calls=None, deadline=None, rank_key=None, keep_candidates=False):
    """Like read_mrz, but returns an MRZResult, with the OCR calls counted and optionally bounded, and an optional
    second pass on rotated or wider views.

    :param file, extra_cmdline_params, lang: see read_mrz. A PDF without an embedded JPEG image gives an MRZResult
        with mrz None and no OCR call, where read_mrz fails.
    :param pass1: run pass 1, the read_mrz pipeline. False runs pass 2 only, so that the caller can run pass 1 on all
        its pages first; ValueError if views is then empty.
    :param views: the views of pass 2, run when pass 1 is off or its MRZ does not satisfy accept: (rotation,
        thumbnail width) pairs, the rotation counterclockwise in degrees (0, 90, 180 or 270, as PIL's Image.rotate)
        and the width a positive integer; ValueError otherwise. ROTATED_VIEWS holds a preset of four views.
        Pass 2 first locates the boxes of every view, without OCR: the loaded image is rotated (np.rot90, no copy),
        scaled down to the width (never up) and processed as in pass 1. A view without box whose image is mostly
        white, the TryOtherMaxWidth condition, adds the view (same rotation, 1000) at the end, once per rotation.
        The views pass 1 covered, (0, 250) and (0, 1000) after TryOtherMaxWidth, are skipped, and so is a view
        given twice. The boxes are then read by decreasing score (see _box_score), each in the full-resolution
        image of its own view, until a read satisfies accept. A read of pass 2 that does not satisfy it never
        replaces the MRZ of pass 1.
    :param accept: a predicate MRZ -> bool, replacing mrz.valid as the condition that stops the search: no further
        variant of a box and no further box once the best read so far satisfies it. Otherwise the pipeline is
        unchanged (best variant by rank, so an accepted read that does not rank strictly higher is not kept; best
        box by rank among those with a positive valid_score; TryOtherMaxWidth when there is none). If the MRZ of
        pass 1 satisfies it, pass 2 does not run.
    :param max_ocr_calls: maximum number of OCR calls for the whole read, both passes; ValueError if negative. 0
        runs no OCR.
    :param deadline: an instant on the time.monotonic() clock. No step starts after it, and each tesseract run gets
        the time left as timeout (killed then, read as empty). Steps without OCR (loading, scaling, box detection,
        ROI extraction, black-tophat) are not interrupted, but no view of pass 2 starts after it. If the deadline
        has already passed, the file is not even loaded.
    :param rank_key: a function MRZ -> comparable key, replacing valid_score wherever reads are compared: between
        the variants of a box (a strictly greater key wins; on a tie, the first read), and between the boxes of
        pass 1 (those with a positive valid_score, the last one on a tie). It is called on every read, including
        empty ones (mrz_type None). None means valid_score, as in 0.2.0. Given, the CNI_FR reading MRZ.from_ocr
        computed but did not return (a standard reading scoring higher, or no valid check digit on line 2) competes
        as a read of its own, so rank_key=lambda m: m.valid_score may pick it where None does not. accept still
        stops the search, whatever rank_key, but is tested on the best read so far: rank accepted reads first so
        that they are kept. Given, it also skips the black-tophat rescale 0.2.0 tries after a valid black-tophat
        read (one OCR call fewer), since that read is accepted. accept and rank_key never see aux['view'], set
        once the result is known. As in 0.2.0, accept sees the '|max_width' suffix TryOtherMaxWidth adds to the
        MRZ it returns; with keep_candidates, so does rank_key when ranking the candidates.
    :param keep_candidates: fill MRZResult.candidates with every read made, one per OCR call (two with the CNI_FR
        reading from_ocr did not return, a candidate even without rank_key), in pass 1 (TryOtherMaxWidth included)
        and pass 2, kept or not, also when mrz is None, ranked as rank_key ranks them. No image is kept. Each one
        has aux['method'] (its variant, e.g. 'black_tophat(rescaled(3))'), aux['raw_text'] and aux['view'], the
        view it was read in, as MRZResult.view. Two reads are no candidates: the text read upside down, replaced
        by the flipped read; and the empty read of a zone whose first OCR call the deadline skipped, so that mrz
        is a candidate unless accept accepts such an empty read. Only observes: mrz, accepted, view and the OCR
        calls are the same as without it (mrz, being one of the candidates, then also has aux['view']).

    When a bound is reached, the read stops without exception and returns the best MRZ read so far, with
    budget_exhausted=True. With the default parameters, the OCR calls and .mrz are those of read_mrz.
    """
    start = time.monotonic()
    views = _check_views(views)
    if not pass1 and not views:
        raise ValueError('pass1=False needs views')
    if pass1:
        p = MRZPipeline(file, extra_cmdline_params, lang=lang, accept=accept, max_ocr_calls=max_ocr_calls,
                        deadline=deadline, rank_key=rank_key, keep_candidates=keep_candidates)
        budget, box_to_mrz = p.budget, p.components['mrz'].box_to_mrz
    else:
        budget = OCRBudget(max_ocr_calls, deadline)
        box_to_mrz = BoxToMRZ(extra_cmdline_params=extra_cmdline_params, lang=lang, accept=accept, budget=budget,
                              rank_key=rank_key, keep_candidates=keep_candidates)
    reads = box_to_mrz.candidates  # None without keep_candidates
    # The view of each read, set on aux only once the result is known, so that accept and rank_key never see it
    read_views = []

    mrz = view = None
    if deadline is not None and budget.time_left() <= 0:
        # Already out of time: the file is not even loaded
        budget.exhausted = True
    elif (img := p['img'] if pass1 else Loader(file)()) is None:
        # A PDF without an embedded JPEG image: no MRZ, instead of failing on img.shape as read_mrz does
        pass
    else:
        covered = set()
        if pass1:
            # The boxes at 250 px first, as p.result would, to tell their reads from those of a TryOtherMaxWidth rerun
            p['mrz']
            _note_view(reads, read_views, (0, 250))
            mrz = p.result
            final_width = p.components['scaler'].max_width
            _note_view(reads, read_views, (0, final_width))
            view = None if mrz is None else (0, final_width)
            covered = {(0, 250), (0, final_width)}
        if views and not (mrz is not None and box_to_mrz.accepts(mrz)):
            for zone, zone_mrz in _read_zones(_locate_zones(img, views, covered, budget), img, box_to_mrz):
                _note_view(reads, read_views, zone.view)
                # Reads not accepted are only candidates
                if box_to_mrz.accepts(zone_mrz):
                    mrz, view = zone_mrz, zone.view
                    break
    accepted = mrz is not None and box_to_mrz.accepts(mrz)
    candidates = ()
    if reads is not None:
        # sorted() is stable with reverse=True too: reading order on a tie
        candidates = tuple(sorted(reads, key=box_to_mrz.rank, reverse=True))
        for read, read_view in zip(reads, read_views):
            read.aux['view'] = read_view
    return MRZResult(
        mrz=mrz,
        accepted=accepted,
        view=view,
        ocr_calls=budget.calls,
        elapsed_s=time.monotonic() - start,
        budget_exhausted=budget.exhausted,
        candidates=candidates,
    )


def _note_view(reads, read_views, view):
    """Appends `view` to read_views for each read of `reads` made since the last call. reads None (no candidates
    kept): nothing to do."""
    if reads is not None:
        read_views.extend([view] * (len(reads) - len(read_views)))


# ---------------------------------------------------------------------------
# Pass 2 of read_mrz_detailed
# ---------------------------------------------------------------------------
def _check_views(views):
    """views as a tuple of (rotation, width) pairs of ints; ValueError if one is not a pair of integers, a rotation
    is not 0, 90, 180 or 270, or a width is not positive."""
    checked = []
    for v in views:
        try:
            rotation, width = v
            rotation, width = operator.index(rotation), operator.index(width)
        except (TypeError, ValueError):
            raise ValueError('view %r: expected a pair (rotation, width) of integers' % (v,)) from None
        if rotation not in (0, 90, 180, 270) or width <= 0:
            raise ValueError('view %r: expected a rotation of 0, 90, 180 or 270 and a positive width' % (v,))
        checked.append((rotation, width))
    return tuple(checked)


@dataclass(frozen=True)
class _Zone(object):
    """A box found in pass 2, with what is needed to read it later: the rotated image is rebuilt from the loaded one
    (np.rot90 makes no copy), and the thumbnail is not kept."""

    view: tuple[int, int]
    box: RotatedBox
    scale_factor: float
    score: float


def _locate_zones(img, views, covered, budget):
    """Locates the boxes of each view, in order, without OCR, and returns them as _Zone, in view then box order.

    Skips the views in `covered` and repeated ones; adds the view of _wider_view at the end. No view starts once the
    budget is spent.
    """
    queue = list(dict.fromkeys(v for v in views if v not in covered))
    widened = set()
    zones = []
    for v in queue:  # grows while iterating
        if budget.spent():
            break
        rotation, width = v
        img_small, scale_factor = Scaler(width)(np.rot90(img, rotation // 90))
        img_binary = BooneTransform()(img_small)
        boxes = MRZBoxLocator()(img_binary)
        zones.extend(_Zone(v, box, scale_factor, _box_score(box, img_small)) for box in boxes)
        wider = _wider_view(v, boxes, img_binary, img)
        if wider is not None and rotation not in widened and wider not in queue and wider not in covered:
            widened.add(rotation)
            queue.append(wider)
    return zones


def _wider_view(view, boxes, img_binary, img):
    """Step 3 of pass 2: for a view without box whose image is mostly white, by the condition of TryOtherMaxWidth,
    the view to add: the same rotation at 1000 px. None otherwise.

    Provisional, to be kept or dropped after measuring what it recovers. As in pass 1, img.mean() > 0.95
    always holds for a uint8 grayscale image (0-255).
    """
    if not boxes and (img_binary.mean() < 0.01 or img.mean() > 0.95):
        return view[0], 1000
    return None


def _box_score(box, img_small):
    """How much `box` looks like an MRZ, from 0 to 1, on the thumbnail of its view, without extraction nor OCR.

    Provisional, to be confirmed against reading the views in order (see _reading_order). The product of two
    terms, each from 0 to 1:
      - elongation, width / height up to 10, about that of a 2-line MRZ box: a longer box is likelier a single text
        line than a better MRZ;
      - relative area, box area over thumbnail area, comparable between widths.

    The angle is left out: extract_from_image straightens a box whatever its angle, and on real pages a rotated view
    often finds the MRZ of an upright page as a vertical box (angle 0), which then reads fine.

    Rounded to 2 significant digits: the thumbnail does not tell more, and the same box found in two views (at 90
    and 270 degrees, the detection is almost symmetric) then gets the same score, so that the order of the views
    decides.
    """
    elongation = min(box.width / box.height / 10.0, 1.0) if box.height > 0 else 0.0
    area = box.area / float(img_small.shape[0] * img_small.shape[1])
    return float('%.2g' % (elongation * area))


def _reading_order(zones):
    """The zones of pass 2 in the order they are read: by decreasing score, then in their order (view, then box).

    Replace by `return list(zones)` to read the views in their order instead.
    """
    return sorted(zones, key=lambda zone: -zone.score)


def _read_zones(zones, img, box_to_mrz):
    """Reads the zones in _reading_order with box_to_mrz, each in the full-resolution image of its view, and yields
    (zone, mrz) for each one read. Stops when the budget of box_to_mrz is spent."""
    for zone in _reading_order(zones):
        if box_to_mrz.budget.spent():
            return
        rotated = np.rot90(img, zone.view[0] // 90)
        # BoxToMRZ reads in img (use_original_image), so img_small is not needed
        _roi, _text, mrz = box_to_mrz(zone.box, rotated, None, zone.scale_factor)
        yield zone, mrz
