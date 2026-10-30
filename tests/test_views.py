"""read_mrz_detailed: pass1 and views (pass 2 on rotated or wider views).

No Tesseract needed: mrzmini.image.ocr is faked. OrientedOCR stands for a tesseract that reads an MRZ only when the
ROI holds it whole and upright, so that a test passing proves the right view was read in the right orientation.
Parity with read_mrz without options is tested in test_pipeline_parity.py.
"""
import json
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import mrzmini
import mrzmini.image
from mrzmini.geometry import RotatedBox
from mrzmini.image import ROTATED_VIEWS, MRZPipeline, Scaler, read_mrz_detailed
from pipeline_recording import (CNI_FR_LINE2, CNI_L1, CNI_L2, SCORE_44_A, TD3_L1, TD3_L2, VALID, draw_card, glyph,
                                image_png, on_a4, run_read_mrz_detailed)
from test_detailed import cni_fr_line2
from test_pipeline_parity import GOLDEN_PATH, SCENARIOS


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    monkeypatch.delenv('MRZMINI_TESSERACT_LANG', raising=False)


# ---------------------------------------------------------------------------
# A tesseract that reads only an upright MRZ
# ---------------------------------------------------------------------------
def mrz_bitmap(lines, cell_w=20):
    """The ink of a 2-line MRZ laid out as draw_card does, cropped to the ink."""
    n = len(lines[0])
    cell_h = int(cell_w * 1.4)
    pad = max(1, cell_w // 6)
    ink = np.zeros((2 * cell_h + cell_h // 2, n * cell_w), bool)
    for r, line in enumerate(lines):
        y = r * (cell_h + cell_h // 2)
        for c, char in enumerate(line):
            x = c * cell_w + pad
            ink[y:y + cell_h, x:x + cell_w - 2 * pad] |= glyph(cell_h, cell_w - 2 * pad, char)
    return crop(ink)


def crop(ink):
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    return ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def ink_features(ink):
    """Width over height of the ink, and its density on a grid of 2 rows (the MRZ lines) by 8 columns."""
    h, w = ink.shape
    grid = [[ink[r * h // 2:(r + 1) * h // 2, c * w // 8:(c + 1) * w // 8].mean() for c in range(8)] for r in range(2)]
    return w / h, np.ravel(grid)


def is_upright(roi, bitmap):
    """True if roi holds the MRZ of bitmap whole and upright.

    The ink (darker than 3/4 of the median, the background) must have about the shape of the MRZ, and its density
    grid must correlate with that of the MRZ. The block glyphs are not symmetric (the filler runs of a TD3 line 1,
    for example), so an MRZ upside down does not correlate; one on its side, or partly cut, does not have its shape.
    """
    roi = np.asarray(roi, np.float64)
    if roi.ndim != 2 or roi.size == 0:
        return False
    ink = roi < np.median(roi) * 0.75
    if not ink.any():
        return False
    aspect, grid = ink_features(crop(ink))
    mrz_aspect, mrz_grid = ink_features(bitmap)
    if not 0.8 <= aspect / mrz_aspect <= 1.25 or grid.std() == 0:
        return False
    return np.corrcoef(grid, mrz_grid)[0, 1] >= 0.9


class OrientedOCR(object):
    """Stands for mrzmini.image.ocr: the text of `lines` if the ROI holds them upright, '' otherwise (no '<', so
    BoxToMRZ stops at that call). Records the shape of each ROI."""

    def __init__(self, lines):
        self.text = '\n'.join(lines)
        self.bitmap = mrz_bitmap(lines)
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(img.shape)
        return self.text if is_upright(img, self.bitmap) else ''


TD3 = [TD3_L1, TD3_L2]
CNI = [CNI_L1, CNI_L2]


def rotated(arr, angle):
    """The image turned counterclockwise by angle degrees, as PIL turns it."""
    return np.asarray(Image.fromarray(arr).rotate(angle, expand=True))


# Pages turned counterclockwise. On the CNI_FR card, the box of the upright MRZ has angle pi/2, so that its ROI is
# upright; on the TD3 card of 640 px, its angle is -1.559, so that its ROI is upside down (BoxToMRZ relies on tesseract
# reading '>>' to flip it, which OrientedOCR does not do) and the view at 180 degrees gives the upright ROI.
PAGES = {
    'cni_90': (lambda: rotated(draw_card(CNI, 640), 90), CNI),
    'cni_270': (lambda: rotated(draw_card(CNI, 640), 270), CNI),
    'td3_90': (lambda: rotated(draw_card(TD3, 640), 90), TD3),
    'td3_270': (lambda: rotated(draw_card(TD3, 640), 270), TD3),
    # MRZ lines too thin for 250 and 500 px thumbnails: only found at 1000 px
    'a4_small_card_90': (lambda: rotated(on_a4(draw_card(TD3, 320)), 90), TD3),
    'a4_small_card_270': (lambda: rotated(on_a4(draw_card(TD3, 320)), 270), TD3),
}


def read(image, lines, **kwargs):
    """read_mrz_detailed(image, **kwargs) with OrientedOCR; returns the MRZResult and the fake."""
    fake = OrientedOCR(lines)
    with mock.patch('mrzmini.image.ocr', fake):
        return read_mrz_detailed(image, **kwargs), fake


def detected_views(image, text='', **kwargs):
    """The views whose boxes read_mrz_detailed(image, **kwargs) locates in pass 2, in order, with an OCR that always
    reads `text`."""
    views = []
    wider_view = mrzmini.image._wider_view

    def recorded(view, *args):
        views.append(view)
        return wider_view(view, *args)

    with mock.patch('mrzmini.image._wider_view', recorded), \
            mock.patch('mrzmini.image.ocr', return_value=text):
        read_mrz_detailed(image, **kwargs)
    return views


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_exports():
    assert 'ROTATED_VIEWS' in mrzmini.__all__
    assert mrzmini.ROTATED_VIEWS == ((270, 250), (90, 250), (90, 500), (270, 500))


@pytest.mark.parametrize('kwargs', [
    {'pass1': False},
    {'pass1': False, 'views': []},
    {'views': [(45, 250)]},
    {'views': [(-90, 250)]},
    {'views': [(360, 250)]},
    {'views': [(90.0, 250)]},
    {'views': [('90', 250)]},
    {'views': [(90, 0)]},
    {'views': [(90, -250)]},
    {'views': [(90, 250.5)]},
    {'views': [(90,)]},
    {'views': [(90, 250, 1)]},
    {'views': [90]},
    {'views': [(90, 250), (270, None)]},
])
def test_invalid_views(kwargs):
    with mock.patch('mrzmini.imageproc.imread_gray', side_effect=AssertionError('image loaded')):
        with pytest.raises(ValueError):
            read_mrz_detailed(image_png('blank_page'), **kwargs)


def test_rot90_turns_like_pil():
    rgb = draw_card(CNI, 200)
    for arr in (rgb, rgb[..., 0]):
        for k in (1, 2, 3):
            assert np.array_equal(np.rot90(arr, k), rotated(arr, 90 * k))


def test_upright_criterion():
    bitmap = mrz_bitmap(TD3)
    page = (~mrz_bitmap(TD3, 13)).astype(np.float64)
    assert is_upright(page, bitmap)
    for k in (1, 2, 3):
        assert not is_upright(np.rot90(page, k), bitmap)
    assert not is_upright(page, mrz_bitmap(CNI))
    assert not is_upright(page[:, :page.shape[1] // 2], bitmap)

    # On the ROIs the pipeline extracts from every view of the upright CNI_FR card: the box of the (0, 250) view
    # only, and the same box seen upside down in the (180, 250) view once flipped
    bitmap = mrz_bitmap(CNI)
    img = mrzmini.image.Loader(draw_card(CNI, 640))()
    upright, flipped = [], []
    for rotation in (0, 90, 180, 270):
        view = np.rot90(img, rotation // 90)
        for width in (250, 500, 1000):
            pipeline = MRZPipeline(None)
            pipeline['img'] = view
            pipeline.replace_component('scaler', Scaler(width))
            for roi in pipeline['rois']:
                if is_upright(roi, bitmap):
                    upright.append((rotation, width))
                if is_upright(roi[::-1, ::-1], bitmap):
                    flipped.append((rotation, width))
    assert upright == [(0, 250)] and flipped == [(180, 250)]


# ---------------------------------------------------------------------------
# Pass 1 accepted: no view
# ---------------------------------------------------------------------------
def golden_scenarios():
    return json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))['scenarios']


ACCEPTED = [p for p in SCENARIOS if p.id in ('valid_first_call', 'valid_first_call_gray', 'flipped', 'cni_fr',
                                              'no_positive_score_then_wider', 'a4_card_wider_only')]


@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', ACCEPTED)
def test_pass1_accepted_skips_views(request, image, texts, extra, zone, n_calls, method):
    with mock.patch('mrzmini.image._locate_zones', side_effect=AssertionError('pass 2 run')):
        record, result = run_read_mrz_detailed(image_png(image), texts, extra, views=ROTATED_VIEWS)
    assert record == golden_scenarios()[request.node.callspec.id]
    assert result.accepted is True and result.ocr_calls == n_calls
    assert result.view == ((0, 1000) if method.endswith('|max_width(1000)') else (0, 250))


# ---------------------------------------------------------------------------
# Rotated pages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('page, view, n_calls', [
    ('cni_90', (270, 250), 1),
    # (270, 250) shows the page upside down: its box is read first (same score, first view), in vain
    ('cni_270', (90, 250), 2),
    # The ROI of the upright TD3 card is upside down (see PAGES): read in the view at 180 degrees
    ('td3_90', (90, 250), 2),
    ('td3_270', (270, 250), 1),
])
def test_rotated_page(page, view, n_calls):
    make, lines = PAGES[page]
    image = make()
    result, fake = read(image, lines, pass1=False, views=ROTATED_VIEWS)

    assert result.view == view and result.accepted is True
    assert result.mrz.valid and result.mrz.to_dict()['raw_text'] == '\n'.join(lines)
    assert result.ocr_calls == len(fake.calls) == n_calls
    assert result.budget_exhausted is False

    # Pass 1 first: its boxes (the MRZ on its side) are read in vain, then pass 2 finds the same view
    pass1, pass1_fake = read(image, lines)
    assert pass1.mrz is None and pass1.ocr_calls > 0
    both, fake = read(image, lines, views=ROTATED_VIEWS)
    assert both.view == view and both.accepted is True
    assert both.ocr_calls == pass1.ocr_calls + n_calls
    assert fake.calls[:pass1.ocr_calls] == pass1_fake.calls


@pytest.mark.parametrize('page, view', [('a4_small_card_90', (270, 1000)), ('a4_small_card_270', (90, 1000))])
def test_small_card_found_at_1000(page, view):
    make, lines = PAGES[page]
    image = make()
    result, fake = read(image, lines, pass1=False, views=ROTATED_VIEWS)

    assert result.view == view and result.accepted is True and result.mrz.valid
    # Not in ROTATED_VIEWS: added by the mostly white (270, 250) and (90, 250) views
    assert detected_views(image, pass1=False, views=ROTATED_VIEWS) == list(ROTATED_VIEWS) + [(270, 1000), (90, 1000)]

    # Without them, not found
    with mock.patch('mrzmini.image._wider_view', return_value=None):
        result, fake = read(image, lines, pass1=False, views=ROTATED_VIEWS)
    assert result.mrz is None


def test_score_ignores_angle():
    # A turned view often finds the MRZ of an upright page as a vertical box, which reads fine once extracted
    thumbnail = np.zeros((160, 250))
    scores = [mrzmini.image._box_score(RotatedBox([80, 125], 180, 29, angle), thumbnail)
              for angle in (0.0, 0.3, np.pi / 2, -np.pi / 2)]
    assert len(set(scores)) == 1 and scores[0] > 0
    # Larger first; elongation counts up to 10
    assert mrzmini.image._box_score(RotatedBox([80, 125], 90, 15, 0.0), thumbnail) < scores[0]
    assert mrzmini.image._box_score(RotatedBox([80, 125], 200, 20, 0.0), thumbnail) == \
        mrzmini.image._box_score(RotatedBox([80, 125], 400, 10, 0.0), thumbnail)


def test_wider_view_once_per_rotation():
    # A uint8 grayscale image always counts as mostly white: each view without box adds its rotation at 1000 px,
    # once per rotation and never twice
    blank = np.full((404, 640), 255, np.uint8)
    assert detected_views(blank, pass1=False, views=[(90, 250), (90, 500), (270, 250), (270, 250)]) == \
        [(90, 250), (90, 500), (270, 250), (90, 1000), (270, 1000)]
    assert detected_views(blank, pass1=False, views=[(90, 250), (90, 1000)]) == [(90, 250), (90, 1000)]
    # Pass 1 covered (0, 250), and (0, 1000) with TryOtherMaxWidth
    assert detected_views(blank, views=[(0, 250), (0, 500), (180, 250)]) == [(0, 500), (180, 250), (180, 1000)]


def test_wider_view_only_when_mostly_white():
    # A page of RGB noise has no box in any view, but is not mostly white: no view is added
    noise = np.random.default_rng(0).integers(0, 256, (404, 640, 3), dtype=np.uint8)
    assert detected_views(noise, pass1=False, views=ROTATED_VIEWS) == list(ROTATED_VIEWS)


def test_covered_views_follow_pass1():
    # Pass 1 reads a 44 at 250 px, so TryOtherMaxWidth does not run: (0, 1000) is not covered
    assert detected_views(image_png('card_td3_rgb'), SCORE_44_A, views=[(0, 1000), (0, 250)]) == [(0, 1000)]
    # Pass 1 reads nothing on the A4 page, so TryOtherMaxWidth reruns it at 1000 px: (0, 1000) is covered
    views = detected_views(image_png('a4_card_480'), views=[(0, 1000), (90, 250)])
    assert views[0] == (90, 250) and (0, 1000) not in views


def test_pass1_accepted_by_custom_accept_skips_views():
    # accept decides, not mrz.valid: a CNI_FR valid on line 2 only (score 71) stops the read after pass 1
    data = image_png('card_cni_fr_rgb')
    with mock.patch('mrzmini.image._locate_zones', side_effect=AssertionError('pass 2 run')):
        record, result = run_read_mrz_detailed(data, [CNI_FR_LINE2], None, views=ROTATED_VIEWS, accept=cni_fr_line2)
    assert record == run_read_mrz_detailed(data, [CNI_FR_LINE2], None, accept=cni_fr_line2)[0]
    assert result.accepted is True and result.mrz.valid is False and result.view == (0, 250)


@pytest.mark.parametrize('pass1', [True, False])
def test_pass2_forwards_ocr_options(pass1):
    fake = mock.Mock(return_value='')
    with mock.patch('mrzmini.image.ocr', fake):
        result = read_mrz_detailed(PAGES['cni_90'][0](), pass1=pass1, views=ROTATED_VIEWS, lang='mrz',
                                   extra_cmdline_params='--psm 6')
    assert result.ocr_calls == fake.call_count > 0
    assert all(c.kwargs == {'extra_cmdline_params': '--psm 6', 'lang': 'mrz'} for c in fake.call_args_list)


def test_pass2_does_not_replace_pass1():
    # Pass 1 reads a 44, pass 2 valid MRZs, which accept rejects: the 44 of pass 1 is kept
    phase = []
    locate_zones = mrzmini.image._locate_zones

    def pass2(*args):
        phase.append(2)
        return locate_zones(*args)

    def ocr(img, **kwargs):
        return VALID if phase else SCORE_44_A

    image = PAGES['cni_90'][0]()
    with mock.patch('mrzmini.image._locate_zones', pass2), mock.patch('mrzmini.image.ocr', ocr):
        pass1 = read_mrz_detailed(image, accept=lambda mrz: False)
        result = read_mrz_detailed(image, views=ROTATED_VIEWS, accept=lambda mrz: False)
    assert result.ocr_calls > pass1.ocr_calls
    assert result.mrz.valid_score == 44 and result.view == (0, 250) and result.accepted is False
    assert result.mrz.to_dict() == pass1.mrz.to_dict()

    # Without pass 1, nothing
    fake = mock.Mock(return_value=VALID)
    with mock.patch('mrzmini.image.ocr', fake):
        result = read_mrz_detailed(image, pass1=False, views=ROTATED_VIEWS, accept=lambda mrz: False)
    assert result.ocr_calls == fake.call_count > 0
    assert result.mrz is None and result.view is None and result.accepted is False


def test_pass2_only_loads_like_pass1():
    with pytest.raises(TypeError):
        read_mrz_detailed(object(), pass1=False, views=ROTATED_VIEWS)


# ---------------------------------------------------------------------------
# A caller's retries, replayed
# ---------------------------------------------------------------------------
def pr_retries(image, lines):
    """A caller's retries around mrzmini 0.2.0, in their order: read_mrz on the page turned by PIL, with the
    Scaler at the width of the view, TryOtherMaxWidth included, until a valid MRZ. Returns it (or None) and the OCR
    calls."""
    fake = OrientedOCR(lines)
    with mock.patch('mrzmini.image.ocr', fake):
        for rotation, width in ROTATED_VIEWS:
            pipeline = MRZPipeline(Image.fromarray(image).rotate(rotation, expand=True))
            pipeline.replace_component('scaler', Scaler(width))
            mrz = pipeline.result
            if mrz is not None and mrz.valid:
                return mrz, fake.calls
    return None, fake.calls


@pytest.mark.parametrize('page', sorted(PAGES))
def test_no_more_calls_than_pr(page):
    make, lines = PAGES[page]
    image = make()
    pr_mrz, pr_calls = pr_retries(image, lines)
    result, fake = read(image, lines, pass1=False, views=ROTATED_VIEWS)

    assert pr_mrz is not None and result.accepted is True
    assert result.ocr_calls <= len(pr_calls)
    # The same read; only the method differs (theirs has '|max_width(1000)' after TryOtherMaxWidth)
    pr_dict, our_dict = pr_mrz.to_dict(), result.mrz.to_dict()
    assert pr_dict.pop('method').startswith(our_dict.pop('method'))
    assert our_dict == pr_dict


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def test_max_ocr_calls_over_both_passes():
    make, lines = PAGES['cni_90']
    image = make()
    unbounded, fake = read(image, lines, views=ROTATED_VIEWS)
    pass1_calls = read(image, lines)[0].ocr_calls
    assert unbounded.ocr_calls == pass1_calls + 1 and unbounded.accepted is True

    # Pass 1 spends the budget: pass 2 locates no view
    assert detected_views(image, views=ROTATED_VIEWS, max_ocr_calls=pass1_calls) == []
    result, bounded_fake = read(image, lines, views=ROTATED_VIEWS, max_ocr_calls=pass1_calls)
    assert result.ocr_calls == pass1_calls and result.mrz is None and result.budget_exhausted is True
    assert bounded_fake.calls == fake.calls[:pass1_calls]

    result, bounded_fake = read(image, lines, views=ROTATED_VIEWS, max_ocr_calls=pass1_calls + 1)
    assert result.accepted is True and result.budget_exhausted is False
    assert bounded_fake.calls == fake.calls


class Clock(object):
    """Stands for the time module in mrzmini.image."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


def test_deadline_between_views():
    # Locating the boxes of a view takes 1 s on this clock: the deadline falls after the first view
    clock = Clock()
    locator = mrzmini.image.MRZBoxLocator

    class SlowLocator(locator):
        def __call__(self, img_binary):
            clock.now += 1
            return super().__call__(img_binary)

    make, lines = PAGES['cni_90']
    with mock.patch('mrzmini.image.time', clock), mock.patch('mrzmini.image.MRZBoxLocator', SlowLocator), \
            mock.patch('mrzmini.image._wider_view', wraps=mrzmini.image._wider_view) as wider:
        result, fake = read(make(), lines, pass1=False, views=ROTATED_VIEWS, deadline=clock.now + 1)

    assert [c.args[0] for c in wider.call_args_list] == [(270, 250)]
    # Its box is located but not read: no step after the deadline
    assert result.mrz is None and result.ocr_calls == 0 and result.budget_exhausted is True
    assert result.elapsed_s == 1


def test_deadline_between_zones():
    # Each OCR call takes 1 s: the first zone read (in vain) uses up the time, the second one is not read
    clock = Clock()
    make, lines = PAGES['cni_270']
    fake = OrientedOCR(lines)

    def slow_ocr(img, **kwargs):
        clock.now += 1
        return fake(img, **kwargs)

    extract = RotatedBox.extract_from_image
    with mock.patch('mrzmini.image.time', clock), mock.patch('mrzmini.image.ocr', slow_ocr), \
            mock.patch.object(RotatedBox, 'extract_from_image', autospec=True, side_effect=extract) as extracted:
        result = read_mrz_detailed(make(), pass1=False, views=ROTATED_VIEWS, deadline=clock.now + 1)

    assert result.ocr_calls == 1 and result.mrz is None and result.budget_exhausted is True
    # Not even extracted: the full-resolution rotation is not paid after the deadline
    assert extracted.call_count == 1

    # With one more second, the second zone is read
    clock = Clock()
    fake = OrientedOCR(lines)
    with mock.patch('mrzmini.image.time', clock), mock.patch('mrzmini.image.ocr', slow_ocr):
        result = read_mrz_detailed(make(), pass1=False, views=ROTATED_VIEWS, deadline=clock.now + 2.5)
    assert result.ocr_calls == 2 and result.view == (90, 250) and result.budget_exhausted is False
