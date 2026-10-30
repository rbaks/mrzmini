"""read_mrz_detailed: rank_key and keep_candidates.

No Tesseract needed: mrzmini.image.ocr is faked. Parity with read_mrz without options is tested in
test_pipeline_parity.py; here, the same scenarios also run with keep_candidates, which must change nothing.
"""
import json
from unittest import mock

import pytest

from mrzmini import MRZ, ROTATED_VIEWS, read_mrz, read_mrz_detailed
from mrzmini.text import MRZCheckDigit
from pipeline_recording import (CNI_FR_LINE2, CNI_L1, CNI_L2, SCORE_25, SCORE_44_A, SCORE_44_B, SCORE_62_A,
                                UPSIDE_DOWN, VALID, ScriptedOCR, image_png, run_read_mrz_detailed)
from test_detailed import ALL_VARIANTS, cni_fr_line2
from test_pipeline_parity import GOLDEN_PATH, SCENARIOS
from test_views import PAGES, TD3, OrientedOCR

# ICAO TD2 specimen with a wrong composite check digit: 3 valid check digits, score 77, not a CNI_FR candidate
TD2_77 = 'I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<\nD231458907UTO7408122F1204159<<<<<<<5'
# Read as a TD2 with 3 valid check digits (77) by MRZ.from_ocr, which also reads it as a CNI_FR whose two check
# digits of line 2 are valid (71) and returns the TD2. The names of the CNI_FR are the TD2 dates, letters for digits.
HIDDEN_CNI_FR = CNI_L1 + '\n' + '1507069011237GSOIOISMGSOIOI5503126M0'


def cni_fr_first(mrz):
    """A key that prefers greenloc's CNI_FR (both check digits of line 2 valid), then valid_score."""
    return cni_fr_line2(mrz), mrz.valid_score


def by_score(mrz):
    return mrz.valid_score


def no_line2_check():
    """A text MRZ.from_ocr reads as a TD2, whose CNI_FR reading scores higher (42) without any valid check digit
    on line 2, so that from_ocr does not return it."""
    b = CNI_L2[:12] + '1' + CNI_L2[13:33] + '1M'
    return CNI_L1 + '\n' + b + MRZCheckDigit.compute(CNI_L1 + b)


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    monkeypatch.delenv('MRZMINI_TESSERACT_LANG', raising=False)


def test_texts():
    td2 = MRZ.from_ocr(TD2_77)
    assert (td2.mrz_type, td2.valid_score) == ('TD2', 77)
    assert len(MRZ._from_ocr_reads(TD2_77)) == 1  # pylint: disable=protected-access
    cni = MRZ.from_ocr(CNI_FR_LINE2)
    assert (cni.mrz_type, cni.valid_score, cni_fr_line2(cni)) == ('CNI_FR', 71, True)

    returned, discarded = MRZ._from_ocr_reads(HIDDEN_CNI_FR)  # pylint: disable=protected-access
    assert MRZ.from_ocr(HIDDEN_CNI_FR).to_dict() == returned.to_dict()
    assert (returned.mrz_type, returned.valid_score) == ('TD2', 77)
    assert (discarded.mrz_type, discarded.valid_score, cni_fr_line2(discarded)) == ('CNI_FR', 71, True)
    assert discarded.aux == {'text': HIDDEN_CNI_FR, 'raw_text': HIDDEN_CNI_FR}

    returned, discarded = MRZ._from_ocr_reads(no_line2_check())  # pylint: disable=protected-access
    assert returned.mrz_type == 'TD2' and discarded.mrz_type == 'CNI_FR'
    assert discarded.valid_score > returned.valid_score
    assert not (discarded.valid_number or discarded.valid_date_of_birth)


def test_from_ocr_reads_one_when_not_discarded():
    # Valid standard reading: no CNI_FR reading computed; CNI_FR returned: the standard reading is not kept
    for text in (VALID, CNI_L1 + '\n' + CNI_L2, CNI_FR_LINE2, ''):
        assert len(MRZ._from_ocr_reads(text)) == 1  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# rank_key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('texts, td2_method, cni_method', [
    ([TD2_77, CNI_FR_LINE2], 'direct', 'rescaled(3)'),
    ([CNI_FR_LINE2, TD2_77], 'rescaled(3)', 'direct'),
], ids=['td2_first', 'cni_fr_first'])
def test_rank_key_between_variants(texts, td2_method, cni_method):
    data = image_png('card_td3_rgb')
    record, result = run_read_mrz_detailed(data, texts, None)
    assert (result.mrz.mrz_type, result.mrz.valid_score) == ('TD2', 77)
    assert record['method'] == td2_method

    ranked, result = run_read_mrz_detailed(data, texts, None, rank_key=cni_fr_first)
    assert (result.mrz.mrz_type, result.mrz.valid_score) == ('CNI_FR', 71)
    assert ranked['method'] == cni_method
    # Same calls: rank_key only chooses
    assert ranked['ocr_calls'] == record['ocr_calls'] and result.ocr_calls == 5
    assert result.accepted is False and result.view == (0, 250)


def test_rank_key_tie_keeps_first_variant():
    # Every read ties: the first variant is kept, as with valid_score
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None, rank_key=lambda m: 0)
    assert result.mrz.valid_score == 44 and record['method'] == 'direct'


def test_rank_key_between_boxes():
    # Box 1 reads the TD2 (77), box 2 the CNI_FR (71)
    data = image_png('card_td3_rotated_90')
    texts = [TD2_77, '', '', '', '', CNI_FR_LINE2]
    record, result = run_read_mrz_detailed(data, texts, None)
    assert result.mrz.mrz_type == 'TD2'

    ranked, result = run_read_mrz_detailed(data, texts, None, rank_key=cni_fr_first)
    assert result.mrz.mrz_type == 'CNI_FR'
    assert ranked['ocr_calls'] == record['ocr_calls'] and result.ocr_calls == 10

    # On a tie, the last box with a positive valid_score, as with valid_score
    record, result = run_read_mrz_detailed(data, [SCORE_62_A, '', '', '', '', SCORE_44_A], None,
                                           rank_key=lambda m: 0)
    assert result.mrz.valid_score == 44


def test_rank_key_picks_cni_fr_discarded_by_from_ocr():
    data = image_png('card_td3_rgb')
    record, result = run_read_mrz_detailed(data, [HIDDEN_CNI_FR], None)
    assert (result.mrz.mrz_type, result.mrz.valid_score) == ('TD2', 77)

    ranked, result = run_read_mrz_detailed(data, [HIDDEN_CNI_FR], None, rank_key=cni_fr_first)
    assert (result.mrz.mrz_type, result.mrz.valid_score) == ('CNI_FR', 71)
    assert ranked['method'] == 'direct' and ranked['result']['raw_text'] == HIDDEN_CNI_FR
    assert ranked['ocr_calls'] == record['ocr_calls']
    # The returned MRZ has no new aux key
    assert set(result.mrz.aux) == {'text', 'raw_text', 'method'}

    # Ranked by valid_score, the TD2 still wins
    _, result = run_read_mrz_detailed(data, [HIDDEN_CNI_FR], None, rank_key=by_score)
    assert result.mrz.mrz_type == 'TD2'


def test_rank_key_by_score_may_pick_a_discarded_cni_fr():
    # from_ocr returns the TD2, since no check digit of line 2 is valid; ranked by valid_score as a read of its
    # own, the CNI_FR scores higher. rank_key=None keeps 0.2.0's choice.
    data = image_png('card_td3_rgb')
    _, result = run_read_mrz_detailed(data, [no_line2_check()], None)
    assert result.mrz.mrz_type == 'TD2'
    _, result = run_read_mrz_detailed(data, [no_line2_check()], None, rank_key=by_score)
    assert result.mrz.mrz_type == 'CNI_FR'


def test_accept_by_discarded_cni_fr_stops():
    data = image_png('card_td3_rgb')
    texts = [HIDDEN_CNI_FR, VALID]
    record, result = run_read_mrz_detailed(data, texts, None, accept=cni_fr_line2, rank_key=cni_fr_first)
    assert result.ocr_calls == len(record['ocr_calls']) == 1
    assert result.accepted is True and result.mrz.mrz_type == 'CNI_FR'

    # Without rank_key, it is not a read of its own: the next variants are tried, and the valid TD3 wins
    record, result = run_read_mrz_detailed(data, texts, None, accept=cni_fr_line2)
    assert result.ocr_calls == 5
    assert result.mrz.mrz_type == 'TD3' and result.accepted is False


def test_accept_stops_whatever_rank_key():
    # The CNI_FR (rescaled(3)) is preferred and accepted: no rescaled(1), black-tophat or its rescale
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), [TD2_77, CNI_FR_LINE2, VALID], None,
                                           accept=cni_fr_line2, rank_key=cni_fr_first)
    assert result.ocr_calls == 2 and result.accepted is True and record['method'] == 'rescaled(3)'


def test_rank_key_in_pass2():
    # Pass 2 only, each zone reads the TD2 then the CNI_FR (5 variants). Ranked first, the CNI_FR is accepted at the
    # second call.
    zone_texts = [TD2_77, CNI_FR_LINE2, '', '', '']
    texts = iter(zone_texts * 20)
    image = PAGES['cni_90'][0]()
    with mock.patch('mrzmini.image.ocr', lambda img, **kwargs: next(texts)):
        result = read_mrz_detailed(image, pass1=False, views=ROTATED_VIEWS, accept=cni_fr_line2,
                                   rank_key=cni_fr_first)
    assert result.ocr_calls == 2 and result.accepted is True
    assert result.mrz.mrz_type == 'CNI_FR' and result.view == (270, 250)

    # Without rank_key, the TD2 (77) beats it in every zone, and nothing is accepted
    texts = iter(zone_texts * 20)
    with mock.patch('mrzmini.image.ocr', lambda img, **kwargs: next(texts)):
        result = read_mrz_detailed(image, pass1=False, views=ROTATED_VIEWS, accept=cni_fr_line2)
    assert result.ocr_calls > 5 and result.ocr_calls % 5 == 0 and result.mrz is None


# ---------------------------------------------------------------------------
# keep_candidates
# ---------------------------------------------------------------------------
def summary(candidates):
    return [(c.mrz_type, c.valid_score, c.aux['method'], c.aux['view']) for c in candidates]


def test_candidates_default_empty():
    _, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None)
    assert result.candidates == ()


def test_candidates_every_variant_with_its_method():
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None, keep_candidates=True)
    assert record['method'] == 'black_tophat'
    view = (0, 250)
    # By score, reading order on a tie (62 of rescaled(1) before the black-tophat rescale)
    assert summary(result.candidates) == [
        ('TD3', 81, 'black_tophat', view),
        ('TD3', 62, 'rescaled(1)', view),
        ('TD3', 62, 'black_tophat(rescaled(3))', view),
        ('TD3', 44, 'direct', view),
        ('TD3', 44, 'rescaled(3)', view),
    ]
    assert result.candidates[0] is result.mrz
    assert all(c.aux['raw_text'] for c in result.candidates)


def test_candidates_black_tophat_methods_when_not_kept():
    # Neither the black-tophat read (25) nor its rescale (44) beats rescaled(3) (62)
    texts = [SCORE_44_A, SCORE_62_A, '', SCORE_25, SCORE_44_B]
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), texts, None, keep_candidates=True)
    assert record['method'] == 'rescaled(3)'
    methods = {c.valid_score: c.aux['method'] for c in result.candidates if c.aux['method'].startswith('black')}
    assert methods == {25: 'black_tophat', 44: 'black_tophat(rescaled(3))'}
    assert [c.aux['method'] for c in result.candidates] == [
        'rescaled(3)', 'direct', 'black_tophat(rescaled(3))', 'black_tophat', 'rescaled(1)']


def test_candidates_when_no_mrz():
    # Two boxes read in vain (no '<'): no MRZ, two candidates
    record, result = run_read_mrz_detailed(image_png('card_td3_rotated_90'), [], None, keep_candidates=True)
    assert result.mrz is None and result.ocr_calls == 2
    assert summary(result.candidates) == [(None, 0, 'direct', (0, 250))] * 2
    assert all(c.aux['raw_text'] == '' for c in result.candidates)


def test_candidates_of_tryothermaxwidth():
    record, result = run_read_mrz_detailed(image_png('a4_card_480'), ['', VALID], None, keep_candidates=True)
    assert result.view == (0, 1000)
    # Only the MRZ returned has the suffix of TryOtherMaxWidth, as in 0.2.0
    assert summary(result.candidates) == [('TD3', 100, 'direct|max_width(1000)', (0, 1000)),
                                          (None, 0, 'direct', (0, 250))]
    assert result.candidates[0] is result.mrz


def test_candidates_include_discarded_cni_fr():
    # Without rank_key, the CNI_FR reading from_ocr did not return is a candidate, but not chosen
    record, result = run_read_mrz_detailed(image_png('card_td3_rgb'), [HIDDEN_CNI_FR], None, keep_candidates=True)
    assert result.mrz.mrz_type == 'TD2'
    assert summary(result.candidates)[:2] == [('TD2', 77, 'direct', (0, 250)), ('CNI_FR', 71, 'direct', (0, 250))]

    # Ranked by rank_key
    _, result = run_read_mrz_detailed(image_png('card_td3_rgb'), [HIDDEN_CNI_FR], None, keep_candidates=True,
                                      rank_key=cni_fr_first)
    assert result.mrz is result.candidates[0] and result.mrz.mrz_type == 'CNI_FR'
    assert result.candidates[1].mrz_type == 'TD2'


def test_candidates_of_pass2():
    make, lines = PAGES['cni_270']
    image = make()
    fake = OrientedOCR(lines)
    with mock.patch('mrzmini.image.ocr', fake):
        result = read_mrz_detailed(image, views=ROTATED_VIEWS, keep_candidates=True)
    assert result.view == (90, 250) and result.accepted is True
    pass1_calls = result.ocr_calls - 2
    assert pass1_calls > 0

    # The MRZ, then the reads in vain: those of pass 1, then the box of (270, 250), the page upside down
    assert result.candidates[0] is result.mrz
    assert summary(result.candidates[:1]) == [('CNI_FR', 100, 'direct', (90, 250))]
    assert [c.aux['view'] for c in result.candidates[1:]] == [(0, 250)] * pass1_calls + [(270, 250)]
    assert len(result.candidates) == result.ocr_calls


def test_candidates_of_pass2_not_accepted():
    # Pass 2 reads the MRZ but accept rejects it: kept among the candidates, with its view
    make, lines = PAGES['td3_270']
    fake = OrientedOCR(TD3)
    with mock.patch('mrzmini.image.ocr', fake):
        result = read_mrz_detailed(make(), pass1=False, views=ROTATED_VIEWS, accept=lambda m: False,
                                   keep_candidates=True)
    assert result.mrz is None and result.ocr_calls > 0
    best = result.candidates[0]
    assert best.valid and best.aux['view'] == (270, 250)
    assert len({c.aux['view'] for c in result.candidates}) > 1


def test_candidates_hold_no_image():
    _, result = run_read_mrz_detailed(image_png('card_td3_rgb'), ALL_VARIANTS, None, keep_candidates=True)
    assert all(set(c.aux) == {'text', 'raw_text', 'method', 'view'} for c in result.candidates)


# ---------------------------------------------------------------------------
# Invariants on the parity scenarios
# ---------------------------------------------------------------------------
def golden_scenarios():
    return json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))['scenarios']


@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', SCENARIOS)
def test_keep_candidates_only_observes(request, image, texts, extra, zone, n_calls, method):
    golden = golden_scenarios()
    data = image_png(image)
    _, default = run_read_mrz_detailed(data, texts, extra)
    for kwargs in ({'keep_candidates': True}, {'keep_candidates': True, 'rank_key': None}, {'rank_key': None}):
        record, result = run_read_mrz_detailed(data, texts, extra, **kwargs)

        # Same OCR calls and result as read_mrz in 0.2.0
        assert record == golden[request.node.callspec.id]
        assert (result.accepted, result.view, result.ocr_calls, result.budget_exhausted) == \
            (default.accepted, default.view, default.ocr_calls, default.budget_exhausted)
        if not kwargs.get('keep_candidates'):
            assert result.candidates == ()
            continue
        # One read per OCR call, but for the text read upside down before flipping the ROI
        assert len(result.candidates) == result.ocr_calls - (1 if texts and texts[0] == UPSIDE_DOWN else 0)
        scores = [c.valid_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)
        assert all({'method', 'raw_text', 'view'} <= set(c.aux) for c in result.candidates)
        if result.mrz is None:
            assert default.mrz is None
        else:
            assert any(c is result.mrz for c in result.candidates)
            assert result.mrz.aux['view'] == result.view
            assert {k: v for k, v in result.mrz.aux.items() if k != 'view'} == default.mrz.aux


@pytest.mark.parametrize('image, texts, extra, zone, n_calls, method', SCENARIOS)
def test_no_new_aux_key_by_default(image, texts, extra, zone, n_calls, method):
    data = image_png(image)
    args = () if extra is None else (False, extra)
    with mock.patch('mrzmini.image.ocr', ScriptedOCR(texts)):
        mrz = read_mrz(data, *args)
    kwargs = {} if extra is None else {'extra_cmdline_params': extra}
    with mock.patch('mrzmini.image.ocr', ScriptedOCR(texts)):
        result = read_mrz_detailed(data, **kwargs)
    for m in (mrz, result.mrz):
        assert m is None or set(m.aux) == {'text', 'raw_text', 'method'}
