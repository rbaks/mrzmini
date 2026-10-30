"""Public CNI_FR API: detection_mode, surname_truncated, names_truncated and looks_like_cni_fr.

All data is fictitious (synthetic CNI_FR of test_cni_fr.py, ICAO Doc 9303 specimens). The doctests of mrzmini/text.py
cover each detection mode too.
"""
import pytest

import mrzmini
from mrzmini import MRZ, looks_like_cni_fr
from mrzmini.text import MRZCheckDigit
from test_cni_fr import L1, L2
from test_icao_regression import DIRECT_CASES, FROM_OCR_CASES

NEW_ATTRS = ('detection_mode', 'surname_truncated', 'names_truncated')

# Surname and names filling their fields (25 and 14 characters), check digits recomputed
FULL_L1 = 'IDFRA' + 'DUPONT<MOREAU<LEFEBVRE<GA' + '069123'
FULL_L2 = '1507069012348' + 'ALEXANDRE<<MAR' + '8503127M8'

# to_dict() of these CNI_FR reads, captured with mrzmini 0.2.0 (git v0.2.0:mrzmini/text.py)
CNI_FR_020 = {
    'prefix': {
        'mrz_type': 'CNI_FR', 'valid_score': 100, 'raw_text': L1 + '\n' + L2, 'type': 'ID', 'country': 'FRA',
        'number': '150706901234', 'issue_date': '1507', 'date_of_birth': '850312', 'nationality': 'FRA', 'sex': 'M',
        'names': 'JEAN PIERRE', 'surname': 'MARTIN', 'administrative_code': '069123', 'check_number': '8',
        'check_date_of_birth': '7', 'check_composite': '2', 'valid_number': True, 'valid_date_of_birth': True,
        'valid_composite': True},
    'full': {
        'mrz_type': 'CNI_FR', 'valid_score': 100, 'raw_text': FULL_L1 + '\n' + FULL_L2, 'type': 'ID',
        'country': 'FRA', 'number': '150706901234', 'issue_date': '1507', 'date_of_birth': '850312',
        'nationality': 'FRA', 'sex': 'M', 'names': 'ALEXANDRE MAR', 'surname': 'DUPONT MOREAU LEFEBVRE GA',
        'administrative_code': '069123', 'check_number': '8', 'check_date_of_birth': '7', 'check_composite': '8',
        'valid_number': True, 'valid_date_of_birth': True, 'valid_composite': True},
    'shifted': {
        'mrz_type': 'CNI_FR', 'valid_score': 65, 'raw_text': 'K' + L1 + '\n' + L2, 'type': 'KI', 'country': 'DFR',
        'number': '150706901234', 'issue_date': '1507', 'date_of_birth': '850312', 'nationality': 'FRA', 'sex': 'M',
        'names': 'JEAN PIERRE', 'surname': 'AMARTIN', 'administrative_code': '<06912', 'check_number': '8',
        'check_date_of_birth': '7', 'check_composite': '2', 'valid_number': True, 'valid_date_of_birth': True,
        'valid_composite': False},
    'line2': {
        'mrz_type': 'CNI_FR', 'valid_score': 68, 'raw_text': 'XYFRA' + L1[5:] + '\n' + L2, 'type': 'XY',
        'country': 'FRA', 'number': '150706901234', 'issue_date': '1507', 'date_of_birth': '850312',
        'nationality': 'FRA', 'sex': 'M', 'names': 'JEAN PIERRE', 'surname': 'MARTIN',
        'administrative_code': '069123', 'check_number': '8', 'check_date_of_birth': '7', 'check_composite': '2',
        'valid_number': True, 'valid_date_of_birth': True, 'valid_composite': False},
}


def test_full_lines_check_digits():
    assert len(FULL_L1) == len(FULL_L2) == 36
    assert MRZCheckDigit.compute(FULL_L1 + FULL_L2[:35]) == FULL_L2[35]


def test_exports():
    assert 'looks_like_cni_fr' in mrzmini.__all__
    assert mrzmini.looks_like_cni_fr is looks_like_cni_fr


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('line1, line2, mode, surname_truncated, names_truncated', [
    (L1, L2, 'prefix', False, False),
    (FULL_L1, FULL_L2, 'prefix', True, True),
    # Only one field full
    (FULL_L1, FULL_L2[:13] + 'JEAN<<PIERRE<<' + FULL_L2[27:], 'prefix', True, False),
    (L1, FULL_L2, 'prefix', False, True),
    # One character off in the prefix, normalised: still reliable
    ('IOFRA' + FULL_L1[5:], FULL_L2, 'prefix', True, True),
    # A line of the wrong length: positions shifted or padded
    (L1[:20] + '<' + L1[20:], L2, 'prefix', None, None),
    (L1, L2[:35], 'prefix', None, None),
    ('K' + FULL_L1, FULL_L2, 'shifted', None, None),
    ('XYFRA' + FULL_L1[5:], FULL_L2, 'line2', None, None),
])
def test_cni_fr_attributes(line1, line2, mode, surname_truncated, names_truncated):
    m = MRZ.from_ocr(line1 + '\n' + line2)
    assert m.mrz_type == 'CNI_FR'
    assert (m.detection_mode, m.surname_truncated, m.names_truncated) == (mode, surname_truncated, names_truncated)


def test_truncation_reads_the_corrected_lines():
    # '0' read for 'O' as the last surname character: corrected to a letter, so the field is full
    line1 = FULL_L1[:29] + '0' + FULL_L1[30:]
    m = MRZ.from_ocr(line1 + '\n' + FULL_L2)
    assert m.surname_truncated is True


def test_truncation_none_when_correction_changes_the_length():
    # upper() turns 'ß' into 'SS': the raw line has 36 characters, the corrected one 37
    line1 = 'IDFRAßTRAUSS' + '<' * 18 + '069123'
    assert len(line1) == 36
    m = MRZ.from_ocr(line1 + '\n' + L2)
    assert (m.mrz_type, m.detection_mode) == ('CNI_FR', 'prefix')
    assert m.surname_truncated is None and m.names_truncated is None


@pytest.mark.parametrize('text', [p.values[0] for p in FROM_OCR_CASES] + [''],
                         ids=[p.id for p in FROM_OCR_CASES] + ['empty'])
def test_attributes_none_on_other_types(text):
    m = MRZ.from_ocr(text)
    assert m.mrz_type != 'CNI_FR'
    assert all(getattr(m, attr) is None for attr in NEW_ATTRS)
    assert not hasattr(m, 'valid_format')


@pytest.mark.parametrize('lines', [p.values[0] for p in DIRECT_CASES] + [[L1, L2], []])
def test_attributes_none_on_direct_constructor(lines):
    m = MRZ(lines)
    assert all(getattr(m, attr) is None for attr in NEW_ATTRS)
    assert not hasattr(m, 'valid_format')


# ---------------------------------------------------------------------------
# to_dict() unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('text, expected', [(p.values[0], p.values[1]) for p in FROM_OCR_CASES] + [
    (L1 + '\n' + L2, CNI_FR_020['prefix']),
    (FULL_L1 + '\n' + FULL_L2, CNI_FR_020['full']),
    ('K' + L1 + '\n' + L2, CNI_FR_020['shifted']),
    ('XYFRA' + L1[5:] + '\n' + L2, CNI_FR_020['line2']),
], ids=[p.id for p in FROM_OCR_CASES] + ['cni_fr_prefix', 'cni_fr_full', 'cni_fr_shifted', 'cni_fr_line2'])
def test_to_dict_unchanged(text, expected):
    m = MRZ.from_ocr(text)
    assert list(m.to_dict().items()) == list(expected.items())

    # Not read by to_dict, whatever their values
    for attr in NEW_ATTRS:
        setattr(m, attr, 'changed')
    assert list(m.to_dict().items()) == list(expected.items())


# ---------------------------------------------------------------------------
# looks_like_cni_fr
# ---------------------------------------------------------------------------
def test_looks_like_cni_fr_is_detect_cni_fr():
    for lines in ([L1, L2], ['K' + L1, L2], ['XYFRA' + L1[5:], L2], [L1], [L1, L2, L2], []):
        assert looks_like_cni_fr(lines) == MRZ._detect_cni_fr(lines)  # pylint: disable=protected-access
    # Any iterable of lines
    assert looks_like_cni_fr(iter([L1, L2])) == 'prefix'
    assert looks_like_cni_fr((L1, L2)) == 'prefix'


def test_looks_like_cni_fr_does_not_split():
    assert looks_like_cni_fr([L1 + '\n' + L2]) is None
    with pytest.raises(TypeError):
        looks_like_cni_fr(L1 + '\n' + L2)


def caller_shape(text):
    """A shape test a caller may write on its own: two lines of 34 to 38 characters, line 1 near IDFRA, maybe after
    one stray character. Returns its verdict and its lines."""
    lines = [ln.replace(' ', '') for ln in text.splitlines() if ln.strip()]

    def near_idfra(s):
        return sum(x != y for x, y in zip(s[:5].upper(), 'IDFRA')) <= 1

    if len(lines) != 2 or not all(34 <= len(ln) <= 38 for ln in lines):
        return False, lines
    return near_idfra(lines[0]) or near_idfra(lines[0][1:]), lines


def cni_l2(line1=L1, names='JEAN<<PIERRE', dob='850312'):
    """A line 2 for line1, with its check digits computed."""
    number = '150706901234'
    line2 = number + MRZCheckDigit.compute(number) + names.ljust(14, '<') + dob + MRZCheckDigit.compute(dob) + 'M'
    return line2 + MRZCheckDigit.compute(line1 + line2)


def broken(line2, *positions):
    """line2 with the check digits at `positions` (12: number, 33: date of birth) made wrong."""
    for p in positions:
        line2 = line2[:p] + str((int(line2[p]) + 1) % 10) + line2[p + 1:]
    return line2


TD3_SPECIMEN = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10'
TD1_SPECIMEN = 'IDAUT10000999<6<<<<<<<<<<<<<<<\n7109094F1112315AUT<<<<<<<<<<<6\nMUSTERFRAU<<ISOLDE<<<<<<<<<<<<'
# Two lines of text that are not an MRZ (fictitious address)
ADDRESS = 'DOUZERUEDESACACIAS7OOOOVILLEFICTIVE<<<<<<<<\nDOMICILIEAVILLEFICTIVEDEPUIS2O1O<<<<<<<<<<<'
FULL_SURNAME_L1 = ('IDFRA' + 'DUPONT<MOREAU<LEFEBVRE<GA').ljust(30, '<') + '069123'

# Cases with fictitious data, then the bounds of the shape test
SHAPE_CASES = {
    'perfect': L1 + '\n' + cni_l2(),
    'line1_one_char_more': L1[:20] + '<' + L1[20:] + '\n' + cni_l2(),
    'dob_check_broken': L1 + '\n' + broken(cni_l2(), 33),
    'junk_in_surname': L1[:20] + '.' + L1[21:] + '\n' + cni_l2(),
    'shifted': '9' + L1 + '\n' + cni_l2(),
    'no_line2_check': L1 + '\n' + broken(cni_l2(), 12, 33),
    'address': ADDRESS,
    'td3': TD3_SPECIMEN,
    'td1': TD1_SPECIMEN,
    'full_surname': FULL_SURNAME_L1 + '\n' + cni_l2(FULL_SURNAME_L1),
    'full_names': L1 + '\n' + cni_l2(names='ALEXANDRE<<MAR'),
    'spaces_and_blank_lines': '\n  ' + L1[:10] + ' ' + L1[10:] + '  \n\n' + cni_l2() + '\n \n',
    'line1_34': L1[:34] + '\n' + L2,
    'line1_33': L1[:33] + '\n' + L2,
    'line1_38': L1 + '<<\n' + L2,
    'line1_39': L1 + '<<<\n' + L2,
    'prefix_two_off': 'XXFRA' + L1[5:] + '\n' + L2,
    'shifted_two_off': 'KXXFRA' + L1[5:] + '\n' + L2,
    'lowercase_prefix': 'idfra' + L1[5:] + '\n' + L2,
    # Only line 2 has the CNI_FR shape: 'line2' for mrzmini, not a candidate for caller_shape
    'line2_only': 'XYZAB' + L1[5:] + '\n' + L2,
    'one_line': L1,
    'empty': '',
}


@pytest.mark.parametrize('text', SHAPE_CASES.values(), ids=SHAPE_CASES.keys())
def test_same_verdict_as_caller_shape(text):
    verdict, lines = caller_shape(text)
    mode = looks_like_cni_fr(lines)
    assert (mode in ('prefix', 'shifted')) == verdict


def test_shape_cases_cover_both_verdicts():
    verdicts = [caller_shape(text)[0] for text in SHAPE_CASES.values()]
    assert verdicts.count(True) >= 5 and verdicts.count(False) >= 5
    assert looks_like_cni_fr(caller_shape(SHAPE_CASES['line2_only'])[1]) == 'line2'
