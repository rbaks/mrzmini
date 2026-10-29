"""CNI_FR (pre-2021 French national identity card) reading through MRZ.from_ocr: spec cases 1 to 16 and 20 to 23, plus checks of the rules they do not pin."""
import pytest

from mrzmini.text import MRZ, MRZCheckDigit, MRZOCRCleaner

# Synthetic example (fictitious identity)
L1 = 'IDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123'
L2 = '1507069012348JEAN<<PIERRE<<8503127M2'
# Official specimen (fictitious identity)
B1 = 'IDFRABERTHIER<<<<<<<<<<<<<<<<<<<<<<<'
B2 = '8806923102858CORINNE<<<<<<<6512068F6'
# ICAO TD2 starting with IDFRA (fictitious identity)
T1 = 'IDFRADUPONT<<MARIE<<<<<<<<<<<<<<<<<<'
T2 = 'AB12345671FRA8503127F3001019<<<<<<<8'
# ICAO 9303 TD2 specimen
E1 = 'I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<'
E2 = 'D231458907UTO7408122F1204159<<<<<<<6'

CNI_FR_KEYS = ['mrz_type', 'valid_score', 'raw_text',
               'type', 'country', 'number', 'issue_date', 'date_of_birth', 'nationality', 'sex', 'names', 'surname',
               'administrative_code',
               'check_number', 'check_date_of_birth', 'check_composite',
               'valid_number', 'valid_date_of_birth', 'valid_composite']
TD2_ONLY_ATTRS = ['optional1', 'optional2', 'personal_number', 'check_expiration_date', 'valid_expiration_date',
                  'check_personal_number', 'valid_personal_number']


def read(a, b):
    return MRZ.from_ocr(a + '\n' + b)


def l2_with_date_of_birth(dob):
    """L2 with another date of birth, its check digit and the composite check digit recomputed."""
    b = L2[:27] + dob + MRZCheckDigit.compute(dob) + 'M'
    return b + MRZCheckDigit.compute(L1 + b)


def test_reference_vectors_check_digits():
    assert MRZCheckDigit.compute(L2[:12]) == '8'
    assert MRZCheckDigit.compute(L2[27:33]) == '7'
    assert MRZCheckDigit.compute(L1 + L2[:35]) == '2'
    assert MRZCheckDigit.compute(B1 + B2[:35]) == '6'
    assert all(len(ln) == 36 for ln in [L1, L2, B1, B2, T1, T2, E1, E2])


def test_case_01_synthetic_example():
    m = read(L1, L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 100
    assert m.valid
    assert m.type == 'ID'
    assert m.country == 'FRA'
    assert m.surname == 'MARTIN'
    assert m.names == 'JEAN PIERRE'
    assert m.number == '150706901234'
    assert m.check_number == '8'
    assert m.issue_date == '1507'
    assert m.administrative_code == '069123'
    assert m.date_of_birth == '850312'
    assert m.check_date_of_birth == '7'
    assert m.sex == 'M'
    assert m.check_composite == '2'
    assert m.nationality == 'FRA'
    assert m.expiration_date is None
    assert m.valid_check_digits == [True, True, True]
    assert m.valid_line_lengths == [True, True]
    assert m.valid_misc == [True, True]
    assert m.aux['text'] == L1 + '\n' + L2
    assert m.aux['raw_text'] == L1 + '\n' + L2


@pytest.mark.parametrize('line1', [
    L1[:30] + '9' + L1[30:],               # case 2: a 9 inserted before 069123
    L1[:30] + '0' + L1[30:],               # case 3: a 0 inserted before 069123
    L1.replace('MARTIN', 'MAR9TIN'),       # case 4
    L1.replace('MARTIN', 'MARXTIN'),       # case 5
], ids=['case_02', 'case_03', 'case_04', 'case_05'])
def test_cases_02_to_05_line1_too_long_is_not_repaired(line1):
    assert len(line1) == 37
    m = read(line1, L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 68
    assert not m.valid
    assert m.valid_line_lengths == [False, True]
    assert m.valid_number and m.valid_date_of_birth
    assert not m.valid_composite
    assert m.valid_misc == [True, True]


def test_case_06_stray_leading_character():
    m = read('K' + L1, L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 65
    assert not m.valid
    assert m.valid_line_lengths == [False, True]
    assert m.valid_number and m.valid_date_of_birth
    assert not m.valid_composite
    assert m.valid_misc == [False, True]
    assert m.type == 'KI'


def test_case_07_prefix_normalisation():
    m = read('IOFRA' + L1[5:], L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 100
    assert m.valid
    assert m.type == 'ID' and m.country == 'FRA'


def test_case_08_wrong_number_check_digit():
    m = read(L1, L2[:12] + '1' + L2[13:])
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 42
    assert not m.valid
    assert m.valid_number is False
    assert m.valid_date_of_birth is True
    assert m.valid_composite is False


def test_case_09_wrong_composite_check_digit():
    m = read(L1, L2[:35] + '5')
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 71
    assert not m.valid
    assert m.valid_check_digits == [True, True, False]
    assert m.valid_line_lengths == [True, True]
    assert m.valid_misc == [True, True]


def test_case_10_impossible_date_of_birth():
    m = read(L1, l2_with_date_of_birth('851312'))
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 71
    assert not m.valid
    assert m.valid_number is True
    assert m.valid_date_of_birth is False
    assert m.valid_composite is True


def test_case_11_line2_too_short():
    m = read(L1, L2[:35])
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 68
    assert not m.valid
    assert m.valid_line_lengths == [True, False]
    assert m.valid_composite is False


def test_case_12_line1_too_short_is_not_repaired():
    line1 = L1[:29] + L1[30:]
    assert len(line1) == 35
    m = read(line1, L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 68
    assert not m.valid
    assert m.valid_line_lengths == [False, True]
    assert m.valid_composite is False


def test_case_13_official_specimen():
    m = read(B1, B2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 100
    assert m.valid
    assert m.surname == 'BERTHIER'
    assert m.names == 'CORINNE'
    assert m.administrative_code == '<<<<<<'
    assert m.number == '880692310285'
    assert m.issue_date == '8806'
    assert m.date_of_birth == '651206'
    assert m.sex == 'F'


# Expected to_dict() outputs, obtained with mrzmini 0.1.0 (git HEAD:mrzmini/text.py)
TD2_010 = {
    14: {'mrz_type': 'TD2', 'valid_score': 100, 'raw_text': E1 + '\n' + E2, 'type': 'I<', 'country': 'UTO',
         'number': 'D23145890', 'date_of_birth': '740812', 'expiration_date': '120415', 'nationality': 'UTO',
         'sex': 'F', 'names': 'ANNA MARIA', 'surname': 'ERIKSSON', 'optional1': '<<<<<<<', 'check_number': '7',
         'check_date_of_birth': '2', 'check_expiration_date': '9', 'check_composite': '6', 'valid_number': True,
         'valid_date_of_birth': True, 'valid_expiration_date': True, 'valid_composite': True},
    15: {'mrz_type': 'TD2', 'valid_score': 100, 'raw_text': T1 + '\n' + T2, 'type': 'ID', 'country': 'FRA',
         'number': 'AB1234567', 'date_of_birth': '850312', 'expiration_date': '300101', 'nationality': 'FRA',
         'sex': 'F', 'names': 'MARIE', 'surname': 'DUPONT', 'optional1': '<<<<<<<', 'check_number': '1',
         'check_date_of_birth': '7', 'check_expiration_date': '9', 'check_composite': '8', 'valid_number': True,
         'valid_date_of_birth': True, 'valid_expiration_date': True, 'valid_composite': True},
    16: {'mrz_type': 'TD2', 'valid_score': 54, 'raw_text': T1 + '\n' + T2[:5] + 'X' + T2[6:], 'type': 'ID',
         'country': 'FRA', 'number': 'AB123X567', 'date_of_birth': '850312', 'expiration_date': '300101',
         'nationality': 'FRA', 'sex': 'F', 'names': 'MARIE', 'surname': 'DUPONT', 'optional1': '<<<<<<<',
         'check_number': '1', 'check_date_of_birth': '7', 'check_expiration_date': '9', 'check_composite': '8',
         'valid_number': False, 'valid_date_of_birth': True, 'valid_expiration_date': True, 'valid_composite': False},
}


@pytest.mark.parametrize('case, lines', [
    (14, (E1, E2)),
    (15, (T1, T2)),
    (16, (T1, T2[:5] + 'X' + T2[6:])),
], ids=['case_14', 'case_15', 'case_16'])
def test_cases_14_to_16_icao_td2_unchanged(case, lines):
    m = read(*lines)
    assert m.mrz_type == 'TD2'
    assert m.valid_score == TD2_010[case]['valid_score']
    assert m.valid == (m.valid_score == 100)
    assert list(m.to_dict().items()) == list(TD2_010[case].items())
    assert not hasattr(m, 'administrative_code')
    assert not hasattr(m, 'issue_date')


def test_to_dict_cni_fr_keys_and_order():
    m = read(L1, L2)
    d = m.to_dict()
    assert list(d.keys()) == CNI_FR_KEYS
    assert d['raw_text'] == L1 + '\n' + L2
    assert d['issue_date'] == '1507'
    assert d['administrative_code'] == '069123'
    for key in ['expiration_date', 'check_expiration_date', 'valid_expiration_date',
                'optional1', 'optional2', 'personal_number']:
        assert key not in d


def test_to_dict_cni_fr_with_method():
    m = read(L1, L2)
    m.aux['method'] = 'direct'
    d = m.to_dict()
    assert list(d.keys()) == CNI_FR_KEYS + ['method']
    assert d['method'] == 'direct'


def test_cni_fr_object_has_no_td2_attributes():
    m = read(L1, L2)
    for attr in TD2_ONLY_ATTRS:
        assert not hasattr(m, attr), attr


def test_partial_cni_fr_has_no_td2_attributes():
    m = read(L1[:30] + '9' + L1[30:], L2)
    assert m.mrz_type == 'CNI_FR'
    for attr in TD2_ONLY_ATTRS:
        assert not hasattr(m, attr), attr


def test_direct_constructor_unchanged():
    # MRZ(lines) keeps the 0.1.0 behaviour: no cleaning, no arbitration, read as TD2
    m = MRZ([L1, L2])
    assert m.mrz_type == 'TD2'
    assert m.aux == {}
    assert list(m.to_dict().items()) == [
        ('mrz_type', 'TD2'), ('valid_score', 9), ('type', 'ID'), ('country', 'FRA'), ('number', '150706901'),
        ('date_of_birth', 'JEAN<<'), ('expiration_date', 'ERRE<<'), ('nationality', '348'), ('sex', 'I'),
        ('names', '069123'), ('surname', 'MARTIN'), ('optional1', '503127M'), ('check_number', '2'),
        ('check_date_of_birth', 'P'), ('check_expiration_date', '8'), ('check_composite', '2'),
        ('valid_number', False), ('valid_date_of_birth', False), ('valid_expiration_date', False),
        ('valid_composite', False)]
    assert not m.valid
    assert not hasattr(m, 'administrative_code')


# Review follow-up: errors the check digits cannot see (cases 20 to 22) and branches not covered by the spec cases

@pytest.mark.parametrize('line1, line2', [
    ('IDFRAMARTIN><<<<<<<<<<<<<<<<<<069123', L2),   # case 20: '>' counts as '<' in the check digits
    (L1, L2.replace('JEAN<<', 'JEAN><')),            # same on line 2
    (L1, L2[:3] + 'R' + L2[4:]),                     # case 22: 'R' (27) for '7' in the card number
], ids=['case_20', 'case_20_line2', 'case_22'])
def test_characters_invisible_to_check_digits_are_never_valid(line1, line2):
    m = read(line1, line2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_check_digits == [True, True, True]
    assert m.valid_format is False
    assert m.valid_score == 99
    assert not m.valid


def test_case_21_country_other_than_fra_is_never_valid():
    # Two characters off IDFRA: detected through line 2 only, so the prefix is not normalised
    m = read('IDPHA' + L1[5:], L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.country == 'PHA'
    assert m.nationality == 'FRA'
    assert m.valid_check_digits == [True, True, True]
    assert m.valid_format is False
    assert m.valid_score == 99
    assert not m.valid


def test_valid_format_not_in_to_dict():
    assert 'valid_format' not in read(L1, L2).to_dict()
    assert read(L1, L2).valid_format is True


def test_line_lengths_measured_on_raw_lines():
    # upper() turns 'ß' into 'SS': the raw line has 36 characters, the corrected one 37
    m = read(L1.replace('MARTIN', 'MARTIß'), L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_line_lengths == [True, True]
    assert m.valid_format is False
    assert not m.valid


def test_corsican_department_kept_in_administrative_code():
    # Class 'c' does not turn B into 8: with the 'n' table, 2B would become 28 and the composite would fail
    line1 = L1[:30] + '2B0012'
    m = read(line1, L2[:35] + MRZCheckDigit.compute(line1 + L2[:35]))
    assert m.mrz_type == 'CNI_FR'
    assert m.administrative_code == '2B0012'
    assert m.valid


def test_lowercase_input():
    m = MRZ.from_ocr((L1 + '\n' + L2).lower())
    assert m.mrz_type == 'CNI_FR'
    assert m.surname == 'MARTIN'
    assert m.names == 'JEAN PIERRE'
    assert m.valid


def test_detection_through_line2_only():
    m = read('XYFRA' + L1[5:], L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.type == 'XY'
    assert m.valid_misc == [False, True]
    assert m.valid_number and m.valid_date_of_birth
    assert not m.valid


def test_not_a_candidate_line2_shape():
    # Line 1 far from IDFRA and only 10 digits in the first 13 characters of line 2
    line1 = 'XYZAB' + L1[5:]
    line2 = 'XYW' + L2[3:]  # letters the numeric fixer leaves alone
    assert MRZ._detect_cni_fr([line1, line2]) is None  # pylint: disable=protected-access
    assert read(line1, line2).mrz_type != 'CNI_FR'


def test_three_lines_are_not_a_candidate():
    assert MRZ.from_ocr(L1 + '\n' + L2 + '\n' + 'A'*30).mrz_type != 'CNI_FR'


def test_arbitration_requires_a_valid_line2_check_digit():
    # Number and date of birth check digits wrong, composite recomputed: CNI_FR scores 42, above the TD2 reading,
    # but neither line 2 check digit is valid, so the standard reading is returned
    b = L2[:12] + '1' + L2[13:33] + '1M'
    b += MRZCheckDigit.compute(L1 + b)
    std = MRZ(MRZOCRCleaner.apply(L1 + '\n' + b))
    cni = MRZ._read_cni_fr([L1, b])  # pylint: disable=protected-access
    assert cni.valid_check_digits == [False, False, True]
    assert cni.valid_score > std.valid_score
    m = read(L1, b)
    assert m.mrz_type == 'TD2'
    assert m.valid_score == std.valid_score


def test_arbitration_falls_back_on_exception(monkeypatch):
    def boom(*args):
        raise ValueError('boom')
    monkeypatch.setattr(MRZ, '_parse_cni_fr', boom)
    m = read(L1, L2)
    assert m.mrz_type == 'TD2'
    assert m.aux['raw_text'] == L1 + '\n' + L2


# Second review follow-up: rules of valid_format, detection bounds and arbitration not pinned by the cases above

@pytest.mark.parametrize('line1, line2', [
    (L1, '1507A69012348JEAN<<PIERRE<<8503127M2'),   # case 23: A (10) for 0 in the department of the card number
    (L1, '1507<69012348JEAN<<PIERRE<<8503127M2'),   # '<' (0) for 0: never allowed in the card number
    (L1.replace('MARTIN', 'MARTıN'), L2),      # dotless i: uppercases to I, only the raw alphabet rule sees it
    (L1.replace('MARTIN', 'MA7TIN'), L2),           # '7' (7) for 'R' (27) in the surname: line 1 class check
], ids=['case_23', 'filler_in_card_number', 'raw_alphabet', 'line1_classes'])
def test_valid_format_rules(line1, line2):
    m = read(line1, line2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_check_digits == [True, True, True]
    assert m.valid_format is False
    assert m.valid_score == 99
    assert not m.valid


def test_corsican_department_in_card_number():
    number = '15072B901234'
    b = number + MRZCheckDigit.compute(number) + L2[13:35]
    m = read(L1, b + MRZCheckDigit.compute(L1 + b))
    assert m.number == '15072B901234'
    assert m.valid


def test_sex_must_be_m_or_f():
    b = L2[:34] + 'X'
    m = read(L1, b + MRZCheckDigit.compute(L1 + b))
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_misc == [True, False]
    assert m.valid_score == 97
    assert not m.valid


def test_surname_and_names_with_single_fillers():
    line1 = 'IDFRALE<GOFF' + '<'*18 + '069123'
    b = L2[:13] + 'ANNE<MARIE<<<<' + L2[27:35]
    m = read(line1, b + MRZCheckDigit.compute(line1 + b))
    assert (m.surname, m.names) == ('LE GOFF', 'ANNE MARIE')
    assert m.valid


def test_shifted_detection():
    # Line 2 without its last character does not have the line-2 shape: only the shifted test finds the candidate
    assert MRZ._detect_cni_fr(['K' + L1, L2[:35]]) == 'shifted'  # pylint: disable=protected-access
    m = read('K' + L1, L2[:35])
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_score == 62


@pytest.mark.parametrize('line1, expected', [
    (L1[:34], 'prefix'), (L1 + '<<', 'prefix'), (L1[:33], None), (L1 + '<<<', None),
], ids=['34', '38', '33', '39'])
def test_detection_line_length_bounds(line1, expected):
    assert MRZ._detect_cni_fr([line1, L2]) == expected  # pylint: disable=protected-access


def test_detection_line2_shape_bounds():
    line1 = 'XYZAB' + L1[5:]
    assert MRZ._detect_cni_fr([line1, 'XY' + L2[2:]]) == 'line2'  # pylint: disable=protected-access  # 11 digits
    assert MRZ._detect_cni_fr([line1, L2 + '<']) is None  # pylint: disable=protected-access  # must end the line


def test_arbitration_requires_a_strictly_higher_score(monkeypatch):
    std = MRZ(MRZOCRCleaner.apply(L1 + '\n' + L2))
    cni = MRZ._read_cni_fr([L1, L2])  # pylint: disable=protected-access
    cni.valid_score = std.valid_score
    monkeypatch.setattr(MRZ, '_read_cni_fr', staticmethod(lambda lines: cni))
    assert read(L1, L2).mrz_type == 'TD2'


# Third review follow-up: the department rule, prefix normalisation bounds, lowercase, line 2 shape and short lines

def card(administrative_code='069123', number='150706901234'):
    """A correct read of the synthetic example with another administrative code or card number."""
    line1 = L1[:30] + administrative_code
    b = number + MRZCheckDigit.compute(number) + L2[13:35]
    return line1, b + MRZCheckDigit.compute(line1 + b)


@pytest.mark.parametrize('line1, line2', [
    (L1, '1512A69012344JEAN<<PIERRE<<8503127M0'),   # case 24: the 2 before the A ends the issue month (12)
    (L1, '1502B09012340JEAN<<PIERRE<<8503127M8'),   # same with B and month 02
    (L1[:30] + '06912A', '1507069012348JEAN<<PIERRE<<8503127M9'),   # A after a 2 of the office, not the department
], ids=['case_24', 'month_02', 'office'])
def test_corsican_letter_only_after_a_2_of_the_same_department(line1, line2):
    m = read(line1, line2)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid_check_digits == [True, True, True]
    assert m.valid_format is False
    assert m.valid_score == 99
    assert not m.valid


@pytest.mark.parametrize('lines', [
    card(administrative_code='2A0123'), card(administrative_code='02A123'),
    card(administrative_code='2B0123'), card(administrative_code='02B123'),
    card(number='15072A901234'), card(number='150702A01234'),
], ids=['admin_2A0', 'admin_02A', 'admin_2B0', 'admin_02B', 'number_2A0', 'number_02A'])
def test_corsican_departments_are_valid(lines):
    m = read(*lines)
    assert m.mrz_type == 'CNI_FR'
    assert m.valid


def test_two_differences_in_prefix_are_not_normalised():
    m = read('IDFKB' + L1[5:], L2)
    assert m.mrz_type == 'CNI_FR'
    assert m.country == 'FKB'
    assert not m.valid


def test_two_differences_in_prefix_corrected_by_the_cleaner():
    # Spec case 21 note: on the line 2 path, the 'a' class still corrects 1 -> I and 4 -> A
    m = read('1DFR4' + L1[5:], L2)
    assert (m.type, m.country) == ('ID', 'FRA')
    assert m.valid


def test_lowercase_prefix_with_one_error():
    m = MRZ.from_ocr(('IOFRA' + L1[5:] + '\n' + L2).lower())
    assert m.mrz_type == 'CNI_FR'
    assert m.valid


def test_lowercase_line2_only_detection():
    assert MRZ._detect_cni_fr([('XYFRA' + L1[5:]).lower(), L2.lower()]) == 'line2'  # pylint: disable=protected-access


@pytest.mark.parametrize('line2, expected', [
    (L2[:27] + 'X' + L2[28:], None),                       # only 6 digits before the sex
    (L2[:34] + 'X' + L2[35:], None),                       # sex neither M nor F
    ('1512069012344JEAN<<PIERRE<<8503127M0', 'line2'),     # composite check digit 0
], ids=['six_digits', 'sex_X', 'composite_0'])
def test_detection_line2_shape(line2, expected):
    assert MRZ._detect_cni_fr(['XYZAB' + L1[5:], line2]) == expected  # pylint: disable=protected-access


def test_short_lines_fields():
    m = read(L1, L2[:35])   # case 11
    assert (m.administrative_code, m.check_composite, m.valid_format) == ('069123', '<', False)
    m = read(L1[:29] + L1[30:], L2)   # case 12
    assert (m.administrative_code, m.check_composite, m.valid_format) == ('69123<', '2', False)
