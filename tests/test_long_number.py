"""Document numbers longer than 9 characters in TD1 and TD2 (ICAO 9303).

The first 9 characters fill the number field, a '<' takes the place of its check digit, and the rest of the number
opens the optional data, followed by the real check digit and a '<'. The check digit is computed on the field as
written, filler included: on the Portuguese card below, 092150918<ZY4 gives 2 (the digit printed), 092150918ZY4
gives 6.

Line 1 of the TD1 is that of a Portuguese ID card read by tesseract; the other lines are made up.
"""
import pytest

from mrzmini import MRZ

PRT_LINE1 = 'I<PRT092150918<ZY42<<<<<<<<<<<'
TD1_LINE3 = 'EXEMPLO<<ANA<MARIA<<<<<<<<<<<<'
TD2_LINE1 = 'I<PRTEXEMPLO<<ANA<MARIA<<<<<<<<<<<<<'


def test_td1_long_number():
    mrz = MRZ([PRT_LINE1, '8501019M3001019PRT<<<<<<<<<<<0', TD1_LINE3])
    assert mrz.mrz_type == 'TD1' and mrz.valid is True and mrz.valid_score == 100
    assert mrz.number == '092150918ZY4' and mrz.check_number == '2' and mrz.valid_number is True
    # Raw text of the line, as before
    assert mrz.optional1 == 'ZY42<<<<<<<<<<<'


def test_td1_long_number_from_ocr():
    mrz = MRZ.from_ocr('\n'.join([PRT_LINE1, '8501019M3001019PRT<<<<<<<<<<<0', TD1_LINE3]))
    assert mrz.mrz_type == 'TD1' and mrz.valid is True and mrz.number == '092150918ZY4'


@pytest.mark.parametrize('line1, line2', [
    # '<' as check digit and nothing after it
    ('I<PRT092150918<<<<<<<<<<<<<<<<', '8501019M3001019PRT<<<<<<<<<<<7'),
    # A single character after it: no rest of the number
    ('I<PRT092150918<4<<<<<<<<<<<<<<', '8501019M3001019PRT<<<<<<<<<<<9'),
    # Wrong final check digit
    ('I<PRT092150918<ZY43<<<<<<<<<<<', '8501019M3001019PRT<<<<<<<<<<<3'),
])
def test_td1_not_a_long_number(line1, line2):
    mrz = MRZ([line1, line2, TD1_LINE3])
    assert mrz.valid_composite is True
    assert mrz.valid_number is False and mrz.valid is False and mrz.valid_score == 77
    assert mrz.number == '092150918' and mrz.check_number == '<'


def test_td2_long_number():
    mrz = MRZ([TD2_LINE1, '092150918<PRT8501019M3001019ZY42<<<2'])
    assert mrz.mrz_type == 'TD2' and mrz.valid is True and mrz.valid_score == 100
    assert mrz.number == '092150918ZY4' and mrz.check_number == '2' and mrz.valid_number is True
    assert mrz.optional1 == 'ZY42<<<'


def test_td2_wrong_final_check_digit():
    mrz = MRZ([TD2_LINE1, '092150918<PRT8501019M3001019ZY43<<<9'])
    assert mrz.valid_composite is True
    assert mrz.valid_number is False and mrz.valid is False
    assert mrz.number == '092150918' and mrz.check_number == '<'
