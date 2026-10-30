'''
PassportEye::MRZ: Machine-readable zone extraction and parsing.

This is a copy of PassportEye's ``passporteye/mrz/text.py``.
It is pure Python (only depends on the standard library) and is reproduced
unchanged so that parsing/validation behavior matches upstream exactly,
with one deliberate addition: the 'CNI_FR' type (pre-2021 French national
identity card, 2x36 non-ICAO MRZ), handled in MRZ.from_ocr only,
and one deliberate change: TD1 and TD2 read document numbers longer than
9 characters (ICAO 9303). Otherwise MRZ(lines) called directly behaves
exactly as upstream.

Author: Konstantin Tretyakov
License: MIT
'''
# pylint: disable=attribute-defined-outside-init,line-too-long
import re
from collections import OrderedDict
from datetime import datetime


class MRZ(object):
    """
    A simple parser for a Type1 or Type3 Machine-readable zone strings from identification documents.
    See:
        - https://en.wikipedia.org/wiki/Machine-readable_passport
        - http://www.icao.int/publications/pages/publication.aspx?docnum=9303

    Usage:
        Represent the MRZ as a list of 2 or 3 lines, create an instance of this class,
        and read off the various fields filled by the parser.
        The first field you should check is .mrz_type.
        It is either None (no parsing done at all) or 'TD1', 'TD2', 'TD3', 'MRVA', 'MRVB' or 'CNI_FR' (see below) depending on the type of the MRZ.
        The next one is 'valid'. If this is true, you may be pretty sure the parsing was successful and
        all the checksum digits passed the test as well. Sometimes the validity check may fail for some trivial reason
        (e.g. nonstandard document type character or one of the checksums wrong while others corect) -
        for this reason there is a field `valid_score`, which is an integer between 0 (nothing is valid) to 100
        (all checksums, line lengths and miscellaneous checks passed).

        Otherwise at least some of the checks failed, the meaning of which is up to you to interpret.
        When given invalid data, the algorithm attempts to do some trivial data clean-up: drop whitespaces from lines,
        and extend short lines with filler characters <, after which the fields are extracted from the lines as if
        they were valid.

        The parsing computes three validation indicators:
            valid_check_digits - a list of booleans indicating which of the "check digits" in the MRZ were valid.
                                TD1/TD2 has four check digits, TD3 - five, MRVA/B - three.
                                The separate booleans are also available as valid_number, valid_date_of_birth, valid_expiration_date, valid_composite
                                and valid_personal_number (TD3 only).
            valid_line_lengths - a list of booleans, indicating which of the lines (3 in TD1, 2 in TD2/TD3) had the expected length.
            valid_misc         - a list of booleans, indicating various additional validity checks (unspecified, see code).
        The valid_score field counts the "validity score" according to the flags above and is an int between 0 and 100.
        When all validation passes, the valid field is set to True as well.
        However, you may attempt reading fields from a "not completely valid" MRZ as well sometimes.

        The reported fields are: type, country, number, date_of_birth, sex, expiration_date, nationality, names, surname
        TD1 MRZ also has fields optional1 and optional2. TD2 MRZ has optional1, TD3 MRZ has personal_number.
        MRVA and MRVB are the same as TD3 except personal_number and check_composite (which are not present)

        The field aux is a dictionary of additional data that may be associated with MRZ by OCR code,
        e.g. aux['roi'], aux['box'] or aux['raw_text'] may be used to carry around the part of the image that was used
        to extract the information, aux['method'] to mark the method used, etc.

    CNI_FR (pre-2021 French national identity card):
        A non-ICAO MRZ of 2 lines of 36 characters, which would otherwise be read as an (invalid) TD2.
        It is only produced by MRZ.from_ocr (hence by read_mrz), never by MRZ(lines) called directly:
        from_ocr reads the raw OCR lines as CNI_FR when they look like one and when this beats the standard reading.
        Fields: type and country (read from line 1: 'ID' and 'FRA' on a good read), surname, names (truncated by the format), number (12 characters),
        issue_date (YYMM, the first 4 characters of number), administrative_code (6 characters, '<' possible),
        date_of_birth, sex, nationality (always 'FRA'), expiration_date (always None: not in this MRZ),
        check_number, check_date_of_birth, check_composite.
        There are no optional1, optional2, personal_number, check_expiration_date or valid_expiration_date fields.
        valid_check_digits has three elements: [valid_number, valid_date_of_birth, valid_composite].
        Beware: index 2 is the composite check digit here, while it is the expiration date for the other types;
        generic code should read the named attributes rather than the indices.
        valid_line_lengths is [line 1 has 36 chars, line 2 has 36 chars], valid_misc is [type == 'ID', sex in M/F].
        valid_format (CNI_FR only, not in to_dict) is True when both raw lines only contain A-Z, a-z, 0-9 and '<',
        every corrected character has the class of its position (letter, digit, or department code: digit, A or B right
        after a 2 of the same department, '<' in the administrative code only) and country is 'FRA'.
        The raw alphabet is checked after the 'IDFRA' prefix normalisation. Check digits may not detect these errors
        (an unknown character usually weighs like '<' in the composite), so when valid_format is False the score is capped at 99.
        Lines are never repaired (no character removed or inserted): a line of the wrong length makes the MRZ invalid,
        since line 1 is only protected by the composite check digit and a repaired line would pass it about 1 time in 10.
        Only the constant 'IDFRA' prefix may be normalised.
        Like any modulo 10 check digit, a single misread whose value differs by a multiple of 10 (e.g. '<' read as 'K')
        still passes every check. This applies after correction: '<' read as 'C' in the administrative code becomes '0',
        and '<' read as '4' in a name becomes 'A', both worth the same as '<'.

        Three more attributes, not in to_dict either, are None on every other type (MRZ(lines) included):
        detection_mode, how the raw lines were recognised (see looks_like_cni_fr): 'prefix', 'shifted' or 'line2';
        surname_truncated and names_truncated, True when the surname field (25 characters) or the names field
        (14 characters) is full, so that its last name may be cut. They are None unless the field positions are
        reliable: 'prefix' mode with two raw lines of 36 characters.

        >>> m = MRZ.from_ocr('IDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123\\n1507069012348JEAN<<PIERRE<<8503127M2')
        >>> m.mrz_type, m.detection_mode, m.surname_truncated, m.names_truncated
        ('CNI_FR', 'prefix', False, False)
        >>> m = MRZ.from_ocr('IDFRADUPONT<MOREAU<LEFEBVRE<GA069123\\n1507069012348ALEXANDRE<<MAR8503127M8')
        >>> m.valid, m.surname, m.surname_truncated, m.names, m.names_truncated
        (True, 'DUPONT MOREAU LEFEBVRE GA', True, 'ALEXANDRE MAR', True)
        >>> m = MRZ.from_ocr('KIDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123\\n1507069012348JEAN<<PIERRE<<8503127M2')
        >>> m.mrz_type, m.detection_mode, m.surname_truncated, m.names_truncated
        ('CNI_FR', 'shifted', None, None)
        >>> m = MRZ.from_ocr('XYFRAMARTIN<<<<<<<<<<<<<<<<<<<069123\\n1507069012348JEAN<<PIERRE<<8503127M2')
        >>> m.mrz_type, m.detection_mode, m.surname_truncated, m.names_truncated
        ('CNI_FR', 'line2', None, None)
        >>> m = MRZ.from_ocr('P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\\nL898902C36UTO7408122F1204159ZE184226B<<<<<10')
        >>> m.mrz_type, m.detection_mode, m.surname_truncated, m.names_truncated
        ('TD3', None, None, None)
    """

    #: CNI_FR only (see above); None on the other types
    detection_mode = None
    surname_truncated = None
    names_truncated = None

    def __init__(self, mrz_lines):
        """
        Parse a TD1/TD2/TD3/MRVA/MRVB MRZ from a single newline-separated string or a list of strings.

        :param mrz_lines: either a single string with newlines, or a list of 2 or 3 strings, representing the lines of an MRZ.
        :return: self
        """
        self._parse(mrz_lines)
        self.aux = {}

    @staticmethod
    def from_ocr(mrz_ocr_string):
        """
        Given a single string which is output from an OCR routine, cleans it up using MRZ.ocr_cleanup and creates a MRZ object.

        If the standard reading is not valid and the raw (uncleaned) lines look like a CNI_FR, they are also read as CNI_FR,
        and that reading is returned if at least one of its line 2 check digits (number, date of birth) is valid
        and its valid_score is strictly higher than the standard one.
        """
        return MRZ._from_ocr_reads(mrz_ocr_string)[0]

    @staticmethod
    def _from_ocr_reads(mrz_ocr_string):
        """
        The readings of from_ocr: a list of the MRZ it returns, then the CNI_FR reading it computed but did not return,
        if any. Each one gets aux['text'] and aux['raw_text'].
        """
        # The TD2 cleaner alters characters irreversibly (e.g. C, D, O, Q -> 0), so the CNI_FR path starts from the raw lines
        raw_lines = MRZOCRCleaner.instance()._split_lines(mrz_ocr_string)  # pylint: disable=protected-access
        result = MRZ(MRZOCRCleaner.apply(mrz_ocr_string))
        reads = [result]
        if not result.valid:
            try:
                cni = MRZ._read_cni_fr(raw_lines)
                if cni is not None:
                    if (cni.valid_number or cni.valid_date_of_birth) and cni.valid_score > result.valid_score:
                        reads = [cni]
                    else:
                        reads.append(cni)
            except Exception:
                pass
        for mrz in reads:
            mrz.aux['text'] = mrz_ocr_string
            mrz.aux['raw_text'] = mrz_ocr_string
        return reads

    @staticmethod
    def _detect_cni_fr(lines):
        """
        Tells whether raw OCR lines may be a CNI_FR MRZ. The test is deliberately loose: the check digits decide afterwards.
        Returns None (not a candidate), 'prefix' (line 1 starts with IDFRA, one character off at most),
        'shifted' (same, after one stray leading character) or 'line2' (only line 2 has the CNI_FR shape).
        """
        if len(lines) != 2 or not all(34 <= len(ln) <= 38 for ln in lines):
            return None

        def near_idfra(s):
            return sum(x != y for x, y in zip(s[:5].upper(), 'IDFRA')) <= 1

        if near_idfra(lines[0]):
            return 'prefix'
        if near_idfra(lines[0][1:]):
            return 'shifted'
        fixer = MRZOCRCleaner.instance().FIXERS['n']
        b = ''.join(fixer.get(ch, ch) for ch in lines[1].upper())
        if sum(ch in '0123456789' for ch in b[:13]) >= 11 and re.search(r'[0-9]{7}[MF][0-9]$', b):
            return 'line2'
        return None

    @staticmethod
    def _read_cni_fr(lines):
        """Reads raw OCR lines as a CNI_FR MRZ. Returns a new MRZ object, or None if the lines are not a CNI_FR candidate."""
        detected = MRZ._detect_cni_fr(lines)
        if detected is None:
            return None
        a, b = lines
        if detected == 'prefix':
            # The only normalisation allowed: a constant of the format, with no change of length
            a = 'IDFRA' + a[5:]
        cleaner = MRZOCRCleaner.instance()
        fixed_a = cleaner._fix_line(a, 'CNI_FR', 0)  # pylint: disable=protected-access
        fixed_b = cleaner._fix_line(b, 'CNI_FR', 1)  # pylint: disable=protected-access
        # A separate object, built without MRZ.__init__ so that no TD2 field is ever set on it
        mrz = MRZ.__new__(MRZ)
        mrz.aux = {}
        mrz.mrz_type = 'CNI_FR'
        mrz.detection_mode = detected
        mrz.valid = mrz._parse_cni_fr(fixed_a, fixed_b, a, b)
        # Field positions hold only on lines of the right length, raw and corrected (upper() may lengthen a character):
        # in 'shifted' mode the stray character is kept, and a short line is padded with '<'
        if detected == 'prefix' and len(a) == len(b) == len(fixed_a) == len(fixed_b) == 36:
            mrz.surname_truncated = fixed_a[29] != '<'
            mrz.names_truncated = fixed_b[26] != '<'
        else:
            mrz.surname_truncated = mrz.names_truncated = None
        return mrz

    def _parse(self, mrz_lines):
        self.mrz_type = MRZ._guess_type(mrz_lines)
        try:
            if self.mrz_type == 'TD1':
                self.valid = self._parse_td1(*mrz_lines)
            elif self.mrz_type == 'TD2':
                self.valid = self._parse_td2(*mrz_lines)
            elif self.mrz_type == 'TD3':
                self.valid = self._parse_td3(*mrz_lines)
            elif self.mrz_type == 'MRVA':
                self.valid = self._parse_mrv(*mrz_lines, length=44)
            elif self.mrz_type == 'MRVB':
                self.valid = self._parse_mrv(*mrz_lines, length=36)
            else:
                self.valid = False
                self.valid_score = 0
        except Exception:
            self.mrz_type = None
            self.valid = False
            self.valid_score = 0

    @staticmethod
    def _guess_type(mrz_lines):
        """Guesses the type of the MRZ from given lines. Returns 'TD1', 'TD2', 'TD3', 'MRVA', 'MRVB' or None."""
        try:
            if len(mrz_lines) == 3:
                return 'TD1'
            elif len(mrz_lines) == 2 and len(mrz_lines[0]) < 40 and len(mrz_lines[1]) < 40:
                return 'MRVB' if mrz_lines[0][0].upper() == 'V' else 'TD2'
            elif len(mrz_lines) == 2:
                return 'MRVA' if mrz_lines[0][0].upper() == 'V' else 'TD3'
            else:
                return None
        except Exception:
            return None

    def to_dict(self):
        """Converts this object to an (ordered) dictionary of field-value pairs."""
        result = OrderedDict()
        result['mrz_type'] = self.mrz_type
        result['valid_score'] = self.valid_score
        if 'raw_text' in self.aux:
            result['raw_text'] = self.aux['raw_text']
        if self.mrz_type == 'CNI_FR':
            # Returns before the generic code, which reads valid_check_digits[2:4] as TD2 flags and personal_number
            result['type'] = self.type
            result['country'] = self.country
            result['number'] = self.number
            result['issue_date'] = self.issue_date
            result['date_of_birth'] = self.date_of_birth
            result['nationality'] = self.nationality
            result['sex'] = self.sex
            result['names'] = self.names
            result['surname'] = self.surname
            result['administrative_code'] = self.administrative_code
            result['check_number'] = self.check_number
            result['check_date_of_birth'] = self.check_date_of_birth
            result['check_composite'] = self.check_composite
            result['valid_number'] = self.valid_number
            result['valid_date_of_birth'] = self.valid_date_of_birth
            result['valid_composite'] = self.valid_composite
            if 'method' in self.aux:
                result['method'] = self.aux['method']
            return result
        if self.mrz_type is not None:
            result['type'] = self.type
            result['country'] = self.country
            result['number'] = self.number
            result['date_of_birth'] = self.date_of_birth
            result['expiration_date'] = self.expiration_date
            result['nationality'] = self.nationality
            result['sex'] = self.sex
            result['names'] = self.names
            result['surname'] = self.surname
            if self.mrz_type == 'TD1':
                result['optional1'] = self.optional1
                result['optional2'] = self.optional2
            elif self.mrz_type in ['TD2', 'MRVA', 'MRVB']:
                result['optional1'] = self.optional1
            else:
                result['personal_number'] = self.personal_number
            result['check_number'] = self.check_number
            result['check_date_of_birth'] = self.check_date_of_birth
            result['check_expiration_date'] = self.check_expiration_date
            if self.mrz_type not in ['MRVA', 'MRVB']:
                result['check_composite'] = self.check_composite
            if self.mrz_type == 'TD3':
                result['check_personal_number'] = self.check_personal_number
            result['valid_number'] = self.valid_check_digits[0]
            result['valid_date_of_birth'] = self.valid_check_digits[1]
            result['valid_expiration_date'] = self.valid_check_digits[2]
            if self.mrz_type not in ['MRVA', 'MRVB']:
                result['valid_composite'] = self.valid_check_digits[3]
            if self.mrz_type == 'TD3':
                result['valid_personal_number'] = self.valid_check_digits[4]
            if 'method' in self.aux:
                result['method'] = self.aux['method']
        return result

    def _parse_td1(self, a, b, c):
        len_a, len_b, len_c = len(a), len(b), len(c)
        if len(a) < 30:
            a = a + '<'*(30 - len(a))
        if len(b) < 30:
            b = b + '<'*(30 - len(b))
        if len(c) < 30:
            c = c + '<'*(30 - len(c))
        self.type = a[0:2]
        self.country = a[2:5]
        self.number = a[5:14]
        self.check_number = a[14]
        self.optional1 = a[15:30]
        self.date_of_birth = b[0:6]
        self.check_date_of_birth = b[6]
        self.sex = b[7]
        self.expiration_date = b[8:14]
        self.check_expiration_date = b[14]
        self.nationality = b[15:18]
        self.optional2 = b[18:29]
        self.check_composite = b[29]
        surname_names = c.split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        self.surname, self.names = surname_names
        self.names = self.names.replace('<', ' ').strip()
        self.surname = self.surname.replace('<', ' ').strip()

        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(a[5:30] + b[0:7] + b[8:15] + b[18:29]) == self.check_composite
        if self.check_number == '<':
            # ICAO 9303 number longer than 9 characters: the rest of it and its check digit open optional1
            ext = a[15:30].split('<', 1)[0]
            if len(ext) >= 2 and MRZCheckDigit.compute(a[5:15] + ext[:-1]) == ext[-1]:
                self.number = a[5:14] + ext[:-1]
                self.check_number = ext[-1]
                self.valid_number = True
        self.valid_check_digits = [self.valid_number,
                                   self.valid_date_of_birth,
                                   self.valid_expiration_date,
                                   self.valid_composite]
        self.valid_line_lengths = [len_a == 30, len_b == 30, len_c == 30]
        self.valid_misc = [a[0] in 'IAC']
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100*self.valid_score//(40+3+1+1)
        return self.valid_score == 100

    def _parse_td2(self, a, b):
        len_a, len_b = len(a), len(b)
        if len(a) < 36:
            a = a + '<'*(36 - len(a))
        if len(b) < 36:
            b = b + '<'*(36 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:36].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        self.surname, self.names = surname_names
        self.names = self.names.replace('<', ' ').strip()
        self.surname = self.surname.replace('<', ' ').strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.optional1 = b[28:35]
        self.check_composite = b[35]
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:35]) == self.check_composite
        if self.check_number == '<':
            # ICAO 9303 number longer than 9 characters: the rest of it and its check digit open optional1
            ext = b[28:35].split('<', 1)[0]
            if len(ext) >= 2 and MRZCheckDigit.compute(b[0:10] + ext[:-1]) == ext[-1]:
                self.number = b[0:9] + ext[:-1]
                self.check_number = ext[-1]
                self.valid_number = True
        self.valid_check_digits = [self.valid_number,
                                   self.valid_date_of_birth,
                                   self.valid_expiration_date,
                                   self.valid_composite]
        self.valid_line_lengths = [len_a == 36, len_b == 36]
        self.valid_misc = [a[0] in 'ACI']
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100*self.valid_score//(40+2+1+1)
        return self.valid_score == 100

    def _parse_td3(self, a, b):
        len_a, len_b = len(a), len(b)
        if len(a) < 44:
            a = a + '<'*(44 - len(a))
        if len(b) < 44:
            b = b + '<'*(44 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:44].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        self.surname, self.names = surname_names
        self.names = self.names.replace('<', ' ').strip()
        self.surname = self.surname.replace('<', ' ').strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.personal_number = b[28:42]
        self.check_personal_number = b[42]
        self.check_composite = b[43]
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:43]) == self.check_composite
        self.valid_personal_number = (((self.check_personal_number == '<' or self.check_personal_number == '0') and self.personal_number == '<<<<<<<<<<<<<<')
                                   or MRZCheckDigit.compute(self.personal_number) == self.check_personal_number)
        self.valid_check_digits = [self.valid_number,
                                   self.valid_date_of_birth,
                                   self.valid_expiration_date,
                                   self.valid_composite,
                                   self.valid_personal_number]
        self.valid_line_lengths = [len_a == 44, len_b == 44]
        self.valid_misc = [a[0] in 'P']
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100*self.valid_score//(50+2+1+1)
        return self.valid_score == 100

    @staticmethod
    def _check_date(ymd):
        try:
            datetime.strptime(ymd, '%y%m%d')
            return True
        except ValueError:
            return False

    def _parse_mrv(self, a, b, length=44):
        len_a, len_b = len(a), len(b)
        if len(a) < length:
            a = a + '<'*(44 - len(a))
        if len(b) < length:
            b = b + '<'*(44 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:length].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        self.surname, self.names = surname_names
        self.names = self.names.replace('<', ' ').strip()
        self.surname = self.surname.replace('<', ' ').strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.optional1 = b[28:length]
        self.valid_check_digits = [MRZCheckDigit.compute(self.number) == self.check_number,
                                   MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth,
                                   MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date]
        self.valid_line_lengths = [len_a == length, len_b == length]
        self.valid_misc = [a[0] == 'V']
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100*self.valid_score//(30+2+1+1)
        self.valid_number, self.valid_date_of_birth, self.valid_expiration_date = self.valid_check_digits
        return self.valid_score == 100

    def _parse_cni_fr(self, a, b, raw_a, raw_b):
        # Lengths are measured on the raw lines, before padding: the padding only makes slicing safe, it never repairs a line
        # (and upper() in the cleaner may change the length of a non-ASCII character, e.g. 'ß' -> 'SS')
        len_a, len_b = len(raw_a), len(raw_b)
        # Check digits may not see characters outside the MRZ alphabet (at a weight 1 position of the composite they count
        # as '<'), nor a character of the wrong class whose value differs by a multiple of 10 (e.g. 'R' for '7'):
        # such a reading can never be valid
        self.valid_format = (all(re.fullmatch(r'[A-Za-z0-9<]*', ln) for ln in (raw_a, raw_b))
                             and MRZ._matches_cni_fr_classes(a, 0) and MRZ._matches_cni_fr_classes(b, 1)
                             and a[2:5] == 'FRA')
        if len(a) < 36:
            a = a + '<'*(36 - len(a))
        if len(b) < 36:
            b = b + '<'*(36 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        self.surname = ' '.join(a[5:30].replace('<', ' ').split())
        self.administrative_code = a[30:36]
        self.number = b[0:12]
        self.issue_date = b[0:4]
        self.check_number = b[12]
        # Names are separated by '<<': collapsing every run of '<' into a single space joins them with one space
        self.names = ' '.join(b[13:27].replace('<', ' ').split())
        self.date_of_birth = b[27:33]
        self.check_date_of_birth = b[33]
        self.sex = b[34]
        self.check_composite = b[35]
        self.nationality = 'FRA'
        self.expiration_date = None
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        # The composite covers the whole line 1, so it can only be valid if line 1 has exactly 36 characters
        self.valid_composite = len_a == 36 and MRZCheckDigit.compute(a[0:36] + b[0:35]) == self.check_composite
        self.valid_check_digits = [self.valid_number,
                                   self.valid_date_of_birth,
                                   self.valid_composite]
        self.valid_line_lengths = [len_a == 36, len_b == 36]
        self.valid_misc = [self.type == 'ID', self.sex in ('M', 'F')]
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100*self.valid_score//(30+2+2+1)
        if self.valid_score == 100 and not self.valid_format:
            # Kept below 100 so that valid == (valid_score == 100) still holds
            self.valid_score = 99
        return self.valid_score == 100

    @staticmethod
    def _matches_cni_fr_classes(line, line_idx):
        """
        Tells whether a corrected CNI_FR line has the right length and a character of the right class at every position.
        In the 3-character departments (line 1 positions 31-33, line 2 positions 5-7), A and B are only allowed
        right after a 2 of the same department (Corsica: 2A, 2B); '<' is only allowed in the administrative code (line 1),
        never in the card number (line 2).
        """
        fmt = MRZOCRCleaner.instance().FORMAT['CNI_FR'][line_idx]
        allowed = {'a': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                   'A': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ<',
                   'n': '0123456789',
                   'c': '0123456789<' if line_idx == 0 else '0123456789'}
        # 0-based positions of the 2nd and 3rd characters of the department, the only ones that may follow its 2
        corsica = (31, 32) if line_idx == 0 else (5, 6)
        if len(line) != len(fmt):
            return False
        for j, (ch, cls) in enumerate(zip(line, fmt)):
            if j in corsica and ch in 'AB' and line[j - 1] == '2':
                continue
            if ch not in allowed[cls]:
                return False
        return True


class MRZOCRCleaner(object):
    """
    The __call__ method of this class implements the "cleaning" of an OCR-obtained string in preparation for MRZ parsing.
    This is a singleton class, so rather than creating an instance, simply use its `apply` static method.

    >>> MRZOCRCleaner.apply('\\nuseless lines\\n  P<POLKOWALSKA < KWIATKOWSKA<<JOANNA<<<<<<<<<<<extrachars \\n  AA0000000OP0L6OOzoB4Fi4iz3I4<<<<<<<<<<<<<<<4  \\n  asdf  ')
    ['P<POLKOWALSKA<KWIATKOWSKA<<JOANNA<<<<<<<<<<<extrachars', 'AA00000000POL6002084F1412314<<<<<<<<<<<<<<<4']

    """

    def __init__(self):
        # Specifications for which characters may be present at each position of each line of each document type.
        #   a  - alpha
        #   A  - alpha+<
        #   n  - numeric
        #   N  - numeric+<
        #   c  - department code: numeric+<, but B is kept (Corsican 2B)
        #   *  - alpha+num+<

        TD1 = ['a*' + 'A'*3 + '*'*9 + 'N' + '*'*15,
               'n'*7 + 'A' + 'n'*7 + 'A'*3 + '*'*11 + 'n',
               'A'*30]
        TD2 = ['a' + 'A'*35,
               '*'*9 + 'n' + 'A'*3 + 'n'*7 + 'A' + 'n'*7 + '*'*7 + 'n'*1]
        TD3 = ['a' + 'A'*43,
               '*'*9 + 'n' + 'A'*3 + 'n'*7 + 'A' + 'n'*7 + '*'*14 + 'n'*2]
        MRV = ['a' + 'A'*43,
               '*'*9 + 'n' + 'A'*3 + 'n'*7 + 'A' + 'n'*7 + '*'*16]
        # Never guessed by MRZ._guess_type: only applied by MRZ.from_ocr to CNI_FR candidates
        CNI_FR = ['a'*5 + 'A'*25 + 'c'*6,
                  'n'*4 + 'c'*3 + 'n'*6 + 'A'*14 + 'n'*7 + 'a' + 'n']
        self.FORMAT = {'TD1': TD1, 'TD2': TD2, 'TD3': TD3, 'MRVA': MRV, 'MRVB': MRV, 'CNI_FR': CNI_FR}

        # Fixers
        a = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
        n = {'B': '8', 'C': '0', 'D': '0', 'G': '6', 'I': '1', 'O': '0', 'Q': '0', 'S': '5', 'Z': '2'}
        c = {k: v for k, v in n.items() if k != 'B'}
        self.FIXERS = {'a': a, 'A': a, 'n': n, 'N': n, 'c': c, '*': {}}

    def _split_lines(self, mrz_ocr_string):
        return [ln for ln in mrz_ocr_string.replace(' ', '').split('\n') if (len(ln) >= 20 or '<<' in ln)]

    def __call__(self, mrz_ocr_string):
        """
        Given a string, which is output from an OCR routine, splits it into lines and performs various ad-hoc cleaning on those.
        In particular:
            - Spaces are removed
            - Lines shorter than 30 non-space characters are removed
            - The type of the document is guessed based on the number of lines and their lengths,
              if it is not-none, OCR-fixup is performed on a character-by-character basis depending on
              what characters are allowed at particular positions.
        """
        lines = self._split_lines(mrz_ocr_string)
        tp = MRZ._guess_type(lines)  # pylint: disable=protected-access
        if tp is not None:
            for i in range(len(lines)):  # pylint: disable=consider-using-enumerate
                lines[i] = self._fix_line(lines[i], tp, i)
        return lines

    def _fix_line(self, line, type, line_idx):
        ln = list(line)
        for j in range(len(ln)):  # pylint: disable=consider-using-enumerate
            ln[j] = self._fix_char(ln[j], type, line_idx, j)
        return ''.join(ln)

    def _fix_char(self, char, type, line_idx, char_idx):
        fmt = self.FORMAT[type][line_idx]
        if char_idx >= len(fmt):
            return char
        else:
            fixer = self.FIXERS[fmt[char_idx]]
            char = char.upper()
            return fixer.get(char, char)

    @staticmethod
    def instance():
        """Returns the singleton instance of the cleaner."""
        if getattr(MRZOCRCleaner, '__instance__', None) is None:
            MRZOCRCleaner.__instance__ = MRZOCRCleaner()
        return MRZOCRCleaner.__instance__

    @staticmethod
    def apply(txt):
        return MRZOCRCleaner.instance()(txt)


class MRZCheckDigit(object):
    """
    The algorithm used to compute "check digits" within MRZ.
    Its __call__ method, given a string, returns either the single character check digit.
    Rather than creating an instance every time, use the static compute(txt) method (which makes use of a singleton instance).

    # Valid codes
    >>> assert MRZCheckDigit.compute('0') == '0'
    >>> assert MRZCheckDigit.compute('0000000000') == '0'
    >>> assert MRZCheckDigit.compute('00A0A<0A0<<0A0A0<0A') == '0'
    >>> assert MRZCheckDigit.compute('111111111') == '3'
    >>> assert MRZCheckDigit.compute('111<<<111111') == '3'
    >>> assert MRZCheckDigit.compute('BBB<<<1B1<<<BB1') == '3'
    >>> assert MRZCheckDigit.compute('1<<1<<1<<1') == '8'
    >>> assert MRZCheckDigit.compute('1<<1<<1<<1') == '8'
    >>> assert MRZCheckDigit.compute('BCDEFGHIJ') == MRZCheckDigit.compute('123456789')

    # Invalid codes
    >>> assert MRZCheckDigit.compute('') == ''
    >>> assert MRZCheckDigit.compute('0000 0') == ''
    >>> assert MRZCheckDigit.compute('0 0') == ''
    >>> assert MRZCheckDigit.compute('onlylowercase') == ''
    >>> assert MRZCheckDigit.compute('BBb<<<1B1<<<BB1') == ''

    """

    def __init__(self):
        self.CHECK_CODES = dict()
        for i in range(10):
            self.CHECK_CODES[str(i)] = i
        for i in range(ord('A'), ord('Z')+1):
            self.CHECK_CODES[chr(i)] = i - 55   # A --> 10, B --> 11, etc
        self.CHECK_CODES['<'] = 0
        self.CHECK_WEIGHTS = [7, 3, 1]

    def __call__(self, txt):
        if txt == '':
            return ''
        res = sum([self.CHECK_CODES.get(c, -1000)*self.CHECK_WEIGHTS[i % 3] for i, c in enumerate(txt)])
        if res < 0:
            return ''
        else:
            return str(res % 10)

    @staticmethod
    def compute(txt):
        if getattr(MRZCheckDigit, '__instance__', None) is None:
            MRZCheckDigit.__instance__ = MRZCheckDigit()
        return MRZCheckDigit.__instance__(txt)


def looks_like_cni_fr(lines):
    """
    Tells whether raw OCR lines may be a CNI_FR MRZ, by the loose test MRZ.from_ocr runs before reading them as CNI_FR.
    Returns None (not a candidate), 'prefix' (2 lines of 34 to 38 characters, line 1 starting with IDFRA, one character
    off at most), 'shifted' (same, after one stray leading character) or 'line2' (only line 2 has the CNI_FR shape).

    `lines` are split already: this function does not split a text, since from_ocr drops the lines shorter than 20
    characters while a caller may want exactly two non-empty lines. Spaces count as characters. A string raises
    TypeError.
    To test only line 1 (near IDFRA, maybe after one stray character), ignore 'line2'.

    >>> looks_like_cni_fr(['IDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123', '1507069012348JEAN<<PIERRE<<8503127M2'])
    'prefix'
    >>> looks_like_cni_fr(['KIDFRAMARTIN<<<<<<<<<<<<<<<<<<<069123', '1507069012348JEAN<<PIERRE<<8503127M2'])
    'shifted'
    >>> looks_like_cni_fr(['XYFRAMARTIN<<<<<<<<<<<<<<<<<<<069123', '1507069012348JEAN<<PIERRE<<8503127M2'])
    'line2'
    >>> looks_like_cni_fr(['I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<', 'D231458907UTO7408122F1204159<<<<<<<6']) is None
    True
    """
    if isinstance(lines, str):
        raise TypeError('lines: expected a sequence of lines, not a string: split it first')
    return MRZ._detect_cni_fr(list(lines))  # pylint: disable=protected-access
