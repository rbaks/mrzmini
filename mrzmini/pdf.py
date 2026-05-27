'''
mrzmini.pdf: extract the first embedded JPEG from a PDF, dependency-free.

Best-effort replacement for PassportEye's ``passporteye/util/pdf.py`` which used
``pdfminer``. PassportEye only ever looked for the first embedded JPEG whose data
begins with the JFIF marker ``\\xff\\xd8\\xff\\xe0`` (``DCTDecode`` image streams
are stored verbatim inside the PDF), so we simply scan the raw bytes for that
marker and return the JPEG up to its end-of-image marker ``\\xff\\xd9``.

This covers the common "scanner produced a PDF wrapping a JPEG" case, matching
upstream's own documented limitation; other embedded image encodings are not
handled.
'''

_SOI_JFIF = b'\xff\xd8\xff\xe0'  # JPEG start-of-image + APP0/JFIF
_EOI = b'\xff\xd9'               # JPEG end-of-image


def extract_first_jpeg_in_pdf(fstream):
    """Scan a binary PDF stream for the first embedded JFIF JPEG.

    :param fstream: a readable binary stream of the PDF.
    :return: ``bytes`` of the JPEG image, or ``None`` if none was found.
    """
    data = fstream.read()
    start = data.find(_SOI_JFIF)
    if start < 0:
        return None
    end = data.find(_EOI, start)
    if end < 0:
        return None
    return data[start:end + len(_EOI)]
