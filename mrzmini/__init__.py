'''
mrzmini: a minimal-footprint reimplementation of PassportEye's MRZ
detection and reading pipeline.

Only depends on numpy + Pillow (plus the external ``tesseract`` binary for OCR).
The public interface mirrors PassportEye:

    from mrzmini import read_mrz
    mrz = read_mrz('passport.jpg')
    print(mrz.to_dict())

Original algorithm and code: Konstantin Tretyakov, PassportEye (MIT License).
'''
from .text import MRZ
from .image import read_mrz, MRZPipeline

__all__ = ['read_mrz', 'MRZ', 'MRZPipeline']
