"""Parity harness: compare mrzmini against the real PassportEye, stage by stage.

Run with the parity dependency group installed:

    uv run --group parity python parity_check.py [image ...]

With no arguments it scans the ``testdata/`` directory.
"""
import sys
import glob
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import passporteye
from passporteye.mrz.image import MRZPipeline as PEPipeline
import mrzmini
from mrzmini.image import MRZPipeline as MMPipeline


def _boxes_summary(boxes):
    return sorted(
        (round(float(b.cx), 3), round(float(b.cy), 3), round(float(b.width), 3),
         round(float(b.height), 3), round(float(b.angle), 4))
        for b in boxes
    )


def compare(path):
    print("=" * 70)
    print(path)
    pe = PEPipeline(path)
    mm = MMPipeline(path)

    # --- stage-by-stage detection arrays ---
    pe_small, mm_small = pe['img_small'], mm['img_small']
    print("  img_small shape %s/%s  maxdiff %.2e"
          % (pe_small.shape, mm_small.shape,
             np.abs(pe_small.astype(float) - mm_small.astype(float)).max()
             if pe_small.shape == mm_small.shape else float('nan')))

    pe_bin, mm_bin = pe['img_binary'], mm['img_binary']
    disagree = int((pe_bin != mm_bin).sum()) if pe_bin.shape == mm_bin.shape else -1
    print("  img_binary disagreements: %d / %d" % (disagree, pe_bin.size))

    pe_boxes, mm_boxes = _boxes_summary(pe['boxes']), _boxes_summary(mm['boxes'])
    print("  boxes match: %s  (n=%d/%d)"
          % (pe_boxes == mm_boxes, len(pe_boxes), len(mm_boxes)))
    if pe_boxes != mm_boxes:
        print("    PE:", pe_boxes)
        print("    MM:", mm_boxes)

    # --- end to end ---
    pe_mrz = passporteye.read_mrz(path)
    mm_mrz = mrzmini.read_mrz(path)
    pe_d = pe_mrz.to_dict() if pe_mrz is not None else None
    mm_d = mm_mrz.to_dict() if mm_mrz is not None else None

    def strip(d):
        if d is None:
            return None
        return {k: v for k, v in d.items() if k not in ('raw_text', 'method')}

    same = strip(pe_d) == strip(mm_d)
    print("  read_mrz to_dict match: %s" % same)
    print("    PE valid/score:", None if pe_d is None else (pe_mrz.valid, pe_d.get('valid_score')),
          " type:", None if pe_d is None else pe_d.get('mrz_type'))
    print("    MM valid/score:", None if mm_d is None else (mm_mrz.valid, mm_d.get('valid_score')),
          " type:", None if mm_d is None else mm_d.get('mrz_type'))
    if not same:
        keys = set((strip(pe_d) or {})) | set((strip(mm_d) or {}))
        for k in sorted(keys):
            pv = (strip(pe_d) or {}).get(k)
            mv = (strip(mm_d) or {}).get(k)
            if pv != mv:
                print("      DIFF %-22s PE=%r  MM=%r" % (k, pv, mv))
    return {'binary': disagree, 'boxes': pe_boxes == mm_boxes, 'mrz': same}


def main():
    paths = sys.argv[1:] or sorted(glob.glob('testdata/*'))
    results = []
    for p in paths:
        try:
            results.append((p, compare(p)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((p, {'error': str(e)}))
    print("=" * 70)
    print("SUMMARY")
    for p, r in results:
        print("  %-40s %s" % (p.split('/')[-1], r))


if __name__ == '__main__':
    main()
