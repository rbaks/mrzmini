"""Command-line entry point for mrzmini.

Usage:
    mrzmini <image-or-pdf> [...]
    python -m mrzmini <image-or-pdf> [...]
"""
import sys
import json

from mrzmini import read_mrz


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    for path in sys.argv[1:]:
        print("=" * 60)
        print(path)
        mrz = read_mrz(path)
        if mrz is None:
            print("  No MRZ found.")
            continue
        print(json.dumps(mrz.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
