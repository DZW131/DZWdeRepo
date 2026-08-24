#!/usr/bin/env python3
"""Build the single 25-epoch worker-independent schedule for WD-CH."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.wdch_common import build_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_schedule(args.train_root, args.output))


if __name__ == "__main__":
    main()
