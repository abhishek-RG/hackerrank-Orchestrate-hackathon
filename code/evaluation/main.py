#!/usr/bin/env python3
"""Evaluation entry point.

Runs the regression evaluator (`code/tools/evaluate.py`) over
`dataset/sample_requests.csv` and prints field-level agreement against its
solved columns. See that module's docstring for what this is (a sanity
check against 25 known-good examples) and is not (the actual grading
mechanism, which compares `output.csv` against hidden ground truth).

Usage:
    python code/evaluation/main.py [-v] [request_NN]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from evaluate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
