#!/usr/bin/env python3
"""Buy or Wait? - entry point.

Reads every request in ``dataset/requests.csv``, runs the deterministic
decision engine (financial-state reconstruction -> forecasting -> candidate
generation -> ranking -> explanation), validates every row, and writes
``output.csv`` at the repository root.

Usage::

    python code/main.py
    python code/main.py --dataset path/to/other/dataset --out somewhere.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bow import dataset, engine, output

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="dataset directory (default: repo dataset/)")
    parser.add_argument("--out", default=None, help="output CSV path (default: repo output.csv)")
    parser.add_argument("--no-cache", action="store_true",
                       help="bypass the evidence cache entirely (forces the generic "
                            "heuristic/LLM extractors to run on every message and image)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(REPO_ROOT, "output.csv")

    started = time.time()
    data = dataset.load(args.dataset)
    eng = engine.Engine(data, use_cache=not args.no_cache)

    rows = []
    all_problems = {}
    for request in data.requests:
        decision = eng.decide(request)
        rows.append(decision.row)
        if decision.problems:
            all_problems[request.request_id] = decision.problems

    output.write_csv(out_path, rows)
    elapsed = time.time() - started

    print(f"wrote {len(rows)} rows to {out_path} in {elapsed:.1f}s")
    if all_problems:
        print(f"WARNING: {len(all_problems)} row(s) failed structural validation:")
        for rid, problems in all_problems.items():
            print(f"  {rid}: {problems}")
        return 1

    print("all rows passed structural validation "
         "(bounds, plan feasibility, schedule match, flexible-only spending changes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
