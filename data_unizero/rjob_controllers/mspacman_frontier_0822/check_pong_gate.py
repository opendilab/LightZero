#!/usr/bin/env python3
"""Fail unless the seed-0 Pong regression retains its known 250k performance."""

import re
import sys
from pathlib import Path


def read_eval_scores(path: Path):
    text = path.read_text(encoding='utf-8', errors='replace')
    return [
        float(score)
        for score in re.findall(
            r'\|\s*Value\s*\|\s*\[[^\n]*\]\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|',
            text,
        )
    ]


def require_latest_score(scores, threshold):
    """Require the final evaluation, rather than an earlier nearby peak, to pass."""
    if len(scores) < 2:
        raise ValueError(f'Pong gate needs at least two evaluations, found {scores}')

    recent = scores[-2:]
    latest = scores[-1]
    print(f'Pong gate scores={scores}; recent={recent}; latest={latest}; threshold={threshold}')
    if latest < threshold:
        raise ValueError(
            f'Pong regression failed: latest={latest:.3f} < {threshold:.3f}'
        )


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f'usage: {sys.argv[0]} EVALUATOR_LOG [MIN_SCORE]')
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f'Pong gate evaluator log does not exist: {path}')

    scores = read_eval_scores(path)
    threshold = float(sys.argv[2]) if len(sys.argv) == 3 else 15.0
    try:
        require_latest_score(scores, threshold)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
