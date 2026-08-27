#!/usr/bin/env python3
"""Print the newest structurally complete learner checkpoint in a run directory."""

import re
import sys
import zipfile
from pathlib import Path


_ITERATION_PATTERN = re.compile(r'^iteration_(\d+)\.pth\.tar$')


def _is_complete_torch_archive(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def find_valid_checkpoint(run_dir: Path):
    checkpoint_dir = run_dir / 'ckpt'
    if not checkpoint_dir.is_dir():
        return None

    periodic = []
    for path in checkpoint_dir.iterdir():
        match = _ITERATION_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            periodic.append((int(match.group(1)), path))
    candidates = [path for _, path in sorted(periodic, reverse=True)]
    best = checkpoint_dir / 'ckpt_best.pth.tar'
    if best.is_file():
        candidates.append(best)

    for path in candidates:
        if _is_complete_torch_archive(path):
            return path
        print(f'skipping incomplete checkpoint: {path}', file=sys.stderr)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(f'usage: {Path(sys.argv[0]).name} RUN_DIR', file=sys.stderr)
        return 2
    checkpoint = find_valid_checkpoint(Path(sys.argv[1]))
    if checkpoint is None:
        return 1
    print(checkpoint.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
