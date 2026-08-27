#!/usr/bin/env python3
"""Print the newest complete periodic checkpoint, without loading its tensors."""

import re
import sys
import zipfile
from pathlib import Path


PATTERN = re.compile(r'^iteration_(\d+)\.pth\.tar$')


def valid(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def find_checkpoint(run_dir):
    checkpoint_dir = Path(run_dir) / 'ckpt'
    if not checkpoint_dir.is_dir():
        return None
    candidates = []
    for path in checkpoint_dir.iterdir():
        match = PATTERN.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    for _, path in sorted(candidates, reverse=True):
        if valid(path):
            return path.resolve()
        print(f'skipping incomplete checkpoint: {path}', file=sys.stderr)
    return None


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {Path(sys.argv[0]).name} RUN_DIR')
    result = find_checkpoint(sys.argv[1])
    if result is None:
        raise SystemExit(1)
    print(result)
