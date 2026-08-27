#!/usr/bin/env python3
"""Read-only TensorBoard archaeology for every MsPacman run under a root."""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2


EVAL_TAG = 'evaluator_step/eval_episode_return_mean'


def scalar(value):
    if value.HasField('simple_value'):
        return float(value.simple_value)
    if value.HasField('tensor') and value.tensor.float_val:
        return float(value.tensor.float_val[0])
    return None


def scan_run(root_text, run_text):
    root, run = Path(root_text), Path(run_text)
    evaluations = {}
    max_env_step = 0
    for event_path in sorted((run / 'log' / 'serial').glob('events.out.tfevents*')):
        for raw in RawEventFileLoader(str(event_path)).Load():
            event = event_pb2.Event.FromString(raw)
            for value in event.summary.value:
                number = scalar(value)
                if number is None:
                    continue
                if value.tag == EVAL_TAG:
                    previous = evaluations.get(int(event.step))
                    if previous is None or event.wall_time >= previous[1]:
                        evaluations[int(event.step)] = (number, event.wall_time)
                elif value.tag.endswith('/total_envstep_count'):
                    max_env_step = max(max_env_step, int(number), int(event.step))
    if not evaluations:
        return None
    ordered = [(step, value) for step, (value, _) in sorted(evaluations.items())]
    max_eval_step = ordered[-1][0]
    tail = [value for step, value in ordered if step >= 0.9 * max_eval_step]
    peak_step, peak = max(ordered, key=lambda pair: pair[1])
    final = ordered[-1][1]
    relative = str(run.relative_to(root))
    match = re.search(r'(?:^|[_/-])seed(\d+)(?:[_/-]|$)', relative, flags=re.I)
    config = run / 'formatted_total_config.py'
    if not config.exists():
        config = run / 'total_config.py'
    return {
        'run': relative,
        'config': str(config.relative_to(root)) if config.exists() else None,
        'seed': int(match.group(1)) if match else None,
        'eval_count': len(ordered),
        'peak_return': peak,
        'peak_env_step': peak_step,
        'tail_10pct_mean_return': sum(tail) / len(tail),
        'tail_eval_count': len(tail),
        'final_return': final,
        'peak_to_final_drop_fraction': 0.0 if peak == 0 else (peak - final) / abs(peak),
        'max_eval_env_step': max_eval_step,
        'max_env_step': max(max_env_step, max_eval_step),
        'retired_frameskip16': 'atari_frameskip16_retired' in relative,
        'evaluations': ordered,
    }


def discover_runs(root):
    runs = set()
    configs = list(root.rglob('formatted_total_config.py')) + list(root.rglob('total_config.py'))
    for config in configs:
        try:
            is_mspacman = 'ALE/MsPacman-v5' in config.read_text(errors='ignore')
        except OSError:
            continue
        if is_mspacman and (config.parent / 'log' / 'serial').is_dir():
            runs.add(config.parent)
    return sorted(runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_run, str(root), str(run)): run
            for run in discover_runs(root)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception as error:  # preserve other runs when one event stream is malformed
                errors.append({'run': str(futures[future]), 'error': repr(error)})
    rows.sort(key=lambda row: (row['tail_10pct_mean_return'], row['peak_return']), reverse=True)
    payload = json.dumps({'root': str(root), 'runs': rows, 'errors': errors}, indent=2)
    if args.output:
        args.output.write_text(payload + '\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
