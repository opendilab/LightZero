#!/usr/bin/env python3
"""Evaluate the preregistered 300k kill recommendation for the 3M matrix."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAG = 'evaluator_step/eval_episode_return_mean'
GROUPS = ('baseline', 'v1', 'v2', 'v3')


def evaluation_tail(run_dir, target_step, points):
    serial = run_dir / 'log' / 'serial'
    if not serial.is_dir():
        return None
    accumulator = EventAccumulator(str(serial), size_guidance={'scalars': 0})
    accumulator.Reload()
    if TAG not in accumulator.Tags().get('scalars', []):
        return None
    by_step = {}
    for event in accumulator.Scalars(TAG):
        if event.step <= target_step:
            previous = by_step.get(event.step)
            if previous is None or event.wall_time >= previous.wall_time:
                by_step[event.step] = event
    ordered = [by_step[step].value for step in sorted(by_step)]
    if not ordered or max(by_step) < target_step:
        return None
    tail = ordered[-points:]
    return sum(tail) / len(tail), tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step', type=int, default=300_000)
    parser.add_argument('--margin', type=float, default=0.35)
    parser.add_argument('--points', type=int, default=3)
    args = parser.parse_args()
    results = {}
    for group in GROUPS:
        run = args.output_root / f'unizero_mspacman_{group}_seed{args.seed}_3M'
        results[group] = evaluation_tail(run, args.step, args.points)
    if results['baseline'] is None:
        print(f'WAIT baseline has not logged an evaluation at {args.step} env steps')
        return 2
    baseline_mean = results['baseline'][0]
    threshold = baseline_mean * (1.0 - args.margin)
    print(f'baseline_tail_mean={baseline_mean:.3f} threshold={threshold:.3f}')
    recommended = []
    for group in GROUPS[1:]:
        result = results[group]
        if result is None:
            print(f'{group}: WAIT')
            continue
        mean, values = result
        decision = 'KILL_RECOMMENDED' if mean < threshold else 'KEEP'
        print(f'{group}: mean={mean:.3f} values={values} decision={decision}')
        if decision == 'KILL_RECOMMENDED':
            recommended.append(group)
    return 10 if recommended else 0


if __name__ == '__main__':
    raise SystemExit(main())
