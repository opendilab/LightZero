"""Measure a Chinese chess checkpoint's win rate against the seeded random bot."""

import argparse
import copy
from typing import Callable, Tuple

import numpy as np


def _load(algo: str) -> Tuple[dict, dict, Callable]:
    if algo == 'alphazero':
        from lzero.entry import eval_alphazero
        from zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config import (
            create_config,
            main_config,
        )

        return copy.deepcopy(main_config), copy.deepcopy(create_config), eval_alphazero

    from lzero.entry import eval_muzero
    from zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config import create_config, main_config

    return copy.deepcopy(main_config), copy.deepcopy(create_config), eval_muzero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('algo', choices=('alphazero', 'muzero'))
    parser.add_argument('checkpoint', help='path to a .pth.tar checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='episodes per seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    config, create_config, evaluate = _load(args.algo)
    config.env.agent_vs_human = False
    all_returns = []
    for seed in args.seeds:
        _, returns = evaluate(
            [config, create_config],
            seed=seed,
            num_episodes_each_seed=args.episodes,
            print_seed_details=True,
            model_path=args.checkpoint,
        )
        all_returns.extend(np.asarray(returns).reshape(-1).tolist())

    returns = np.asarray(all_returns, dtype=np.float32)
    print(
        f'episodes={len(returns)} win_rate={np.mean(returns == 1):.3f} '
        f'draw_rate={np.mean(returns == 0):.3f} loss_rate={np.mean(returns == -1):.3f}'
    )


if __name__ == '__main__':
    main()
