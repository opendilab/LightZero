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
    parser.add_argument('--simulations', type=int, default=None, help='MCTS simulations per agent move')
    parser.add_argument('--num-res-blocks', type=int, default=None, help='override checkpoint model depth')
    parser.add_argument('--num-channels', type=int, default=None, help='override checkpoint model width')
    args = parser.parse_args()

    config, create_config, evaluate = _load(args.algo)
    config.env.agent_vs_human = False
    if args.num_res_blocks is not None:
        config.policy.model.num_res_blocks = args.num_res_blocks
    if args.num_channels is not None:
        config.policy.model.num_channels = args.num_channels
    if args.simulations is not None:
        if args.simulations <= 0:
            parser.error('--simulations must be positive')
        if args.algo == 'alphazero':
            if args.simulations < 4 or args.simulations % 4:
                parser.error('--simulations must be a positive multiple of 4 for AlphaZero evaluation')
            config.policy.mcts.num_simulations = args.simulations // 4
        else:
            config.policy.num_simulations = args.simulations
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
