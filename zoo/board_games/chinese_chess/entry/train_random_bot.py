"""Train an AlphaZero or MuZero Xiangqi agent against the built-in random bot."""

import argparse
import copy
from typing import Any, Tuple


def _load_config(algo: str) -> Tuple[Any, Any, Any]:
    if algo == 'alphazero':
        from lzero.entry import train_alphazero
        from zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config import (
            create_config,
            main_config,
        )

        return copy.deepcopy(main_config), copy.deepcopy(create_config), train_alphazero

    from lzero.entry import train_muzero
    from zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config import create_config, main_config

    return copy.deepcopy(main_config), copy.deepcopy(create_config), train_muzero


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('algo', choices=('alphazero', 'muzero'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-env-step', type=int, default=None)
    parser.add_argument('--num-simulations', type=int, default=None)
    parser.add_argument('--collector-env-num', type=int, default=None)
    parser.add_argument('--evaluator-env-num', type=int, default=None)
    parser.add_argument('--n-episode', type=int, default=None)
    parser.add_argument('--update-per-collect', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--eval-freq', type=int, default=None)
    parser.add_argument(
        '--stop-value', type=float, default=None, help='stop when mean evaluation return reaches this value'
    )
    parser.add_argument('--max-episode-steps', type=int, default=None, help='maximum physical plies per game')
    parser.add_argument('--num-res-blocks', type=int, default=None)
    parser.add_argument('--num-channels', type=int, default=None)
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--cpu', action='store_true', help='disable CUDA')
    parser.add_argument('--smoke-test', action='store_true', help='apply a tiny startup-only configuration')
    return parser


def _set_if_not_none(obj: Any, key: str, value: Any) -> None:
    if value is not None:
        obj[key] = value


def main() -> None:
    args = _parser().parse_args()
    config, create_config, train = _load_config(args.algo)

    if args.exp_name is not None:
        config.exp_name = args.exp_name
    _set_if_not_none(config.env, 'stop_value', args.stop_value)
    _set_if_not_none(config.env, 'max_episode_steps', args.max_episode_steps)
    _set_if_not_none(config.policy, 'update_per_collect', args.update_per_collect)
    _set_if_not_none(config.policy, 'batch_size', args.batch_size)
    _set_if_not_none(config.policy, 'eval_freq', args.eval_freq)
    _set_if_not_none(config.policy.model, 'num_res_blocks', args.num_res_blocks)
    _set_if_not_none(config.policy.model, 'num_channels', args.num_channels)

    if args.num_simulations is not None:
        if args.algo == 'alphazero':
            config.policy.mcts.num_simulations = args.num_simulations
        else:
            config.policy.num_simulations = args.num_simulations
    if args.collector_env_num is not None:
        config.env.collector_env_num = args.collector_env_num
        config.policy.collector_env_num = args.collector_env_num
    if args.evaluator_env_num is not None:
        config.env.evaluator_env_num = args.evaluator_env_num
        config.env.n_evaluator_episode = args.evaluator_env_num
        config.policy.evaluator_env_num = args.evaluator_env_num
    _set_if_not_none(config.policy, 'n_episode', args.n_episode)
    config.policy.cuda = not args.cpu

    max_env_step = args.max_env_step
    if max_env_step is None:
        max_env_step = int(2e6)

    if args.smoke_test:
        config.exp_name = args.exp_name or f'/tmp/chinese_chess_{args.algo}_smoke'
        config.env.collector_env_num = 1
        config.env.evaluator_env_num = 1
        config.env.n_evaluator_episode = 1
        config.env.max_episode_steps = 12
        config.env.stop_value = 2.0
        config.policy.collector_env_num = 1
        config.policy.evaluator_env_num = 1
        config.policy.n_episode = 1
        config.policy.update_per_collect = 1
        config.policy.batch_size = 4
        config.policy.eval_freq = 10000
        config.policy.model.num_res_blocks = 1
        config.policy.model.num_channels = 16
        config.policy.cuda = False
        if args.algo == 'alphazero':
            config.policy.mcts.num_simulations = 2
        else:
            config.policy.num_simulations = 2
            config.policy.game_segment_length = 6
            config.policy.td_steps = 6
            config.policy.num_unroll_steps = 2
            config.policy.replay_buffer_size = 100
        create_config.env_manager.type = 'base'
        max_env_step = 1

    print(
        f'XIANGQI_TRAIN_START algo={args.algo} seed={args.seed} max_env_step={max_env_step} '
        f'exp_name={config.exp_name}',
        flush=True,
    )
    train([config, create_config], seed=args.seed, max_env_step=max_env_step)
    print(f'XIANGQI_TRAIN_FINISHED algo={args.algo} exp_name={config.exp_name}', flush=True)


if __name__ == '__main__':
    main()
