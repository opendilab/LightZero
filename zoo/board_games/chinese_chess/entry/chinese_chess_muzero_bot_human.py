"""Let a human (Black) play a MuZero checkpoint (Red) in the terminal."""

import argparse

from lzero.entry import eval_muzero
from zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config import create_config, main_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='path to a MuZero .pth.tar checkpoint')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per bot move')
    args = parser.parse_args()

    main_config.env.agent_vs_human = True
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    main_config.policy.evaluator_env_num = 1
    main_config.policy.num_simulations = args.simulations
    create_config.env_manager.type = 'base'
    eval_muzero(
        [main_config, create_config],
        seed=0,
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=args.checkpoint
    )


if __name__ == '__main__':
    main()
