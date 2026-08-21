"""Play terminal Chinese chess: human (Red) versus the built-in random bot (Black)."""

import argparse

from zoo.board_games.chinese_chess.envs.chinese_chess_env import (
    HUMAN_QUIT_MESSAGE,
    ChineseChessEnv,
    HumanQuitError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0, help='random-bot seed')
    args = parser.parse_args()

    env = ChineseChessEnv({'battle_mode': 'play_with_bot_mode'})
    env.seed(args.seed)
    env.reset()
    print('You are Red; the built-in bot plays Black.')
    print('Enter ICCS moves such as h9g7, or q to quit.')

    while True:
        try:
            action = env.human_to_action()
            timestep = env.step(action)
        except HumanQuitError as error:
            if str(error) != HUMAN_QUIT_MESSAGE:
                raise
            print('Game exited.')
            return

        if timestep.done:
            env.render('human')
            result = float(timestep.info.get('eval_episode_return', 0.0))
            print(f'Game over. Return for Red: {result:.1f}')
            return


if __name__ == '__main__':
    main()
