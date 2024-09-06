from easydict import EasyDict

from zoo.board_games.chess.envs.chess_env import ChessEnv


class EvalChessEnvHuman:

    def eval_bot_vs_human(self):
        cfg = EasyDict({
            'render_mode': 'ansi',
            # 'render_mode': 'human',
            'replay_path': None,
        })
        env = ChessEnv(cfg)
        obs = env.reset()
        print('Initial board state:')
        env.render()

        done = False
        step = 0
        while not done:
            step += 1
            print(f'Step {step}:')

            # Player 1 (bot action)
            action = env.bot_action()
            print(f'Player 1 action: {action}')
            obs, reward, done, info = env.step(action)
            self.check_step_outputs(obs, reward, done, info)
            env.render()
            if done:
                self.print_game_result(reward, 'Player 1 (Bot)')
                break

            # Player 2 (human action)
            action = env.human_to_action(interactive=True)
            print(f'Player 2 action: {action}')
            obs, reward, done, info = env.step(action)
            self.check_step_outputs(obs, reward, done, info)
            env.render()
            if done:
                self.print_game_result(reward, 'Player 2 (Human)')
                break

        env.close()

    def check_step_outputs(self, obs, reward, done, info):
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert isinstance(reward, int)
        assert isinstance(info, dict)

    def print_game_result(self, reward, player):
        if reward > 0:
            print(f'{player} wins!')
        elif reward < 0:
            print(f'{player} loses!')
        else:
            print('The game is a draw!')

if __name__ == '__main__':
    eval_env = EvalChessEnvHuman()
    eval_env.eval_bot_vs_human()