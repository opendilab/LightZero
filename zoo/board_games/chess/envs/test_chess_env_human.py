import pytest
from zoo.board_games.chess.envs.chess_env import ChessEnv


@pytest.mark.envtest
class TestChessEnvHuman:

    def test_human_vs_random(self):
        env = ChessEnv()
        obs = env.reset()
        print('Initial board state:')
        env.render()

        done = False
        step = 0
        while not done:
            step += 1
            print(f'Step {step}:')

            # Player 1 (human action)
            action = env.human_to_action()
            print(f'Player 1 action: {action}')
            obs, reward, done, info = env.step(action)
            self.check_step_outputs(obs, reward, done, info)
            env.render()
            if done:
                self.print_game_result(reward, 'Player 1 (Human)')
                break

            # Player 2 (random action)
            action = env.random_action()
            print(f'Player 2 action: {action}')
            obs, reward, done, info = env.step(action)
            self.check_step_outputs(obs, reward, done, info)
            env.render()
            if done:
                self.print_game_result(reward, 'Player 2 (Random)')
                break

        env.close()

    def test_bot_vs_human(self):
        env = ChessEnv()
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
            action = env.human_to_action()
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