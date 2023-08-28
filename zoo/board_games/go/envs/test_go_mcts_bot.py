from easydict import EasyDict
from zoo.board_games.go.envs.go_env import GoEnv, flatten_action_to_gtp_action
from zoo.board_games.go.envs.katago_policy import KatagoPolicy
from zoo.board_games.mcts_bot import MCTSBot

import pytest
import time
import numpy as np

cfg = EasyDict(dict(
    board_size=6,
    num_simulations=80,
    # board_size=5,
    # num_simulations=20,
    save_gif_replay=False,
    render_in_ui=False,
    # katago_checkpoint_path="/Users/puyuan/code/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
    katago_checkpoint_path="/mnt/nfs/puyuan/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
    ignore_pass_if_have_other_legal_actions=True,
    save_gif_path='./',
    komi=7.5,
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    scale=True,
    channel_last=True,
    agent_vs_human=False,
    bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
    katago_device='cpu',
))

cfg.katago_policy = KatagoPolicy(checkpoint_path=cfg.katago_checkpoint_path, board_size=cfg.board_size,
                                 ignore_pass_if_have_other_legal_actions=cfg.ignore_pass_if_have_other_legal_actions,
                                 device=cfg.katago_device)


@pytest.mark.envtest
class TestGoBot:

    # def test_go_mcts_vs_human(self):
    #     # player_0  num_simulation=1000, will win
    #     # player_1  num_simulation=1
    #     env = GoEnv(cfg)
    #     obs = env.reset()
    #     state = obs['board']
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #
    #     player_index = 0  # player 1 first
    #     print('#' * 15)
    #     print(state)
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             # action = player_2.get_actions(state, player_index=player_index)
    #             # action = env.random_action()
    #             action = env.human_to_action()
    #             player_index = 0
    #
    #         timestep = env.step(action)
    #         env.render('human')
    #         # time.sleep(0.1)
    #         state = timestep.obs['board']
    #         print('-' * 40)
    #         print(state)
    #     assert env.get_done_winner()[1] == 1, f'winner is {env.get_done_winner()[1]}, player 1 should win'
    #
    # def test_go_mcts_vs_random(self):
    #     # player_0  num_simulation=1000, will win
    #     # player_1  num_simulation=1
    #     env = GoEnv(cfg)
    #     obs = env.reset()
    #     state = obs['board']
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #
    #     player_index = 0  # player 1 first
    #     print('#' * 15)
    #     print(state)
    #     print('#' * 15)
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             # action = player_2.get_actions(state, player_index=player_index)
    #             action = env.random_action()
    #             player_index = 0
    #
    #         timestep = env.step(action)
    #         # env.render('human')
    #         # time.sleep(0.1)
    #         state = timestep.obs['board']
    #         print('-' * 40)
    #         print(state)
    #     assert env.get_done_winner()[1] == 1, f'winner is {env.get_done_winner()[1]}, player 1 should win'
    #
    # def test_go_self_play_mode_player1_win(self):
    #     # player_0  num_simulation=1000, will win
    #     # player_1  num_simulation=1
    #     env = GoEnv(cfg)
    #     obs = env.reset()
    #     state = obs['board']
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #     player_2 = MCTSBot(GoEnv, cfg, 'player 2', int(cfg.num_simulations/2))  # player_index = 1, player = 2
    #
    #     player_index = 0  # player 1 first
    #     print('#' * 15)
    #     print(state)
    #     print('#' * 15)
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             action = player_2.get_actions(state, player_index=player_index)
    #             player_index = 0
    #         timestep = env.step(action)
    #         state = timestep.obs['board']
    #         print('-' * 40)
    #         print(state)
    #     assert env.get_done_winner()[1] == 1, f'winner is {env.get_done_winner()[1]}, player 1 should win'
    #
    # def test_go_self_play_mode_player2_win(self):
    #     # player_0  num_simulation=1
    #     # player_1  num_simulation=1000, will win
    #     env = GoEnv(cfg)
    #     obs = env.reset()
    #     state = obs['board']
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 1', 1)  # player_index = 0, player = 1
    #     player_2 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
    #
    #     player_index = 0  # player 1 first
    #     print('#' * 15)
    #     print(state)
    #     print('#' * 15)
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             action = player_2.get_actions(state, player_index=player_index)
    #             player_index = 0
    #         timestep = env.step(action)
    #         state = timestep.obs['board']
    #         print('-' * 40)
    #         print(state)
    #     assert env.get_done_winner()[1] == 2, f'winner is {env.get_done_winner()[1]}, player 2 should win'
    #
    # def test_go_self_play_mode_draw(self):
    #     # player_0  num_simulation=1000
    #     # player_1  num_simulation=1000, will draw
    #     cfg.num_simulations = 50

    #     env = GoEnv(cfg)
    #     obs = env.reset()
    #     state = obs['board']

    #     player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #     player_2 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2

    #     player_index = 0  # player 1 fist
    #     print(state)
    #     print('-' * 40)
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             action = player_2.get_actions(state, player_index=player_index)
    #             player_index = 0
    #         timestep = env.step(action)

    #         env.render('human')
    #         # time.sleep(0.1)

    #         state = timestep.obs['board']
    #         print('-' * 40)
    #         print(state)
    #     assert env.get_done_winner()[1] == -1, f'winner is {env.get_done_winner()[1]}, two players should draw'

    def test_go_self_play_mode_case_1(self):
        env = GoEnv(cfg)
        init_state = np.array([
            [-1, -1, 1, -1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 1, 0],
            [-1, 0, 0, -1, -1, 1],
            [0, -1, 0, 1, 1, 0],
        ])

        # TODO
        cfg.num_simulations = 50

        player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
        player_2 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
        player_index = 0  # player 1 fist

        obs = env.reset(player_index, init_state)
        state = obs['board']
        print(state)
        print('#' * 15)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_1.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 1")
            else:
                action = player_2.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 1")
                player_index = 0

    def test_go_self_play_mode_case_2(self):
        env = GoEnv(cfg)
        init_state = np.array([
            [0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, -1, 0],
            [0, -1, 0, 0, -1, 0],
            [-1, -1, 1, 1, -1, -1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ])
        # TODO
        cfg.num_simulations = 50

        player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
        player_2 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
        player_index = 0  # player 1 fist

        obs = env.reset(player_index, init_state)
        state = obs['board']
        print(state)
        print('#' * 15)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_1.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 2")
            else:
                action = player_2.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 2")
                player_index = 0

    def test_go_self_play_mode_case_3(self):
        env = GoEnv(cfg)
        init_state = np.array([
            [1, 1, 1, 1, 0, 0],
            [1, -1, -1, 1, 1, 1],
            [1, 0, 0, -1, -1, 1],
            [1, -1, 0, 1, -1, 1],
            [1, -1, 1, 1, -1, 1],
            [-1, -1, -1, -1, -1, 1],
        ])
        # TODO
        cfg.num_simulations = 50

        player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
        player_2 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
        player_index = 0  # player 1 fist

        obs = env.reset(player_index, init_state)
        state = obs['board']
        print(state)
        print('#' * 15)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_1.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 3")
            else:
                action = player_2.get_actions(state, player_index=player_index)
                print(f"mcts_gtp_action: {flatten_action_to_gtp_action(action, cfg.board_size)} for case 3")
                player_index = 0

# test = TestGoBot().test_go_self_play_mode_player1_win()
# test = TestGoBot().test_go_self_play_mode_draw()
# test = TestGoBot().test_go_mcts_vs_human()

# test = TestGoBot().test_go_self_play_mode_case_1()
# test = TestGoBot().test_go_self_play_mode_case_2()
# test = TestGoBot().test_go_self_play_mode_case_3()


