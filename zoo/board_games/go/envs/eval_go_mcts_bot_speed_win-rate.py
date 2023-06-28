from easydict import EasyDict
from zoo.board_games.go.envs.go_env import GoEnv
from zoo.board_games.mcts_bot import MCTSBot

import pytest
import time
import numpy as np


def test_go_mcts_vs_random(num_simulations):
    mcts_bot_time_list = []
    random_bot_time_list = []
    winner = []
    for i in range(10):
        print('-' * 10 + 'episode' + str(i) + '-' * 10)

        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        env = GoEnv(cfg)
        obs = env.reset()
        state = obs['board']
        player_1 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1

        player_index = 0  # player 1 first
        # print('#' * 15)
        # print(state)
        # print('#' * 15)
        while not env.get_done_reward()[0]:
            if player_index == 0:
                t1 = time.time()
                action = player_1.get_actions(state, player_index=player_index)
                t2 = time.time()
                mcts_bot_time_list.append(t2 - t1)

                player_index = 1
            else:
                # print('-' * 40)
                t1 = time.time()
                action = env.random_action()
                t2 = time.time()
                random_bot_time_list.append(t2 - t1)
                player_index = 0

            timestep = env.step(action)
            # env.render('human')
            # time.sleep(0.1)
            state = timestep.obs['board']
            # print('-' * 40)
            # print(state)
        winner.append(env.get_done_winner()[1])

    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_std = np.std(mcts_bot_time_list)

    random_bot_mu = np.mean(random_bot_time_list)
    random_bot_std = np.std(random_bot_time_list)

    print('num_simulations={}\n'.format(num_simulations))

    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, bot_action_std={}\n'.format(mcts_bot_mu, mcts_bot_std))

    print('random_bot_time_list={}\n'.format(random_bot_time_list))
    print('random_bot_mu={}, random_bot_std={}\n'.format(random_bot_mu, random_bot_std))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_go_mcts_vs_mcts(num_simulations_player_1, num_simulations_player_2):
    mcts_bot_1_time_list = []
    mcts_bot_2_time_list = []
    winner = []
    for i in range(10):
        print('-' * 10 + 'episode' + str(i) + '-' * 10)

        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        env = GoEnv(cfg)
        obs = env.reset()
        state = obs['board']
        player_1 = MCTSBot(GoEnv, cfg, 'player 1', num_simulations_player_1)  # player_index = 0, player = 1
        player_2 = MCTSBot(GoEnv, cfg, 'player 2', num_simulations_player_2)  # player_index = 1, player = 2

        player_index = 0  # player 1 first
        # print('#' * 15)
        # print(state)
        # print('#' * 15)
        while not env.get_done_reward()[0]:
            if player_index == 0:
                t1 = time.time()
                action = player_1.get_actions(state, player_index=player_index)
                t2 = time.time()
                mcts_bot_1_time_list.append(t2 - t1)

                player_index = 1
            else:
                # print('-' * 40)
                t1 = time.time()
                action = player_2.get_actions(state, player_index=player_index)
                t2 = time.time()
                mcts_bot_2_time_list.append(t2 - t1)

                player_index = 0

            timestep = env.step(action)
            # env.render('human')
            # time.sleep(0.1)
            state = timestep.obs['board']
            # print('-' * 40)
            # print(state)
        winner.append(env.get_done_winner()[1])

    mcts_bot_1_mu = np.mean(mcts_bot_1_time_list)
    mcts_bot_1_std = np.std(mcts_bot_1_time_list)

    mcts_bot_2_mu = np.mean(mcts_bot_2_time_list)
    mcts_bot_2_std = np.std(mcts_bot_2_time_list)

    print('num_simulations_player_1 ={}\n'.format(num_simulations_player_1))
    print('num_simulations_player_2 ={}\n'.format(num_simulations_player_2))

    print('mcts_bot_1_time_list={}\n'.format(mcts_bot_1_time_list))
    print('mcts_bot_1_mu={}, mcts_bot_1_std={}\n'.format(mcts_bot_1_mu, mcts_bot_1_std))

    print('mcts_bot_2_time_list={}\n'.format(mcts_bot_2_time_list))
    print('mcts_bot_2_mu={}, bot_action_std={}\n'.format(mcts_bot_2_mu, mcts_bot_2_std))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


if __name__ == '__main__':
    cfg = EasyDict(dict(
        board_size=5,
        komi=0.5,
        num_simulations=2,
        num_simulations_player_1=2,
        num_simulations_player_2=2,

        # board_size=6,
        # komi=2.5,
        # num_simulations=80,
        # num_simulations_player_1=80,
        # num_simulations_player_2=80,

        # board_size=9,
        # komi=4.5,
        # num_simulations=180,
        # num_simulations_player_1=180,
        # num_simulations_player_2=180,

        # board_size=19,
        # komi=7.5,
        # num_simulations=800,
        # num_simulations_player_1=800,
        # num_simulations_player_2=800,
        
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='self_play_mode',
        scale=True,
        channel_last=True,
        agent_vs_human=False,
        bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
        prob_random_action_in_bot=0.,
        check_action_to_connect4_in_bot_v0=False,
    ))

    # ==============================================================
    # test win rate between mcts_bot and random_bot
    # ==============================================================
    test_go_mcts_vs_random(num_simulations=cfg.num_simulations)

    # ==============================================================
    # test win rate between mcts_bot and mcts_bot
    # ==============================================================
    test_go_mcts_vs_mcts(num_simulations_player_1=cfg.num_simulations_player_1,
                         num_simulations_player_2=cfg.num_simulations_player_2)
