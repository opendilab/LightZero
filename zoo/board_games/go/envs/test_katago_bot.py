import logging

import pytest
from easydict import EasyDict

from zoo.board_games.go.envs.go_env import GoEnv, flatten_action_to_gtp_action
from zoo.board_games.go.envs.katago_policy import str_coord, KatagoPolicy, GameState

cfg = EasyDict(
    # board_size=19,
    board_size=9,
    # board_size=6,
    save_gif_replay=True,
    # render_in_ui=True,
    # save_gif_replay=False,
    render_in_ui=False,
    katago_checkpoint_path="/Users/puyuan/code/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
    ignore_pass_if_have_other_legal_actions=True,
    save_gif_path='./',
    komi=7.5,
    battle_mode='self_play_mode',
    prob_random_agent=0,
    channel_last=False,
    scale=True,
    agent_vs_human=False,
    bot_action_type='v0',
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
    stop_value=1,
    katago_device='cpu',
    mcts_ctree=True,
)

cfg.katago_policy = KatagoPolicy(checkpoint_path=cfg.katago_checkpoint_path, board_size=cfg.board_size,
                                 ignore_pass_if_have_other_legal_actions=cfg.ignore_pass_if_have_other_legal_actions,
                                 device=cfg.katago_device)


@pytest.mark.envtest
class TestKataGoBot:

    def test_katago_bot(self):

        env = GoEnv(cfg)
        test_episodes = 1
        for i in range(test_episodes):
            print('=' * 20)
            print(f'episode {i}')
            print('=' * 20)

            obs = env.reset()
            # print(obs['observation'].shape, obs['action_mask'].shape)
            # print(obs['observation'], obs['action_mask'])
            # env.render()
            # TODO(pu): katago_game_state init
            turn = 0
            while True:
                turn += 1
                print('turn: ', turn)
                # ****** player 1's turn ******
                # bot_action = env.random_action()
                # bot_action = env.human_to_action()
                # bot_action = env.human_to_gtp_action()
                bot_action = env.get_katago_action(to_play=1)

                if bot_action not in env.legal_actions:
                    # logging.warning(
                    #     f"You input illegal *bot* action: {bot_action}, the legal_actions are {env.legal_actions}. "
                    #     f"Now we randomly choice a action from self.legal_actions."
                    # )
                    # bot_action = np.random.choice(env.legal_actions)
                    logging.warning(
                        f"You input illegal *bot* action: {bot_action}, the legal_actions are {env.legal_actions}. "
                        f"Now we choice the first action from self.legal_actions."
                    )
                    bot_action = env.legal_actions[0]

                katago_flatten_action = env.lz_flatten_to_katago_flatten(bot_action, env.board_size)
                print('player 1:', str_coord(katago_flatten_action, env.katago_game_state.board))

                # TODO(pu): wheather to keep the history boards and moves?
                # env.katago_game_state = GameState(env.board_size)
                # env.katago_game_state.board = env.katago_board

                # env.update_katago_internal_game_state(katago_flatten_action, to_play=1)

                action = bot_action
                # action = actions_black[i]
                # print('player 1 (black): ', action)
                obs, reward, done, info = env.step(action)
                # time.sleep(0.1)
                # print(obs, reward, done, info)
                assert isinstance(obs, dict)
                assert isinstance(done, bool)
                assert isinstance(reward, float) or isinstance(reward, int)
                # env.render('board')

                if done:
                    if reward > 0:
                        print('player 1 (black) win')
                    elif reward < 0:
                        print('player 2 (white) win')
                    else:
                        print('draw')
                    break

                # ****** player 2's turn ******
                # bot_action = env.human_to_gtp_action()
                bot_action = env.get_katago_action(to_play=2)
                if bot_action not in env.legal_actions:
                    # logging.warning(
                    #     f"You input illegal *bot* action: {bot_action}, the legal_actions are {env.legal_actions}. "
                    #     f"Now we randomly choice a action from self.legal_actions."
                    # )
                    # bot_action = np.random.choice(env.legal_actions)
                    logging.warning(
                        f"You input illegal *bot* action: {bot_action}, the legal_actions are {env.legal_actions}. "
                        f"Now we choice the first action from self.legal_actions."
                    )
                    bot_action = env.legal_actions[0]

                # ****** update katago internal game state ******
                # TODO(pu): how to avoid this?
                katago_flatten_action = env.lz_flatten_to_katago_flatten(bot_action, env.board_size)
                print('player 2:', str_coord(katago_flatten_action, env.katago_game_state.board))
                # TODO(pu): wheather to keep the history boards and moves?
                # env.update_katago_internal_game_state(katago_flatten_action, to_play=2)

                # action = env.random_action()
                action = bot_action

                # action = actions_white[i]
                # print('player 2 (white): ', action)
                obs, reward, done, info = env.step(action)
                # time.sleep(0.1)
                # print(self.board)
                # print(obs, reward, done, info)
                # env.render('board')

                if done:
                    if reward > 0:
                        print('player 2 (white) win')
                    elif reward < 0:
                        print('player 1 (black) win')
                    else:
                        print('draw')
                    break


TestKataGoBot().test_katago_bot()
