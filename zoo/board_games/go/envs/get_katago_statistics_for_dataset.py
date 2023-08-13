import numpy as np
from easydict import EasyDict

from get_episode_gtp_actions import read_episode_gtp_actions_from_txt
from zoo.board_games.go.envs.go_env import GoEnv
from zoo.board_games.go.envs.katago_policy import KatagoPolicy

cfg = EasyDict(
    board_size=19,
    # board_size=9,
    save_gif_replay=True,
    render_in_ui=False,
    # render_in_ui=True,
    # save_gif_replay=False,
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


def get_katago_statistics_for_dataset():
    """
    Overview:
        Get katago statistics for dataset.
    Returns:
        - katago_statistics_episode (:obj:`list`): The katago statistics for each step in the episode.
        - katago_statistics_step (:obj:`dict`): The katago statistics for each step.

    Note:
        - katago_statistics_step.keys() (:obj:`list`): The keys of katago_statistics_step.
            - dict_keys(['policy0', 'policy1', 'moves_and_probs0', 'moves_and_probs1', 'value', 'td_value', 'td_value2', 'td_value3', 'scoremean', 'td_score', 'scorestdev', 'lead', 'vtime', 'estv', 'ests', 'ownership', 'ownership_by_loc', 'scoring', 'scoring_by_loc', 'futurepos', 'futurepos0_by_loc', 'futurepos1_by_loc', 'seki', 'seki_by_loc', 'seki2', 'seki_by_loc2', 'scorebelief', 'genmove_result'])
        - katago_statistics_step['moves_and_probs0'] (:obj:`list`): The moves and probs for player 0.
        - katago_statistics_step['moves_and_probs1'] (:obj:`list`): The moves and probs for player 1.
        - each element in 'moves_and_probs0' (:obj:`tuple`): (katago_flatten_action, prior).
    """
    env = GoEnv(cfg)
    dataset_file_dir = '/Users/puyuan/code/LightZero/go_sgf_dataset/1_result_pos.txt'
    episode_gtp_actions = read_episode_gtp_actions_from_txt(dataset_file_dir)
    test_episodes = 1
    for i in range(test_episodes):
        print('=' * 20)
        print(f'episode {i}')
        print('=' * 20)
        katago_statistics_episode = []

        turn = 0
        obs = env.reset()
        while True:
            turn += 1
            print('turn: ', turn)

            # ****** player 1's turn ******
            katago_statistics_step = env.katago_policy.get_katago_statistics(env.katago_game_state)

            katago_statistics_episode.append(katago_statistics_step)
            try:
                dataset_gtp_action = episode_gtp_actions[2 * (turn - 1)]
            except IndexError:
                # in dataset, the last action is not passed, so the env is not done, we need to break the loop.
                print('IndexError')
                break
            flatten_action = env.gtp_action_to_lz_flatten_action(dataset_gtp_action, board_size=env.board_size)
            action = flatten_action
            obs, reward, done, info = env.step(action)
            # env.render('board')
            # print(obs, reward, done, info)

            if done:
                if reward > 0:
                    print('player 1 (black) win')
                elif reward < 0:
                    print('player 2 (white) win')
                else:
                    print('draw')
                break

            # ****** player 2's turn ******
            katago_statistics_step = env.katago_policy.get_katago_statistics(env.katago_game_state)
            katago_statistics_episode.append(katago_statistics_step)
            try:
                dataset_gtp_action = episode_gtp_actions[2 * (turn - 1) + 1]
            except IndexError:
                # in dataset, the last action is not passed, so the env is not done, we need to break the loop.
                print('IndexError')
                break

            flatten_action = env.gtp_action_to_lz_flatten_action(dataset_gtp_action, board_size=env.board_size)
            action = flatten_action

            obs, reward, done, info = env.step(action)
            # env.render('board')
            # print(obs, reward, done, info)

            if done:
                if reward > 0:
                    print('player 2 (white) win')
                elif reward < 0:
                    print('player 1 (black) win')
                else:
                    print('draw')
                break

            # if turn > 10:
            #     break

        np.save(f'./katago_statistics_episode_{i}.npy', katago_statistics_episode)
        print('katago_statistics_episode saved!')


if __name__ == "__main__":
    get_katago_statistics_for_dataset()
