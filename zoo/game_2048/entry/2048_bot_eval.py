import numpy as np
from easydict import EasyDict
from rich import print

from zoo.game_2048.envs.expectimax_search_based_bot import expectimax_search
from zoo.game_2048.envs.game_2048_env import Game2048Env

# Define game configuration
config = EasyDict(dict(
    env_name="game_2048",
    save_replay=False,
    replay_format='gif',
    replay_name_suffix='bot',
    replay_path=None,
    render_real_time=False,
    act_scale=True,
    channel_last=True,
    obs_type='raw_board',  # options=['raw_board', 'raw_encoded_board', 'dict_encoded_board']
    reward_type='raw',  # options=['raw', 'merged_tiles_plus_log_max_tile_num']
    reward_normalize=False,
    reward_norm_scale=100,
    max_tile=int(2 ** 16),
    delay_reward_step=0,
    prob_random_agent=0.,
    max_episode_steps=int(1e4),
    is_collect=False,
    ignore_legal_actions=True,
    need_flatten=False,
    num_of_possible_chance_tile=2,
    possible_tiles=np.array([2, 4]),
    tile_probabilities=np.array([0.9, 0.1]),
))

if __name__ == "__main__":
    game_2048_env = Game2048Env(config)
    obs = game_2048_env.reset()
    print('init board state: ')
    game_2048_env.render()
    step = 0
    while True:
        print('=' * 40)
        grid = obs.astype(np.int64)
        # action = game_2048_env.human_to_action()  # which obtain about 10000 score
        # action = game_2048_env.random_action()  # which obtain about 1000 score
        action = expectimax_search(grid)  # which obtain about 300000~70000 score
        obs, reward, done, info = game_2048_env.step(action)
        step += 1
        print(f"step: {step}, action: {action}, reward: {reward}, raw_reward: {info['raw_reward']}")
        game_2048_env.render(mode='human')
        if done:
            print('total_step_number: {}'.format(step))
            break
