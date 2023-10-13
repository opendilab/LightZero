# According to the model you want to evaluate, import the corresponding config.
import numpy as np

from lzero.entry import eval_muzero
from zoo.game_2048.config.muzero_2048_config import main_config, create_config
from zoo.game_2048.config.stochastic_muzero_2048_config import main_config, create_config

if __name__ == "__main__":
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    """

    model_path = "./ckpt/ckpt_best.pth.tar"

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1   # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
    main_config.env.save_replay = True  # Whether to save the replay, if save the video render_mode_human must to be True
    main_config.env.replay_format = 'gif'
    main_config.env.replay_name_suffix = 'muzero_ns100_s0'
    # main_config.env.replay_name_suffix = 'stochastic_muzero_ns100_s0'

    main_config.env.replay_path = None
    main_config.env.max_episode_steps = int(1e9)  # Adjust according to different environments

    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=False,
            model_path=model_path
        )
        print(returns_mean, returns)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean())
    print("=" * 20)
