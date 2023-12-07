# According to the model you want to evaluate, import the corresponding config.
import numpy as np

from lzero.entry import eval_muzero
from zoo.game_2048.config.muzero_2048_config import main_config, create_config
from zoo.game_2048.config.stochastic_muzero_2048_config import main_config, create_config

if __name__ == "__main__":
    """
    Entry point for the evaluation of the muzero or stochastic_muzero model on the 2048 environment. 

    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should point to the ckpt file of the 
        pretrained model. An absolute path is recommended. In LightZero, the path is usually something like 
        ``exp_name/ckpt/ckpt_best.pth.tar``.
        - returns_mean_seeds (:obj:`List[float]`): List to store the mean returns for each seed.
        - returns_seeds (:obj:`List[float]`): List to store the returns for each seed.
        - seeds (:obj:`List[int]`): List of seeds for the environment.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to run for each seed.
        - total_test_episodes (:obj:`int`): Total number of test episodes, computed as the product of the number of 
        seeds and the number of episodes per seed.
    """

    # model_path = './ckpt/ckpt_best.pth.tar'
    model_path = None

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 1

    # main_config.env.render_mode = 'image_realtime_mode'
    main_config.env.render_mode = 'image_savefile_mode'
    main_config.env.replay_path = './video'
    main_config.env.replay_format = 'gif'
    main_config.env.replay_name_suffix = 'muzero_ns100_s0'
    # main_config.env.replay_name_suffix = 'stochastic_muzero_ns100_s0'

    main_config.env.max_episode_steps = int(1e9)  # Adjust according to different environments
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1   # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
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
