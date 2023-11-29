from lzero.entry import eval_muzero
import numpy as np

if __name__ == "__main__":
    """
    Overview:
        Main script to evaluate the MuZero model on Atari games. The script will loop over multiple seeds,
        evaluating a certain number of episodes per seed. Results are aggregated and printed.

    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, pointing to the ckpt file of the pretrained model. 
          The path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
        - seeds (:obj:`List[int]`): List of seeds to use for the evaluations.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to evaluate for each seed.
        - total_test_episodes (:obj:`int`): Total number of test episodes, calculated as num_episodes_each_seed * len(seeds).
        - returns_mean_seeds (:obj:`np.array`): Array of mean return values for each seed.
        - returns_seeds (:obj:`np.array`): Array of all return values for each seed.
    """
    # Take the config of MuZero as an example
    from zoo.atari.config.atari_muzero_config import main_config, create_config

    # model_path = "/path/ckpt/ckpt_best.pth.tar"
    model_path = None

    seeds = [0]
    num_episodes_each_seed = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1  # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
    main_config.env.render_mode_human = False  # Whether to enable real-time rendering

    main_config.env.save_replay = True  # Whether to save the video
    main_config.env.save_path = './video'
    main_config.env.eval_max_episode_steps = int(20)  # Adjust according to different environments

    returns_mean_seeds = []
    returns_seeds = []

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