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
    # Importing the necessary configuration files from the atari muzero configuration in the zoo directory.
    # module_path = '/home/nighoodRen/LightZero/result/new_env/new_CrowdSim_vt20_muzero_md_ssl_step300000_uav2__human59_seed0'
    # import sys
    # if module_path not in sys.path:
    #     sys.path.append(module_path)
    # # 导入模块中的内容
    # from formatted_total_config import main_config, create_config
    # from result.new_env.new_CrowdSim_vt20_muzero_md_ssl_step300000_uav2__human59_seed0.formatted_total_config import main_config, create_config
    from zoo.CrowdSim.config.crowdsim_muzero_md_config import main_config, create_config

    # model_path is the path to the trained MuZero model checkpoint.
    # If no path is provided, the script will use the default model.
    model_path = '/home/nighoodRen/LightZero/result/old_env/CrowdSim_muzeromd_ssl_step300000_uav2__human59_seed0_240503_022923/ckpt/ckpt_best.pth.tar'
    main_config.exp_name = '/home/nighoodRen/LightZero/result/old_env/CrowdSim_muzeromd_ssl_step300000_uav2__human59_seed0_240503_022923/' + 'eval'   # original result folder/eval
    # seeds is a list of seed values for the random number generator, used to initialize the environment.
    seeds = [0]
    # num_episodes_each_seed is the number of episodes to run for each seed.
    num_episodes_each_seed = 1
    # total_test_episodes is the total number of test episodes, calculated as the product of the number of seeds and the number of episodes per seed
    total_test_episodes = num_episodes_each_seed * len(seeds)

    # Setting the type of the environment manager to 'base' for the visualization purposes.
    create_config.env_manager.type = 'base'
    # The number of environments to evaluate concurrently. Set to 1 for visualization purposes.
    main_config.env.evaluator_env_num = 1
    # The total number of evaluation episodes that should be run.
    main_config.env.n_evaluator_episode = total_test_episodes
    # A boolean flag indicating whether to render the environments in real-time.
    main_config.env.render_mode_human = False

    # A boolean flag indicating whether to save the video of the environment.
    main_config.env.save_replay = True
    # The path where the recorded video will be saved.
    main_config.env.replay_path = main_config.exp_name + '/video'   # current result folder/eval
    
    # The maximum number of steps for each episode during evaluation. This may need to be adjusted based on the specific characteristics of the environment.
    main_config.env.eval_max_episode_steps = int(20)

    # These lists will store the mean and total rewards for each seed.
    returns_mean_seeds = []
    returns_seeds = []

    # The main evaluation loop. For each seed, the MuZero model is evaluated and the mean and total rewards are recorded.
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

    # Convert the list of mean and total rewards into numpy arrays for easier statistical analysis.
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Printing the evaluation results. The average reward and the total reward for each seed are displayed, followed by the mean reward across all seeds.
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)