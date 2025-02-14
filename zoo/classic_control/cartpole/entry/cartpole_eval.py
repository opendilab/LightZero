from zoo.classic_control.cartpole.config.cartpole_muzero_config import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

if __name__ == "__main__":
    """
    Entry point for the evaluation of the MuZero model on the CartPole environment. 

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
    # model_path = "./ckpt/ckpt_best.pth.tar"
    model_path = None
    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 2
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1  # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
    main_config.env.save_replay_gif = True
    main_config.env.replay_path_gif = './cartpole_gif'

    for seed in seeds:
        """
        - returns_mean (:obj:`float`): The mean return of the evaluation.
        - returns (:obj:`List[float]`): The returns of the evaluation.
        """
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=False,
            model_path=model_path
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Print evaluation results
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)