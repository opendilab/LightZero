# Import the necessary libraries and configs based on the model you want to evaluate
from zoo.box2d.lunarlander.config.lunarlander_disc_muzero_config import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

if __name__ == "__main__":
    """
    Overview:
        Evaluate the model performance by running multiple episodes with different seeds using the MuZero algorithm.
        The evaluation results (returns and mean returns) are printed out for each seed and summarized for all seeds. 
    Variables:
        - model_path (:obj:`str`): Path to the pretrained model's checkpoint file. Usually something like 
          "exp_name/ckpt/ckpt_best.pth.tar". Absolute path is recommended.
        - seeds (:obj:`List[int]`): List of seeds to use for evaluation. Each seed will run for a specified number 
          of episodes.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to be run for each seed.
        - main_config (:obj:`EasyDict`): Main configuration for the evaluation, imported from the model's config file.
        - returns_mean_seeds (:obj:`List[float]`): List to store the mean returns for each seed.
        - returns_seeds (:obj:`List[List[float]]`): List to store the returns for each episode from each seed.
    Outputs:
        Prints out the mean returns and returns for each seed, along with the overall mean return across all seeds.

    .. note::
        The eval_muzero function is used here for evaluation. For more details about this function and its parameters, 
        please refer to its own documentation.
    """
    # model_path = './ckpt/ckpt_best.pth.tar'
    model_path = None

    seeds = [0]
    num_episodes_each_seed = 1
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    main_config.env.replay_path = './video'
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
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean())
    print("=" * 20)