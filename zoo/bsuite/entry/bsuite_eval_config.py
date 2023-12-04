from bsuite_muzero_config import main_config, create_config
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

    # Initialize a list with a single seed for the experiment
    seeds = [0]

    # Set the number of episodes to run for each seed
    num_episodes_each_seed = 1

    # Specify the number of environments for the evaluator to use
    main_config.env.evaluator_env_num = 1

    # Set the number of episodes for the evaluator to run
    main_config.env.n_evaluator_episode = 1

    # The total number of test episodes is the product of the number of episodes per seed and the number of seeds
    total_test_episodes = num_episodes_each_seed * len(seeds)

    # Uncomment the following lines to save a replay of the episodes as an mp4 video
    # main_config.env.replay_path = './video'

    # Enable saving of replay as a gif, specify the path to save the replay gif
    main_config.env.save_replay_gif = True
    main_config.env.replay_path_gif = './video'

    # Initialize lists to store the mean and total returns for each seed
    returns_mean_seeds = []
    returns_seeds = []

    # For each seed, run the evaluation function and store the resulting mean and total returns
    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],  # Configuration parameters for the evaluation
            seed=seed,  # The seed for the random number generator
            num_episodes_each_seed=num_episodes_each_seed,  # The number of episodes to run for this seed
            print_seed_details=False,  # Whether to print detailed information for each seed
            model_path=model_path  # The path to the trained model to be evaluated
        )
        # Append the mean and total returns to their respective lists
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    # Convert the lists of returns to numpy arrays for easier statistical analysis
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Print evaluation results
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)