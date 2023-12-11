# Import the necessary libraries and configs based on the model you want to evaluate
from zoo.pooltool.sum_to_three.config.sum_to_three_config import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

NUM = 10

if __name__ == "__main__":
    model_path = './ckpt/iteration_0.pth.tar'

    seeds = np.random.randint(low=0, high=100, size=NUM)

    # Specify the number of environments for the evaluator to use
    main_config.env.evaluator_env_num = 1

    # Set the number of episodes for the evaluator to run
    main_config.env.n_evaluator_episode = 1

    # If save_replay_path is not None, num_episodes_each_seed _must_ be 1!
    #main_config.env.save_replay_path = "./video"
    main_config.env.save_replay_path = None
    num_episodes_each_seed = 1

    # The total number of test episodes is the product of the number of episodes per seed and the number of seeds
    total_test_episodes = num_episodes_each_seed * len(seeds)

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
