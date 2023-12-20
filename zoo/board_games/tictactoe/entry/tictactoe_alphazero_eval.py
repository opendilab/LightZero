from zoo.board_games.tictactoe.config.tictactoe_alphazero_bot_mode_config import main_config, create_config
from lzero.entry import eval_alphazero
import numpy as np

if __name__ == '__main__':
    """
    Entry point for the evaluation of the AlphaZero model on the TicTacToe environment. 

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
    seeds = [0]
    num_episodes_each_seed = 1

    # Enable saving of replay as a gif, specify the path to save the replay gif
    main_config.env.replay_path = './video'

    main_config.policy.mcts_ctree = False
    # If True, you can play with the agent.
    main_config.env.agent_vs_human = False
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)

    returns_mean_seeds = []
    returns_seeds = []
    for seed in seeds:
        returns_mean, returns = eval_alphazero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
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
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}'
    )
    print("=" * 20)
