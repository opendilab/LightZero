from zoo.board_games.gomoku.config.gomoku_alphazero_bot_mode_config import main_config, create_config


if __name__ == '__main__':
    from lzero.entry import eval_alphazero
    import numpy as np

    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
     """
    model_path = '/Users/user/code/LightZero/zoo/board_games/gomoku/gomoku_alphazero_bot-mode_rand0.5_ns50_upc50_rr0.3_seed0/ckpt/ckpt_best.pth.tar'
    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 3
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = eval_alphazero([main_config, create_config], seed=seed,
                                               num_episodes_each_seed=num_episodes_each_seed,
                                               print_seed_details=True, model_path=model_path)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}')
    print("=" * 20)
