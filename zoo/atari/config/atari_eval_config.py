# According to the model you want to evaluate, import the corresponding config.

from .atari_muzero_config import main_config, create_config
# from .atari_efficientzero_config import main_config, create_config
# from .atari_sampled_efficientzero_config import main_config, create_config

if __name__ == "__main__":
    from lzero.entry import eval_muzero
    import numpy as np
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
     """
    model_path = "./ckpt/ckpt_best.pth.tar"

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 5
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = eval_muzero([main_config, create_config], seed=seed,
                                                            num_episodes_each_seed=num_episodes_each_seed,
                                                            print_seed_details=False, model_path=model_path)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean())
    print("=" * 20)