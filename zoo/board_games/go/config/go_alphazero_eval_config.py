from zoo.board_games.go.config.go_alphazero_bot_mode_config import main_config, create_config
# from zoo.board_games.go.config.go_alphazero_sp_mode_config import main_config, create_config

from lzero.entry import eval_alphazero
import numpy as np

from zoo.board_games.go.envs.katago_policy import KatagoPolicy

if __name__ == '__main__':
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    """
    # model_path = './ckpt/ckpt_best.pth.tar'
    model_path = '/Users/puyuan/code/LightZero/tb_go_b9_bot/go_b9-komi-7.5_alphazero_bot-mode_rand0_nb-5-nc-64_ns200_upc200_rbs1e6_seed0/ckpt_best.pth.tar'
    # model_path = None
    main_config.env.use_katago_bot = True
    seeds = [0]
    num_episodes_each_seed = 1
    # If True, you can play with the agent.
    main_config.env.agent_vs_human = False

    # main_config.env.agent_vs_human = True
    main_config.env.save_gif_replay = True
    main_config.env.render_in_ui = True
    # main_config.env.render_in_ui = False
    main_config.env.save_gif_path = '/Users/puyuan/code/LightZero/tb_go_b9_bot/'


    # main_config.env.save_gif_replay = False
    # main_config.env.render_in_ui = False

    main_config.env.collector_env_num = 1
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    create_config.env_manager.type = 'base'
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

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}'
    )
    print("=" * 20)
