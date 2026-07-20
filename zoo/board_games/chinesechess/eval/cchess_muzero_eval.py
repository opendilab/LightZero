from zoo.board_games.chinesechess.config.cchess_muzero_sp_mode_config import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

if __name__ == '__main__':
    """
    中国象棋 MuZero 模型的评估入口

    变量说明:
        - model_path: 预训练模型路径，应指向 ckpt 文件
        - returns_mean_seeds: 每个种子的平均回报列表
        - returns_seeds: 每个种子的回报列表
        - seeds: 环境种子列表
        - num_episodes_each_seed: 每个种子运行的局数
        - total_test_episodes: 总测试局数

    python -m LightZero.zoo.board_games.chinesechess.eval.cchess_muzero_eval
    
    """
    # model_path = './ckpt/ckpt_best.pth.tar'
    model_path = r'./data_muzero/cchess_self-play-mode_seed0/ckpt/iteration_0.pth.tar'
    seeds = [0]
    num_episodes_each_seed = 1

    # 如果设置为 True，可以与 agent 对弈
    # main_config.env.agent_vs_human = True
    main_config.env.agent_vs_human = True

    # 渲染模式
    main_config.env.render_mode = 'image_realtime_mode'
    # main_config.env.render_mode = None
    main_config.env.replay_path = './video'

    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1

    total_test_episodes = num_episodes_each_seed * len(seeds)
    returns_mean_seeds = []
    returns_seeds = []

    for seed in seeds:
        returns_mean, returns = eval_muzero(
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
    print(f"总共评估了 {len(seeds)} 个种子。每个种子评估了 {num_episodes_each_seed} 局。")
    print(f"种子 {seeds} 的平均回报为 {returns_mean_seeds}，回报为 {returns_seeds}。")
    print("所有种子的平均奖励:", returns_mean_seeds.mean())
    print(
        f'胜率: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes:.2%}, '
        f'和率: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes:.2%}, '
        f'负率: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes:.2%}'
    )
    print("=" * 20)

