# According to the model you want to evaluate, import the corresponding config.
from zoo.gobigger.entry import eval_muzero_gobigger
import numpy as np

if __name__ == "__main__":
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    """
    # ez
    # from gobigger_efficientzero_config import main_config, create_config

    # sez
    # from gobigger_sampled_efficientzero_config import main_config, create_config
    
    # mz
    from gobigger_muzero_config import main_config, create_config
    model_path = "exp_name/ckpt/ckpt_best.pth.tar"

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    create_config.env_manager.type = 'base'  # when visualize must set as  base
    main_config.env.evaluator_env_num = 1  # when visualize must set as 1
    main_config.env.n_evaluator_episode = 2  # each seed eval episodes num
    main_config.env.playback_settings.by_frame.save_frame = True
    main_config.env.playback_settings.by_frame.save_name_prefix = 'gobigger'

    for seed in seeds:
        returns_selfplay_mean, returns_vsbot_mean = eval_muzero_gobigger(
            [main_config, create_config],
            seed=seed,
            model_path=model_path,
        )
        print('seed: {}'.format(seed))
        print('returns_selfplay_mean: {}'.format(returns_selfplay_mean))
        print('returns_vsbot_mean: {}'.format(returns_vsbot_mean))
