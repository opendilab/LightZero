from copy import deepcopy

from zoo.box2d.lunarlander.config.lunarlander_disc_unizero_config import (
    create_config as _unizero_create_config,
    main_config as _unizero_main_config,
    max_env_step,
)


main_config = deepcopy(_unizero_main_config)
create_config = deepcopy(_unizero_create_config)
main_config.exp_name = 'data_unizero_ppo/lunarlander_disc_unizero_ppo_seed0'
# Gymnasium removed v2; v3 keeps the discrete observation/action interface.
main_config.env.env_name = 'LunarLander-v3'
main_config.env.env_id = 'LunarLander-v3'
main_config.policy.policy_improvement = 'ppo'
main_config.policy.collect_with_pure_policy = True
main_config.policy.ppo = dict(
    gamma=0.997,
    gae_lambda=0.95,
    clip_ratio=0.2,
    entropy_weight=0.01,
    epochs=4,
    minibatch_size=256,
    normalize_advantage=True,
    target_kl=0.03,
    fresh_ratio_tolerance=1e-5,
    world_model_update_per_collect=None,
)


def main(seed=0, max_env_step_override=max_env_step, cuda=True):
    from lzero.entry import train_unizero
    run_config = deepcopy(main_config)
    run_config.exp_name = f'data_unizero_ppo/lunarlander_disc_unizero_ppo_seed{seed}'
    if not cuda:
        run_config.policy.cuda = False
        run_config.policy.model.world_model_cfg.device = 'cpu'
    train_unizero(
        [run_config, deepcopy(create_config)], seed=seed,
        max_env_step=max_env_step_override,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train UniZero+PPO on LunarLander.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-env-step', type=int, default=max_env_step)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args.seed, args.max_env_step, cuda=not args.cpu)
