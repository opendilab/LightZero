"""One-million-step optimized MuZero+SSL+RND run for KeyCorridorS3R3."""

from copy import deepcopy

from zoo.minigrid.config.minigrid_muzero_rnd_config import (
    create_config as baseline_create_config,
    main_config as baseline_main_config,
    seed,
)


max_env_step = int(1e6)
main_config = deepcopy(baseline_main_config)
create_config = deepcopy(baseline_create_config)

main_config.exp_name = (
    'data_mz_rnd_ctree/MiniGrid-KeyCorridorS3R3-v0_muzero-rnd_'
    'opt-v2_rnd-decay_td20_rb3e5_temp5e4_1m_seed0'
)

# A score above 0.8 means that all three deterministic evaluation layouts are
# being solved efficiently; stop immediately once that target is reached.
main_config.env.stop_value = 0.8

# The baseline first discovers success around 250k env steps, but its fixed
# novelty bonus remains active during evaluation planning.  Estimate count is
# one-to-one with learner updates (~10 learner updates per 100 env steps in the
# observed run): retain full RND through ~350k env steps and decay it to zero by
# ~700k so the second half of training consolidates the extrinsic objective.
main_config.reward_model.intrinsic_reward_weight_final = 0.0
main_config.reward_model.intrinsic_reward_weight_decay_start = int(3.5e4)
main_config.reward_model.intrinsic_reward_weight_decay_steps = int(3.5e4)

# The entry schedules visit-count temperature by learner iteration.  The old
# 5e5 threshold kept temperature at 1.0 throughout the first million env steps.
# This schedule reaches 0.5 around 250k and the final 0.25 around 375k while RND
# and root Dirichlet noise continue to provide exploration.
main_config.policy.threshold_training_steps_for_final_temperature = int(5e4)

# Propagate each rare terminal reward farther per update and discard the oldest
# random failures once the buffer reaches 300k transitions.  PER still retains
# and emphasizes surprising successful transitions within the recent window.
main_config.policy.td_steps = 20
main_config.policy.replay_buffer_size = int(3e5)


if __name__ == '__main__':
    from lzero.entry import train_muzero_with_reward_model

    train_muzero_with_reward_model(
        [main_config, create_config], seed=seed, max_env_step=max_env_step
    )
