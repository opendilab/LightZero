from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 6
n_episode = 6
evaluator_env_num = 6
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = None
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.0
eval_freq = 1000
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

sumtothree_cont_sampled_efficientzero_config = dict(
    exp_name=f"data_pooltool_ctree/sumtothree_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0",
    env=dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(
            shared_memory=False,
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type="conditioned",
            model_type="mlp",
            lstm_hidden_size=512,
            latent_state_dim=256,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type="BN",
        ),
        cuda=True,
        env_type="not_board_games",
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type="Adam",
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        # NOTE: this parameter is important for stability in bipedalwalker.
        grad_clip_value=0.5,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=eval_freq,
        # the size/capacity of replay_buffer, in the terms of transitions.
        replay_buffer_size=int(3e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
sumtothree_cont_sampled_efficientzero_config = EasyDict(
    sumtothree_cont_sampled_efficientzero_config
)
main_config = sumtothree_cont_sampled_efficientzero_config
sumtothree_cont_sampled_efficientzero_create_config = dict(
    env=dict(
        type="pooltool_sumtothree",
        import_names=["zoo.pooltool.sum_to_three.envs.sum_to_three_env"],
    ),
    env_manager=dict(type="subprocess"),
    #env_manager=dict(type="base"),
    policy=dict(
        type="sampled_efficientzero",
        import_names=["lzero.policy.sampled_efficientzero"],
    ),
    collector=dict(
        type="episode_muzero",
        get_train_sample=True,
        import_names=["lzero.worker.muzero_collector"],
    ),
)
sumtothree_cont_sampled_efficientzero_create_config = EasyDict(
    sumtothree_cont_sampled_efficientzero_create_config
)
create_config = sumtothree_cont_sampled_efficientzero_create_config

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from lzero.entry import train_muzero

    parser = argparse.ArgumentParser(
        description="Train MuZero with an optional path to the checkpoint file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the checkpoint file (.pth.tar)",
        required=False,
    )
    args = parser.parse_args()

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"The specified checkpoint file does not exist: {model_path}"
            )
        model_path = model_path.resolve()

    train_muzero(
        [main_config, create_config],  # type: ignore
        seed=0,
        max_env_step=max_env_step,
        model_path=None if model_path is None else str(model_path),
    )
