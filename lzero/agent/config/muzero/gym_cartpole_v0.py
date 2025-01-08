from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e5)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='CartPole-v0-MuZero',
        seed=0,
        env=dict(
            env_id='CartPole-v0',
            continuous=False,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=4,
                action_space_size=2,
                model_type='mlp',
                lstm_hidden_size=128,
                latent_state_dim=128,
                self_supervised_learning_loss=True,  # NOTE: default is False.
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=50,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            ssl_loss_weight=2,  # NOTE: default is 0.
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e2),
            replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config=dict(
        env=dict(
            type='cartpole_lightzero',
            import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='muzero',
            import_names=['lzero.policy.muzero'],
        ),
    ),
)

cfg = EasyDict(cfg)


if __name__ == "__main__":
    # Note: Install the `huggingface_ding` package using the following shell commands
    # git clone https://github.com/opendilab/huggingface_ding.git
    # cd huggingface_ding
    # pip3 install -e .

    # Import the required modules for downloading a pretrained model from Hugging Face Model Zoo
    from lzero.agent import MuZeroAgent
    from huggingface_ding import pull_model_from_hub

    # Pull the pretrained model and its configuration from the Hugging Face Hub
    policy_state_dict, cfg = pull_model_from_hub(repo_id="OpenDILabCommunity/CartPole-v0-MuZero")

    # Instantiate the agent (MuZeroAgent) with the environment, configuration, and policy state
    agent = MuZeroAgent(
        env_id="CartPole-v0",  # Environment ID
        exp_name="CartPole-v0-MuZero",  # Experiment name
        cfg=cfg.exp_config,  # Configuration for the experiment
        policy_state_dict=policy_state_dict  # Pretrained policy states
    )

    # Train the agent for 5000 steps
    agent.train(step=5000)

    # Render the performance of the trained agent and save the replay
    agent.deploy(enable_save_replay=True)