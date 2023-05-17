from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6  # default_size is 15
prob_random_action_in_bot = 0.5

collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)
sp_prob = 0.  # TODO(pu): 0, 0.8, 1
snapshot_the_player_in_iter_zero = False
one_phase_step = int(1e5)

# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 2
# update_per_collect = 5
# batch_size = 2
# max_env_step = int(2e5)
# sp_prob = 0.
# snapshot_the_player_in_iter_zero = False
# one_phase_step = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_alphazero_league_config = dict(
    exp_name=f"data_az_ptree_league/gomoku_alphazero_league_sp-{sp_prob}_iter-zero-init-{snapshot_the_player_in_iter_zero}_phase-step-{one_phase_step}_seed0",
    env=dict(
        stop_value=2,
        env_name="Gomoku",
        board_size=board_size,
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        channel_last=False,
        scale=True,
        agent_vs_human=False,
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        prob_random_action_in_bot=prob_random_action_in_bot,
        check_action_to_connect4_in_bot_v0=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        prob_random_agent=0,
        prob_expert_agent=0,
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            num_res_blocks=1,
            num_channels=32,
        ),
        cuda=True,
        board_size=board_size,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        # eval_freq=int(100),  # debug
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        league=dict(
            log_freq_for_payoff_rank=50,
            # log_freq=2,  # debug
            player_category=['gomoku'],
            # path to save policy of league player, user can specify this field
            path_policy=f"data_az_ptree_league/gomoku_alphazero_league_sp-{sp_prob}_iter-zero-init-{snapshot_the_player_in_iter_zero}_phase-step-{one_phase_step}_policy_ckpt_seed0",
            active_players=dict(main_player=1, ),
            main_player=dict(
                # An active player will be considered trained enough for snapshot after two phase steps.
                one_phase_step=one_phase_step,
                # A namedtuple of probabilities of selecting different opponent branch.
                # branch_probs=dict(pfsp=0.2, sp=0.8),
                branch_probs=dict(pfsp=1 - sp_prob, sp=sp_prob),
                # If win rates between this player and all the opponents are greater than this value, this player can
                # be regarded as strong enough to these opponents. If also already trained for one phase step,
                # this player can be regarded as trained enough for snapshot.
                strong_win_rate=0.7,
            ),
            use_pretrain=False,
            use_pretrain_init_historical=False,
            # "use_bot_init_historica" means whether to use bot as an init historical player
            use_bot_init_historical=True,
            # "snapshot_the_player_in_iter_zero" means whether to snapshot the player in iter zero as historical_player.
            snapshot_the_player_in_iter_zero=snapshot_the_player_in_iter_zero,
            payoff=dict(
                type='battle',
                decay=0.99,
                min_win_rate_games=4,
            ),
            metric=dict(
                mu=0,
                sigma=25 / 3,
                beta=25 / 3 / 2,
                tau=0.0,
                draw_probability=0.02,
            ),
        ),
    ),
)

gomoku_alphazero_league_config = EasyDict(gomoku_alphazero_league_config)
main_config = gomoku_alphazero_league_config

gomoku_alphazero_league_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        get_train_sample=False,
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
gomoku_alphazero_league_create_config = EasyDict(gomoku_alphazero_league_create_config)
create_config = gomoku_alphazero_league_create_config

if __name__ == "__main__":
    from lzero.entry import train_alphazero_league
    from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
    train_alphazero_league(main_config, GomokuEnv, max_env_step=max_env_step)