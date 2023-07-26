from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# board_size = 6
board_size = 9

if board_size in [9, 19]:
    komi = 7.5
elif board_size == 6:
    komi = 4

if board_size == 19:
    num_simulations = 800
elif board_size == 9:
    num_simulations = 180
    # num_simulations = 50
elif board_size == 6:
    # num_simulations = 80
    num_simulations = 50

collector_env_num = 8
n_episode = 8
evaluator_env_num = 1
update_per_collect = 200
batch_size = 256
max_env_step = int(100e6)
snapshot_the_player_in_iter_zero = True
one_phase_step = int(5e3)
# TODO(pu)
sp_prob = 0.5  # 0, 0.5, 1
use_bot_init_historical = False

# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# update_per_collect = 2
# batch_size = 2
# max_env_step = int(2e5)
# sp_prob = 0.
# snapshot_the_player_in_iter_zero = True
# one_phase_step = int(5)
# num_simulations = 2

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

go_alphazero_league_config = dict(
    exp_name=f"data_az_ptree_league/go_b{board_size}-komi-{komi}_alphazero_ns{num_simulations}_upc{update_per_collect}_league-sp-{sp_prob}_bot-init-{use_bot_init_historical}_phase-step-{one_phase_step}_seed0",
    env=dict(
        stop_value=2,
        env_name="Go",
        board_size=board_size,
        komi=7.5,
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        scale=True,
        agent_vs_human=False,
        use_katago_bot=True,
        # katago_checkpoint_path="/Users/puyuan/code/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
        katago_checkpoint_path="/mnt/nfs/puyuan/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
        ignore_pass_if_have_other_legal_actions=True,
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        prob_random_action_in_bot=0,
        channel_last=True,
        check_action_to_connect4_in_bot_v0=False,
        save_gif_replay=False,
        save_gif_path='./',
        render_in_ui=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        prob_random_agent=0,
        prob_expert_agent=0,
    ),
    policy=dict(
        model=dict(
            observation_shape=(board_size, board_size, 17),
            action_space_size=int(board_size * board_size + 1),
            num_res_blocks=1,
            num_channels=64,
        ),
        mcts_ctree=False,
        # mcts_ctree=True,
        cuda=True,
        env_type='board_games',
        board_size=board_size,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=500,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        # eval_freq=int(100),  # debug
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        league=dict(
            log_freq_for_payoff_rank=50,
            player_category=['go'],
            # path to save policy of league player, user can specify this field
            path_policy=f"data_az_ptree_league/go_alphazero_league_sp-{sp_prob}_bot-init-{use_bot_init_historical}_phase-step-{one_phase_step}_ns{num_simulations}_policy_ckpt_seed0",
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
            # "use_bot_init_historical" means whether to use bot as an init historical player
            use_bot_init_historical=use_bot_init_historical,
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

go_alphazero_league_config = EasyDict(go_alphazero_league_config)
main_config = go_alphazero_league_config

go_alphazero_league_create_config = dict(
    env=dict(
        type='go_lightzero',
        import_names=['zoo.board_games.go.envs.go_env'],
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
go_alphazero_league_create_config = EasyDict(go_alphazero_league_create_config)
create_config = go_alphazero_league_create_config

if __name__ == "__main__":
    from lzero.entry import train_alphazero_league
    from zoo.board_games.go.envs.go_env import GoEnv
    train_alphazero_league(main_config, GoEnv, max_env_step=max_env_step)