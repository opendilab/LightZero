from easydict import EasyDict
import torch
torch.cuda.set_device(0)
# torch.cuda.empty_cache()
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================


collector_env_num = 8
n_episode = 8
evaluator_env_num = 1
num_simulations = 25
# update_per_collect = 200
update_per_collect = None
model_update_ratio = 0.5
max_env_step = int(2e5)

reanalyze_ratio = 0
# num_unroll_steps = 20

# batch_size = 32
# num_unroll_steps = 5

batch_size = 32
num_unroll_steps = 5



# # debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# model_update_ratio = 1
# batch_size = 2
# max_env_step = int(1e5)
# reanalyze_ratio = 0
# num_unroll_steps = 5

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_muzero_gpt_config = dict(
    # TODO: world_model.py decode_obs_tokens
    # TODO: tokenizer: lpips loss
    exp_name=f'data_mz_gpt_ctree_1226/cartpole_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_nlayers2_emd64_smallnet_bs{batch_size}_mcs500_bs{batch_size}_contembdings_ez-ssl-loss-k1_lsd256_fixmask_seed0',

    # exp_name=f'data_mz_gpt_ctree/cartpole_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_nlayers2_emd64_smallnet_bs{batch_size}_mcs50_batch8_obs-token-lw2_recons-obs_bs{batch_size}_indep0_trans-wd0.01_pt2_argmaxtoken_orig-sdpa_onestep_seed0',
    # exp_name=f'data_mz_gpt_ctree_debug/cartpole_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_nlayers2_emd64_smallnet_bs{batch_size}_mcs500_batch8_obs-token-lw2_recons-obs_bs{batch_size}_indep0_trans-wd0.01_pt2_argmaxtokenp_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_mz_gpt_ctree/cartpole_muzero_gpt_ns25_upc200-mur1_rr0_H5_nlayers2_emd64_smallnet_bs64_mcs50_batch8_obs-token-lw2_recons-obs_bs64_indep0_trans-wd0.01_pt2_argmaxtoken_pt2sdpa-drop0ineval_seed0/ckpt/ckpt_best.pth.tar',
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        # transformer_start_after_envsteps=int(5e3),
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_mz_gpt_ctree/cartpole_muzero_gpt_ns25_upc20-mur1_rr0_H5_nlayers2_emd128_mediumnet_bs64_mcs25_fixedtokenizer_fixloss_fixlatent_seed0/ckpt/ckpt_best.pth.tar',
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=4,
            action_space_size=2,
            model_type='mlp',
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
        ),
        cuda=True,
        # cuda=False,
        env_type='not_board_games',
        game_segment_length=50,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cartpole_muzero_gpt_config = EasyDict(cartpole_muzero_gpt_config)
main_config = cartpole_muzero_gpt_config

cartpole_muzero_gpt_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_gpt',
        import_names=['lzero.policy.muzero_gpt'],
    ),
)
cartpole_muzero_gpt_create_config = EasyDict(cartpole_muzero_gpt_create_config)
create_config = cartpole_muzero_gpt_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_gpt
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

    # 下面为cprofile的代码
    # from lzero.entry import train_muzero_gpt
    # def run(max_env_step: int):
    #     train_muzero_gpt([main_config, create_config], seed=0, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({10000})", filename="cartpole_muzero_gpt_ctree_cprofile_10k_envstep", sort="cumulative")