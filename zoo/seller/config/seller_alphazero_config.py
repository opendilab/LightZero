from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 10
update_per_collect = 20
batch_size = 20
max_env_step = int(1e5)
mcts_ctree = False

# for debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 1
# update_per_collect = 2
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

seller_alphazero_config = dict(
    exp_name=f'data_az_ptree/seller_alphazero_ns{num_simulations}_upc{update_per_collect}_goods-train10test20_persona10_seed0',
    # exp_name=f'data_az_ptree_debug/seller_alphazero_ns{num_simulations}_upc{update_per_collect}_adam_fixreturn_seed0',
    env=dict(
        agent='deepseek',
        # api_key='sk-7866ab6ea8ca408a91971ef18eed4b75',
        api_key='sk-c4a8fe52693a4aaab64e648c42f40be6',
        commands=[
            '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他',
            '询问用户的具体使用情景', '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
        ],
        max_round=5,

        # debug
        # max_round=2, 
        # commands=[
        #     '将你的产品推销给用户'
        # ],
        lang='zh',
        log_suffix='az_a9_goods0_persona5',

        # save_replay=True,  # TODO

        battle_mode='play_with_bot_mode',
        battle_mode_in_simulation_env='play_with_bot_mode',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_az_ptree/seller_alphazero_ns10_upc50_adam_fixreturn_seed0_240902_123328/ckpt/ckpt_best.pth.tar',
        model_path=None,
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='seller',
        simulation_env_config_type='play_with_bot',
        # ==============================================================
        model=dict(
            action_space_size=9, # debug
            # action_space_size=1, # debug
        ),
        cuda=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        grad_clip_value=5,
        value_weight=1.0,
        entropy_weight=1e-4,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

seller_alphazero_config = EasyDict(seller_alphazero_config)
main_config = seller_alphazero_config

seller_alphazero_create_config = dict(
    env=dict(
        type='seller',
        import_names=['zoo.seller.seller_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
seller_alphazero_create_config = EasyDict(seller_alphazero_create_config)
create_config = seller_alphazero_create_config

if __name__ == '__main__':
    # from lzero.entry import train_alphazero
    # train_alphazero([main_config, create_config], model_path=main_config.policy.model_path, seed=0, max_env_step=max_env_step)


    from lzero.entry import train_alphazero
    def run(max_env_step: int):
        train_alphazero([main_config, create_config], model_path=main_config.policy.model_path, seed=0, max_env_step=max_env_step)
    import cProfile
    cProfile.run(f"run({100000})", filename="seller_az_cprofile_100k_envstep", sort="cumulative")
