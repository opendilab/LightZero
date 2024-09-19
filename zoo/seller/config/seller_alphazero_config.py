from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 4
n_episode = 4
evaluator_env_num = 20
num_simulations = 10
update_per_collect = 50
batch_size = 32
max_env_step = int(1e5)
mcts_ctree = False

# for debug
collector_env_num = 1
n_episode = 1
evaluator_env_num = 2
num_simulations = 1
update_per_collect = 2
batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

seller_alphazero_config = dict(
    exp_name=f'data_alphazero/seller_alphazero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        agent='lmdeploy',
        # agent='deepseek',
        api_key=[
            'sk-f50d634a123f4c84bc08fa880387ff76', 'sk-f8e6d25f99e5434c9ebda6e447fa8a7a',
            'sk-d020afbebe1e4d1ba1db7d32700c068c', 'sk-514a633560104439a4324dc30deab907',
        ],
        commands=[
            '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他',
            '询问用户的具体使用情景', '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
        ],
        max_round=5,
        lang='zh',
        log_suffix='az_a9_qwen2',
        save_replay=False,  # TODO
        dynamic_action_space=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='seller',
        # ==============================================================
        model=dict(
            action_space_size=9, # NOTE
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
        type='alphazero_seller',
        import_names=['lzero.policy.alphazero_seller'],
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
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], model_path=main_config.policy.model_path, seed=0, max_env_step=max_env_step)

