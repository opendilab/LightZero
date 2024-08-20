from easydict import EasyDict

seller_prompt_pg_config = dict(
    exp_name='seller_prompt_pg_seed0',
    env=dict(
        collector_env_num=3,
        evaluator_env_num=6,
        n_evaluator_episode=6,
        stop_value=1,
        agent='deepseek',
        api_key='sk-c4a8fe52693a4aaab64e648c42f40be6',
        commands=[
            '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他', '询问用户的具体使用情景',
            '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
        ],
        max_round=2,
        seed=0,
        lang='zh',
    ),
    policy=dict(
        cuda=True,
        shot_number=1,
        model=dict(
            model_name="bert-base-uncased",
            add_linear=True,
            freeze_encoder=True,
            embedding_size=128,
        ),
        learn=dict(
            batch_size=8,
            # (bool) Whether to normalize advantage. Default to False.
            learning_rate=0.001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            entropy_weight=0.001,
            weight_decay=5e-3,
            grad_norm=0.5,
        ),
        collect=dict(
            # (int) collect n_sample data, train model 1 times
            n_sample=3,
            discount_fact=1.,
        ),
        eval=dict(evaluator=dict(eval_freq=10, )),
    ),
)
main_config = EasyDict(seller_prompt_pg_config)

seller_prompt_pg_config = dict(
    env=dict(
        type='seller',
        import_names=['dizoo.seller.seller_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='prompt_pg'),
    replay_buffer=dict(type='naive'),
)
create_config = EasyDict(seller_prompt_pg_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0, max_env_step=1000)
