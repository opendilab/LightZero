from easydict import EasyDict
import torch.nn as nn


model_name = 'BAAI/bge-base-en-v1.5'
evaluator_env_num = 2

# env_id = 'detective.z5'
action_space_size = 10
max_steps = 50

env_id = 'zork1.z5'
action_space_size = 10
max_steps = 400


# proj train
# collector_env_num = 18
# batch_size = 320
collector_env_num = 4
batch_size = 32

# all train
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# batch_size = 4 
# num_unroll_steps = 5
# infer_context_length = 2
jericho_ppo_config = dict(
    exp_name=f"data_ppo_detective/jericho_{env_id}_ms{max_steps}_ass{action_space_size}_ppo_projtrain_bs{batch_size}_seed0",
    # exp_name=f"data_ppo_detective_debug/jericho_add-loc-inv_ppo_projtrain_bs{batch_size}_seed0",
    env=dict(
        remove_stuck_actions=False,
        # remove_stuck_actions=True,
        # add_location_and_inventory=True,
        add_location_and_inventory=False,
        stop_value=int(1e6),
        observation_shape=512,
        max_steps=max_steps,
        max_action_num=action_space_size,
        tokenizer_path=model_name,
        # tokenizer_path="/mnt/afs/zhangshenghan/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        max_seq_len=512,
        # game_path="z-machine-games-master/jericho-game-suite/" + env_id,
        game_path="/mnt/afs/niuyazhe/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/" + env_id,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='discrete',
        model=dict(
            obs_shape=(26, 5, 4), # 没有起作用
            action_shape=action_space_size,
            action_space='discrete',
            encoder_hidden_size_list = [512], # encoder_hidden_size_list[-1]是head的输入维度
            actor_head_hidden_size= 512,
            critic_head_hidden_size = 512,
        ),
        learn=dict(
            epoch_per_collect=4,
            batch_size=batch_size, 
            learning_rate=0.0005,
            entropy_weight=0.05,
            value_norm=True,
            grad_clip_value=10,
        ),
        collect=dict(
            n_sample=320, # TODO: DEBUG
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=5000, )),
    ),
)
jericho_ppo_config = EasyDict(jericho_ppo_config)
main_config = jericho_ppo_config
cartpole_ppo_create_config = dict(
    env=dict(
        type='jericho',
        import_names=['zoo.jericho.envs.jericho_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
cartpole_ppo_create_config = EasyDict(cartpole_ppo_create_config)
create_config = cartpole_ppo_create_config


if __name__ == "__main__":
    from ding.entry import serial_pipeline_onpolicy
    from ding.model.template import VAC
    m = main_config.policy.model
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from lzero.model.common import HFLanguageRepresentationNetwork
    encoder = HFLanguageRepresentationNetwork(url=model_name, embedding_size=512)

    model = VAC(obs_shape=m.obs_shape, action_shape=m.action_shape, action_space=m.action_space, encoder_hidden_size_list=m.encoder_hidden_size_list,
            actor_head_hidden_size=m.actor_head_hidden_size,
            critic_head_hidden_size =m.critic_head_hidden_size, encoder=encoder)
    serial_pipeline_onpolicy([main_config, create_config], seed=0, model=model)
