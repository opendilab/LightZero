from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import torch.distributed as dist


env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
gpu_num = 8
update_per_collect = 1000 # Very import for ddp seting
replay_ratio = 0.25
collector_env_num = 8
num_segments = int(8*gpu_num)
n_episode = int(8*gpu_num)
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(2e5)
batch_size = 64
num_unroll_steps = 10
infer_context_length = 4
seed = 0

# ====== only for debug =====
# collector_env_num = 2
# num_segments = int(2*gpu_num)
# n_episode = int(2*gpu_num)
# num_simulations = 3
# max_env_step = int(2e5)
# batch_size = 2
# num_unroll_steps = 10
# replay_ratio = 0.005
# update_per_collect = 8

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_unizero_config = dict(
    exp_name = f'data_unizero_ddp_1110/{env_id[:-14]}/{env_id[:-14]}_stack1_unizero_ddp_{gpu_num}gpu_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: only for debug
        # collect_max_episode_steps=int(200),
        # eval_max_episode_steps=int(200),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 64, 64),
            action_space_size=action_space_size,
            world_model_cfg=dict(
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                # device='cpu',
                action_space_size=action_space_size,
                num_layers=2,
                num_heads=8,
                embed_dim=768,
                obs_type='image',
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        multi_gpu=True,
        num_unroll_steps=num_unroll_steps,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        num_simulations=num_simulations,
        num_segments=num_segments,
        n_episode=n_episode,
        replay_buffer_size=int(1e6),
        eval_freq=int(5e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_unizero_config = EasyDict(atari_unizero_config)
main_config = atari_unizero_config

atari_unizero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
atari_unizero_create_config = EasyDict(atari_unizero_create_config)
create_config = atari_unizero_create_config

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./zoo/atari/config/atari_unizero_ddp_config.py
        torchrun --nproc_per_node=2 ./zoo/atari/config/atari_unizero_ddp_config.py

    """
    from ding.utils import DDPContext
    from lzero.entry import train_unizero
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        # 确保每个 Rank 分配到正确的 collector_env_num
        print(f"Rank {dist.get_rank()} Collector Env Num: {main_config.policy.collector_env_num}")
        # TODO: first test muzero_collector
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)
