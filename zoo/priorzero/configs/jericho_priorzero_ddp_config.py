"""Distributed PriorZero training configuration for Jericho.

Launch from the LightZero repository root::

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONFAULTHANDLER=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
    torchrun --nproc_per_node=4 --master-port=24554 \
    ./zoo/priorzero/configs/jericho_priorzero_ddp_config.py
"""

import argparse
import os
import sys
from pathlib import Path

from easydict import EasyDict


os.environ.setdefault('PYTHONFAULTHANDLER', '1')
os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
os.environ.setdefault('NCCL_DEBUG', 'INFO')

LIGHTZERO_ROOT = Path(__file__).resolve().parents[3]
PRIORZERO_SRC = LIGHTZERO_ROOT / 'zoo' / 'priorzero' / 'src'
sys.path.insert(0, str(LIGHTZERO_ROOT))
sys.path.insert(0, str(PRIORZERO_SRC))

from zoo.priorzero.src.priorzero_config import (  # noqa: E402
    get_available_models,
    get_priorzero_config,
    get_priorzero_debug_config,
)


# -----------------------------------------------------------------------------
# Frequently modified experiment parameters
# Defaults reproduce all_experiments/data_priorzero_paper/llm_train.
# Keep launcher-only settings (GPU IDs, process count and port) in torchrun.
# -----------------------------------------------------------------------------
WORLD_MODEL_CONFIG = dict(
    collector_env_num=1,
    evaluator_env_num=2,
    batch_size=64,
    replay_ratio=0.1,
    num_unroll_steps=10,
    infer_context_length=4,
    game_segment_length=50,
    num_layers=2,
    num_heads=24,
    embed_dim=768,
    num_simulations=50,
    collect_num_simulations=25,
    eval_num_simulations=25,
    eval_freq=30000,
    replay_buffer_size=300000,
    learning_rate=3e-4,
    weight_decay=1e-4,
    grad_clip_value=10.0,
)

LLM_PRIOR_CONFIG = dict(
    enable_rft=True,
    train_mode_dict=dict(
        mode='full',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_bias='none',
        lora_target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
    ),
    train_schedule=dict(
        alternate=True,
        wm_update_iters=2000,
        llm_update_iters=200,
        start_phase='wm',
        wm_warmup_updates=0,
        # The paper run predates this field and collected during the LLM phase.
        llm_collect_mode='wm_llm_collect',
    ),
    llm_prior_temperature=2.0,
    mcts_root_logits_dict=dict(
        mode='llm_plus_wm_logits',
        plus_method='fixed',
        wm_weight=0.5,
        llm_max_weight=0.7,
        llm_min_weight=0.3,
        max_envsteps=1e5,
    ),
    eval_dict=dict(
        world_model=True,
        world_model_llm_prior=True,
        llm_prior=True,
        wm_eval_freq=500,
        llm_eval_freq=50,
    ),
    user_prompt_dict=dict(
        history_with_reward=True,
        observation_with_valid_actions=False,
    ),
    train_batch_size=128,
    micro_train_batch_size=4,
    learning_rate=1e-6,
    history_length=10,
    gradient_checkpointing=False,
    max_rollout_staleness=1,
    reward_func=dict(
        format_reward=True,
        format_param=dict(format_weight=0.5),
    ),
    value_norm_cfg=dict(
        enable_stability_optimizer=True,
        value_norm_init_momentum=0.9,
        value_norm_final_momentum=0.99,
        value_norm_warmup_steps=100,
        value_norm_clip_percentile=0.95,
        value_norm_clip_method='soft',
        value_norm_history_size=1000,
    ),
)


def _apply_experiment_config(main_config, llm_config, model: str) -> None:
    """Apply the editable parameters above to all dependent config fields."""
    wm = WORLD_MODEL_CONFIG
    env_config = main_config.env
    policy_config = main_config.policy
    world_model_config = policy_config.model.world_model_cfg

    env_config.collector_env_num = wm['collector_env_num']
    env_config.evaluator_env_num = wm['evaluator_env_num']
    env_config.n_evaluator_episode = wm['evaluator_env_num']

    policy_config.collector_env_num = wm['collector_env_num']
    policy_config.evaluator_env_num = wm['evaluator_env_num']
    policy_config.n_episode = wm['collector_env_num']
    policy_config.num_segments = wm['collector_env_num']
    policy_config.batch_size = wm['batch_size']
    policy_config.replay_ratio = wm['replay_ratio']
    policy_config.num_unroll_steps = wm['num_unroll_steps']
    policy_config.game_segment_length = wm['game_segment_length']
    policy_config.num_simulations = wm['num_simulations']
    policy_config.collect_num_simulations = wm['collect_num_simulations']
    policy_config.eval_num_simulations = wm['eval_num_simulations']
    policy_config.eval_freq = wm['eval_freq']
    policy_config.replay_buffer_size = wm['replay_buffer_size']
    policy_config.learning_rate = wm['learning_rate']
    policy_config.weight_decay = wm['weight_decay']
    policy_config.grad_clip_value = wm['grad_clip_value']

    world_model_config.max_blocks = wm['num_unroll_steps']
    world_model_config.max_tokens = 2 * wm['num_unroll_steps']
    world_model_config.context_length = 2 * wm['infer_context_length']
    world_model_config.game_segment_length = wm['game_segment_length']
    world_model_config.num_layers = wm['num_layers']
    world_model_config.num_heads = wm['num_heads']
    world_model_config.embed_dim = wm['embed_dim']
    world_model_config.env_num = max(wm['collector_env_num'], wm['evaluator_env_num'])

    prior = LLM_PRIOR_CONFIG
    llm_config.enable_rft = prior['enable_rft']
    llm_config.train_mode_dict = EasyDict(prior['train_mode_dict'])
    llm_config.train_schedule = EasyDict(prior['train_schedule'])
    llm_config.mcts_root_logits_dict = EasyDict(prior['mcts_root_logits_dict'])
    llm_config.eval_dict = EasyDict(prior['eval_dict'])
    llm_config.user_prompt_dict = EasyDict(prior['user_prompt_dict'])
    llm_config.train_batch_size = prior['train_batch_size']
    llm_config.micro_train_batch_size = prior['micro_train_batch_size']
    llm_config.learning_rate = prior['learning_rate']
    llm_config.llm_prior_temperature = prior['llm_prior_temperature']
    llm_config.history_length = prior['history_length']
    llm_config.gradient_checkpointing = prior['gradient_checkpointing']
    llm_config.max_rollout_staleness = prior['max_rollout_staleness']
    llm_config.reward_func = EasyDict(prior['reward_func'])
    llm_config.value_norm_cfg = EasyDict(prior['value_norm_cfg'])


def _set_experiment_name(main_config, llm_config, model: str) -> None:
    """Build the experiment path after all command-line overrides."""
    env_config = main_config.env
    env_name = env_config.env_id.replace('.z5', '')
    if llm_config.enable_rft:
        main_config.exp_name = (
            f'all_experiments/data_priorzero_latest/llm_rft/'
            f'priorzero_{env_name}_{model}_train_{llm_config.train_mode_dict.mode}/'
            f'useCot_{llm_config.use_cot}_alternate_{llm_config.train_schedule.alternate}/'
            f'mcts_{llm_config.mcts_root_logits_dict.mode}_staleness_{llm_config.max_rollout_staleness}_'
            f'tbs_{llm_config.train_batch_size}_use_mispo_{llm_config.use_mispo}'
        )
    else:
        main_config.exp_name = (
            f'all_experiments/data_priorzero_latest/llm_frozen/'
            f'priorzero_{env_name}_{model}_train_{llm_config.train_mode_dict.mode}'
            f'useCot_{llm_config.use_cot}_seed{main_config.seed}'
        )


def main(
    env_id: str = 'detective.z5',
    seed: int = 0,
    max_train_iter: int = int(1e6),
    model: str = 'qwen2.5-3b',
    use_cot: bool = True,
    cot_weight: float = 0.1,
    mcts_mode: str = LLM_PRIOR_CONFIG['mcts_root_logits_dict']['mode'],
    quick_test: bool = False,
    enable_profile: bool = False,
) -> None:
    """Build the Jericho PriorZero DDP configuration and launch training."""
    from zoo.priorzero.src.priorzero_entry_sync_ddp import train_priorzero

    if quick_test:
        main_config, create_config, llm_config = get_priorzero_debug_config(
            env_id=env_id,
            seed=seed,
            exp_name=f'all_experiments/data_priorzero/priorzero_debug_{env_id}',
            use_cot=use_cot,
            model_key=model,
        )
        main_config.policy.multi_gpu = True
    else:
        main_config, create_config, llm_config = get_priorzero_config(
            env_id=env_id,
            seed=seed,
            use_cot=use_cot,
            model_key=model,
            multi_gpu=True,
        )

    if not quick_test:
        _apply_experiment_config(main_config, llm_config, model)
    llm_config.cot_weight = cot_weight
    llm_config.mcts_root_logits_dict.mode = mcts_mode
    if not quick_test:
        _set_experiment_name(main_config, llm_config, model)

    train_priorzero(
        main_config,
        create_config,
        llm_config,
        seed=seed,
        max_train_iter=max_train_iter,
        enable_profile=enable_profile,
    )


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser(description='Launch distributed PriorZero training on Jericho.')
    parser.add_argument('--env', '--env_id', dest='env_id', default='detective.z5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=int(1e6))
    parser.add_argument('--model', choices=get_available_models(), default='qwen2.5-3b')
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--enable_profile', action='store_true')
    cot_group = parser.add_mutually_exclusive_group()
    cot_group.add_argument('--use_cot', dest='use_cot', action='store_true')
    cot_group.add_argument('--no_cot', dest='use_cot', action='store_false')
    parser.set_defaults(use_cot=True)
    parser.add_argument('--cot_weight', type=float, default=0.1)
    parser.add_argument(
        '--mcts_mode',
        choices=['llm_logits', 'wm_logits', 'llm_plus_wm_logits'],
        default=LLM_PRIOR_CONFIG['mcts_root_logits_dict']['mode'],
    )
    args = parser.parse_args()
    main(
        env_id=args.env_id,
        seed=args.seed,
        max_train_iter=args.max_iter,
        model=args.model,
        use_cot=args.use_cot,
        cot_weight=args.cot_weight,
        mcts_mode=args.mcts_mode,
        quick_test=args.quick_test,
        enable_profile=args.enable_profile,
    )
