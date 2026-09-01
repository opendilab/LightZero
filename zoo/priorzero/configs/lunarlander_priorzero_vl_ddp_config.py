"""Distributed visual PriorZero training configuration for LunarLander.

Launch from the LightZero repository root::

    CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
    torchrun --nproc_per_node=2 --master-port=29501 \
    ./zoo/priorzero/configs/lunarlander_priorzero_vl_ddp_config.py

Experiment outputs are grouped under ``data_priorzero/vl_rft`` or
``data_priorzero/vl_frozen`` according to whether the VL model is trained.
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

from zoo.priorzero.src.vl_config import get_available_vl_models, get_priorzero_vl_config  # noqa: E402


# -----------------------------------------------------------------------------
# Frequently modified experiment parameters
# Keep launcher-only settings (GPU IDs, process count and port) in torchrun.
# -----------------------------------------------------------------------------
WORLD_MODEL_CONFIG = dict(
    collector_env_num=4,
    evaluator_env_num=3,
    batch_size=256,
    replay_ratio=0.25,
    num_unroll_steps=10,
    infer_context_length=4,
    game_segment_length=200,
    num_layers=2,
    num_heads=4,
    embed_dim=256,
    num_simulations=25,
    collect_num_simulations=50,
    eval_num_simulations=25,
    eval_freq=20000,
    learning_rate=1e-4,
    grad_clip_value=5,
)

VL_PRIOR_CONFIG = dict(
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
        wm_update_iters=1000,
        llm_update_iters=100,
        start_phase='wm',
        wm_warmup_updates=0,
    ),
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
        wm_eval_freq=1000,
        llm_eval_freq=100,
        eval_freq=20000,
    ),
    user_prompt_dict=dict(
        history_with_reward=True,
        observation_with_valid_actions=False,
    ),
    train_batch_size=128,
    micro_train_batch_size=4,
    learning_rate=1e-6,
    history_length=3,
    llm_prior_temperature=2.0,
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


def _apply_experiment_config(main_config, vl_config) -> None:
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
    policy_config.learning_rate = wm['learning_rate']
    policy_config.grad_clip_value = wm['grad_clip_value']

    world_model_config.max_blocks = wm['num_unroll_steps']
    world_model_config.max_tokens = 2 * wm['num_unroll_steps']
    world_model_config.context_length = 2 * wm['infer_context_length']
    world_model_config.game_segment_length = wm['game_segment_length']
    world_model_config.num_layers = wm['num_layers']
    world_model_config.num_heads = wm['num_heads']
    world_model_config.embed_dim = wm['embed_dim']
    world_model_config.num_simulations = wm['num_simulations']
    world_model_config.env_num = max(wm['collector_env_num'], wm['evaluator_env_num'])

    prior = VL_PRIOR_CONFIG
    vl_config.train_mode_dict = EasyDict(prior['train_mode_dict'])
    vl_config.train_schedule = EasyDict(prior['train_schedule'])
    vl_config.mcts_root_logits_dict = EasyDict(prior['mcts_root_logits_dict'])
    vl_config.eval_dict = EasyDict(prior['eval_dict'])
    vl_config.user_prompt_dict = EasyDict(prior['user_prompt_dict'])
    vl_config.train_batch_size = prior['train_batch_size']
    vl_config.micro_train_batch_size = prior['micro_train_batch_size']
    vl_config.learning_rate = prior['learning_rate']
    vl_config.history_length = prior['history_length']
    vl_config.llm_prior_temperature = prior['llm_prior_temperature']
    vl_config.reward_func = EasyDict(prior['reward_func'])
    vl_config.value_norm_cfg = EasyDict(prior['value_norm_cfg'])


def _set_experiment_name(main_config, vl_config, vl_model: str) -> None:
    """Build the experiment path after all command-line overrides."""
    env_name = main_config.env.env_id
    if vl_config.enable_rft:
        main_config.exp_name = (
            f'data_priorzero/vl_rft/'
            f'priorzero_{env_name}_{vl_model}_train_{vl_config.train_mode_dict.mode}/'
            f'useCot_{vl_config.use_cot}_alternate_{vl_config.train_schedule.alternate}/'
            f'mcts_{vl_config.mcts_root_logits_dict.mode}_image_{vl_config.vlm_image_mode}_'
            f'staleness_{vl_config.max_rollout_staleness}_tbs_{vl_config.train_batch_size}_'
            f'use_mispo_{vl_config.use_mispo}_seed{main_config.seed}'
        )
    else:
        main_config.exp_name = (
            f'data_priorzero/vl_frozen/'
            f'priorzero_{env_name}_{vl_model}_train_{vl_config.train_mode_dict.mode}/'
            f'useCot_{vl_config.use_cot}_mcts_{vl_config.mcts_root_logits_dict.mode}_'
            f'image_{vl_config.vlm_image_mode}_seed{main_config.seed}'
        )


def main(
    env_id: str = 'LunarLander-v2',
    seed: int = 0,
    max_train_iter: int = int(1e6),
    vl_model: str = 'Qwen2.5-VL-3b',
    use_cot: bool = True,
    cot_weight: float = 0.1,
    mcts_mode: str = VL_PRIOR_CONFIG['mcts_root_logits_dict']['mode'],
    vlm_image_mode: str = 'current_only',
    prompt_style: str = 'legacy',
    logprob_mode: str = 'approximate',
    vl_fixed: bool = True,
    quick_test: bool = False,
    enable_profile: bool = False,
) -> None:
    """Build the LunarLander visual PriorZero configuration and launch training."""
    from zoo.priorzero.src.priorzero_entry_sync_ddp import train_priorzero

    main_config, create_config, vl_config = get_priorzero_vl_config(
        env_id=env_id,
        seed=seed,
        vl_model_key=vl_model,
        use_prior=True,
        multi_gpu=int(os.environ.get('WORLD_SIZE', '1')) > 1,
        quick_test=quick_test,
    )
    if not quick_test:
        _apply_experiment_config(main_config, vl_config)
    vl_config.use_cot = use_cot
    vl_config.cot_weight = cot_weight
    vl_config.mcts_root_logits_dict.mode = mcts_mode
    vl_config.vlm_image_mode = vlm_image_mode
    vl_config.prompt_style = prompt_style
    vl_config.logprob_extraction_mode = logprob_mode
    vl_config.vl_fixed = vl_fixed
    if vl_fixed:
        vl_config.enable_rft = False
    if not quick_test:
        _set_experiment_name(main_config, vl_config, vl_model)

    train_priorzero(
        main_config,
        create_config,
        vl_config,
        seed=seed,
        max_train_iter=max_train_iter,
        enable_profile=enable_profile,
    )


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser(description='Launch visual PriorZero training on LunarLander.')
    parser.add_argument('--env', '--env_id', dest='env_id', default='LunarLander-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=int(1e6))
    parser.add_argument('--vl_model', choices=get_available_vl_models(), default='Qwen2.5-VL-3b')
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
        default=VL_PRIOR_CONFIG['mcts_root_logits_dict']['mode'],
    )
    parser.add_argument(
        '--vlm_image_mode',
        choices=['current_only', 'first_and_current', 'all_history'],
        default='current_only',
    )
    parser.add_argument('--prompt_style', choices=['concise', 'legacy'], default='legacy')
    parser.add_argument('--logprob_mode', choices=['exact', 'approximate'], default='approximate')
    vl_fixed_group = parser.add_mutually_exclusive_group()
    vl_fixed_group.add_argument('--vl_fixed', dest='vl_fixed', action='store_true')
    vl_fixed_group.add_argument('--no_vl_fixed', dest='vl_fixed', action='store_false')
    parser.set_defaults(vl_fixed=True)
    args = parser.parse_args()
    main(
        env_id=args.env_id,
        seed=args.seed,
        max_train_iter=args.max_iter,
        vl_model=args.vl_model,
        use_cot=args.use_cot,
        cot_weight=args.cot_weight,
        mcts_mode=args.mcts_mode,
        vlm_image_mode=args.vlm_image_mode,
        prompt_style=args.prompt_style,
        logprob_mode=args.logprob_mode,
        vl_fixed=args.vl_fixed,
        quick_test=args.quick_test,
        enable_profile=args.enable_profile,
    )
