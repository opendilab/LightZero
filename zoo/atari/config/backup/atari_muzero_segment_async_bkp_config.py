import argparse
import sys
from pathlib import Path

LIGHTZERO_ROOT = Path(__file__).resolve().parents[3]
if str(LIGHTZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTZERO_ROOT))

from zoo.atari.config.atari_muzero_segment_config import main as segment_main  # noqa: E402


def main(
        env_id='ALE/Pong-v5',
        seed=0,
        exp_root='data_lz_muzero',
        run_tag=None,
        max_env_step_override=None,
        num_collector_actors=2,
        max_policy_lag=0,
        max_train_chunk_steps=4,
        weight_sync_interval=1,
        collector_num_gpus=0,
        evaluator_num_gpus=0,
        smoke_test=False,
):
    """
    Run the Ray-based async MuZero Atari segment pipeline.

    This entry keeps the algorithm and replay buffer configuration in
    ``atari_muzero_segment_config.py`` and only enables async actor scheduling.
    """
    return segment_main(
        env_id,
        seed,
        exp_root=exp_root,
        run_tag=run_tag,
        max_env_step_override=max_env_step_override,
        async_pipeline=True,
        num_collector_actors=num_collector_actors,
        max_policy_lag=max_policy_lag,
        max_train_chunk_steps=max_train_chunk_steps,
        weight_sync_interval=weight_sync_interval,
        collector_num_gpus=collector_num_gpus,
        evaluator_num_gpus=evaluator_num_gpus,
        smoke_test=smoke_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ray async MuZero Atari segment training.')
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/Pong-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument('--exp-root', type=str, help='Experiment root directory', default='data_lz_muzero')
    parser.add_argument('--run-tag', type=str, help='Optional grouped run tag for experiment layout', default=None)
    parser.add_argument('--max-env-step', type=int, help='Override max env steps for smoke/debug runs', default=None)
    parser.add_argument('--num-collector-actors', type=int, default=2, help='Number of Ray collector actors')
    parser.add_argument('--max-policy-lag', type=int, default=0, help='Allowed collector policy version lag')
    parser.add_argument('--max-train-chunk-steps', type=int, default=4, help='Max learner updates before yielding to async tasks')
    parser.add_argument('--weight-sync-interval', type=int, default=1, help='Learner steps between collect/eval weight publishes')
    parser.add_argument('--collector-num-gpus', type=float, default=0, help='Ray GPU resource per collector actor')
    parser.add_argument('--evaluator-num-gpus', type=float, default=0, help='Ray GPU resource per evaluator actor')
    parser.add_argument('--smoke-test', action='store_true', help='Use a tiny config for startup/rjob smoke validation')
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        exp_root=args.exp_root,
        run_tag=args.run_tag,
        max_env_step_override=args.max_env_step,
        num_collector_actors=args.num_collector_actors,
        max_policy_lag=args.max_policy_lag,
        max_train_chunk_steps=args.max_train_chunk_steps,
        weight_sync_interval=args.weight_sync_interval,
        collector_num_gpus=args.collector_num_gpus,
        evaluator_num_gpus=args.evaluator_num_gpus,
        smoke_test=args.smoke_test,
    )
