"""Search defaults for fresh single-task Atari MCTS UniZero runs.

Pong segment runs support this choice; other games and training pipelines still
need performance validation. PPO, multitask search and checkpoint diagnostics
must select their own inference protocol.
"""


def full_context_world_model_config(num_unroll_steps):
    """Match inference capacity to training; a state prediction uses 2H-1 tokens."""
    return dict(
        context_length=2 * num_unroll_steps,
        rebuild_kv_window_from_tokens=True,
        mcts_cache_by_node=True,
        replay_context_use_prediction_window=True,
    )
