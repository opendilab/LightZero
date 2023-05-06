from typing import Callable, TYPE_CHECKING
from easydict import EasyDict
from ding.utils import one_time_warning

if TYPE_CHECKING:
    from ding.policy import Policy
    from lzero.mcts import GameBuffer


def data_pusher(replay_buffer: 'GameBuffer') -> Callable:

    def _push(ctx):
        if ctx.trajectories is not None:  # collector will skip when not reach update_per_collect
            new_data = ctx.trajectories
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

    return _push


def data_reanalyze_fetcher(cfg: EasyDict, policy: 'Policy', replay_buffer: 'GameBuffer') -> Callable:
    B = cfg.policy.batch_size

    def _fetch(ctx):
        if replay_buffer.get_num_of_transitions() > B:
            ctx.train_data = replay_buffer.sample(B, policy)

            yield

            if cfg.policy.use_priority:
                replay_buffer.update_priority(ctx.train_data, ctx.train_output['td_error_priority'])
        else:
            one_time_warning(
                f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                f'batch_size: {B}, '
                f'{replay_buffer} '
                f'continue to collect now ....'
            )

    return _fetch
