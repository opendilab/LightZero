from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
# batch_size = 256
batch_size = 5
max_env_step = int(1e5)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_efficientzero_config = dict(
    exp_name='data_ez_ctree/cartpole_efficientzero_task_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,
            action_space_size=2,
            model_type='mlp',  # options={'mlp', 'conv'}
            lstm_hidden_size=128,
            latent_state_dim=128,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cartpole_efficientzero_config = EasyDict(cartpole_efficientzero_config)
main_config = cartpole_efficientzero_config

cartpole_efficientzero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
cartpole_efficientzero_create_config = EasyDict(cartpole_efficientzero_create_config)
create_config = cartpole_efficientzero_create_config

from functools import partial
from ditk import logging
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import ContextExchanger, ModelExchanger, CkptSaver, trainer, \
    termination_checker, online_logger
from ding.utils import set_pkg_seed
from lzero.policy import EfficientZeroPolicy
from lzero.mcts import EfficientZeroGameBuffer
from lzero.middleware import MuZeroEvaluator, MuZeroCollector, temperature_handler, data_reanalyze_fetcher, \
    lr_scheduler, data_pusher

logging.getLogger().setLevel(logging.INFO)
main_config.policy.device = 'cpu'  # ['cpu', 'cuda']
cfg = compile_config(main_config, create_cfg=create_config, auto=True, save_cfg=task.router.node_id == 0)
ding_init(cfg)

env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
policy = EfficientZeroPolicy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
replay_buffer = EfficientZeroGameBuffer(cfg.policy)


def main():

    with task.start(ctx=OnlineRLContext()):

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            elif task.router.node_id == 2:
                task.add_role(task.role.REANALYZER)
            else:
                task.add_role(task.role.COLLECTOR)

            # Sync their context and model between each worker.
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(policy._model))

        # import os
        # print(f"os.getpid():{os.getpid()}")

        # Here is the part of single process pipeline.
        task.use(MuZeroEvaluator(cfg, policy.eval_mode, evaluator_env, eval_freq=100))
        task.use(temperature_handler(cfg, collector_env))
        task.use(MuZeroCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(replay_buffer))
        task.use(data_reanalyze_fetcher(cfg, policy, replay_buffer))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(lr_scheduler(cfg, policy))
        task.use(online_logger(train_show_freq=10))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=int(1e4)))
        task.use(termination_checker(max_env_step=int(max_env_step)))
        task.run()


if __name__ == "__main__":
    from ding.framework import Parallel
    Parallel.runner(n_parallel_workers=4, startup_interval=0.1)(main)