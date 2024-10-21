from easydict import EasyDict

dmc2gym_sac_config = dict(
    # exp_name='dmc2gym_cheetah_run_sac_state_seed0',
    exp_name='dmc2gym_walker_walk_sac_state_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        # domain_name="cartpole",
        # task_name="swingup",
        # frame_skip=8,
        # domain_name="cheetah",
        # task_name="run",
        domain_name="walker",
        task_name="walk",
        frame_skip=2,
        frame_stack=1,
        from_pixels=False,  # state obs
        channels_first=False,  # obs shape (height, width, 3)
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=1e6,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_type='state',
        cuda=True,
        random_collect_size=10000,
        model=dict(
            # obs_shape=5,
            # action_shape=1,
            obs_shape=24,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            ignore_done=True,
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

dmc2gym_sac_config = EasyDict(dmc2gym_sac_config)
main_config = dmc2gym_sac_config

dmc2gym_sac_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
dmc2gym_sac_create_config = EasyDict(dmc2gym_sac_create_config)
create_config = dmc2gym_sac_create_config



from ditk import logging
from ding.model import ContinuousQAC
from ding.policy import SACPolicy
from ding.envs import BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import data_pusher, StepCollector, interaction_evaluator, \
    CkptSaver, OffPolicyLearner, termination_checker
from ding.utils import set_pkg_seed
from dizoo.dmc2gym.envs.dmc2gym_env import DMC2GymEnv
# from dizoo.dmc2gym.config.dmc2gym_sac_state_config import main_config, create_config
import numpy as np
from tensorboardX import SummaryWriter
import os


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'dmc2gym_sac_state_nseed_5M'
    main_config.policy.cuda = True
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    num_seed = 4
    for seed_i in range(num_seed):
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'seed' + str(seed_i)))

        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = ContinuousQAC(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = SACPolicy(cfg.policy, model=model)

            def _add_scalar(ctx):
                if ctx.eval_value != -np.inf:
                    tb_logger.add_scalar('evaluator_step/reward', ctx.eval_value, global_step=ctx.env_step)
                    collector_rewards = [ctx.trajectories[i]['reward'] for i in range(len(ctx.trajectories))]
                    collector_mean_reward = sum(collector_rewards) / len(ctx.trajectories)
                    # collector_max_reward = max(collector_rewards)
                    # collector_min_reward = min(collector_rewards)
                    tb_logger.add_scalar('collecter_step/mean_reward', collector_mean_reward, global_step=ctx.env_step)
                    # tb_logger.add_scalar('collecter_step/max_reward', collector_max_reward, global_step= ctx.env_step)
                    # tb_logger.add_scalar('collecter_step/min_reward', collector_min_reward, global_step= ctx.env_step)
                    tb_logger.add_scalar(
                        'collecter_step/avg_env_step_per_episode',
                        ctx.env_step / ctx.env_episode,
                        global_step=ctx.env_step
                    )

            def _add_train_scalar(ctx):
                len_train = len(ctx.train_output)
                cur_lr_q_avg = sum([ctx.train_output[i]['cur_lr_q'] for i in range(len_train)]) / len_train
                cur_lr_p_avg = sum([ctx.train_output[i]['cur_lr_p'] for i in range(len_train)]) / len_train
                critic_loss_avg = sum([ctx.train_output[i]['critic_loss'] for i in range(len_train)]) / len_train
                policy_loss_avg = sum([ctx.train_output[i]['policy_loss'] for i in range(len_train)]) / len_train
                total_loss_avg = sum([ctx.train_output[i]['total_loss'] for i in range(len_train)]) / len_train
                tb_logger.add_scalar('learner_step/cur_lr_q_avg', cur_lr_q_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/cur_lr_p_avg', cur_lr_p_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/critic_loss_avg', critic_loss_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/policy_loss_avg', policy_loss_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/total_loss_avg', total_loss_avg, global_step=ctx.env_step)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(
                StepCollector(
                    cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size
                )
            )
            task.use(_add_scalar)
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(_add_train_scalar)
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=int(1e5)))
            task.use(termination_checker(max_env_step=int(5e6)))
            task.run()


if __name__ == "__main__":
    main()
