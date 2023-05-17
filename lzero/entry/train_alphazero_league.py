import copy
import logging
import os
import shutil
from functools import partial
from typing import Optional

import torch
from ding.config import compile_config
from ding.envs import SyncSubprocessEnvManager
from ding.league import BaseLeague, ActivePlayer
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, create_buffer
from ding.worker import NaiveReplayBuffer
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.policy.alphazero import AlphaZeroPolicy
from lzero.worker import AlphaZeroEvaluator
from lzero.worker import BattleAlphaZeroCollector
from lzero.policy import visit_count_temperature


def win_loss_draw(episode_info):
    """
    Overview:
        Get win/loss/draw result from episode info
    Arguments:
        - episode_info (:obj:`list`): List of episode info
    Returns:
        - win_loss_result (:obj:`list`): List of win/loss/draw result
    Examples:
        >>> episode_info = [{'eval_episode_return': 1}, {'eval_episode_return': 0}, {'eval_episode_return': -1}]
        >>> win_loss_draw(episode_info)
        ['wins', 'draws', 'losses']
    """
    win_loss_result = []
    for e in episode_info:
        if e['eval_episode_return'] == 1:
            result = 'wins'
        elif e['eval_episode_return'] == 0:
            result = 'draws'
        else:
            result = 'losses'
        win_loss_result.append(result)

    return win_loss_result


class AlphaZeroLeague(BaseLeague):
    # override
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job(eval_flag))
        return {
            'agent_num': 2,
            # home player_id
            'launch_player': player.player_id,
            # include home and away player_id
            'player_id': [player.player_id, player_job_info.opponent.player_id],
            'checkpoint_path': [player.checkpoint_path, player_job_info.opponent.checkpoint_path],
            'player_active_flag': [isinstance(p, ActivePlayer) for p in [player, player_job_info.opponent]],
        }

    # override
    def _mutate_player(self, player: ActivePlayer):
        # no mutate operation
        pass

    # override
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        assert isinstance(player, ActivePlayer)
        if 'learner_step' in player_info:
            player.total_agent_step = player_info['learner_step']
        # torch.save(player_info['state_dict'], player.checkpoint_path)

    # override
    @staticmethod
    def save_checkpoint(src_checkpoint_path: str, dst_checkpoint_path: str) -> None:
        shutil.copy(src_checkpoint_path, dst_checkpoint_path)


def train_alphazero_league(cfg, Env, seed=0, max_train_iter: Optional[int] = int(1e10), max_env_step: Optional[int] = int(1e10)) -> None:
    """
    Overview:
        Train alphazero league
    Arguments:
        - cfg (:obj:`EasyDict`): Config dict
        - Env (:obj:`BaseEnv`): Env class
        - seed (:obj:`int`): Random seed
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - None
    """
    # prepare config
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        AlphaZeroPolicy,
        BaseLearner,
        BattleAlphaZeroCollector,
        AlphaZeroEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.battle_mode = 'eval_mode'
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # TODO(pu): use different replay buffer for different players
    # create replay buffer
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)

    # create league
    league = AlphaZeroLeague(cfg.policy.league)
    policies, learners, collectors = {}, {}, {}

    # create players
    for player_id in league.active_players_ids:
        policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
        policies[player_id] = policy
        collector_env = SyncSubprocessEnvManager(
            env_fn=[partial(Env, collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
        )
        collector_env.seed(seed)

        learners[player_id] = BaseLearner(
            cfg.policy.learn.learner,
            policy.learn_mode,
            tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_learner'
        )
        collectors[player_id] = BattleAlphaZeroCollector(
            cfg.policy.collect.collector,
            collector_env, [policy.collect_mode, policy.collect_mode],
            tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_collector'
        )

    # create policy
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    main_key = [k for k in learners.keys() if k.startswith('main_player')][0]
    main_player = league.get_player_by_id(main_key)
    main_learner = learners[main_key]
    main_collector = collectors[main_key]

    policies['historical'] = policy

    # create bot policy
    cfg.policy.type = cfg.policy.league.player_category[0] + '_bot_v0'
    bot_policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    policies['bot'] = bot_policy

    # create evaluator
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(Env, evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env.seed(seed, dynamic_seed=False)
    evaluator_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator_cfg.stop_value = cfg.env.stop_value
    evaluator = AlphaZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name='vs_bot_evaluator'
    )

    def load_checkpoint_fn(player_id: str, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        policies[player_id].learn_mode.load_state_dict(state_dict)

    league.load_checkpoint = load_checkpoint_fn

    if cfg.policy.league.snapshot_the_player_in_iter_zero:
        # snapshot the initial player as the first historical player
        for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
            torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
            league.judge_snapshot(player_id, force=True)

    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    league_iter = 0
    while True:
        if evaluator.should_eval(main_learner.train_iter):
            stop_flag, eval_episode_info = evaluator.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            win_loss_result = win_loss_draw(eval_episode_info)

            # set eval bot rating as 100.
            main_player.rating = league.metric_env.rate_1vsC(
                main_player.rating, league.metric_env.create_rating(mu=100, sigma=1e-8), win_loss_result
            )
            if stop_flag:
                break

        for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
            tb_logger.add_scalar(
                'league/{}_trueskill'.format(player_id),
                league.get_player_by_id(player_id).rating.exposure, main_collector.envstep
            )
            collector, learner = collectors[player_id], learners[player_id]

            job = league.get_job_info(player_id)
            opponent_player_id = job['player_id'][1]
            # print('job player: {}'.format(job['player_id']))
            if 'historical' in opponent_player_id and 'bot' not in opponent_player_id:
                opponent_policy = policies['historical'].collect_mode
                opponent_path = job['checkpoint_path'][1]
                opponent_policy.load_state_dict(torch.load(opponent_path, map_location='cpu'))
                opponent_policy_info = {
                    'policy': opponent_policy,
                    'policy_id': opponent_player_id,
                    'policy_type': 'historical'
                }
            elif 'bot' in opponent_player_id:
                opponent_policy = policies['bot'].collect_mode
                opponent_policy_info = {
                    'policy': opponent_policy,
                    'policy_id': opponent_player_id,
                    'policy_type': 'bot'
                }
            else:
                opponent_policy = policies[opponent_player_id].collect_mode
                opponent_policy_info = {
                    'policy': opponent_policy,
                    'policy_id': opponent_player_id,
                    'policy_type': 'main'
                }

            collector.reset_policy([policies[player_id].collect_mode, opponent_policy_info])

            collect_kwargs = {}
            # set temperature for visit count distributions according to the train_iter,
            # please refer to Appendix D in MuZero paper for details.
            collect_kwargs['temperature'] = visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            )

            new_data, episode_info = collector.collect(
                train_iter=learner.train_iter, n_episode=cfg.policy.n_episode, policy_kwargs=collect_kwargs
            )
            # TODO(pu): new_data[1]
            new_data = sum(new_data[0], [])
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            # Learn policy from collected data
            for i in range(cfg.policy.update_per_collect):
                # Learner will train ``update_per_collect`` times in one iteration.
                train_data = replay_buffer.sample(cfg.policy.batch_size, learner.train_iter)
                if train_data is None:
                    logging.warning(
                        'The data in replay_buffer is not sufficient to sample a mini-batch.'
                        'continue to collect now ....'
                    )
                    break
                learner.train(train_data, collector.envstep)

            # update the learner_step for the current active player, i.e. the main player in most cases.
            player_info = learner.learn_info
            player_info['player_id'] = player_id
            league.update_active_player(player_info)

            # player_info['state_dict'] = policies[player_id].learn_mode.state_dict()

            league.judge_snapshot(player_id)
            # set eval_flag=True to enable trueskill update

            win_loss_result = win_loss_draw(episode_info[0])

            job_finish_info = {
                'eval_flag': True,
                'launch_player': job['launch_player'],
                'player_id': job['player_id'],
                'result': win_loss_result,
            }
            league.finish_job(job_finish_info, league_iter)

        if league_iter % cfg.policy.league.log_freq_for_payoff_rank == 0:
            payoff_string = repr(league.payoff)
            rank_string = league.player_rank(string=True)
            tb_logger.add_text('payoff_step', payoff_string, main_collector.envstep)
            tb_logger.add_text('rank_step', rank_string, main_collector.envstep)

        league_iter += 1

        if main_collector.envstep >= max_env_step or main_learner.train_iter >= max_train_iter:
            break
