from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union

import torch.distributions
import torch.nn.functional as F
import torch.optim as optim

from ding.model import model_wrap
from ding.policy.base_policy import Policy
from lzero.mcts.ptree.ptree_az import MCTS
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate
from ding.rl_utils import get_nstep_return_data, get_train_sample


@POLICY_REGISTRY.register('alphazero')
class AlphaZeroPolicy(Policy):
    """
    Overview:
        The policy class for AlphaZero.
    """
    config = dict(
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=False,
        priority=False,
        model=dict(
            observation_shape=(3, 6, 6),
            num_res_blocks=1,
            num_channels=32,
        ),
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu.
            multi_gpu=False,
            batch_size=256,
            optim_type='Adam',
            learning_rate=0.001,
            weight_decay=0.0001,
            grad_clip_value=10,
            value_weight=1.0,
        ),
        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        # threshold_training_steps_for_final_lr=int(
        #     threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_lr=int(5e5),
        # lr: 0.2 -> 0.02 -> 0.002

        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        # threshold_training_steps_for_final_temperature=int(
        #     threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_temperature=int(1e5),
        # temperature: 1 -> 0.5 -> 0.25
        auto_temperature=True,
        # ``fixed_temperature_value`` is effective only when auto_temperature=False
        fixed_temperature_value=0.25,
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For DQN, ``ding.model.template.q_learning.DQN``
        """
        return 'AlphaZeroModel', ['lzero.model.alphazero_model']

    def _init_learn(self):
        if 'optim_type' not in self._cfg.learn.keys() or self._cfg.learn.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                momentum=self._cfg.learn.momentum,
                weight_decay=self._cfg.learn.weight_decay,
            )
        elif self._cfg.learn.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self._cfg.learn.weight_decay
            )

        if self._cfg.learn.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        # Main and target models
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    def _forward_learn(self, inputs: dict) -> Dict[str, Any]:
        inputs = default_collate(inputs)
        if self._cuda:
            inputs = to_device(inputs, self._device)
        self._learn_model.train()

        state_batch = inputs['obs']['observation']
        mcts_probs = inputs['probs']
        reward = inputs['reward']

        state_batch = state_batch.to(device=self._device, dtype=torch.float)
        mcts_probs = mcts_probs.to(device=self._device, dtype=torch.float)
        reward = reward.to(device=self._device, dtype=torch.float)

        action_probs, values = self._learn_model.compute_prob_value(state_batch)
        log_probs = torch.log(action_probs)

        # calculate policy entropy, for monitoring only
        entropy = torch.mean(-torch.sum(action_probs * log_probs, 1))
        entropy_loss = -entropy

        # ============
        # policy loss
        # ============
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_probs, 1))

        # ============
        # value loss
        # ============
        value_loss = F.mse_loss(values.view(-1), reward)

        total_loss = self._value_weight * value_loss + policy_loss + self._entropy_weight * entropy_loss
        self._optimizer.zero_grad()
        total_loss.backward()

        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(list(self._model.parameters()), max_norm=self._cfg.learn.grad_clip_value,)
        self._optimizer.step()
        if self._cfg.learn.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_grad_norm_before_clip': total_grad_norm_before_clip,
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_mcts = MCTS(self._cfg.collect.mcts)
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()
        self.collect_mcts_temperature = 1


    @torch.no_grad()
    def _forward_collect(self, envs, obs, temperature: list = None):
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
            - temperature: shape: (N, ), where N is the number of collect_env.
        Returns:
            - data (:obj:`dict`): The collected data
        """
        self.collect_mcts_temperature = temperature[0]
        ready_env_id = list(envs.keys())
        init_state = {env_id: obs[env_id]['board'] for env_id in ready_env_id}
        start_player_index = {env_id: obs[env_id]['current_player_index'] for env_id in ready_env_id}
        output = {}
        self._policy_model = self._collect_model
        for env_id in ready_env_id:
            # print('[collect] start_player_index={}'.format(start_player_index[env_id]))
            # print('[collect] init_state=\n{}'.format(init_state[env_id]))
            envs[env_id].reset(
                start_player_index=start_player_index[env_id],
                init_state=init_state[env_id],
            )
            action, mcts_probs = self._collect_mcts.get_next_action(
                envs[env_id], policy_forward_fn=self._policy_value_fn, temperature=self.collect_mcts_temperature, sample=True
            )
            output[env_id] = {
                'action': action,
                'probs': mcts_probs,
            }
        return output

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        return {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'probs': model_output['probs'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_mcts = MCTS(self._cfg.eval.mcts)
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, envs: dict, obs) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        ready_env_id = list(obs.keys())
        init_state = {env_id: obs[env_id]['board'] for env_id in ready_env_id}
        start_player_index = {env_id: obs[env_id]['current_player_index'] for env_id in ready_env_id}
        output = {}
        self._policy_model = self._eval_model
        for env_id in ready_env_id:
            # print('[eval] start_player_index={}'.format(start_player_index[env_id]))
            # print('[eval] init_state=\n {}'.format(init_state[env_id]))
            envs[env_id].reset(
                start_player_index=start_player_index[env_id],
                init_state=init_state[env_id],
            )
            action, mcts_probs = self._collect_mcts.get_next_action(
                envs[env_id], policy_forward_fn=self._policy_value_fn, temperature=1.0, sample=False
            )
            output[env_id] = {
                'action': action,
                'probs': mcts_probs,
            }
        return output

    @torch.no_grad()
    def _policy_value_fn(self, env):
        """
        Overview:
            - input: env
            - output: a list of (action, probability) tuples for each available
                action and the score of the env state
        """
        legal_actions = env.legal_actions
        current_state, current_state_scale = env.current_state()
        current_state_scale = torch.from_numpy(current_state_scale).to(device=self._device, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = self._policy_model.compute_prob_value(current_state_scale)
        action_probs_dict = dict(zip(legal_actions, action_probs.squeeze(0)[legal_actions].detach().cpu().numpy()))
        return action_probs_dict, value.item()

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'cur_lr', 'total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'total_grad_norm_before_clip'
        ]
