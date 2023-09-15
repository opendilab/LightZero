from collections import namedtuple
from typing import List, Dict, Tuple

import numpy as np
import torch.distributions
import torch.nn.functional as F
import torch.optim as optim
from ding.policy.base_policy import Policy
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate

from lzero.mcts.ptree.ptree_az import MCTS
from lzero.policy import configure_optimizers


@POLICY_REGISTRY.register('alphazero')
class AlphaZeroPolicy(Policy):
    """
    Overview:
        The policy class for AlphaZero.
    """

    # The default_config for AlphaZero policy.
    config = dict(
        # (bool) Whether to use torch.compile method to speed up our model, which required torch>=2.0.
        torch_compile=False,
        # (bool) Whether to use TF32 for our model.
        tensor_float_32=False,
        model=dict(
            # (tuple) The stacked obs shape.
            observation_shape=(3, 6, 6),
            # (int) The number of res blocks in AlphaZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in AlphaZero model.
            num_channels=32,
        ),
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.model_update_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        model_update_ratio=0.1,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam', 'AdamW']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (float) The weight of value loss.
        value_weight=1.0,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e5),
        # (bool) Whether to use manually temperature decay.
        # i.e. temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        mcts=dict(
            # (int) The number of simulations to perform at each move.
            num_simulations=50,
            # (int) The maximum number of moves to make in a game.
            max_moves=512,  # for chess and shogi, 722 for Go.
            # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
            root_dirichlet_alpha=0.3,
            # (float) The noise weight at the root node of the search tree.
            root_noise_weight=0.25,
            # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_base=19652,
            # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_init=1.25,
        ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=int(1e6),
            save_episode=False,
        )),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
            - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        """
        return 'AlphaZeroModel', ['lzero.model.alphazero_model']

    def _init_learn(self) -> None:
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        if self._cfg.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay,
            )
        elif self._cfg.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._cfg.learning_rate, weight_decay=self._cfg.weight_decay
            )
        elif self._cfg.optim_type == 'AdamW':
            self._optimizer = configure_optimizers(
                model=self._model,
                weight_decay=self._cfg.weight_decay,
                learning_rate=self._cfg.learning_rate,
                device_type=self._cfg.device
            )

        if self._cfg.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # Algorithm config
        self._value_weight = self._cfg.value_weight
        self._entropy_weight = self._cfg.entropy_weight
        # Main and target models
        self._learn_model = self._model

        # TODO(pu): test the effect of torch 2.0
        if self._cfg.torch_compile:
            self._learn_model = torch.compile(self._learn_model)

    def _forward_learn(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
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

        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=self._cfg.grad_clip_value,
        )
        self._optimizer.step()
        if self._cfg.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            'collect_mcts_temperature': self._collect_mcts_temperature,
        }

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._collect_mcts = MCTS(self._cfg.mcts)
        self._collect_model = self._model
        self._collect_mcts_temperature = 1

    @torch.no_grad()
    def _forward_collect(self, envs: Dict, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The forward function for collecting data in collect mode. Use real env to execute MCTS search.
        Arguments:
            - envs (:obj:`Dict`): The dict of colletor envs, the key is env_id and the value is the env instance.
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
            - temperature (:obj:`float`): The temperature for MCTS search.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
        """
        self._collect_mcts_temperature = temperature
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
                envs[env_id],
                policy_forward_fn=self._policy_value_fn,
                temperature=self._collect_mcts_temperature,
                sample=True
            )
            output[env_id] = {
                'action': action,
                'probs': mcts_probs,
            }
        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_mcts = MCTS(self._cfg.mcts)
        self._eval_model = self._model

    def _forward_eval(self, envs: Dict, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode, similar to ``self._forward_collect``.
        Arguments:
            - envs (:obj:`Dict`): The dict of colletor envs, the key is env_id and the value is the env instance.
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
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
            action, mcts_probs = self._eval_mcts.get_next_action(
                envs[env_id], policy_forward_fn=self._policy_value_fn, temperature=1.0, sample=False
            )
            output[env_id] = {
                'action': action,
                'probs': mcts_probs,
            }
        return output

    @torch.no_grad()
    def _policy_value_fn(self, env: 'Env') -> Tuple[Dict[int, np.ndarray], float]:  # noqa
        legal_actions = env.legal_actions
        current_state, current_state_scale = env.current_state()
        current_state_scale = torch.from_numpy(current_state_scale).to(
            device=self._device, dtype=torch.float
        ).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = self._policy_model.compute_prob_value(current_state_scale)
        action_probs_dict = dict(zip(legal_actions, action_probs.squeeze(0)[legal_actions].detach().cpu().numpy()))
        return action_probs_dict, value.item()

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return super()._monitor_vars_learn() + [
            'cur_lr', 'total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'total_grad_norm_before_clip',
            'collect_mcts_temperature'
        ]

    def _process_transition(self, obs: Dict, model_output: Dict[str, torch.Tensor], timestep: namedtuple) -> Dict:
        """
        Overview:
            Generate the dict type transition (one timestep) data from policy learning.
        """
        return {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'probs': model_output['probs'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass
