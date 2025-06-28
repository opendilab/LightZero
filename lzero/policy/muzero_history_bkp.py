import copy
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.optim as optim
import wandb
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss

from lzero.entry.utils import initialize_zeros_batch
from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.model.utils import cal_dormant_ratio
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs_history, configure_optimizers


@POLICY_REGISTRY.register('muzero_history')
class MuZeroHistoryPolicy(Policy):
    """
    Overview:
        if self._cfg.model.model_type in ["conv", "mlp"]:
            The policy class for MuZero.
        if self._cfg.model.model_type == ["conv_context", "mlp_context"]:
            The policy class for MuZero w/ Context, a variant of MuZero.
            This variant retains the same training settings as MuZero but diverges during inference
            by employing a k-step recursively predicted latent representation at the root node,
            proposed in the UniZero paper https://arxiv.org/abs/2406.10667.
    """

    # The default_config for MuZero policy.
    config = dict(
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            # reference: http://proceedings.mlr.press/v80/imani18a/imani18a.pdf, https://arxiv.org/abs/2403.03950
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=300,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (str) The type of action encoding. Options are ['one_hot', 'not_one_hot']. Default to 'one_hot'.
            discrete_action_encoding_type='one_hot',
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'LN'.
            norm_type='BN',
            # (bool) Whether to analyze simulation normalization.
            analysis_sim_norm=False,
            # (bool) Whether to analyze dormant ratio.
            analysis_dormant_ratio=False,
            # (bool) Whether to use HarmonyDream to balance weights between different losses. Default to False.
            # More details can be found in https://arxiv.org/abs/2310.00344.
            harmony_balance=False
        ),
        # ****** common ******
        # (bool) Whether to use wandb to log the training process.
        use_wandb=False,
        # (bool) whether to use rnd model.
        use_rnd_model=False,
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled EfficientZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero)
        gumbel_algo=False,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. Options are ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. Options are ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=200,
        # (bool): Indicates whether to perform an offline evaluation of the checkpoint (ckpt).
        # If set to True, the checkpoint will be evaluated after the training process is complete.
        # IMPORTANT: Setting eval_offline to True requires configuring the saving of checkpoints to align with the evaluation frequency.
        # This is done by setting the parameter learn.learner.hook.save_ckpt_after_iter to the same value as eval_freq in the train_muzero.py automatically.
        eval_offline=False,
        # (bool) Whether to calculate the dormant ratio.
        cal_dormant_ratio=False,
        # (bool) Whether to analyze simulation normalization.
        analysis_sim_norm=False,
        # (bool) Whether to analyze dormant ratio.
        analysis_dormant_ratio=False,

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (int) Frequency of target network update.
        target_update_freq=100,
        # (int) Frequency of target network update.
        target_update_freq_for_intrinsic_reward=1000,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (int) The number of episodes in each collecting stage when use muzero_collector.
        n_episode=8,
        # (int) The number of num_segments in each collecting stage when use muzero_segment_collector.
        num_segments=8,
        # (int) the number of simulations in MCTS.
        num_simulations=50,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of steps for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of policy entropy loss.
        policy_entropy_weight=0,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        piecewise_decay_lr_scheduler=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,
        # (bool) Whether to add noise to roots during reanalyze process.
        reanalyze_noise=True,
        # (bool) Whether to reuse the root value between batch searches.
        reuse_search=False,
        # (bool) whether to use the pure policy to collect data. If False, use the MCTS guided with policy.
        collect_with_pure_policy=False,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and model import_names.
                - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
                - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For MuZero, ``lzero.model.muzero_model.MuZeroModel``
        """
        if self._cfg.model.model_type in ["conv_history"]:
            return 'MuZeroHistoryModel', ['lzero.model.muzero_model_history']
        elif self._cfg.model.model_type == "mlp":
            return 'MuZeroModelMLP', ['lzero.model.muzero_model_mlp']
        elif self._cfg.model.model_type in ["conv_context"]:
            return 'MuZeroContextModel', ['lzero.model.muzero_context_model']
        else:
            raise ValueError("model type {} is not supported".format(self._cfg.model.model_type))

    def set_train_iter_env_step(self, train_iter, env_step) -> None:
        """
        Overview:
            Set the train_iter and env_step for the policy.
        Arguments:
            - train_iter (:obj:`int`): The train_iter for the policy.
            - env_step (:obj:`int`): The env_step for the policy.
        """
        self.train_iter = train_iter
        self.env_step = env_step

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        # NOTE: in board_games, for fixed lr 0.003, 'Adam' is better than 'SGD'.
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
            self._optimizer = configure_optimizers(model=self._model, weight_decay=self._cfg.weight_decay,
                                                   learning_rate=self._cfg.learning_rate, device_type=self._cfg.device)

        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.target_update_freq}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        self.value_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.reward_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )
        
        # ==============================================================
        # harmonydream (learnable weights for different losses)
        # ==============================================================
        if self._cfg.model.harmony_balance:
            # List of parameter names
            harmony_names = ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
            # Initialize and name each parameter
            for name in harmony_names:
                param = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
                setattr(self, name, param)
            
        if self._cfg.use_rnd_model:
            if self._cfg.target_model_for_intrinsic_reward_update_type == 'assign':
                self._target_model_for_intrinsic_reward = model_wrap(
                    self._target_model,
                    wrapper_name='target',
                    update_type='assign',
                    update_kwargs={'freq': self._cfg.target_update_freq_for_intrinsic_reward}
                )
            elif self._cfg.target_model_for_intrinsic_reward_update_type == 'momentum':
                self._target_model_for_intrinsic_reward = model_wrap(
                    self._target_model,
                    wrapper_name='target',
                    update_type='momentum',
                    update_kwargs={'theta': self._cfg.target_update_theta_for_intrinsic_reward}
                )

        # ========= logging for analysis =========
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        self.dormant_ratio_encoder = 0.
        self.dormant_ratio_dynamics = 0.

        if self._cfg.use_wandb:
            # TODO: add the model to wandb
            wandb.watch(self._learn_model.representation_network, log="all")

    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning policy in learn mode, which is the core of the learning process.
            The data is sampled from replay buffer.
            The loss is calculated by the loss function and the loss is backpropagated to update the model.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): The data sampled from replay buffer, which is a tuple of tensors.
                The first tensor is the current_batch, the second tensor is the target_batch.
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): The information dict to be logged, which contains \
                current learning loss and learning statistics.
        """
        self._learn_model.train()
        self._target_model.train()
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.train()

        current_batch, target_batch = data
        obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
        target_reward, target_value, target_policy = target_batch

        # import ipdb;ipdb.set_trace()
        # TODO
        obs_batch, obs_target_batch = prepare_obs_history(obs_batch_ori, self._cfg)

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if self._cfg.model.self_supervised_learning_loss:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .long() is only  for discrete action space.
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        data_list = [mask_batch, target_reward,
            target_value, target_policy, weights
        ]
        [mask_batch, target_reward, target_value, target_policy,
         weights] = to_torch_float_tensor(data_list, self._cfg.device)

        target_reward = target_reward.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        assert obs_batch.size(0) == self._cfg.batch_size == target_reward.size(0)

        # ``scalar_transform`` to transform the original value to the scaled value,
        # i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # transform a scalar to its categorical_distribution. After this transformation, each scalar is
        # represented as the linear combination of its two adjacent supports.
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        # ==============================================================
        # the core initial_inference in MuZero policy.
        # ==============================================================
        network_output = self._learn_model.initial_inference(obs_batch)

        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

        # ========= logging for analysis =========
        # calculate dormant ratio of encoder
        if self._cfg.cal_dormant_ratio:
            self.dormant_ratio_encoder = cal_dormant_ratio(self._learn_model.representation_network, obs_batch.detach(),
                                                           percentage=self._cfg.dormant_threshold)
        # calculate L2 norm of latent state
        latent_state_l2_norms = torch.norm(latent_state.view(latent_state.shape[0], -1), p=2, dim=1).mean()
        # ========= logging for analysis ===============

        # transform the scaled value or its categorical representation to its original value,
        # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        original_value = self.inverse_scalar_transform_handle(value)

        # Note: The following lines are just for debugging.
        predicted_rewards = []
        if self._cfg.monitor_extra_statistics:
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition.
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + 1e-6

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])
        value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])

        prob = torch.softmax(policy_logits, dim=-1)
        entropy = -(prob * prob.log()).sum(-1)
        policy_entropy_loss = -entropy

        reward_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        target_policy_entropy = 0

        # ==============================================================
        # the core recurrent_inference in MuZero policy.
        # ==============================================================
        for step_k in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``latent_state``, ``reward``,
            # given current ``latent_state`` and ``action``.
            # And then predict policy_logits and value with the prediction function.
            network_output = self._learn_model.recurrent_inference(latent_state, action_batch[:, step_k+self.history_length-1])
            latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

            # ========= logging for analysis ===============
            if step_k == self._cfg.num_unroll_steps - 1 and self._cfg.cal_dormant_ratio:
                # calculate dormant ratio of encoder
                action_tmp = action_batch[:, step_k+self.history_length-1]
                if len(action_tmp.shape) == 1:
                    action = action.unsqueeze(-1)
                # transform action to one-hot encoding.
                # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
                action_one_hot = torch.zeros(action_tmp.shape[0], policy_logits.shape[-1], device=action_tmp.device)
                # transform action to torch.int64
                action_tmp = action_tmp.long()
                action_one_hot.scatter_(1, action_tmp, 1)
                action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
                action_encoding = action_encoding_tmp.expand(
                    latent_state.shape[0], policy_logits.shape[-1], latent_state.shape[2], latent_state.shape[3]
                )
                state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)
                self.dormant_ratio_dynamics = cal_dormant_ratio(self._learn_model.dynamics_network,
                                                                state_action_encoding.detach(),
                                                                percentage=self._cfg.dormant_threshold)
            # ========= logging for analysis ===============

            # transform the scaled value or its categorical representation to its original value,
            # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
            original_value = self.inverse_scalar_transform_handle(value)

            if self._cfg.model.self_supervised_learning_loss:
                # ==============================================================
                # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
                # ==============================================================
                if self._cfg.ssl_loss_weight > 0:
                    # obtain the oracle latent states from representation function.
                    beg_index, end_index = self._get_target_obs_index_in_step_k(step_k)
                    network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index])

                    latent_state = to_tensor(latent_state)
                    representation_state = to_tensor(network_output.latent_state)

                    # NOTE: no grad for the representation_state branch
                    dynamic_proj = self._learn_model.project(latent_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_k]
                    consistency_loss += temp_loss

            # NOTE: the target policy, target_value_categorical, target_reward_categorical is calculated in
            # game buffer now.
            # ==============================================================
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += cross_entropy_loss(policy_logits, target_policy[:, step_k + 1])

            # Here we take the hypothetical step k = step_k + 1
            prob = torch.softmax(policy_logits, dim=-1)
            entropy = -(prob * prob.log()).sum(-1)
            policy_entropy_loss += -entropy

            target_normalized_visit_count = target_policy[:, step_k + 1]

            # ******* NOTE: target_policy_entropy is only for debug.  ******
            non_masked_indices = torch.nonzero(mask_batch[:, step_k + 1]).squeeze(-1)
            # Check if there are any unmasked rows
            if len(non_masked_indices) > 0:
                target_normalized_visit_count_masked = torch.index_select(
                    target_normalized_visit_count, 0, non_masked_indices
                )
                target_policy_entropy += -((target_normalized_visit_count_masked + 1e-6) * (
                        target_normalized_visit_count_masked + 1e-6).log()).sum(-1).mean()
            else:
                # Set target_policy_entropy to log(|A|) if all rows are masked
                target_policy_entropy += torch.log(torch.tensor(target_normalized_visit_count.shape[-1]))

            value_loss += cross_entropy_loss(value, target_value_categorical[:, step_k + 1])
            reward_loss += cross_entropy_loss(reward, target_reward_categorical[:, step_k])

            if self._cfg.monitor_extra_statistics:
                original_rewards = self.inverse_scalar_transform_handle(reward)
                original_rewards_cpu = original_rewards.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, self.inverse_scalar_transform_handle(value).detach().cpu())
                )
                predicted_rewards.append(original_rewards_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))

        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        # weighted loss with masks (some invalid states which are out of trajectory.)
        # Nan appear when consistency loss or policy entropy loss uses harmony parameter as coefficient.
        
        # Please refer to https://github.com/thuml/HarmonyDream/blob/main/wmlib-torch/wmlib/agents/dreamerv2.py#L161
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            loss = (
                  (consistency_loss.mean() * self._cfg.ssl_loss_weight)
                + (policy_loss.mean() / torch.exp(self.harmony_policy))
                + (value_loss.mean() / torch.exp(self.harmony_value)) 
                + (reward_loss.mean() / torch.exp(self.harmony_reward))
            ) 
            weighted_total_loss = loss.mean()
            weighted_total_loss += (
                torch.log(torch.exp(self.harmony_policy) + 1) +
                torch.log(torch.exp(self.harmony_value) + 1) + 
                torch.log(torch.exp(self.harmony_reward) + 1) 
            )
        else:  
            loss = (
                    self._cfg.ssl_loss_weight * consistency_loss + self._cfg.policy_loss_weight * policy_loss +
                    self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * reward_loss +
                    self._cfg.policy_entropy_weight * policy_entropy_loss
            )
            weighted_total_loss = (weights * loss).mean()

        gradient_scale = 1 / self._cfg.num_unroll_steps
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()

        # ============= for analysis =============
        if self._cfg.analysis_sim_norm:
            del self.l2_norm_before
            del self.l2_norm_after
            del self.grad_norm_before
            del self.grad_norm_after
            self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
            self._target_model.encoder_hook.clear_data()
        # ============= for analysis =============

        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(),
                                                                     self._cfg.grad_clip_value)
        self._optimizer.step()
        if self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.update(self._learn_model.state_dict())

        if self._cfg.monitor_extra_statistics:
            predicted_rewards = torch.stack(predicted_rewards).transpose(1, 0).squeeze(-1)
            predicted_rewards = predicted_rewards.reshape(-1).unsqueeze(-1)

        return_log_dict = {
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self.collect_epsilon,
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': loss.mean().item(),
            'policy_loss': policy_loss.mean().item(),
            'policy_entropy': - policy_entropy_loss.mean().item() / (self._cfg.num_unroll_steps + 1),
            'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
            'reward_loss': reward_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'consistency_loss': consistency_loss.mean().item() / self._cfg.num_unroll_steps,
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'predicted_rewards': predicted_rewards.mean().item(),
            'predicted_values': predicted_values.mean().item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            # ==============================================================
            # priority related
            # ==============================================================
            'value_priority': value_priority.mean().item(),
            'value_priority_orig': value_priority,  # torch.tensor compatible with ddp settings

            'analysis/dormant_ratio_encoder': self.dormant_ratio_encoder,
            'analysis/dormant_ratio_dynamics': self.dormant_ratio_dynamics,
            'analysis/latent_state_l2_norms': latent_state_l2_norms.item(),
            'analysis/l2_norm_before': self.l2_norm_before,
            'analysis/l2_norm_after': self.l2_norm_after,
            'analysis/grad_norm_before': self.grad_norm_before,
            'analysis/grad_norm_after': self.grad_norm_after,
        }
        
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            harmony_dict = {
                "harmony_dynamics": self.harmony_dynamics.item(), 
                "harmony_dynamics_exp_recip": (1 / torch.exp(self.harmony_dynamics)).item(),
                "harmony_policy": self.harmony_policy.item(),
                "harmony_policy_exp_recip": (1 / torch.exp(self.harmony_policy)).item(),
                "harmony_value": self.harmony_value.item(),
                "harmony_value_exp_recip": (1 / torch.exp(self.harmony_value)).item(),
                "harmony_reward": self.harmony_reward.item(),
                "harmony_reward_exp_recip": (1 / torch.exp(self.harmony_reward)).item(),
                "harmony_entropy": self.harmony_entropy.item(),
                "harmony_entropy_exp_recip": (1 / torch.exp(self.harmony_entropy)).item(),
            }
            return_log_dict.update(harmony_dict)

        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_log_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_log_dict
    
    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)
        self._collect_mcts_temperature = 1.
        self.collect_epsilon = 0.0
        self.collector_env_num = self._cfg.collector_env_num
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs_collect = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0], self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)
            self.last_batch_action_collect = [-1 for i in range(self.collector_env_num)]
        if self._cfg.model.model_type in [ "conv_history"]:
            self.last_batch_obs_collect = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0]*self._cfg.model.history_length, self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)
            self.last_batch_obs_ready_collect = self.last_batch_obs_collect
            self.last_batch_action_collect = [-1 for i in range(self.collector_env_num)]

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
    ) -> Dict:
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Choosing the action through sampling during the collect mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - epsilon (:obj:`float`): The epsilon of the eps greedy exploration.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - epsilon: :math:`(1, )`.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if active_collect_env_num < self.collector_env_num:
            print(f"active_collect_env_num:{active_collect_env_num}")
            # import ipdb;ipdb.set_trace()

        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._collect_model.initial_inference(data)
            elif self._cfg.model.model_type in ["conv_context", "conv_history"]:
                network_output = self._collect_model.initial_inference(self.last_batch_obs_ready_collect, self.last_batch_action_collect,
                                                                       data)

            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            if not self._cfg.collect_with_pure_policy:
                # the only difference between collect and eval is the dirichlet noise
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(active_collect_env_num, legal_actions)
                
                if len(reward_roots) != len(policy_logits):
                    import ipdb;ipdb.set_trace()
                if len(reward_roots) != len(noises):
                    import ipdb;ipdb.set_trace()
                

                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)

                # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
                roots_visit_count_distributions = roots.get_distributions()
                roots_values = roots.get_values()  # shape: {list: batch_size}

                batch_action = []
                for i, env_id in enumerate(ready_env_id):
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    if self._cfg.eps.eps_greedy_exploration_in_collect:
                        # eps greedy collect
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=True
                        )
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else:
                        # normal collect
                        # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                        # the index within the legal action set, rather than the index in the entire action set.
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=False
                        )
                        # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    output[env_id] = {
                        'action': action,
                        'visit_count_distributions': distributions,
                        'visit_count_distribution_entropy': visit_count_distribution_entropy,
                        'searched_value': value,
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }
                    if self._cfg.model.model_type in ["conv_context"]:
                        batch_action.append(action)

                if self._cfg.model.model_type in ["conv_context"]:
                    self.last_batch_action_collect = batch_action

                    # import ipdb;ipdb.set_trace()
                    # 先更新全局的 self.last_batch_obs：
                    # 对于 ready_env_id 中的每个环境，将最新的观测 data 拼接到之前的历史观测上，然后仅保留最后的 history 个时间步对应的通道。
                    # 为了确保不同环境的顺序一致，先对 ready_env_id 排序（如果 ready_env_id 不是顺序递增的）
                    ready_env_ids = sorted(ready_env_id)

                    # 假设 data 的顺序与 ready_env_ids 对应，即 data[i] 为环境 ready_env_ids[i] 最新的观测。
                    for idx, env_id in enumerate(ready_env_ids):
                        # self.last_batch_obs[env_id]: shape [total_channels, H, W]
                        # data[idx]: shape [num_obs_channels, H, W]
                        # 拼接后通道数为 total_channels + num_obs_channels
                        combined_obs = torch.cat([self.last_batch_obs_collect[env_id], data[idx]], dim=0)
                        # 仅保留最新的 total_channels 个通道
                        self.last_batch_obs_collect[env_id] = combined_obs[-self.history_channels:]
                    
                    # 从全局历史张量中取出当前 ready 环境对应的更新后的观测
                    self.last_batch_obs_ready_collect = self.last_batch_obs_collect[ready_env_ids]
            else:
                for i, env_id in enumerate(ready_env_id):
                    policy_values = torch.softmax(torch.tensor([policy_logits[i][a] for a in legal_actions[i]]),
                                                  dim=0).tolist()
                    policy_values = policy_values / np.sum(policy_values)
                    action_index_in_legal_action_set = np.random.choice(len(legal_actions[i]), p=policy_values)
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    output[env_id] = {
                        'action': action,
                        'searched_value': pred_values[i],
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)
        
        self.evaluator_env_num = self._cfg.evaluator_env_num

        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs_eval = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0], self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)
            self.last_batch_action_eval = [-1 for _ in range(self.evaluator_env_num)]
        if self._cfg.model.model_type in [ "conv_history"]:
            self.last_batch_obs_eval = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0]*self._cfg.model.history_length, self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)
            self.last_batch_obs_ready_eval = self.last_batch_obs_eval
            self.last_batch_action_eval = [-1 for i in range(self.evaluator_env_num)]
        
        num_obs_channels = self._cfg.model.observation_shape[0]
        self.history_length = self._cfg.model.history_length
        self.history_channels = num_obs_channels * self.history_length

        # elif self._cfg.model.model_type == 'mlp_context':
        #     self.last_batch_obs = torch.zeros([3, self._cfg.model.observation_shape]).to(self._cfg.device)
        #     self.last_batch_action = [-1 for _ in range(3)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, ) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}
        if active_eval_env_num < self.evaluator_env_num:
            print(f"active_eval_env_num:{active_eval_env_num}")
            # import ipdb;ipdb.set_trace()

        with torch.no_grad():
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._eval_model.initial_inference(data)
            elif self._cfg.model.model_type in ["conv_history"]:
                # 调用 initial_inference 时，传入更新后的 ready 环境观测；
                # 注意：这里假定 self.last_batch_action 在对应模型中已经维护好（例如前一次记录的动作历史）。
                network_output = self._eval_model.initial_inference(self.last_batch_obs_ready_eval, self.last_batch_action_eval, data)

            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
                # entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                if self._cfg.model.model_type in ["conv_history"]:
                    batch_action.append(action)

            if self._cfg.model.model_type in ["conv_history"]:
                self.last_batch_action_eval = batch_action

                # 先更新全局的 self.last_batch_obs：
                # 对于 ready_env_id 中的每个环境，将最新的观测 data 拼接到之前的历史观测上，然后仅保留最后的 history 个时间步对应的通道。
                # 为了确保不同环境的顺序一致，先对 ready_env_id 排序（如果 ready_env_id 不是顺序递增的）
                ready_env_ids = sorted(ready_env_id)
                # 假设 data 的顺序与 ready_env_ids 对应，即 data[i] 为环境 ready_env_ids[i] 最新的观测。
                for idx, env_id in enumerate(ready_env_ids):
                    # self.last_batch_obs[env_id]: shape [total_channels, H, W]
                    # data[idx]: shape [num_obs_channels, H, W]
                    # 拼接后通道数为 total_channels + num_obs_channels
                    combined_obs = torch.cat([self.last_batch_obs_eval[env_id], data[idx]], dim=0)
                    
                    # 仅保留最新的 total_channels 个通道
                    self.last_batch_obs_eval[env_id] = combined_obs[-self.history_channels:]
                # 从全局历史张量中取出当前 ready 环境对应的更新后的观测
                self.last_batch_obs_ready_eval = self.last_batch_obs_eval[ready_env_ids]


        return output

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset the observation and action for the collector environment.
        Arguments:
            - data_id (`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
        """
        if self._cfg.model.model_type in ["conv_context", "conv_history"]:
            # self.last_batch_obs_collect = initialize_zeros_batch(
            #     self._cfg.model.observation_shape,
            #     self._cfg.collector_env_num,
            #     self._cfg.device
            # )
            self.last_batch_obs_collect = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0]*self._cfg.model.history_length, self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)

            self.last_batch_action_collect = [-1 for _ in range(self._cfg.collector_env_num)]
        else:
            raise ValueError(f"Unsupported model type in collect: {self._cfg.model.model_type}")

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset the observation and action for the evaluator environment.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
        """
        if self._cfg.model.model_type in ["conv_context", "conv_history"]:
            # self.last_batch_obs_eval = initialize_zeros_batch(
            #     self._cfg.model.observation_shape,
            #     self._cfg.evaluator_env_num,
            #     self._cfg.device
            # )
            self.last_batch_obs_eval = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0]*self._cfg.model.history_length, self._cfg.model.observation_shape[1],self._cfg.model.observation_shape[2]]).to(self._cfg.device)
            self.last_batch_action_eval = [-1 for _ in range(self._cfg.evaluator_env_num)]
        else:
            raise ValueError(f"Unsupported model type in eval: {self._cfg.model.model_type}")
    
    def _get_target_obs_index_in_step_k(self, step):
        """
        Overview:
            Get the begin index and end index of the target obs in step k.
        Arguments:
            - step (:obj:`int`): The current step k.
        Returns:
            - beg_index (:obj:`int`): The begin index of the target obs in step k.
            - end_index (:obj:`int`): The end index of the target obs in step k.
        Examples:
            >>> self._cfg.model.model_type = 'conv'
            >>> self._cfg.model.image_channel = 3
            >>> self._cfg.model.frame_stack_num = 4
            >>> self._get_target_obs_index_in_step_k(0)
            >>> (0, 12)
        """
        if self._cfg.model.model_type in ['conv', 'conv_context', 'conv_history']:
            beg_index = self._cfg.model.image_channel * step
            end_index = self._cfg.model.image_channel * (step + self._cfg.model.frame_stack_num)
        elif self._cfg.model.model_type in ['mlp', 'mlp_context', 'mlp_history']:
            beg_index = self._cfg.model.observation_shape * step
            end_index = self._cfg.model.observation_shape * (step + self._cfg.model.frame_stack_num)
        return beg_index, end_index

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return_list = [
            'analysis/dormant_ratio_encoder',
            'analysis/dormant_ratio_dynamics',
            'analysis/latent_state_l2_norms',
            'analysis/l2_norm_before',
            'analysis/l2_norm_after',
            'analysis/grad_norm_before',
            'analysis/grad_norm_after',

            'collect_mcts_temperature',
            'cur_lr',
            'weighted_total_loss',
            'total_loss',
            'policy_loss',
            'policy_entropy',
            'target_policy_entropy',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'predicted_rewards',
            'predicted_values',
            'transformed_target_reward',
            'transformed_target_value',
            'total_grad_norm_before_clip',
        ]
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            harmony_list = [
                'harmony_dynamics', 'harmony_dynamics_exp_recip',
                'harmony_policy', 'harmony_policy_exp_recip',
                'harmony_value', 'harmony_value_exp_recip',
                'harmony_reward', 'harmony_reward_exp_recip',
                'harmony_entropy', 'harmony_entropy_exp_recip',
            ]
            return_list.extend(harmony_list)
        return return_list

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def __del__(self):
        if self._cfg.model.analysis_sim_norm:
            # Remove hooks after training.
            self._collect_model.encoder_hook.remove_hooks()
            self._target_model.encoder_hook.remove_hooks()

    def _process_transition(self, obs, policy_output, timestep):
        # be compatible with DI-engine Policy class
        pass

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass

