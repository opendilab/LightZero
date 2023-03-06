import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import treetensor.torch as ttorch
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.rl_utils import get_nstep_return_data, get_train_sample
from ding.torch_utils import to_tensor, to_device
from ding.utils import POLICY_REGISTRY
from torch.distributions import Categorical, Independent, Normal
from torch.nn import L1Loss

# python mcts_tree
import lzero.mcts.ptree.ptree_sez as ptree
from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts import ImageTransforms, modified_cross_entropy_loss, value_phi, reward_phi, DiscreteSupport
from lzero.mcts import scalar_transform, InverseScalarTransform
from lzero.mcts import select_action
# cpp mcts_tree
from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as ctree


@POLICY_REGISTRY.register('sampled_efficientzero')
class SampledEfficientZeroPolicy(Policy):
    """
    Overview:
        The policy class for Sampled EfficientZero.
    """

    # The default_config for Sampled fEficientZero policy.
    config = dict(
        # (bool) Whether learning policy is the same as collecting data policy(on-policy)
        on_policy=False,
        model=dict(
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 96, 96, 3] -> [4*3, 96, 96]
            # observation_shape=(12, 96, 96),  # if frame_stack_num=4, gray_scale=False
            # observation_shape=(3, 96, 96),  # if frame_stack_num=1, gray_scale=False
            observation_shape=(4, 96, 96),  # if frame_stack_num=4, gray_scale=True
            action_space_size=6,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            # the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=4,
            # ==============================================================
            # the default config is large size model, same as the EfficientZero original paper.
            # ==============================================================
            num_res_blocks=1,
            num_channels=64,
            lstm_hidden_size=512,
            support_scale=300,
            # the above model para. is usually fixed
            # ==============================================================
            # the sampled specific config
            # ==============================================================
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            fixed_sigma_value=0.3,
        ),
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            # For different env, we have different episode_length,
            # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
            update_per_collect=100,
            # (int) How many samples in a training batch
            batch_size=256,
            # (int) Frequency of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            weight_decay=2e-5,
            momentum=0.9,
            grad_clip_value=10,

            # optim_type='Adam',
            # lr_manually=False,
            # cos_lr_scheduler=True,
            # learning_rate=0.0001,  # init lr for Adam optimizer coupled with cos_lr_scheduler

            # lr_manually=False,
            # cos_lr_scheduler=False,
            # optim_type='Adam',
            # learning_rate=0.003,  # lr for Adam optimizer

            lr_manually=True,
            cos_lr_scheduler=False,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule

            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,

            normalize_prob_of_sampled_actions=False,
            policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_episode" episodes per collect.
            n_episode=8,
            unroll_len=1,
        ),

        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,
        game_block_length=200,

        ## observation
        # the key difference setting between image-input and vector input.
        cvt_string=False,
        use_augmentation=True,
        # style of augmentation
        augmentation=['shift', 'intensity'],

        ## learn
        num_simulations=50,
        discount_factor=0.997,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        policy_entropy_loss_weight=0,
        ssl_loss_weight=2,

        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        # threshold_training_steps_for_final_lr=int(
        #     threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_lr=int(1e5),
        # lr: 0.2 -> 0.02 -> 0.002

        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        # threshold_training_steps_for_final_temperature=int(
        #     threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_temperature=int(1e5),
        # temperature: 1 -> 0.5 -> 0.25
        auto_temperature=True,
        # ``fixed_temperature_value`` is effective only when auto_temperature=False
        fixed_temperature_value=0.25,

        ## reanalyze
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # how much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # how much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,

        ## UCB
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For Sampled EfficientZero, ``lzero.model.sampled_efficientzero_model.SampledEfficientZeroModel``
        """
        return 'SampledEfficientZeroModel', ['lzero.model.sampled_efficientzero_model']

    def _init_learn(self) -> None:
        if self._cfg.model.continuous_action_space:
            # Weight Init for the last output layer of gaussian policy in prediction network.
            init_w = self._cfg.learn.init_w
            self._model.prediction_network.sampled_fc_policy.mu.weight.data.uniform_(-init_w, init_w)
            self._model.prediction_network.sampled_fc_policy.mu.bias.data.uniform_(-init_w, init_w)
            self._model.prediction_network.sampled_fc_policy.log_sigma_layer.weight.data.uniform_(-init_w, init_w)
            try:
                self._model.prediction_network.sampled_fc_policy.log_sigma_layer.bias.data.uniform_(-init_w, init_w)
            except Exception as exception:
                print(exception)

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

        if self._cfg.learn.cos_lr_scheduler is True:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.rl_scheduler = CosineAnnealingLR(self._optimizer, 1e6, eta_min=0, last_epoch=-1)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        # self._learn_model = self._model
        self._learn_model.reset()
        self._target_model.reset()
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
        self.collect_mcts_temperature = 1


    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        self._target_model.train()

        current_batch, targets_batch = data
        # ==============================================================
        # sampled related code
        # ==============================================================
        obs_batch_ori, action_batch, child_sampled_actions_batch, mask_batch, indices, weights, make_time = current_batch
        target_value_prefix, target_value, target_policy = targets_batch

        # [:, 0: config.model.frame_stack_num * 3,:,:]
        # obs_batch_ori is the original observations in a batch
        # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
        # obs_target_batch is the observations for s_t (hidden states from representation function)

        # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames

        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(self._cfg.device).float()

        # collector data process:
        # (batch_size, stack_num+num_unroll_steps, W, H, C) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H )

        # e.g. in pong: stack_num=4, num_unroll_steps=5
        # (4, 9, 96, 96, 3) -> (4, 9*3, 96, 96) = 4,27,96,96

        # in the second dim:
        # timestep t: 1,2,3,4,5,6,7,8,9
        # channel_num:    3    3     3   3     3    3    3   3       3
        #                ---, ---, ---, ---,  ---, ---, ---, ---,   ---

        # (4, 4*3, 96, 96) = (4, 12, 96, 96)
        # take the first stacked obs at timestep t: o_t_stack
        # used in initial_inference
        obs_batch = obs_batch_ori[:, 0:self._cfg.model.frame_stack_num * self._cfg.model.image_channel, :, :]

        # take the all obs other than timestep t1:
        # obs_target_batch is used for calculate consistency loss, which is only performed in the last 8 timesteps
        # for i in rnage(num_unroll_steeps):
        #   beg_index = self._cfg.model.image_channel * step_i
        #   end_index = self._cfg.model.image_channel * (step_i + self._cfg.model.frame_stack_num)
        obs_target_batch = obs_batch_ori[:, self._cfg.model.image_channel:, :, :]

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # batch_size, num_unroll_steps, action_dim
        # e.g. 4, 5, 1
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).float().unsqueeze(-1)
        # TODO: .float()
        # action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1)

        # batch_size, num_unroll_steps+1, num_of_sampled_actions, action_dim, 1
        # e.g. 4, 6, 5, 1, 1
        child_sampled_actions_batch = torch.from_numpy(child_sampled_actions_batch).to(self._cfg.device).unsqueeze(-1)
        mask_batch = torch.from_numpy(mask_batch).to(self._cfg.device).float()
        target_value_prefix = torch.from_numpy(target_value_prefix.astype('float64')).to(self._cfg.device).float()
        target_value = torch.from_numpy(target_value.astype('float64')).to(self._cfg.device).float()
        target_policy = torch.from_numpy(target_policy).to(self._cfg.device).float()
        weights = torch.from_numpy(weights).to(self._cfg.device).float()

        # TODO
        target_value_prefix = target_value_prefix.view(self._cfg.learn.batch_size, -1)
        target_value = target_value.view(self._cfg.learn.batch_size, -1)

        batch_size = obs_batch.size(0)
        assert batch_size == self._cfg.learn.batch_size == target_value_prefix.size(0)
        metric_loss = torch.nn.L1Loss()

        # some logs preparation
        other_log = {}
        other_dist = {}

        extra_statistics = {
            'l1': -1,
            'l1_1': -1,
            'l1_-1': -1,
            'l1_0': -1,
        }
        for i in range(self._cfg.num_unroll_steps):
            key = 'unroll_' + str(i + 1) + '_l1'
            extra_statistics[key] = -1
            extra_statistics[key + '_1'] = -1
            extra_statistics[key + '_-1'] = -1
            extra_statistics[key + '_0'] = -1

        # scalar transform to transformed Q scale, h(.) function
        transformed_target_value_prefix = scalar_transform(target_value_prefix)
        transformed_target_value = scalar_transform(target_value)
        if self._cfg.model.categorical_distribution:
            # scalar to categorical_distribution
            # Under this transformation, each scalar is represented as the linear
            # combination of its two adjacent supports
            target_value_prefix_phi = reward_phi(self.reward_support, transformed_target_value_prefix)
            target_value_phi = value_phi(self.value_support, transformed_target_value)

        network_output = self._learn_model.initial_inference(obs_batch)

        value = network_output.value
        value_prefix = network_output.value_prefix
        hidden_state = network_output.hidden_state  # （2, 64, 6, 6）
        reward_hidden_state = network_output.reward_hidden_state  # {tuple:2} (1,2,512)
        policy_logits = network_output.policy_logits  # {list: 2} {list:6}

        reward_hidden_state = to_device(reward_hidden_state, self._cfg.device)

        # h^-1(.) function
        # transform categorical representation to original_value

        original_value = self.inverse_scalar_transform_handle(value)

        # TODO(pu)
        if not self._learn_model.training:
            # if not in training, obtain the scalars of the value/reward
            original_value = original_value.detach().cpu().numpy()
            # original_value_prefix = original_value_prefix.detach().cpu().numpy()

            hidden_state = hidden_state.detach().cpu().numpy()
            reward_hidden_state = (
                reward_hidden_state[0].detach().cpu().numpy(), reward_hidden_state[1].detach().cpu().numpy()
            )
            policy_logits = policy_logits.detach().cpu().numpy()

        if self._cfg.monitor_statistics:
            state_lst = hidden_state.detach().cpu().numpy()

        predicted_value_prefixs = []
        # Note: Following line is just for logging.
        if self._cfg.monitor_statistics:
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + self._cfg.prioritized_replay_eps

        # calculate loss for the first step
        if self._cfg.model.categorical_distribution:
            value_loss = modified_cross_entropy_loss(value, target_value_phi[:, 0])
        else:
            value_loss = torch.nn.MSELoss(reduction='none')(value.squeeze(-1), transformed_target_value[:, 0])

        # ==============================================================
        # sampled related code
        # ==============================================================

        # calculate policy loss: KL loss
        if self._cfg.model.continuous_action_space:
            """continuous action space"""
            (mu, sigma) = policy_logits[:, :self._cfg.model.
                action_space_size], policy_logits[:, -self._cfg.model.action_space_size:]

            # try:
            dist = Independent(Normal(mu, sigma), 1)
            # except Exception as error:
            #     # Sometimes, random hyperparams can generate NaN
            #     print(error)

            # take the init hypothetical step k=0
            target_normalized_visit_count_init_step = target_policy[:, 0]

            target_normalized_visit_count_init_step_masked = torch.index_select(
                target_normalized_visit_count_init_step, 0,
                torch.nonzero(mask_batch[:, 0]).squeeze(-1)
            )
            # only for debug
            target_dist = Categorical(target_normalized_visit_count_init_step_masked)
            target_policy_entropy = target_dist.entropy().mean()

            # batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1 -> batch_size, num_of_sampled_actions, action_dim
            # e.g. 4, 6, 20, 2, 1 ->  4, 20, 2
            target_sampled_actions = child_sampled_actions_batch[:, 0].squeeze(-1)

            policy_entropy = dist.entropy().mean()
            policy_entropy_loss = -dist.entropy()

            # project the sampled-based improved policy back onto the space of representable policies
            # calculate KL loss
            # batch_size, num_of_sampled_actions -> 4,20
            # target_normalized_visit_count_init_step is categorical distribution, the range of target_log_prob_sampled_actions is (-inf,0)
            target_log_prob_sampled_actions = torch.log(target_normalized_visit_count_init_step + 1e-6)
            log_prob_sampled_actions = []
            for k in range(self._cfg.model.num_of_sampled_actions):
                # target_sampled_actions[:,i,:].shape: batch_size, action_dim -> 4,2
                # dist.log_prob(target_sampled_actions[:,i,:]).shape: batch_size -> 4
                # dist is normal distribution, the range of log_prob_sampled_actions is (-inf, inf)

                # way 1:
                # log_prob = dist.log_prob(target_sampled_actions[:, k, :])

                # way 2: SAC-like
                y = 1 - target_sampled_actions[:, k, :].pow(2)
                # for numerical stability
                target_sampled_actions_clamped = torch.clamp(target_sampled_actions[:, k, :], torch.tensor(-1 + 1e-6),
                                                             torch.tensor(1 - 1e-6))
                target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)

                # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
                log_prob = dist.log_prob(target_sampled_actions_before_tanh).unsqueeze(-1)
                log_prob = log_prob - torch.log(y + 1e-6).sum(-1, keepdim=True)
                log_prob = log_prob.squeeze(-1)

                log_prob_sampled_actions.append(log_prob)
            # batch_size, num_of_sampled_actions -> 4,20
            log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

            if self._cfg.learn.normalize_prob_of_sampled_actions:
                # normalize the prob of sampled actions
                prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                    log_prob_sampled_actions
                ).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach()
                # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1, eps=1e-6)
                log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

            if self._cfg.learn.policy_loss_type == 'KL':
                # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
                # policy_loss = (torch.exp(log_prob_sampled_actions) * (log_prob_sampled_actions - target_log_prob_sampled_actions.detach())).sum(-1).mean(0)

                # policy_loss = (torch.exp(target_log_prob_sampled_actions.detach()) * (
                #         target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)).sum(-1).mean(0)
                policy_loss = (
                                      torch.exp(target_log_prob_sampled_actions.detach()) *
                                      (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                              ).sum(-1) * mask_batch[:, 0]
            elif self._cfg.learn.policy_loss_type == 'cross_entropy':
                # cross_entropy loss: - sum(p * log (q) )
                # policy_loss = - torch.mean(
                #     torch.sum(torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1))
                policy_loss = -torch.sum(
                    torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
                ) * mask_batch[:, 0]
        else:
            """discrete action space"""
            prob = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(prob)

            # take the init hypothetical step k=0
            target_normalized_visit_count_init_step = target_policy[:, 0]

            target_normalized_visit_count_init_step_masked = torch.index_select(
                target_normalized_visit_count_init_step, 0,
                torch.nonzero(mask_batch[:, 0]).squeeze(-1)
            )

            # only for debug
            target_dist = Categorical(target_normalized_visit_count_init_step_masked)
            target_policy_entropy = target_dist.entropy().mean()

            # batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1 -> batch_size, num_of_sampled_actions, action_dim
            # e.g. 4, 6, 20, 2, 1 ->  4, 20, 2
            target_sampled_actions = child_sampled_actions_batch[:, 0].squeeze(-1)

            policy_entropy = dist.entropy().mean()
            policy_entropy_loss = -dist.entropy()

            # project the sampled-based improved policy back onto the space of representable policies
            # calculate KL loss
            # batch_size, num_of_sampled_actions -> 4,20
            # target_normalized_visit_count_init_step is categorical distribution, the range of target_log_prob_sampled_actions is (-inf,0)
            target_log_prob_sampled_actions = torch.log(target_normalized_visit_count_init_step + 1e-6)

            log_prob_sampled_actions = []
            for k in range(self._cfg.model.num_of_sampled_actions):
                # target_sampled_actions[:,i,:].shape: batch_size, action_dim -> 4,2
                # dist.log_prob(target_sampled_actions[:,i,:]).shape: batch_size -> 4
                # dist is normal distribution, the range of log_prob_sampled_actions is (-inf, inf)

                if len(target_sampled_actions.shape) == 2:
                    target_sampled_actions = target_sampled_actions.unsqueeze(-1)

                # way 1:
                log_prob = torch.log(prob.gather(-1, target_sampled_actions[:, k].long()).squeeze(-1) + 1e-6)

                log_prob_sampled_actions.append(log_prob)

            # batch_size, num_of_sampled_actions -> 4,20
            log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

            if self._cfg.learn.normalize_prob_of_sampled_actions:
                # normalize the prob of sampled actions
                prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                    log_prob_sampled_actions
                ).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach()
                # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1, eps=1e-6)
                log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

            if self._cfg.learn.policy_loss_type == 'KL':
                # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
                policy_loss = (
                                      torch.exp(target_log_prob_sampled_actions.detach()) *
                                      (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                              ).sum(-1) * mask_batch[:, 0]
            elif self._cfg.learn.policy_loss_type == 'cross_entropy':
                # cross_entropy loss: - sum(p * log (q) )
                policy_loss = -torch.sum(
                    torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
                ) * mask_batch[:, 0]

        value_prefix_loss = torch.zeros(batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(batch_size, device=self._cfg.device)

        target_value_prefix_cpu = target_value_prefix.detach().cpu()
        gradient_scale = 1 / self._cfg.num_unroll_steps

        # loss of the unrolled steps
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function
            network_output = self._learn_model.recurrent_inference(
                hidden_state, reward_hidden_state, action_batch[:, step_i]
            )
            value = network_output.value
            value_prefix = network_output.value_prefix
            policy_logits = network_output.policy_logits  # {list: 2} {list:6}
            hidden_state = network_output.hidden_state  # （2, 64, 6, 6）
            reward_hidden_state = network_output.reward_hidden_state  # {tuple:2} (1,2,512)

            # h^-1(.) function
            # first transform categorical representation to scalar, then transform to original_value
            original_value = self.inverse_scalar_transform_handle(value)
            original_value_prefix = self.inverse_scalar_transform_handle(value_prefix)

            # TODO(pu)
            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                original_value = original_value.detach().cpu().numpy()
                original_value_prefix = original_value_prefix.detach().cpu().numpy()

                hidden_state = hidden_state.detach().cpu().numpy()
                reward_hidden_state = (
                    reward_hidden_state[0].detach().cpu().numpy(), reward_hidden_state[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy()

            beg_index = self._cfg.model.image_channel * step_i
            end_index = self._cfg.model.image_channel * (step_i + self._cfg.model.frame_stack_num)

            # consistency loss
            if self._cfg.ssl_loss_weight > 0:
                # obtain the oracle hidden states from representation function
                network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                presentation_state = network_output.hidden_state

                hidden_state = to_tensor(hidden_state)
                presentation_state = to_tensor(presentation_state)

                # no grad for the presentation_state branch
                dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                observation_proj = self._learn_model.project(presentation_state, with_grad=False)
                temp_loss = self._consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                extra_statistics['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            # the target policy, target_value_phi, target_value_prefix_phi is calculated in game buffer now
            # ==============================================================
            # sampled related code
            # ==============================================================
            # calculate policy loss: KL loss
            if self._cfg.model.continuous_action_space:

                (mu, sigma) = policy_logits[:, :self._cfg.model.
                    action_space_size], policy_logits[:, -self._cfg.model.action_space_size:]
                dist = Independent(Normal(mu, sigma), 1)

                # take th hypothetical step k= step_i + 1
                target_normalized_visit_count = copy.deepcopy(target_policy[:, step_i + 1])
                # exclude the null target policy
                target_normalized_visit_count_masked = torch.index_select(
                    target_normalized_visit_count, 0,
                    torch.nonzero(mask_batch[:, step_i + 1]).squeeze(-1)
                )
                # only for debug
                try:
                    target_dist = Categorical(target_normalized_visit_count_masked)
                except Exception as error:
                    print(error)
                target_policy_entropy += target_dist.entropy().mean()

                # batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1 -> batch_size, num_of_sampled_actions, action_dim
                # e.g. 4, 3, 20, 2, 1 ->  4, 20, 2
                target_sampled_actions = child_sampled_actions_batch[:, step_i + 1].squeeze(-1)

                policy_entropy = dist.entropy().mean()
                policy_entropy_loss += -dist.entropy()

                # project the sampled-based improved policy back onto the space of representable policies

                # batch_size, num_of_sampled_actions -> 4,20
                target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
                log_prob_sampled_actions = []
                for k in range(self._cfg.model.num_of_sampled_actions):
                    # target_sampled_actions[:,k,:].shape: (batch_size, action_dim) e.g. (4,2)
                    # dist.log_prob(target_sampled_actions[:,k,:]).shape: (batch_size,) e.g. (4,)

                    # way 1:
                    # log_prob = dist.log_prob(target_sampled_actions[:, k, :])

                    # way 2:
                    y = 1 - target_sampled_actions[:, k, :].pow(2)
                    # for numerical stability
                    target_sampled_actions_clamped = torch.clamp(target_sampled_actions[:, k, :],
                                                                 torch.tensor(-1 + 1e-6),
                                                                 torch.tensor(1 - 1e-6))
                    target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)
                    # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
                    log_prob = dist.log_prob(target_sampled_actions_before_tanh).unsqueeze(-1)

                    log_prob = log_prob - torch.log(y + 1e-6).sum(-1, keepdim=True)
                    log_prob = log_prob.squeeze(-1)

                    log_prob_sampled_actions.append(log_prob)
                # (batch_size, num_of_sampled_actions) e.g. (4,20)
                log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

                if self._cfg.learn.normalize_prob_of_sampled_actions:
                    # normalize the prob of sampled actions
                    prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                        log_prob_sampled_actions).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[
                        -1]).detach()
                    # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1, eps=1e-6)
                    log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

                if self._cfg.learn.policy_loss_type == 'KL':
                    # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
                    # NOTE: accumulate policy loss!!! should be +=
                    policy_loss += (
                                           torch.exp(target_log_prob_sampled_actions.detach()) *
                                           (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                                   ).sum(-1) * mask_batch[:, step_i + 1]
                elif self._cfg.learn.policy_loss_type == 'cross_entropy':
                    # cross_entropy loss: - sum(p * log (q) )
                    # NOTE: accumulate policy loss!!! should be +=
                    policy_loss += -torch.sum(
                        torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
                    ) * mask_batch[:, step_i + 1]

            else:
                """discrete action space"""
                prob = torch.softmax(policy_logits, dim=-1)
                dist = Categorical(prob)

                # take th hypothetical step k= step_i + 1
                target_normalized_visit_count = copy.deepcopy(target_policy[:, step_i + 1])
                # exclude the null target policy
                target_normalized_visit_count_masked = torch.index_select(
                    target_normalized_visit_count, 0,
                    torch.nonzero(mask_batch[:, step_i + 1]).squeeze(-1)
                )

                # only for debug
                try:
                    target_dist = Categorical(target_normalized_visit_count_masked)
                except Exception as error:
                    print(error)
                target_policy_entropy += target_dist.entropy().mean()

                # batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1 -> batch_size, num_of_sampled_actions, action_dim
                # e.g. 4, 6, 20, 2, 1 ->  4, 20, 2
                target_sampled_actions = child_sampled_actions_batch[:, step_i + 1].squeeze(-1)

                policy_entropy += dist.entropy().mean()
                policy_entropy_loss += -dist.entropy()

                # project the sampled-based improved policy back onto the space of representable policies
                # shape: (batch_size, num_of_sampled_actions) -> 4,20
                # target_normalized_visit_count is categorical distribution [0,1], the range of target_log_prob_sampled_actions is (-inf,0)
                target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
                log_prob_sampled_actions = []
                for k in range(self._cfg.model.num_of_sampled_actions):
                    # target_sampled_actions[:,i,:].shape: batch_size, action_dim -> 4,2
                    # dist.log_prob(target_sampled_actions[:,i,:]).shape: batch_size -> 4
                    # dist is normal distribution, the range of log_prob_sampled_actions is (-inf, inf)

                    if len(target_sampled_actions.shape) == 2:
                        target_sampled_actions = target_sampled_actions.unsqueeze(-1)

                    # way 1:
                    # log_prob = dist.log_prob(target_sampled_actions[:, k, :])
                    log_prob = torch.log(prob.gather(-1, target_sampled_actions[:, k].long()).squeeze(-1) + 1e-6)

                    log_prob_sampled_actions.append(log_prob)

                # batch_size, num_of_sampled_actions -> 4,20
                log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

                if self._cfg.learn.normalize_prob_of_sampled_actions:
                    # normalize the prob of sampled actions
                    prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                        log_prob_sampled_actions
                    ).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach()
                    # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1,
                    #                                         eps=1e-6)
                    log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

                if self._cfg.learn.policy_loss_type == 'KL':
                    # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
                    # policy_loss = (torch.exp(log_prob_sampled_actions) * (log_prob_sampled_actions - target_log_prob_sampled_actions.detach())).sum(-1).mean(0)

                    # NOTE: accumulate policy loss!!! should be +=
                    policy_loss += (
                                           torch.exp(target_log_prob_sampled_actions.detach()) *
                                           (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                                   ).sum(-1) * mask_batch[:, step_i + 1]

                elif self._cfg.learn.policy_loss_type == 'cross_entropy':
                    # cross_entropy loss: - sum(p * log (q) )
                    # NOTE: accumulate policy loss!!! should be +=
                    policy_loss += -torch.sum(
                        torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
                    ) * mask_batch[:, step_i + 1]
                elif self._cfg.learn.policy_loss_type == 'original':
                    # TODO: debug the original policy loss
                    # calculate loss for the first step
                    policy_loss += modified_cross_entropy_loss(policy_logits, target_policy[:, step_i + 1])

            if self._cfg.model.categorical_distribution:
                value_loss += modified_cross_entropy_loss(value, target_value_phi[:, step_i + 1])
                value_prefix_loss += modified_cross_entropy_loss(value_prefix, target_value_prefix_phi[:, step_i])
            else:
                value_loss += torch.nn.MSELoss(reduction='none'
                                               )(value.squeeze(-1), transformed_target_value[:, step_i + 1])
                value_prefix_loss += torch.nn.MSELoss(
                    reduction='none'
                )(value_prefix.squeeze(-1), transformed_target_value_prefix[:, step_i])

            # reset hidden states
            if (step_i + 1) % self._cfg.lstm_horizon_len == 0:
                reward_hidden_state = (
                    torch.zeros(1, self._cfg.learn.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device),
                    torch.zeros(1, self._cfg.learn.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device)
                )

            if self._cfg.monitor_statistics:
                original_value_prefixs = self.inverse_scalar_transform_handle(value_prefix)
                original_value_prefixs_cpu = original_value_prefixs.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, self.inverse_scalar_transform_handle(value).detach().cpu())
                )
                predicted_value_prefixs.append(original_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                extra_statistics[key] = metric_loss(original_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    extra_statistics[key + '_1'] = metric_loss(
                        original_value_prefixs_cpu[value_prefix_indices_1],
                        target_value_prefix_base[value_prefix_indices_1]
                    )
                if value_prefix_indices_n1.any():
                    extra_statistics[key + '_-1'] = metric_loss(
                        original_value_prefixs_cpu[value_prefix_indices_n1],
                        target_value_prefix_base[value_prefix_indices_n1]
                    )
                if value_prefix_indices_0.any():
                    extra_statistics[key + '_0'] = metric_loss(
                        original_value_prefixs_cpu[value_prefix_indices_0],
                        target_value_prefix_base[value_prefix_indices_0]
                    )

        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
                self._cfg.ssl_loss_weight * consistency_loss + self._cfg.policy_loss_weight * policy_loss +
                self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * value_prefix_loss +
                self._cfg.policy_entropy_loss_weight * policy_entropy_loss
        )
        weighted_total_loss = (weights * loss).mean()

        # backward
        parameters = self._learn_model.parameters()

        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()

        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(parameters, self._cfg.learn.grad_clip_value)
        self._optimizer.step()
        if self._cfg.learn.cos_lr_scheduler is True:
            self.rl_scheduler.step()

        # ==============================================================
        # target model update
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())

        loss_data = (
            weighted_total_loss.item(), loss.mean().item(), policy_loss.mean().item(),
            value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean()
        )
        if self._cfg.monitor_statistics:

            # reward l1 loss
            value_prefix_indices_0 = (
                    target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0
            )
            value_prefix_indices_n1 = (
                    target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1
            )
            value_prefix_indices_1 = (
                    target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1
            )

            target_value_prefix_base = target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1)

            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
            extra_statistics['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
            if value_prefix_indices_1.any():
                extra_statistics['l1_1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1]
                )
            if value_prefix_indices_n1.any():
                extra_statistics['l1_-1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1]
                )
            if value_prefix_indices_0.any():
                extra_statistics['l1_0'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0]
                )

            if self._cfg.model.categorical_distribution:
                td_data = (
                    value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                    transformed_target_value_prefix.detach().cpu().numpy(),
                    transformed_target_value.detach().cpu().numpy(), target_value_prefix_phi.detach().cpu().numpy(),
                    target_value_phi.detach().cpu().numpy(), predicted_value_prefixs.detach().cpu().numpy(),
                    predicted_values.detach().cpu().numpy(), target_policy.detach().cpu().numpy(),
                    predicted_policies.detach().cpu().numpy(), state_lst, extra_statistics, other_log, other_dist
                )
            else:
                td_data = (
                    value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                    transformed_target_value_prefix.detach().cpu().numpy(),
                    transformed_target_value.detach().cpu().numpy(), predicted_value_prefixs.detach().cpu().numpy(),
                    predicted_values.detach().cpu().numpy(), target_policy.detach().cpu().numpy(),
                    predicted_policies.detach().cpu().numpy(), state_lst, extra_statistics, other_log, other_dist
                )

        if self._cfg.model.categorical_distribution:
            if self._cfg.model.continuous_action_space:
                return {
                    'cur_lr': self._optimizer.param_groups[0]['lr'],
                    'weighted_total_loss': loss_data[0],
                    'total_loss': loss_data[1],
                    'policy_loss': loss_data[2],
                    'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'value_prefix_loss': loss_data[3],
                    'value_loss': loss_data[4],
                    'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                    'value_priority': td_data[0].flatten().mean().item(),
                    # ==============================================================
                    # priority related
                    'value_priority_orig': value_priority,
                    # ==============================================================
                    'target_value_prefix': td_data[1].flatten().mean().item(),
                    'target_value': td_data[2].flatten().mean().item(),
                    'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                    'transformed_target_value': td_data[4].flatten().mean().item(),
                    'predicted_value_prefixs': td_data[7].flatten().mean().item(),
                    'predicted_values': td_data[8].flatten().mean().item(),

                    ######################
                    # sampled related code
                    ######################
                    'policy_mu_max': mu[:, 0].max().item(),
                    'policy_mu_min': mu[:, 0].min().item(),
                    'policy_mu_mean': mu[:, 0].mean().item(),
                    'policy_sigma_max': sigma.max().item(),
                    'policy_sigma_min': sigma.min().item(),
                    'policy_sigma_mean': sigma.mean().item(),
                    # take the fist dim in action space
                    'target_sampled_actions_max': target_sampled_actions[:, :, 0].max().item(),
                    'target_sampled_actions_min': target_sampled_actions[:, :, 0].min().item(),
                    'target_sampled_actions_mean': target_sampled_actions[:, :, 0].mean().item(),

                    # 'target_policy':td_data[9],
                    # 'predicted_policies':td_data[10]
                    'total_grad_norm_before_clip': total_grad_norm_before_clip

                }
            else:
                return {
                    'cur_lr': self._optimizer.param_groups[0]['lr'],
                    'weighted_total_loss': loss_data[0],
                    'total_loss': loss_data[1],
                    'policy_loss': loss_data[2],
                    'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'value_prefix_loss': loss_data[3],
                    'value_loss': loss_data[4],
                    'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                    'value_priority': td_data[0].flatten().mean().item(),
                    # ==============================================================
                    # priority related
                    'value_priority_orig': value_priority,
                    # ==============================================================
                    'target_value_prefix': td_data[1].flatten().mean().item(),
                    'target_value': td_data[2].flatten().mean().item(),
                    'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                    'transformed_target_value': td_data[4].flatten().mean().item(),
                    'predicted_value_prefixs': td_data[7].flatten().mean().item(),
                    'predicted_values': td_data[8].flatten().mean().item(),

                    # ==============================================================
                    # sampled related code
                    # ==============================================================
                    # take the fist dim in action space
                    'target_sampled_actions_max': target_sampled_actions[:, :].float().max().item(),
                    'target_sampled_actions_min': target_sampled_actions[:, :].float().min().item(),
                    'target_sampled_actions_mean': target_sampled_actions[:, :].float().mean().item(),
                    # 'target_policy':td_data[9],
                    # 'predicted_policies':td_data[10]
                    'total_grad_norm_before_clip': total_grad_norm_before_clip
                }
        else:
            if self._cfg.model.continuous_action_space:
                return {
                    'collect_mcts_temperature': self.collect_mcts_temperature,
                    'cur_lr': self._optimizer.param_groups[0]['lr'],
                    'weighted_total_loss': loss_data[0],
                    'total_loss': loss_data[1],
                    'policy_loss': loss_data[2],
                    'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'value_prefix_loss': loss_data[3],
                    'value_loss': loss_data[4],
                    'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                    'value_priority': td_data[0].flatten().mean().item(),
                    'target_value_prefix': td_data[1].flatten().mean().item(),
                    'target_value': td_data[2].flatten().mean().item(),
                    'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                    'transformed_target_value': td_data[4].flatten().mean().item(),
                    'predicted_value_prefixs': td_data[5].flatten().mean().item(),
                    'predicted_values': td_data[6].flatten().mean().item(),

                    ######################
                    # sampled related code
                    ######################
                    'policy_mu_max': mu[:, 0].max().item(),
                    'policy_mu_min': mu[:, 0].min().item(),
                    'policy_mu_mean': mu[:, 0].mean().item(),
                    'policy_sigma_max': sigma.max().item(),
                    'policy_sigma_min': sigma.min().item(),
                    'policy_sigma_mean': sigma.mean().item(),
                    # take the fist dim in action space
                    'target_sampled_actions_max': target_sampled_actions[:, :, 0].max().item(),
                    'target_sampled_actions_min': target_sampled_actions[:, :, 0].min().item(),
                    'target_sampled_actions_mean': target_sampled_actions[:, :, 0].mean().item(),

                    # 'target_policy':td_data[9],
                    # 'predicted_policies':td_data[10]
                    # 'td_data': td_data,
                    # 'priority_data_weights': priority_data[0],
                    # 'priority_data_indices': priority_data[1]
                    'total_grad_norm_before_clip': total_grad_norm_before_clip

                }
            else:
                return {
                    'collect_mcts_temperature': self.collect_mcts_temperature,
                    'cur_lr': self._optimizer.param_groups[0]['lr'],
                    'weighted_total_loss': loss_data[0],
                    'total_loss': loss_data[1],
                    'policy_loss': loss_data[2],
                    'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                    'value_prefix_loss': loss_data[3],
                    'value_loss': loss_data[4],
                    'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                    'value_priority': td_data[0].flatten().mean().item(),
                    'target_value_prefix': td_data[1].flatten().mean().item(),
                    'target_value': td_data[2].flatten().mean().item(),
                    'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                    'transformed_target_value': td_data[4].flatten().mean().item(),
                    'predicted_value_prefixs': td_data[5].flatten().mean().item(),
                    'predicted_values': td_data[6].flatten().mean().item(),

                    ######################
                    # sampled related code
                    ######################
                    # take the fist dim in action space
                    'target_sampled_actions_max': target_sampled_actions[:, :].float().max().item(),
                    'target_sampled_actions_min': target_sampled_actions[:, :].float().min().item(),
                    'target_sampled_actions_mean': target_sampled_actions[:, :].float().mean().item(),
                    'total_grad_norm_before_clip': total_grad_norm_before_clip
                }

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        # self._collect_model = model_wrap(self._model, 'base')
        self._collect_model = self._learn_model
        self._collect_model.reset()
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)

    # @profile
    def _forward_collect(
            self, data: ttorch.Tensor, action_mask: list = None, temperature: list = None, to_play=None,
            ready_env_id=None
    ):
        """
        Shapes:
            obs: (B, S, C, H, W), where S is the stack num
            temperature: (N1, ), where N1 is the number of collect_env.
        """
        self.collect_mcts_temperature = temperature[0]
        self._collect_model.eval()
        stack_obs = data
        active_collect_env_num = stack_obs.shape[0]
        with torch.no_grad():
            network_output = self._collect_model.initial_inference(stack_obs)
            hidden_state_roots = network_output.hidden_state  # （2, 64, 6, 6）
            pred_values_pool = network_output.value  # {list: 2}
            policy_logits_pool = network_output.policy_logits  # {list: 2} {list:6}
            value_prefix_pool = network_output.value_prefix  # {list: 2}
            reward_hidden_roots = network_output.reward_hidden_state  # {tuple:2} (1,2,512)

            # TODO(pu)
            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values_pool = self.inverse_scalar_transform_handle(pred_values_pool).detach().cpu().numpy()
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

            # TODO(pu): for board games, when action_num is a list, adapt the Roots method
            # cpp mcts_tree
            if self._cfg.mcts_ctree:
                ######################
                # sampled related code
                ######################
                if to_play[0] is None:
                    # we use to_play=0 means play_with_bot_mode game in mcts_ctree
                    to_play = [0 for i in range(active_collect_env_num)]
                if action_mask[0] is None:
                    # continuous action space env: all -1
                    legal_actions = [
                        [-1 for i in range(self._cfg.model.num_of_sampled_actions)]
                        for _ in range(active_collect_env_num)
                    ]
                else:
                    legal_actions = [
                        [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                    ]
                roots = ctree.Roots(
                    active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                ######################
                # sampled related code
                ######################
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a policy (argmax in testing)
                self._mcts_collect.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots, to_play)
            else:
                # python mcts_tree
                ######################
                # sampled related code
                ######################
                if to_play[0] is None:
                    # we use to_play=None means play_with_bot_mode game in mcts_ptree
                    to_play = [None for i in range(active_collect_env_num)]
                if action_mask[0] is None:
                    # continuous action space
                    roots = ptree.Roots(
                        active_collect_env_num,
                        None,
                        action_space_size=self._cfg.model.action_space_size,
                        num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                        continuous_action_space=self._cfg.model.continuous_action_space
                    )
                    # the only difference between collect and eval is the dirichlet noise
                    # TODO(pu):  int(self._cfg.model.action_space_size)
                    noises = [
                        np.random.dirichlet(
                            [self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                    ]
                else:
                    legal_actions = [
                        [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                    ]
                    roots = ptree.Roots(
                        active_collect_env_num,
                        legal_actions,
                        action_space_size=self._cfg.model.action_space_size,
                        num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                        continuous_action_space=self._cfg.model.continuous_action_space
                    )
                    # the only difference between collect and eval is the dirichlet noise
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                            ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                    ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a policy (argmax in testing)
                self._mcts_collect.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots, to_play)

            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            roots_values = roots.get_values()  # {list: 1}
            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}

            # TODO
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception as error:
                    # print(error)
                    # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])

                # if root_sampled_actions.max() > 1 or root_sampled_actions.min() < -1:
                #     print('here')

                # select the argmax, not sampling
                # TODO(pu):
                # only legal actions have visit counts
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=temperature[i], deterministic=False
                )
                # action, _ = select_action(distributions, temperature=1, deterministic=True)
                if action_mask[0] is not None and not self._cfg.model.continuous_action_space:
                    # only discrete action space have action mask
                    try:
                        action = roots_sampled_actions[i][action].value
                        # print('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # print(error)
                        # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])

                    # TODO(pu): transform to the real action index in legal action set
                    # sampled action is in legal actions
                    # action = np.where(action_mask[i] == 1.0)[0][int(action)]
                else:
                    try:
                        action = roots_sampled_actions[i][action].value
                        # print('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # print(error)
                        # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values_pool[i],
                    'policy_logits': policy_logits_pool[i],
                }
                # print('collect:', output[i])

        return output

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        # self._eval_model = self._learn_model
        # TODO(pu)
        self._eval_model = model_wrap(self._model, wrapper_name='base')

        self._eval_model.reset()
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)

    # @profile
    def _forward_eval(self, data: ttorch.Tensor, action_mask: list, to_play: None, ready_env_id=None):
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        """
        self._eval_model.eval()
        stack_obs = data
        active_eval_env_num = stack_obs.shape[0]
        with torch.no_grad():
            # stack_obs shape [B, S x C, W, H] e.g. {Tensor:(B,12,96,96)}
            network_output = self._eval_model.initial_inference(stack_obs)
            hidden_state_roots = network_output.hidden_state  # for atari, shape（B, 64, 6, 6）
            pred_values_pool = network_output.value  # for atari, shape（B, 601）
            reward_hidden_roots = network_output.reward_hidden_state  # {tuple:2} each element (1,2,512)
            value_prefix_pool = network_output.value_prefix  # shape（B, 1）
            policy_logits_pool = network_output.policy_logits  # shape（B, A）

            # TODO(pu)
            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values_pool = self.inverse_scalar_transform_handle(pred_values_pool).detach().cpu().numpy(
                )  # shape（B, 1）
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()  # list shape（B, A）

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                ######################
                # sampled related code
                ######################
                if to_play[0] is None:
                    # we use to_play=0 means play_with_bot_mode game in mcts_ctree
                    to_play = [0 for i in range(active_eval_env_num)]
                if action_mask[0] is None:
                    # continuous action space env: all -1
                    legal_actions = [
                        [-1 for i in range(self._cfg.model.num_of_sampled_actions)] for _ in range(active_eval_env_num)
                    ]
                else:
                    legal_actions = [
                        [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                    ]
                roots = ctree.Roots(
                    active_eval_env_num,
                    legal_actions,
                    self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions,
                    continuous_action_space=self._cfg.model.continuous_action_space
                )

                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                # do MCTS for a policy (argmax in testing)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)
            else:
                # python mcts_tree
                ######################
                # sampled related code
                ######################
                if to_play[0] is None:
                    # we use to_play=None means play_with_bot_mode game
                    to_play = [None for i in range(active_eval_env_num)]
                if action_mask[0] is None:
                    # continuous action space
                    roots = ptree.Roots(
                        active_eval_env_num,
                        None,
                        action_space_size=self._cfg.model.action_space_size,
                        num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                        continuous_action_space=self._cfg.model.continuous_action_space
                    )
                    # the only difference between collect and eval is the dirichlet noise
                else:
                    legal_actions = [
                        [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                    ]
                    roots = ptree.Roots(
                        active_eval_env_num,
                        legal_actions,
                        action_space_size=self._cfg.model.action_space_size,
                        num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                        continuous_action_space=self._cfg.model.continuous_action_space
                    )
                    # the only difference between collect and eval is the dirichlet noise

                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                # do MCTS for a policy (argmax in testing)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)

            # root visit count
            roots_distributions = roots.get_distributions()  # {list: 1} each element {list:6}
            ######################
            # sampled related code
            ######################
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            roots_values = roots.get_values()  # {list: 1}
            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception as error:
                    # print(error)
                    # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])

                # select the argmax, not sampling
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                ######################
                # sampled related code
                ######################
                if action_mask[0] is not None and not self._cfg.model.continuous_action_space:
                    # only discrete action space have action mask
                    try:
                        action = roots_sampled_actions[i][action].value
                        # print('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # print(error)
                        # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])
                    # TODO(pu): transform to the real action index in legal action set
                    # sampled action is in legal actions
                    # action = np.where(action_mask[i] == 1.0)[0][int(action)]
                else:
                    try:
                        action = roots_sampled_actions[i][action].value
                        # print('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # print(error)
                        # print('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values_pool[i],
                    'policy_logits': policy_logits_pool[i],
                }

        return output

    def _monitor_vars_learn(self) -> List[str]:
        if self._cfg.model.continuous_action_space:
            return [
                'collect_mcts_temperature',
                'cur_lr',
                'total_loss',
                'weighted_total_loss',
                'policy_loss',
                'value_prefix_loss',
                'value_loss',
                'consistency_loss',
                'value_priority',
                'target_value_prefix',
                'target_value',
                'predicted_value_prefixs',
                'predicted_values',
                'transformed_target_value_prefix',
                'transformed_target_value',

                ######################
                # sampled related code
                ######################
                'policy_entropy',
                'target_policy_entropy',
                'policy_mu_max',
                'policy_mu_min',
                'policy_mu_mean',
                'policy_sigma_max',
                'policy_sigma_min',
                'policy_sigma_mean',
                # take the fist dim in action space
                'target_sampled_actions_max',
                'target_sampled_actions_min',
                'target_sampled_actions_mean',
                # 'visit_count_distribution_entropy',
                # 'target_policy',
                # 'predicted_policies'
                'total_grad_norm_before_clip',
            ]
        else:
            return [
                'collect_mcts_temperature',
                'cur_lr',
                'total_loss',
                'weighted_total_loss',
                'loss_mean',
                'policy_loss',
                'value_prefix_loss',
                'value_loss',
                'consistency_loss',
                'value_priority',
                'target_value_prefix',
                'target_value',
                'predicted_value_prefixs',
                'predicted_values',
                'transformed_target_value_prefix',
                'transformed_target_value',

                ######################
                # sampled related code
                ######################
                'policy_entropy',
                'target_policy_entropy',

                # take the fist dim in action space
                'target_sampled_actions_max',
                'target_sampled_actions_min',
                'target_sampled_actions_mean',

                # 'visit_count_distribution_entropy',
                # 'target_policy',
                # 'predicted_policies'
                'total_grad_norm_before_clip',
            ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
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
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _process_transition(
            self, obs: ttorch.Tensor, policy_output: ttorch.Tensor, timestep: ttorch.Tensor
    ) -> ttorch.Tensor:
        return ttorch.as_tensor(
            {
                'obs': obs,
                'action': policy_output.action,
                'distribution': policy_output.distribution,
                'value': policy_output.value,
                'next_obs': timestep.obs,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        )

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _data_preprocess_learn(self, data: ttorch.Tensor):
        # TODO data augmentation before learning
        data = data.cuda(self._cfg.device)
        data = ttorch.stack(data)
        return data

    @staticmethod
    def _consist_loss_func(f1, f2):
        """
        Consistency loss function: similarity loss
        """
        f1 = F.normalize(f1, p=2., dim=-1, eps=1e-6)
        f2 = F.normalize(f2, p=2., dim=-1, eps=1e-6)
        return -(f1 * f2).sum(dim=1)

    @staticmethod
    def _get_max_entropy(action_shape: int) -> None:
        p = 1.0 / action_shape
        return -action_shape * p * np.log2(p)
