import copy
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import wandb
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY

from lzero.entry.utils import initialize_zeros_batch
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, prepare_obs, \
    prepare_obs_stack4_for_unizero
from lzero.policy.muzero import MuZeroPolicy
from .utils import configure_optimizers_nanogpt


@POLICY_REGISTRY.register('unizero')
class UniZeroPolicy(MuZeroPolicy):
    """
    Overview:
        The policy class for UniZero, official implementation for paper UniZero: Generalized and Efficient Planning
        with Scalable LatentWorld Models. UniZero aims to enhance the planning capabilities of reinforcement learning agents
        by addressing the limitations found in MuZero-style algorithms, particularly in environments requiring the
        capture of long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    """

    # The default_config for UniZero policy.
    config = dict(
        type='unizero',
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The obs shape.
            observation_shape=(3, 64, 64),
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=True,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=3,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=50,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'BN'.
            norm_type='BN',
            # (bool) Whether to analyze simulation normalization.
            analysis_sim_norm=False,
            # (int) The save interval of the model.
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=10000, ), ), ),
            world_model_cfg=dict(
                # (bool) If True, the action space of the environment is continuous, otherwise discrete.
                continuous_action_space=False,
                # (int) The number of tokens per block.
                tokens_per_block=2,
                # (int) The maximum number of blocks.
                max_blocks=10,
                # (int) The maximum number of tokens, calculated as tokens per block multiplied by max blocks.
                max_tokens=2 * 10,
                # (int) The context length, usually calculated as twice the number of some base unit.
                context_length=2 * 4,
                # (bool) Whether to use GRU gating mechanism.
                gru_gating=False,
                # (str) The device to be used for computation, e.g., 'cpu' or 'cuda'.
                device='cpu',
                # (bool) Whether to analyze simulation normalization.
                analysis_sim_norm=False,
                # (bool) Whether to analyze dormant ratio.
                analysis_dormant_ratio=False,
                # (int) The shape of the action space.
                action_space_size=6,
                # (int) The size of the group, related to simulation normalization.
                group_size=8,  # NOTE: sim_norm
                # (str) The type of attention mechanism used. Options could be ['causal'].
                attention='causal',
                # (int) The number of layers in the model.
                num_layers=2,
                # (int) The number of attention heads.
                num_heads=8,
                # (int) The dimension of the embedding.
                embed_dim=768,
                # (float) The dropout probability for the embedding layer.
                embed_pdrop=0.1,
                # (float) The dropout probability for the residual connections.
                resid_pdrop=0.1,
                # (float) The dropout probability for the attention mechanism.
                attn_pdrop=0.1,
                # (int) The size of the support set for value and reward heads.
                support_size=101,
                # (int) The maximum size of the cache.
                max_cache_size=5000,
                # (int) The number of environments.
                env_num=8,
                # (float) The weight of the latent reconstruction loss.
                latent_recon_loss_weight=0.,
                # (float) The weight of the perceptual loss.
                perceptual_loss_weight=0.,
                # (float) The weight of the policy entropy loss.
                policy_entropy_weight=0,
                # (str) The type of loss for predicting latent variables. Options could be ['group_kl', 'mse'].
                predict_latent_loss_type='group_kl',
                # (str) The type of observation. Options are ['image', 'vector'].
                obs_type='image',
                # (float) The discount factor for future rewards.
                gamma=1,
                # (float) The threshold for a dormant neuron.
                dormant_threshold=0.025,
            ),
        ),
        # ****** common ******
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
        game_segment_length=400,
        # (bool) Whether to analyze simulation normalization.
        analysis_sim_norm=False,
        # (bool) Whether to use the pure policy to collect data.
        collect_with_pure_policy=False,
        # (int) The evaluation frequency.
        eval_freq=int(2e3),
        # (str) The sample type. Options are ['episode', 'transition'].
        sample_type='transition',
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
        # (str) Optimizer for training policy network.
        optim_type='AdamW',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.0001,
        # (int) Frequency of hard target network update.
        target_update_freq=100,
        # (int) Frequency of soft target network update.
        target_update_theta=0.05,
        # (int) Frequency of target network update.
        target_update_freq_for_intrinsic_reward=1000,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=20,
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
        num_unroll_steps=10,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        piecewise_decay_lr_scheduler=False,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(5e4),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,
        # (int) The initial Env Steps for training.
        train_start_after_envsteps=int(0),

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
            by import_names path. For MuZero, ``lzero.model.unizero_model.MuZeroModel``
        """
        return 'UniZeroModel', ['lzero.model.unizero_model']

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        # NOTE: nanoGPT optimizer
        self._optimizer_world_model = configure_optimizers_nanogpt(
            model=self._model.world_model,
            learning_rate=self._cfg.learning_rate,
            weight_decay=self._cfg.weight_decay,
            device_type=self._cfg.device,
            betas=(0.9, 0.95),
        )

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        # Ensure that the installed torch version is greater than or equal to 2.0
        assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "We need torch version >= 2.0"
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)
        # NOTE: soft target
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.target_update_theta}
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
        self.intermediate_losses = defaultdict(float)
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        
        if self._cfg.use_wandb:
            # TODO: add the model to wandb
            wandb.watch(self._learn_model.representation_network, log="all")

    # @profile
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

        current_batch, target_batch, _ = data
        obs_batch_ori, action_batch,  target_action_batch, mask_batch, indices, weights, make_time = current_batch
        target_reward, target_value, target_policy = target_batch

        # Prepare observations based on frame stack number
        if self._cfg.model.frame_stack_num == 4:
            obs_batch, obs_target_batch = prepare_obs_stack4_for_unizero(obs_batch_ori, self._cfg)
        else:
            obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)  # TODO: optimize

        # Apply augmentations if needed
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if self._cfg.model.self_supervised_learning_loss:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # Prepare action batch and convert to torch tensor
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(
            -1).long()  # For discrete action space
        data_list = [mask_batch, target_reward, target_value, target_policy, weights]
        mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list,
                                                                                                self._cfg.device)
        target_reward = target_reward.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        # Transform rewards and values to their scaled forms
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # Convert to categorical distributions
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        # print(f'transformed_target_value:{transformed_target_value}')
        # print("self.value_support:", self.value_support)
        
        try:
            target_value_categorical = phi_transform(self.value_support, transformed_target_value)
        except Exception as e:
            print('='*20)
            print(e)
            # print(f'transformed_target_value:{transformed_target_value}')
            # print("self.value_support:", self.value_support)
            print('='*20)
            # target_value_categorical = phi_transform(self.value_support, transformed_target_value)


        # Prepare batch for GPT model
        batch_for_gpt = {}
        if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape) == 1:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, self._cfg.model.observation_shape)
        elif len(self._cfg.model.observation_shape) == 3:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, *self._cfg.model.observation_shape)

        batch_for_gpt['actions'] = action_batch.squeeze(-1)
        batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
        batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data
        batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
        batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
        batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long,
                                            device=self._cfg.device)
        batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
        batch_for_gpt['target_policy'] = target_policy[:, :-1]

        # Extract valid target policy data and compute entropy
        try:
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
            average_target_policy_entropy = target_policy_entropy.mean()
        except Exception as e:
            print('='*20)
            print(e)
            average_target_policy_entropy = 0.


        # Update world model
        losses = self._learn_model.world_model.compute_loss(
            batch_for_gpt, self._target_model.world_model.tokenizer, self.inverse_scalar_transform_handle
        )

        weighted_total_loss = losses.loss_total
        for loss_name, loss_value in losses.intermediate_losses.items():
            self.intermediate_losses[f"{loss_name}"] = loss_value

        obs_loss = self.intermediate_losses['loss_obs']
        reward_loss = self.intermediate_losses['loss_rewards']
        policy_loss = self.intermediate_losses['loss_policy']
        value_loss = self.intermediate_losses['loss_value']
        latent_recon_loss = self.intermediate_losses['latent_recon_loss']
        perceptual_loss = self.intermediate_losses['perceptual_loss']
        orig_policy_loss = self.intermediate_losses['orig_policy_loss']
        policy_entropy = self.intermediate_losses['policy_entropy']
        first_step_losses = self.intermediate_losses['first_step_losses']
        middle_step_losses = self.intermediate_losses['middle_step_losses']
        last_step_losses = self.intermediate_losses['last_step_losses']
        dormant_ratio_encoder = self.intermediate_losses['dormant_ratio_encoder']
        dormant_ratio_world_model = self.intermediate_losses['dormant_ratio_world_model']
        latent_state_l2_norms = self.intermediate_losses['latent_state_l2_norms']

        # assert not torch.isnan(losses.loss_total).any(), "Loss contains NaN values"
        # assert not torch.isinf(losses.loss_total).any(), "Loss contains Inf values"

        # Core learn model update step
        self._optimizer_world_model.zero_grad()
        weighted_total_loss.backward()

        #  ========== for debugging ==========
        # for name, param in self._learn_model.world_model.tokenizer.encoder.named_parameters():
        #     print('name, param.mean(), param.std():', name, param.mean(), param.std())
        #     if param.requires_grad:
        #         print(name, param.grad.norm())

        if self._cfg.analysis_sim_norm:
            del self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after
            self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
            self._target_model.encoder_hook.clear_data()

        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(self._learn_model.world_model.parameters(),
                                                                        self._cfg.grad_clip_value)
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)

        self._optimizer_world_model.step()
        if self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Core target model update step
        self._target_model.update(self._learn_model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        else:
            current_memory_allocated_gb = 0.
            max_memory_allocated_gb = 0.

        return_log_dict = {
            'analysis/first_step_loss_value': first_step_losses['loss_value'].item(),
            'analysis/first_step_loss_policy': first_step_losses['loss_policy'].item(),
            'analysis/first_step_loss_rewards': first_step_losses['loss_rewards'].item(),
            'analysis/first_step_loss_obs': first_step_losses['loss_obs'].item(),

            'analysis/middle_step_loss_value': middle_step_losses['loss_value'].item(),
            'analysis/middle_step_loss_policy': middle_step_losses['loss_policy'].item(),
            'analysis/middle_step_loss_rewards': middle_step_losses['loss_rewards'].item(),
            'analysis/middle_step_loss_obs': middle_step_losses['loss_obs'].item(),

            'analysis/last_step_loss_value': last_step_losses['loss_value'].item(),
            'analysis/last_step_loss_policy': last_step_losses['loss_policy'].item(),
            'analysis/last_step_loss_rewards': last_step_losses['loss_rewards'].item(),
            'analysis/last_step_loss_obs': last_step_losses['loss_obs'].item(),

            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self._collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'obs_loss': obs_loss.item(),
            'latent_recon_loss': latent_recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'policy_loss': policy_loss.item(),
            'orig_policy_loss': orig_policy_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'target_policy_entropy': average_target_policy_entropy.item(),
            'reward_loss': reward_loss.item(),
            'value_loss': value_loss.item(),
            # 'value_priority_orig': np.zeros(self._cfg.batch_size),  # TODO
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
            'analysis/dormant_ratio_encoder': dormant_ratio_encoder.item(),
            'analysis/dormant_ratio_world_model': dormant_ratio_world_model.item(),
            'analysis/latent_state_l2_norms': latent_state_l2_norms.item(),
            'analysis/l2_norm_before': self.l2_norm_before,
            'analysis/l2_norm_after': self.l2_norm_after,
            'analysis/grad_norm_before': self.grad_norm_before,
            'analysis/grad_norm_after': self.grad_norm_after,
        }
        
        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_log_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_log_dict

    def monitor_weights_and_grads(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | "
                      f"Weight mean: {param.data.mean():.4f} | "
                      f"Weight std: {param.data.std():.4f} | "
                      f"Grad mean: {param.grad.mean():.4f} | "
                      f"Grad std: {param.grad.std():.4f}")

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
        self._collect_epsilon = 0.0
        self.collector_env_num = self._cfg.collector_env_num
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(self.collector_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(self.collector_env_num)]

    # @profile
    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None
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
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()

        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
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
                    if np.random.rand() < self._collect_epsilon:
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

                # ============== TODO: only for visualize ==============
                # action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                #     distributions, temperature=self._collect_mcts_temperature, deterministic=True
                # )
                # action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                # ============== TODO: only for visualize ==============

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                batch_action.append(action)

            self.last_batch_obs = data
            self.last_batch_action = batch_action

            # ========= TODO: for muzero_segment_collector now =========
            if active_collect_env_num < self.collector_env_num:
                print('==========collect_forward============')
                print(f'len(self.last_batch_obs) < self.collector_env_num, {active_collect_env_num}<{self.collector_env_num}')
                self._reset_collect(reset_init_data=True)
                if getattr(self._cfg, 'sample_type', '') == 'episode':
                    print('BUG: sample_type is episode, but len(self.last_batch_obs) < self.collector_env_num')

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

        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None) -> Dict:
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
        with torch.no_grad():
            network_output = self._eval_model.initial_inference(self.last_batch_obs, self.last_batch_action, data)
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
                # print("roots_visit_count_distributions:", distributions, "root_value:", value)

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
                batch_action.append(action)

            self.last_batch_obs = data
            self.last_batch_action = batch_action

        return output

    def _reset_collect(self, env_id: int = None, current_steps: int = None, reset_init_data: bool = True) -> None:
        """
        Overview:
            This method resets the collection process for a specific environment. It clears caches and memory
            when certain conditions are met, ensuring optimal performance. If reset_init_data is True, the initial data
            will be reset.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. If None or list, the function returns immediately.
            - current_steps (:obj:`int`, optional): The current step count in the environment. Used to determine
              whether to clear caches.
            - reset_init_data (:obj:`bool`, optional): Whether to reset the initial data. If True, the initial data will be reset.
        """
        if reset_init_data:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

        # Return immediately if env_id is None or a list
        if env_id is None or isinstance(env_id, list):
            return

        # Determine the clear interval based on the environment's sample type
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200

        # Clear caches if the current steps are a multiple of the clear interval
        if current_steps % clear_interval == 0:
            print(f'clear_interval: {clear_interval}')

            # Clear various caches in the collect model's world model
            world_model = self._collect_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up GPU memory
            torch.cuda.empty_cache()

            print('collector: collect_model clear()')
            print(f'eps_steps_lst[{env_id}]: {current_steps}')

    def _reset_eval(self, env_id: int = None, current_steps: int = None, reset_init_data: bool = True) -> None:
        """
        Overview:
            This method resets the evaluation process for a specific environment. It clears caches and memory
            when certain conditions are met, ensuring optimal performance. If reset_init_data is True,
            the initial data will be reset.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. If None or list, the function returns immediately.
            - current_steps (:obj:`int`, optional): The current step count in the environment. Used to determine
              whether to clear caches.
            - reset_init_data (:obj:`bool`, optional): Whether to reset the initial data. If True, the initial data will be reset.
        """
        if reset_init_data:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.evaluator_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.evaluator_env_num)]

        # Return immediately if env_id is None or a list
        if env_id is None or isinstance(env_id, list):
            return

        # Determine the clear interval based on the environment's sample type
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200

        # Clear caches if the current steps are a multiple of the clear interval
        if current_steps % clear_interval == 0:
            print(f'clear_interval: {clear_interval}')

            # Clear various caches in the eval model's world model
            world_model = self._eval_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up GPU memory
            torch.cuda.empty_cache()

            print('evaluator: eval_model clear()')
            print(f'eps_steps_lst[{env_id}]: {current_steps}')

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return [
            'analysis/dormant_ratio_encoder',
            'analysis/dormant_ratio_world_model',
            'analysis/latent_state_l2_norms',
            'analysis/l2_norm_before',
            'analysis/l2_norm_after',
            'analysis/grad_norm_before',
            'analysis/grad_norm_after',

            'analysis/first_step_loss_value',
            'analysis/first_step_loss_policy',
            'analysis/first_step_loss_rewards',
            'analysis/first_step_loss_obs',

            'analysis/middle_step_loss_value',
            'analysis/middle_step_loss_policy',
            'analysis/middle_step_loss_rewards',
            'analysis/middle_step_loss_obs',

            'analysis/last_step_loss_value',
            'analysis/last_step_loss_policy',
            'analysis/last_step_loss_rewards',
            'analysis/last_step_loss_obs',

            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'cur_lr_tokenizer',

            'weighted_total_loss',
            'obs_loss',
            'policy_loss',
            'orig_policy_loss',
            'policy_entropy',
            'latent_recon_loss',
            'target_policy_entropy',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'total_grad_norm_before_clip_wm',
            # tokenizer
            'commitment_loss',
            'reconstruction_loss',
            'perceptual_loss',
        ]

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
            'optimizer_world_model': self._optimizer_world_model.state_dict(),
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
        self._optimizer_world_model.load_state_dict(state_dict['optimizer_world_model'])

    def recompute_pos_emb_diff_and_clear_cache(self) -> None:
        """
        Overview:
            Clear the caches and precompute positional embedding matrices in the model.
        """
        for model in [self._collect_model, self._target_model]:
            model.world_model.precompute_pos_emb_diff_kv()
            model.world_model.clear_caches()
        torch.cuda.empty_cache()