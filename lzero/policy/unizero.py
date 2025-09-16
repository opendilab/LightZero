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
    prepare_obs_stack_for_unizero
from lzero.policy.muzero import MuZeroPolicy
from .utils import configure_optimizers_nanogpt

def configure_optimizer_unizero(model, learning_rate, weight_decay, device_type, betas):
    """
    为UniZero模型配置带有差异化学习率的优化器。
    """
    # 1. 定义需要特殊处理的参数
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # 2. 将参数分为三组：Transformer主干、Tokenizer、Heads
    transformer_params = {pn: p for pn, p in param_dict.items() if 'transformer' in pn}
    tokenizer_params = {pn: p for pn, p in param_dict.items() if 'tokenizer' in pn}
    
    # Heads的参数是那些既不属于transformer也不属于tokenizer的
    head_params = {
        pn: p for pn, p in param_dict.items() 
        if 'transformer' not in pn and 'tokenizer' not in pn
    }

    # 3. 为每组设置不同的优化器参数（特别是学习率）
    #    这里我们仍然使用AdamW，但学习率设置更合理
    optim_groups = [
        {
            'params': list(transformer_params.values()),
            'lr': learning_rate * 0.1,  # 为Transformer主干设置一个较小的学习率，例如 1e-5
            'weight_decay': weight_decay
        },
        {
            'params': list(tokenizer_params.values()),
            'lr': learning_rate,  # Tokenizer使用基础学习率，例如 1e-4
            'weight_decay': weight_decay
        },
        {
            'params': list(head_params.values()),
            'lr': learning_rate,  # Heads也使用基础学习率
            'weight_decay': 0.0  # 通常Heads的权重不做衰减
        }
    ]

    print("--- Optimizer Groups ---")
    print(f"Transformer LR: {learning_rate * 0.1}")
    print(f"Tokenizer/Heads LR: {learning_rate}")
    
    optimizer = torch.optim.AdamW(optim_groups, betas=betas)
    return optimizer


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
            # (tuple) The range of supports used in categorical distribution.
            # These variables are only effective when ``model.categorical_distribution=True``.
            reward_support_range=(-50., 51., 1.),
            value_support_range=(-50., 51., 1.),
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
                final_norm_option_in_encoder="SimNorm", # "SimNorm"对应"group_kl",  "LayerNorm"对应"mse", 
                final_norm_option_in_obs_head="SimNorm",
                # (str) The type of loss for predicting latent variables. Options could be ['group_kl', 'mse'].
                predict_latent_loss_type='group_kl',
                # (str) The type of observation. Options are ['image', 'vector'].
                obs_type='image',
                # (float) The discount factor for future rewards.
                gamma=1,
                # (float) The threshold for a dormant neuron.
                dormant_threshold=0.025,
                # (bool) Whether to use Rotary Position Embedding (RoPE) for relative position encoding.
                # If False, nn.Embedding is used for absolute position encoding.
                # For more details on RoPE, refer to the author's blog: https://spaces.ac.cn/archives/8265/
                # TODO: If you want to use rotary_emb in an environment, you need to include the timestep as a return key from the environment.
                rotary_emb=False,
                # (int) The base value for calculating RoPE angles. Commonly set to 10000.
                rope_theta=10000,
                # (int) The maximum sequence length for position encoding.
                max_seq_len=8192,
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
        # # (int) the number of simulations in MCTS for renalyze.
        num_simulations=50,
        # (int) The number of simulations in MCTS for the collect phase.
        collect_num_simulations=25,
        # (int) The number of simulations in MCTS for the eval phase.
        eval_num_simulations=50,
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
        # (bool) Whether to use the cosine learning rate decay.
        cos_lr_scheduler=False,
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
        # (int) The number of steps to accumulate gradients before performing an optimization step.
        accumulation_steps=1,

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
        if self._cfg.optim_type == 'SGD':
            # --- 改为SGD优化器 ---
            self._optimizer_world_model = torch.optim.SGD(
                self._model.world_model.parameters(),
                lr=self._cfg.learning_rate,  # 初始学习率，在配置中设为 0.2
                momentum=self._cfg.momentum, # 在配置中设为 0.9
                weight_decay=self._cfg.weight_decay # 在配置中设为 1e-4
            )
        elif self._cfg.optim_type == 'AdamW':
            # NOTE: nanoGPT optimizer
            self._optimizer_world_model = configure_optimizers_nanogpt(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )
        elif self._cfg.optim_type == 'AdamW_mix_lr':
            self._optimizer_world_model = configure_optimizer_unizero(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,  # 使用一个合理的AdamW基础学习率
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )



        if self._cfg.cos_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # TODO: check the total training steps
            self.lr_scheduler = CosineAnnealingLR(self._optimizer_world_model, 1e5, eta_min=0, last_epoch=-1)

        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer_world_model, lr_lambda=lr_lambda)


        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        # Ensure that the installed torch version is greater than or equal to 2.0
        assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "We need torch version >= 2.0"
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)
        if self._cfg.target_model_update_option=="soft":
            # NOTE: soft target
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='momentum',
                update_kwargs={'theta': self._cfg.target_update_theta}
            )
        elif self._cfg.target_model_update_option=="hard":
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
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        # assert self.value_support.size == self._learn_model.value_support_size          # if these assertions fails, somebody introduced...
        # assert self.reward_support.size == self._learn_model.reward_support_size        # ...incoherence between policy and model
        self.value_inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)
        self.reward_inverse_scalar_transform_handle = InverseScalarTransform(self.reward_support, self._cfg.model.categorical_distribution)

        self.intermediate_losses = defaultdict(float)
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        
        if self._cfg.use_wandb:
            # TODO: add the model to wandb
            wandb.watch(self._learn_model.representation_network, log="all")

        self.accumulation_steps = self._cfg.accumulation_steps
        # 从配置中获取阈值，如果未设置则使用一个合理的默认值（例如20.0）
        # 设置为0或负数则禁用此功能
        self.latent_norm_clip_threshold = self._cfg.get('latent_norm_clip_threshold', 20.0) # TODO


        # 从配置中获取阈值，例如 15.0 或 20.0
        self.logit_clip_threshold = self._cfg.get('logit_clip_threshold', 10.0)
                # 1. 获取 world_model 的引用，方便后续操作
        world_model = self._learn_model.world_model
        # 2. 将参数明确地分为两组：预测头 (heads) 和 主干网络 (backbone)
        #    - a. 获取所有预测头的参数
        self.head_params = list(world_model.head_value.parameters()) + \
                    list(world_model.head_rewards.parameters()) + \
                    list(world_model.head_policy.parameters())
                    # 如果有其他头，也一并加入
        #    - b. 为了高效分离，我们使用参数的ID
        self.head_param_ids = {id(p) for p in self.head_params}
        #    - c. 获取主干网络的参数（所有不在 head_param_ids 中的参数）
        self.backbone_params = [p for p in world_model.parameters() if id(p) not in self.head_param_ids]

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

        current_batch, target_batch, train_iter = data
        obs_batch_ori, action_batch,  target_action_batch, mask_batch, indices, weights, make_time, timestep_batch = current_batch
        target_reward, target_value, target_policy = target_batch

        # Prepare observations based on frame stack number
        if self._cfg.model.frame_stack_num > 1:
            obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, self._cfg)
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
        timestep_batch = torch.from_numpy(timestep_batch).to(self._cfg.device).unsqueeze(
            -1).long()
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
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        # TODO
        # ==================== 核心修复：标签平滑 ====================
        # alpha 是平滑系数，一个小的超参数，例如 0.01 或 0.1
        # alpha = 0.1 
        # num_classes = target_value_categorical.shape[-1]
        # # (1 - alpha) * original_target + alpha / num_classes
        # target_value_categorical = target_value_categorical * (1 - alpha) + (alpha / num_classes)
        # target_reward_categorical = target_reward_categorical * (1 - alpha) + (alpha / num_classes)
        # =============================================================

        # Prepare batch for GPT model
        batch_for_gpt = {}
        if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape) == 1:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, self._cfg.model.observation_shape)
        elif len(self._cfg.model.observation_shape) == 3:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, *self._cfg.model.observation_shape)

        batch_for_gpt['actions'] = action_batch.squeeze(-1)
        batch_for_gpt['timestep'] = timestep_batch.squeeze(-1)

        batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
        batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data
        batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
        batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
        batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long,
                                            device=self._cfg.device)
        batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
        batch_for_gpt['target_policy'] = target_policy[:, :-1]

        # ==================== START MODIFICATION 1 ====================
        # Pass the original scalar target_value to compute_loss for priority calculation.
        batch_for_gpt['scalar_target_value'] = target_value
        # ===================== END MODIFICATION 1 =====================

        # Extract valid target policy data and compute entropy
        valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
        target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
        average_target_policy_entropy = target_policy_entropy.mean()

        # Update world model
        losses = self._learn_model.world_model.compute_loss(
            batch_for_gpt, self._target_model.world_model.tokenizer, self.value_inverse_scalar_transform_handle, global_step=train_iter
        )           # NOTE : compute_loss third argument is now a dead argument. If this changes, it could need adaptation between value_inverse and reward_inverse.

        # ==================== START MODIFICATION 2 ====================
        # Extract the calculated value_priority from the returned losses.
        value_priority_tensor = losses.intermediate_losses['value_priority']
        # Convert to numpy array for the replay buffer, adding a small epsilon.
        value_priority_np = value_priority_tensor.detach().cpu().numpy() + 1e-6
        # ===================== END MODIFICATION 2 =====================

        # weighted_total_loss = losses.loss_total
        # TODO:
        weighted_total_loss = (weights * losses.loss_total).mean()

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
        latent_action_l2_norms = self.intermediate_losses['latent_action_l2_norms']


        logits_value_mean=self.intermediate_losses['logits_value_mean']
        logits_value_max=self.intermediate_losses['logits_value_max']
        logits_value_min=self.intermediate_losses['logits_value_min']

        logits_policy_mean=self.intermediate_losses['logits_policy_mean']
        logits_policy_max=self.intermediate_losses['logits_policy_max']
        logits_policy_min=self.intermediate_losses['logits_policy_min']

        assert not torch.isnan(losses.loss_total).any(), "Loss contains NaN values"
        assert not torch.isinf(losses.loss_total).any(), "Loss contains Inf values"

        # Core learning model update step
        # Reset gradients at the start of each accumulation cycle
        if (train_iter % self.accumulation_steps) == 0:
            self._optimizer_world_model.zero_grad()

        # Scale the loss by the number of accumulation steps
        weighted_total_loss = weighted_total_loss / self.accumulation_steps

        if self._cfg.gradient_scale:
            # ==============================================================
            # START OF THE FIX: Add gradient scaling just like in MuZero
            # ==============================================================
            # This is the key to stabilizing the latent norm. It averages the gradients
            # accumulated over the unroll steps, preventing the exploding gradient problem
            # in the recurrent world model (Transformer).
            gradient_scale = 1.0 / self._cfg.num_unroll_steps
            weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
            # ==============================================================
            # END OF THE FIX
            # ==============================================================


        weighted_total_loss.backward()

        # # ======================= 学习率真实性检查 =======================
        # if (train_iter % 1000) == 0:
        #     print("\n--- Optimizer Learning Rate Analysis ---")
        #     # self._optimizer_world_model 是唯一的优化器
        #     for i, param_group in enumerate(self._optimizer_world_model.param_groups):
        #         # configure_optimizers_nanogpt 可能会创建多个参数组（例如，一个用于带权重衰减的参数，一个用于不带的）
        #         print(f"  Param Group {i}: LR = {param_group['lr']:.6f}")
        # # =================================================================

        # ======================= 梯度检查代码 =======================
        # 我们可以只关注 Encoder 的梯度
        encoder = self._learn_model.world_model.tokenizer.encoder
        total_grad_norm = 0.0

        # if (train_iter % 5000) == 0:
        if (train_iter % 10000) == 0: # 10k
        # if (train_iter % 1) == 0:
            print(f"\n--- Gradient Analysis for Step {train_iter} ---")
            for name, param in encoder.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    
                    # 打印每一层的梯度范数，以定位问题层
                    print(f"  Layer: {name} | Grad Norm: {grad_norm:.6f}")
                else:
                    print(f"  Layer: {name} | Grad is None")
                    
            total_grad_norm = total_grad_norm ** 0.5
            print(f"--- Total Grad Norm for Encoder: {total_grad_norm:.6f} ---\n")
        # =============================================================
    

        # Check if the current iteration completes an accumulation cycle
        if (train_iter + 1) % self.accumulation_steps == 0:
            # Analyze gradient norms if simulation normalization analysis is enabled
            if self._cfg.analysis_sim_norm:
                # Clear previous analysis results to prevent memory overflow
                del self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after
                self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
                self._target_model.encoder_hook.clear_data()
            
            # Clip gradients to prevent exploding gradients
            # total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(
            #     self._learn_model.world_model.parameters(), self._cfg.grad_clip_value
            # )
            total_grad_norm_before_clip_wm = torch.tensor(0.)


            # 3. 对两组参数分别进行梯度裁剪
            #    - a. 对预测头应用一个更严格（更小）的裁剪阈值
            #      您需要在配置文件中新增 `head_grad_clip_value`，例如设置为 1.0 或 0.5
            head_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.head_params, self._cfg.get('head_grad_clip_value', 1.0)  # 示例：严格的阈值
            )
            #    - b. 对主干网络应用一个相对宽松的裁剪阈值
            #      您可以在配置文件中新增 `backbone_grad_clip_value`，例如设置为 10.0
            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backbone_params, self._cfg.get('backbone_grad_clip_value', 10.0) # 示例：标准的阈值
            )


            # Synchronize gradients across multiple GPUs if enabled
            if self._cfg.multi_gpu:
                self.sync_gradients(self._learn_model)

            # Update model parameters
            self._optimizer_world_model.step()

            # Clear CUDA cache if using gradient accumulation
            if self.accumulation_steps > 1:
                torch.cuda.empty_cache()
        else:
            total_grad_norm_before_clip_wm = torch.tensor(0.)

        # =================================================================
        #              Encoder-Clip: Inspired by QK-Clip
        # -----------------------------------------------------------------
        # 直接控制Encoder输出的范数，防止其无界增长，以稳定训练。
        # =================================================================
        if self.latent_norm_clip_threshold > 0 and 'obs_embeddings' in losses.intermediate_losses:
            with torch.no_grad():
                # 1. 从loss字典中获取已分离的encoder输出
                obs_embeddings = losses.intermediate_losses['obs_embeddings']
                if obs_embeddings is None:
                    raise ValueError

                # 2. 计算这批数据中，encoder输出L2范数的最大值
                # obs_embeddings 的形状通常是 (B*L, 1, E) 或 (B*L, E)
                # 我们在最后一个维度（embedding_dim）上计算范数
                latent_norms = obs_embeddings.norm(p=2, dim=-1)
                max_latent_norm = latent_norms.max()

                # 3. 检查最大范数是否超过了我们设定的阈值
                if max_latent_norm > self.latent_norm_clip_threshold:
                    
                    # 4. 计算缩放因子
                    scale_factor = self.latent_norm_clip_threshold / max_latent_norm.item()
                    
                    # (可选) 打印日志，方便调试
                    print(f"[Encoder-Clip] Max latent norm {max_latent_norm.item():.2f} > {self.latent_norm_clip_threshold}. Scaling encoder weights by {scale_factor:.4f}.")

                    # 5. 将缩放因子应用到Encoder的所有权重上
                    encoder = self._model.world_model.tokenizer.encoder
                    for param in encoder.parameters():
                        if param.requires_grad:
                            param.data.mul_(scale_factor)

        # =================================================================
        #              Head-Clip: 直接控制预测头的权重
        # -----------------------------------------------------------------
        # 如果Value或Reward的Logits绝对值过大，则按比例缩放对应头的权重。
        # =================================================================


        if self.logit_clip_threshold > 0:
            with torch.no_grad():
                # 从模型输出中获取原始的Logits (需要确保WorldModel的forward或compute_loss返回了它们)
                # 假设它们存储在 losses.intermediate_losses 中
                logits_value = losses.intermediate_losses.get('logits_value')
                logits_reward = losses.intermediate_losses.get('logits_reward')

                if logits_value is not None and logits_reward is not None:
                    # 计算Value和Reward Logits中的最大绝对值
                    max_abs_logit = max(logits_value.abs().max(), logits_reward.abs().max())

                    # 检查是否超过阈值
                    if max_abs_logit > self.logit_clip_threshold:
                        # 计算缩放因子
                        scale_factor = self.logit_clip_threshold / max_abs_logit.item()
                        
                        print(f"[Head-Clip] Max abs logit {max_abs_logit.item():.2f} > {self.logit_clip_threshold}. Scaling head weights by {scale_factor:.4f}.")

                        # 获取需要裁剪的预测头
                        head_value_module = self._model.world_model.head_value
                        head_reward_module = self._model.world_model.head_rewards

                        # 将缩放因子应用到这两个头的所有权重上
                        for head_module in [head_value_module, head_reward_module]:
                            for param in head_module.parameters():
                                if param.requires_grad:
                                    param.data.mul_(scale_factor)
        # =================================================================

        # Update learning rate scheduler if applicable
        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Update the target model with the current model's parameters
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
            # ==================== START MODIFICATION 3 ====================
            # Add value_priority to the log dictionary.
            'value_priority': value_priority_np.mean().item(),
            'value_priority_orig': value_priority_np,
            # ===================== END MODIFICATION 3 =====================
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
            "head_grad_norm":head_grad_norm,
            "backbone_grad_norm":backbone_grad_norm,

            'analysis/dormant_ratio_encoder': dormant_ratio_encoder.item(),
            'analysis/dormant_ratio_world_model': dormant_ratio_world_model.item(),
            'analysis/latent_state_l2_norms': latent_state_l2_norms.item(),
            'analysis/latent_action_l2_norms': latent_action_l2_norms.item(),
            'analysis/l2_norm_before': self.l2_norm_before,
            'analysis/l2_norm_after': self.l2_norm_after,
            'analysis/grad_norm_before': self.grad_norm_before,
            'analysis/grad_norm_after': self.grad_norm_after,

        "logits_value_mean":logits_value_mean,
        "logits_value_max":logits_value_max,
        "logits_value_min":logits_value_min,
        "logits_policy_mean":logits_policy_mean,
        "logits_policy_max":logits_policy_max,
        "logits_policy_min":logits_policy_min,
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
        # 为 collect MCTS 创建一个配置副本，并设置特定的模拟次数
        mcts_collect_cfg = copy.deepcopy(self._cfg)
        mcts_collect_cfg.num_simulations = self._cfg.collect_num_simulations
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(mcts_collect_cfg)
        else:
            self._mcts_collect = MCTSPtree(mcts_collect_cfg)

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
            action_mask: List = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.ndarray = None,
            timestep: List = [0]
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
            - timestep (:obj:`list`): The step index of the env in one episode.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
            - timestep: :math:`(N, 1)`, where N is the number of collect_env.
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
            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, timestep)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
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

            next_latent_state_with_env = self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, timestep)
            
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

                next_latent_state = next_latent_state_with_env[i][action]
                
                if self._cfg.model.world_model_cfg.obs_type == 'text':
                    # Output the plain text content decoded by the decoder from the next latent state
                    predicted_next = self._collect_model.tokenizer.decode_to_plain_text_for_decoder(embeddings=next_latent_state, max_length=256)
                else:
                    predicted_next = None

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
                    'timestep': timestep[i],
                    'predicted_next_text': predicted_next,
                }
                batch_action.append(action)

            self.last_batch_obs = data
            self.last_batch_action = batch_action

            # ========= TODO: for muzero_segment_collector now =========
            if active_collect_env_num < self.collector_env_num: # 先有环境done,再到下一步的forward出现这个这个条件满足
                print('==========collect_forward============')
                print(f'len(self.last_batch_obs) < self.collector_env_num, {active_collect_env_num}<{self.collector_env_num}')
                self._reset_collect(reset_init_data=True) # TODO(pu): 所有环境全部重置是否合理呢？
                if getattr(self._cfg, 'sample_type', '') == 'episode':
                    print('BUG: sample_type is episode, but len(self.last_batch_obs) < self.collector_env_num')

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model

        # 为 eval MCTS 创建一个配置副本，并设置特定的模拟次数
        mcts_eval_cfg = copy.deepcopy(self._cfg)
        mcts_eval_cfg.num_simulations = self._cfg.eval_num_simulations

        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(mcts_eval_cfg)
        else:
            self._mcts_eval = MCTSPtree(mcts_eval_cfg)
        self.evaluator_env_num = self._cfg.evaluator_env_num

        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: List = [-1],
                      ready_env_id: np.array = None, timestep: List = [0]) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to eval.
            - timestep (:obj:`list`): The step index of the env in one episode.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of eval_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of eval_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of eval_env.
            - to_play: :math:`(N, 1)`, where N is the number of eval_env.
            - ready_env_id: None
            - timestep: :math:`(N, 1)`, where N is the number of eval_env.

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
            network_output = self._eval_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, timestep)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            # if not in training, obtain the scalars of the value/reward
            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
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
            next_latent_state_with_env = self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, timestep)

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

                # Predict the next latent state based on the selected action and policy
                next_latent_state = next_latent_state_with_env[i][action]

                if self._cfg.model.world_model_cfg.obs_type == 'text':
                    # Output the plain text content decoded by the decoder from the next latent state
                    predicted_next = self._eval_model.tokenizer.decode_to_plain_text_for_decoder(embeddings=next_latent_state, max_length=256)
                else:
                    predicted_next = None

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                    'timestep': timestep[i],
                    'predicted_next_text': predicted_next,
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

        # --- BEGIN ROBUST FIX ---
        # This logic handles the crucial end-of-episode cache clearing.
        # The collector calls `_policy.reset([env_id])` when an episode is done,
        # which results in `current_steps` being None and `env_id` being a list.
        
        # We must handle both single int and list of ints for env_id.
        if env_id is not None:
            if isinstance(env_id, int):
                env_ids_to_reset = [env_id]
            else: # Assumes it's a list
                env_ids_to_reset = env_id
                
            # The key condition: `current_steps` is None only on the end-of-episode reset call from the collector.
            if current_steps is None:
                world_model = self._collect_model.world_model
                for eid in env_ids_to_reset:
                    # Clear the specific environment's initial inference cache.
                    if eid < len(world_model.past_kv_cache_init_infer_envs):
                        world_model.past_kv_cache_init_infer_envs[eid].clear()
                    
                    print(f'>>> [Collector] Cleared KV cache for env_id: {eid} at episode end.')

                # TODO
                # The recurrent cache is global, which is problematic.
                # A full clear is heavy-handed but safer than leaving stale entries.
                # world_model.past_kv_cache_recurrent_infer.clear()
                # if hasattr(world_model, 'keys_values_wm_list'):
                #     world_model.keys_values_wm_list.clear()
                # torch.cuda.empty_cache()
            # --- END ROBUST FIX ---


        # # Return immediately if env_id is None or a list
        # if env_id is None or isinstance(env_id, list):
        #     return

        # Determine the clear interval based on the environment's sample type
        # clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200
        # TODO:==========
        # clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 40

        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else self._cfg.game_segment_length


        # Clear caches if the current steps are a multiple of the clear interval
        if current_steps is not None and current_steps % clear_interval == 0:
            print(f'clear_interval: {clear_interval}')

            # Clear various caches in the collect model's world model
            world_model = self._collect_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up GPU memory
            torch.cuda.empty_cache()

            print(f'eps_steps_lst[{env_id}]: {current_steps}, collector: collect_model clear()')


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

        # --- BEGIN ROBUST FIX ---
        # This logic handles the crucial end-of-episode cache clearing for evaluation.
        # The evaluator calls `_policy.reset([env_id])` when an episode is done.
        if env_id is not None:
            if isinstance(env_id, int):
                env_ids_to_reset = [env_id]
            else: # Assumes it's a list
                env_ids_to_reset = env_id

            # The key condition: `current_steps` is None only on the end-of-episode reset call from the evaluator.
            if current_steps is None:
                world_model = self._eval_model.world_model
                for eid in env_ids_to_reset:
                    # Clear the specific environment's initial inference cache.
                    if eid < len(world_model.past_kv_cache_init_infer_envs):
                        world_model.past_kv_cache_init_infer_envs[eid].clear()
                    
                    print(f'>>> [Evaluator] Cleared KV cache for env_id: {eid} at episode end.')

                # The recurrent cache is global.
                world_model.past_kv_cache_recurrent_infer.clear()
                
                if hasattr(world_model, 'keys_values_wm_list'):
                    world_model.keys_values_wm_list.clear()

                torch.cuda.empty_cache()
                return
            # --- END ROBUST FIX ---

        # Return immediately if env_id is None or a list
        # if env_id is None or isinstance(env_id, list):
        #     return

        # Determine the clear interval based on the environment's sample type
        # clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200
        # TODO:==========
        # clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 40
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else self._cfg.game_segment_length

        # # Clear caches if the current steps are a multiple of the clear interval
        if current_steps is not None and current_steps % clear_interval == 0:
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
            'analysis/latent_action_l2_norms',

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
            # ==================== START MODIFICATION 4 ====================
            'value_priority',
            # ===================== END MODIFICATION 4 =====================
            'target_reward',
            'target_value',
            'total_grad_norm_before_clip_wm',
                        "head_grad_norm",
            "backbone_grad_norm",
            # tokenizer
            'commitment_loss',
            'reconstruction_loss',
            'perceptual_loss',


        "logits_value_mean",
        "logits_value_max",
        "logits_value_min",
        "logits_policy_mean",
        "logits_policy_max",
        "logits_policy_min",
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
        # self._optimizer_world_model.load_state_dict(state_dict['optimizer_world_model'])

    def recompute_pos_emb_diff_and_clear_cache(self) -> None:
        """
        Overview:
            Clear the caches and precompute positional embedding matrices in the model.
        """
        for model in [self._collect_model, self._target_model]:
            if not self._cfg.model.world_model_cfg.rotary_emb:
                # If rotary_emb is False, nn.Embedding is used for absolute position encoding.
                model.world_model.precompute_pos_emb_diff_kv()
            model.world_model.clear_caches()
        torch.cuda.empty_cache()