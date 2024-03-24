import copy
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.distributions import Categorical
from torch.nn import L1Loss
import inspect
from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, prepare_obs_stack4_for_gpt
from line_profiler import line_profiler


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

@POLICY_REGISTRY.register('muzero_gpt_multi_task_v2')
class MuZeroGPTMTV2Policy(Policy):
    """
    Overview:
        The policy class for MuZero.
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
        game_segment_length=200,

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
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.model_update_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        model_update_ratio=0.25,
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
        # (int) The number of episodes in each collecting stage.
        n_episode=8,
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
        policy_entropy_loss_weight=0,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
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

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=True,
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
            by import_names path. For MuZero, ``lzero.model.muzero_gpt_model.MuZeroModel``
        """
        if self._cfg.model.model_type == "conv":
            # return 'MuZeroModel', ['lzero.model.muzero_gpt_model']
            return 'MuZeroModelGPTMT', ['lzero.model.muzero_gpt_model_multi_task']
        elif self._cfg.model.model_type == "mlp":
            return 'MuZeroModelGPT', ['lzero.model.muzero_gpt_model_vector_obs']
        else:
            raise ValueError("model type {} is not supported".format(self._cfg.model.model_type))

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        # TODO: nanoGPT optimizer
        self._optimizer_world_model = configure_optimizers(
            model=self._model.world_model,
            learning_rate=1e-4,
            weight_decay=self._cfg.weight_decay,
            device_type=self._cfg.device,
            betas=(0.9, 0.95),
        )

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)

        # TODO: torch 2.0
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)

        # TODO: hard target
        # self._target_model = model_wrap(
        #     self._target_model,
        #     wrapper_name='target',
        #     update_type='assign',
        #     update_kwargs={'freq': self._cfg.target_update_freq}
        # )
        # TODO: soft target
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            # update_kwargs={'theta': 0.01} # MOCO:0.001,  DDPG:0.005, TD-MPC:0.01
            update_kwargs={'theta': 0.05} # MOCO:0.001,  DDPG:0.005, TD-MPC:0.01
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

    #@profile
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
        return_loss_dict = self._forward_learn_transformer(data)
        return return_loss_dict

    #@profile
    def monitor_weights_and_grads(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | "
                    f"Weight mean: {param.data.mean():.4f} | "
                    f"Weight std: {param.data.std():.4f} | "
                    f"Grad mean: {param.grad.mean():.4f} | "
                    f"Grad std: {param.grad.std():.4f}")
                    

    #@profile
    def _forward_learn_transformer(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
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

        obs_loss_multi_task = []
        reward_loss_multi_task = []
        policy_loss_multi_task = []
        value_loss_multi_task = []
        latent_recon_loss_multi_task = []
        perceptual_loss_multi_task = []
        orig_policy_loss_multi_task = []
        policy_entropy_multi_task = []
        weighted_total_loss = torch.tensor(0., device=self._cfg.device)
        weighted_total_loss.requires_grad = True
        
        for task_id, data_one_task in enumerate(data):
            current_batch, target_batch, task_id = data_one_task
            obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
            target_reward, target_value, target_policy = target_batch

            if self._cfg.model.frame_stack_num == 4:
                obs_batch, obs_target_batch = prepare_obs_stack4_for_gpt(obs_batch_ori, self._cfg)
            else:
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

            # do augmentations
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

            # shape: (batch_size, num_unroll_steps, action_dim)
            # NOTE: .long(), in discrete action space.
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
            data_list = [
                mask_batch,
                target_reward.astype('float32'),
                target_value.astype('float32'), target_policy, weights
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

            # compute_loss(self, batch: Batch, tokenizer: Tokenizer, ** kwargs: Any)

            batch_for_gpt = {}
            # TODO: for cartpole self._cfg.model.observation_shape
            if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape)==1:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape( self._cfg.batch_size, -1,  self._cfg.model.observation_shape)  # (B, T, O) or (B, T, C, H, W)
            elif len(self._cfg.model.observation_shape)==3:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape( self._cfg.batch_size, -1,  *self._cfg.model.observation_shape)  # (B, T, O) or (B, T, C, H, W)

            batch_for_gpt['actions'] = action_batch.squeeze(-1)  # (B, T-1, A) -> (B, T-1)
            batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]  # (B, T, R) -> (B, T-1, R)
            batch_for_gpt['mask_padding'] = mask_batch == 1.0  # (B, T) NOTE: 0 means invalid padding data
            batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]  # (B, T-1) TODO
            batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]  # (B, T-1, O) or (B, T-1, C, H, W)
            batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long, device=self._cfg.device) # (B, T-1)
            batch_for_gpt['target_value'] = target_value_categorical[:, :-1]  # (B, T-1, V)
            batch_for_gpt['target_policy'] = target_policy[:, :-1]  # (B, T-1, A)
            # NOTE: TODO: next latent state's policy value
            # batch_for_gpt['target_value'] = target_value_categorical[:, 1:]  # (B, T-1, V)
            # batch_for_gpt['target_policy'] = target_policy[:, 1:]  # (B, T-1, A)

            # get valid target_policy data
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            # compute entropy of each policy
            target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
            # compute average entropy
            average_target_policy_entropy = target_policy_entropy.mean().item()
            # print(f'Average entropy: {average_entropy}')

            # ==============================================================
            # update world model
            # ==============================================================
            intermediate_losses = defaultdict(float)
            losses = self._learn_model.world_model.compute_loss(batch_for_gpt, self._target_model.world_model.tokenizer, task_id)

            weighted_total_loss += losses.loss_total
            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{loss_name}"] = loss_value

            obs_loss = intermediate_losses['loss_obs']
            reward_loss = intermediate_losses['loss_rewards']
            policy_loss = intermediate_losses['loss_policy']
            orig_policy_loss = intermediate_losses['orig_policy_loss']
            policy_entropy = intermediate_losses['policy_entropy']
            value_loss = intermediate_losses['loss_value']
            latent_recon_loss = intermediate_losses['latent_recon_loss']
            perceptual_loss = intermediate_losses['perceptual_loss']
            
            obs_loss_multi_task.append(obs_loss)
            reward_loss_multi_task.append(reward_loss)
            policy_loss_multi_task.append(policy_loss)
            orig_policy_loss_multi_task.append(orig_policy_loss)
            policy_entropy_multi_task.append(policy_entropy)
            reward_loss_multi_task.append(reward_loss)
            value_loss_multi_task.append(value_loss)
            latent_recon_loss_multi_task.append(latent_recon_loss)
            perceptual_loss_multi_task.append(perceptual_loss)


        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        """
        for name, parameter in self._learn_model.tokenizer.named_parameters():
            print(name)
        """
        gradient_scale = 1 / self._cfg.num_unroll_steps
        # TODO(pu): test the effect of gradient scale.
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer_world_model.zero_grad()
        weighted_total_loss.backward()

        # 在训练循环中使用
        # self.monitor_weights_and_grads(self._learn_model.tokenizer.representation_network)
        # print('torch.cuda.memory_summary():', torch.cuda.memory_summary())

        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(
            self._learn_model.world_model.parameters(), self._cfg.grad_clip_value
        )

        self._optimizer_world_model.step()
        if self._cfg.lr_piecewise_constant_decay:
                self.lr_scheduler.step()


        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.update(self._learn_model.state_dict())


        # 确保所有的CUDA核心完成工作，以便准确统计显存使用情况
        torch.cuda.synchronize()
        # 获取当前分配的显存总量（字节）
        current_memory_allocated = torch.cuda.memory_allocated()
        # 获取程序运行到目前为止分配过的最大显存量（字节）
        max_memory_allocated = torch.cuda.max_memory_allocated()

        # 将显存使用量从字节转换为GB
        current_memory_allocated_gb = current_memory_allocated / (1024**3)
        max_memory_allocated_gb = max_memory_allocated / (1024**3)
        # 使用SummaryWriter记录当前和最大显存使用量


        return_loss_dict = {
            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self.collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],

            'weighted_total_loss': weighted_total_loss.item(),
            # 'obs_loss': mean(obs_loss_multi_task),
            'obs_loss': obs_loss,
            'latent_recon_loss':latent_recon_loss,
            'perceptual_loss':perceptual_loss,
            'policy_loss': policy_loss,
            'orig_policy_loss':orig_policy_loss,
            'policy_entropy':policy_entropy,
            'target_policy_entropy': average_target_policy_entropy,
            'reward_loss': reward_loss,
            'value_loss': value_loss,

            # ==============================================================
            # priority related
            # ==============================================================
            # 'value_priority_orig': value_priority,
            'value_priority_orig': np.zeros(self._cfg.batch_size),  # TODO
            # 'value_priority': value_priority.mean().item(),
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            # 'predicted_rewards': predicted_rewards.detach().cpu().numpy().mean().item(),
            # 'predicted_values': predicted_values.detach().cpu().numpy().mean().item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
        }

        return return_loss_dict


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
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([8,self._cfg.model.observation_shape[0],64,64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(8)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([8,self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(8)]

    #@profile
    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            task_id=0
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
        self._collect_model.tokenizer.eval() # TODO
        self._collect_model.world_model.transformer.eval() # TODO


        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        # if active_collect_env_num == 1:
        #     print('debug')
        with torch.no_grad():

            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, task_id=task_id)

            # network_output = self._collect_model.initial_inference(data)
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
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
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, task_id)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)
            
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
                batch_action.append(action)

            self.last_batch_obs = data
            self.last_batch_action = batch_action

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
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([3,self._cfg.model.observation_shape[0],64,64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(3)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([3,self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(3)]

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
        if self._cfg.model.model_type == 'conv':
            beg_index = self._cfg.model.image_channel * step
            end_index = self._cfg.model.image_channel * (step + self._cfg.model.frame_stack_num)
        elif self._cfg.model.model_type == 'mlp':
            beg_index = self._cfg.model.observation_shape * step
            end_index = self._cfg.model.observation_shape * (step + self._cfg.model.frame_stack_num)
        return beg_index, end_index

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1, ready_env_id: np.array = None, task_id=0) -> Dict:
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
        self._eval_model.tokenizer.eval() # TODO
        self._eval_model.world_model.transformer.eval() # TODO

        active_eval_env_num = data.shape[0]
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            # network_output = self._collect_model.initial_inference(data)
            network_output = self._eval_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            # print(f"latent_state_roots:{latent_state_roots}")  # TODO

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
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, task_id)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)
            
            batch_action = []

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # print("roots_visit_count_distributions:", distributions, "root_value:", value)  # TODO

                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                
                # action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                #     distributions, temperature=1, deterministic=True
                # )
                # TODO: eval for breakout
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=0.25, deterministic=False
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

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return [
            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'weighted_total_loss',
            'obs_loss',
            'orig_policy_loss',
            'policy_loss',
            'latent_recon_loss',
            'policy_entropy',
            'target_policy_entropy',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'total_grad_norm_before_clip_wm',
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

    # TODO:
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

    # def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
    #     """
    #     Overview:
    #         Load the state_dict variable into policy learn mode, specifically loading only the 
    #         representation network of the tokenizer within model and target_model.
    #     Arguments:
    #         - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.
    #     """
    #     # Extract the relevant sub-state-dicts for representation_network from the state_dict
    #     # model_rep_network_state = state_dict['model']['tokenizer']['representation_network']
    #     # target_model_rep_network_state = state_dict['target_model']['tokenizer']['representation_network']

    #     # # Load the state into the model's representation network
    #     # self._learn_model.tokenizer.representation_network.load_state_dict(model_rep_network_state)
    #     # self._target_model.tokenizer.representation_network.load_state_dict(target_model_rep_network_state)

    #     # Assuming self._learn_model and self._target_model have a 'representation_network' submodule
    #     self._load_representation_network_state(state_dict['model'], self._learn_model.tokenizer.representation_network)
    #     self._load_representation_network_state(state_dict['target_model'], self._target_model.tokenizer.representation_network)


    def _load_representation_network_state(self, state_dict, model_submodule):
        """
        This function filters the state_dict to only include the state of the representation_network
        and loads it into the given model submodule.
        """
        from collections import OrderedDict

        # Filter the state_dict to only include keys that start with 'representation_network'
        representation_network_keys = {k: v for k, v in state_dict.items() if k.startswith('representation_network')}
        
        # Load the state into the model's representation_network submodule
        # model_submodule.load_state_dict(OrderedDict(representation_network_keys))

        # 去掉键名前缀
        new_state_dict = OrderedDict()
        for key, value in representation_network_keys.items():
            new_key = key.replace('representation_network.', '')  # 去掉前缀
            new_state_dict[new_key] = value

        # # 如果模型在特定的设备上，确保状态字典也在那个设备上
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # new_state_dict = {key: value.to(device) for key, value in new_state_dict.items()}

        # 尝试加载状态字典
        try:
            # model_submodule.load_state_dict(new_state_dict)
            # 使用 strict=False 参数忽略缺少的键
            model_submodule.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print("加载失败: ", e)


    def _process_transition(self, obs, policy_output, timestep):
        # be compatible with DI-engine Policy class
        pass

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass
