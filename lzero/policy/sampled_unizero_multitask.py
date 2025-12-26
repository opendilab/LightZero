import copy
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import wandb
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, set_pkg_seed, get_rank, get_world_size

from lzero.entry.utils import initialize_zeros_batch
from lzero.mcts import SampledUniZeroMCTSCtree as MCTSCtree
from lzero.model import ImageTransforms
from lzero.policy import (
    scalar_transform,
    InverseScalarTransform,
    phi_transform,
    DiscreteSupport,
    to_torch_float_tensor,
    mz_network_output_unpack,
    select_action,
    prepare_obs,
    prepare_obs_stack_for_unizero
)
from lzero.policy.unizero import UniZeroPolicy
from .utils import configure_optimizers_nanogpt
import torch.nn.functional as F
import torch.distributed as dist

# Please add the path to your LibMTL library.
# For example: sys.path.append('/path/to/your/LibMTL/')
import sys
# sys.path.append('/path/to/your/LibMTL/') # Template path
from LibMTL.weighting.MoCo_unizero import MoCo as GradCorrect


def generate_task_loss_dict(multi_task_losses: List[Union[torch.Tensor, float]], task_name_template: str, task_id: int) -> Dict[str, float]:
    """
    Overview:
        Generates a dictionary for losses of each task.
    Arguments:
        - multi_task_losses (:obj:`List[Union[torch.Tensor, float]]`): A list containing the loss for each task.
        - task_name_template (:obj:`str`): A template for the task name, e.g., 'obs_loss_task{}'.
        - task_id (:obj:`int`): The base task ID.
    Returns:
        - (:obj:`Dict[str, float]`): A dictionary containing the loss for each task.
    """
    task_loss_dict = {}
    for task_idx, task_loss in enumerate(multi_task_losses):
        task_name = task_name_template.format(task_idx + task_id)
        try:
            # Convert tensor to float if it has .item(), otherwise cast to float.
            task_loss_dict[task_name] = task_loss.item() if hasattr(task_loss, 'item') else float(task_loss)
        except Exception as e:
            # Fallback for cases where conversion fails.
            task_loss_dict[task_name] = task_loss
    return task_loss_dict


class WrappedModelV2:
    """
    Overview:
        A wrapper class to conveniently manage different parts of a larger model,
        such as the tokenizer, transformer, and various embedding layers. This allows for
        easier handling of parameters and gradients for these components.
    """
    def __init__(self, tokenizer: torch.nn.Module, transformer: torch.nn.Module, pos_emb: torch.nn.Module, task_emb: torch.nn.Module, act_embedding_table: torch.nn.Module):
        """
        Overview:
            Initializes the WrappedModelV2 with model components.
        Arguments:
            - tokenizer (:obj:`torch.nn.Module`): The tokenizer module.
            - transformer (:obj:`torch.nn.Module`): The main transformer module.
            - pos_emb (:obj:`torch.nn.Module`): The positional embedding layer.
            - task_emb (:obj:`torch.nn.Module`): The task embedding layer.
            - act_embedding_table (:obj:`torch.nn.Module`): The action embedding table.
        """
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.pos_emb = pos_emb
        self.task_emb = task_emb
        self.act_embedding_table = act_embedding_table

    def parameters(self) -> List[torch.Tensor]:
        """
        Overview:
            Collects and returns all parameters from the wrapped model components.
        Returns:
            - (:obj:`List[torch.Tensor]`): A list of all parameters.
        """
        return (
            list(self.tokenizer.parameters()) +
            list(self.transformer.parameters()) +
            list(self.pos_emb.parameters()) +
            # list(self.task_emb.parameters()) +
            list(self.act_embedding_table.parameters())
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Overview:
            Sets the gradients of all wrapped model components to zero.
        Arguments:
            - set_to_none (:obj:`bool`): Whether to set gradients to None instead of zero. Defaults to False.
        """
        self.tokenizer.zero_grad(set_to_none=set_to_none)
        self.transformer.zero_grad(set_to_none=set_to_none)
        self.pos_emb.zero_grad(set_to_none=set_to_none)
        # self.task_emb.zero_grad(set_to_none=set_to_none)
        self.act_embedding_table.zero_grad(set_to_none=set_to_none)
    
    def get_group_parameters(self) -> Dict[str, List[torch.Tensor]]:
        """
        Overview:
            Returns a dictionary where keys are module names (or finer-grained layers)
            and values are the corresponding parameter lists. The order of parameters in the
            returned dictionary's values should be consistent with the `parameters()` method.
        Returns:
            - (:obj:`Dict[str, List[torch.Tensor]]`): A dictionary of grouped parameters.
        """
        groups = {}
        groups['tokenizer'] = list(self.tokenizer.parameters())
        groups['transformer'] = list(self.transformer.parameters())
        groups['pos_emb'] = list(self.pos_emb.parameters())
        groups['act_embedding_table'] = list(self.act_embedding_table.parameters())

        # Example of how to add parameters from sub-layers within the transformer.
        # This is for demonstration; ensure the order in parameters() is consistent if used.
        if hasattr(self.transformer, 'blocks'):
            for i, layer in enumerate(self.transformer.blocks):
                groups[f'transformer_layer_{i}'] = list(layer.parameters())
        return groups


@POLICY_REGISTRY.register('sampled_unizero_multitask')
class SampledUniZeroMTPolicy(UniZeroPolicy):
    """
    Overview:
        The policy class for Sampled UniZero Multitask, combining multi-task learning with sampled-based MCTS.
        This implementation extends the UniZeroPolicy to handle multiple tasks simultaneously while utilizing
        sampled MCTS for action selection. It ensures scalability and correctness in multi-task environments.
    """

    # The default_config for Sampled UniZero Multitask policy.
    config = dict(
        type='sampled_unizero_multitask',
        model=dict(
            model_type='conv',  # options={'mlp', 'conv'}
            continuous_action_space=False,
            observation_shape=(3, 64, 64),
            self_supervised_learning_loss=True,
            categorical_distribution=True,
            image_channel=3,
            frame_stack_num=1,
            num_res_blocks=1,
            num_channels=64,
            support_scale=50,
            bias=True,
            res_connection_in_dynamics=True,
            norm_type='LN',
            analysis_sim_norm=False,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=10000, ), ), ),
            world_model_cfg=dict(
                tokens_per_block=2,
                max_blocks=10,
                max_tokens=20,
                context_length=8,
                gru_gating=False,
                device='cpu',
                analysis_sim_norm=False,
                analysis_dormant_ratio=False,
                action_space_size=6,
                group_size=8,
                attention='causal',
                num_layers=2,
                num_heads=8,
                embed_dim=768,
                embed_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                support_size=101,
                max_cache_size=5000,
                env_num=8,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                policy_entropy_weight=5e-3,
                predict_latent_loss_type='group_kl',
                obs_type='image',
                gamma=1,
                dormant_threshold=0.01,
                policy_loss_type='kl',
            ),
        ),
        use_rnd_model=False,
        multi_gpu=True,
        sampled_algo=True,
        gumbel_algo=False,
        mcts_ctree=True,
        cuda=True,
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        action_type='fixed_action_space',
        battle_mode='play_with_bot_mode',
        monitor_extra_statistics=True,
        game_segment_length=400,
        analysis_sim_norm=False,
        collect_with_pure_policy=False,
        eval_freq=int(5e3),
        sample_type='transition',

        transform2string=False,
        gray_scale=False,
        use_augmentation=False,
        augmentation=['shift', 'intensity'],

        ignore_done=False,
        update_per_collect=None,
        replay_ratio=0.25,
        batch_size=256,
        optim_type='AdamW',
        learning_rate=0.0001,
        init_w=3e-3,
        target_update_freq=100,
        target_update_theta=0.05,
        target_update_freq_for_intrinsic_reward=1000,
        weight_decay=1e-4,
        momentum=0.9,
        grad_clip_value=5,
        n_episode=8,
        num_simulations=50,
        discount_factor=0.997,
        td_steps=5,
        num_unroll_steps=10,
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=0,
        cos_lr_scheduler=False,
        piecewise_decay_lr_scheduler=False,
        threshold_training_steps_for_final_lr=int(5e4),
        manual_temperature_decay=False,
        threshold_training_steps_for_final_temperature=int(1e5),
        fixed_temperature_value=0.25,
        use_ture_chance_label_in_chance_encoder=False,

        use_priority=False,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        train_start_after_envsteps=0,

        root_dirichlet_alpha=0.3,
        root_noise_weight=0.25,

        random_collect_episode_num=0,

        eps=dict(
            eps_greedy_exploration_in_collect=False,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm's default model setting for demonstration.
        Returns:
            - (:obj:`Tuple[str, List[str]]`): A tuple containing the model name and the import paths.
        """
        return 'SampledUniZeroMTModel', ['lzero.model.sampled_unizero_model_multitask']

    def _init_learn(self) -> None:
        """
        Overview:
            Initializes the learning mode. This method sets up the learn model, optimizer,
            target model, and other utilities required for training, such as LR schedulers
            and gradient correction methods (e.g., MoCo).
        """
        # Configure optimizer for the world model using NanoGPT's configuration utility.
        self._optimizer_world_model = configure_optimizers_nanogpt(
            model=self._model.world_model,
            learning_rate=self._cfg.learning_rate,
            weight_decay=self._cfg.weight_decay,
            device_type=self._cfg.device,
            betas=(0.9, 0.95),
        )

        # Initialize learning rate schedulers if configured.
        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

            if self._cfg.cos_lr_scheduler:
                self.lr_scheduler = CosineAnnealingLR(
                    self._optimizer_world_model, T_max=int(1e5), eta_min=0, last_epoch=-1
                )
            elif self._cfg.piecewise_decay_lr_scheduler:
                self.lr_scheduler = StepLR(
                    self._optimizer_world_model, step_size=int(5e4), gamma=0.1
                )

        # Initialize weights for continuous action spaces.
        if self._cfg.model.continuous_action_space:
            init_w = self._cfg.init_w
            self._model.world_model.fc_policy_head.mu.weight.data.uniform_(-init_w, init_w)
            self._model.world_model.fc_policy_head.mu.bias.data.uniform_(-init_w, init_w)
            try:
                self._model.world_model.fc_policy_head.log_sigma_layer.weight.data.uniform_(-init_w, init_w)
                self._model.world_model.fc_policy_head.log_sigma_layer.bias.data.uniform_(-init_w, init_w)
            except Exception as exception:
                logging.warning(exception)

        # Initialize and compile the target model.
        self._target_model = copy.deepcopy(self._model)
        assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "Torch version 2.0 or higher is required."
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)
        
        # Wrap the target model for soft updates (momentum-based).
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.target_update_theta}
        )
        self._learn_model = self._model

        # Initialize utilities for loss calculation and transformations.
        
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        
        self.inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)

        self.intermediate_losses = defaultdict(float)
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.

        self.task_id = self._cfg.task_id
        self.task_num_for_current_rank = self._cfg.task_num
        print(f'self._cfg.only_use_moco_stats:{self._cfg.only_use_moco_stats}')

        # Initialize gradient correction method (MoCo) if enabled.
        if self._cfg.use_moco or self._cfg.only_use_moco_stats:
            # Wrap model components for gradient correction. Note: Heads are not included.
            wrapped_model = WrappedModelV2(
                self._learn_model.world_model.tokenizer.encoder,  # TODO: This might contain one or multiple encoders.
                self._learn_model.world_model.transformer,
                self._learn_model.world_model.pos_emb,
                self._learn_model.world_model.task_emb,
                self._learn_model.world_model.act_embedding_table,
            )

            # TODO: The GradCorrect class might need adjustments for multi-GPU training compatibility.
            # Initialize the gradient correction mechanism.
            self.grad_correct = GradCorrect(wrapped_model, self._cfg.total_task_num, self._cfg.device, self._cfg.multi_gpu)

            self.grad_correct.init_param()
            self.grad_correct.rep_grad = False


        encoder_tokenizer = getattr(self._model.tokenizer.encoder, 'tokenizer', None)
        self.pad_token_id = encoder_tokenizer.pad_token_id if encoder_tokenizer is not None else 0
        

    def _forward_learn(self, data: Tuple[torch.Tensor], task_weights: Any = None, ignore_grad: bool = False) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward pass for training. This method processes a batch of data for multiple tasks,
            computes losses, and updates the model weights.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): A tuple of data batches, one for each task.
            - task_weights (:obj:`Any`): Weights for each task's loss. Defaults to None.
            - ignore_grad (:obj:`bool`): If True, gradients are zeroed out after computation, effectively skipping the update. Defaults to False.
        Returns:
            - (:obj:`Dict[str, Union[float, int]]`): A dictionary containing various loss values and training statistics.
        """
        self._learn_model.train()
        self._target_model.train()

        # Initialize lists to store losses and metrics for each task.
        task_weight_multi_task, obs_loss_multi_task, reward_loss_multi_task = [], [], []
        policy_loss_multi_task, orig_policy_loss_multi_task, policy_entropy_multi_task = [], [], []
        value_loss_multi_task, latent_recon_loss_multi_task, perceptual_loss_multi_task = [], [], []
        latent_state_l2_norms_multi_task, average_target_policy_entropy_multi_task = [], []
        value_priority_multi_task, value_priority_mean_multi_task = [], []

        weighted_total_loss = 0.0
        losses_list = []  # Stores the individual loss tensor for each task.

        for task_id, data_one_task in enumerate(data):
            # Unpack data for the current task.
            current_batch, target_batch, task_id = data_one_task
            obs_batch_ori, action_batch, child_sampled_actions_batch, target_action_batch, mask_batch, indices, weights, make_time, timestep_batch = current_batch
            target_reward, target_value, target_policy = target_batch

            # Prepare observations.
            if self._cfg.model.frame_stack_num == 4:
                obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, self._cfg)
            else:
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg, task_id)

            # Apply data augmentation if enabled.
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

            # Prepare actions and convert data to torch tensors.
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1)
            if not self._cfg.model.continuous_action_space:
                action_batch = action_batch.long()
            
            data_list = [mask_batch, target_reward.astype('float32'), target_value.astype('float32'), target_policy, weights]
            mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list, self._cfg.device)

            cur_batch_size = target_reward.size(0)
            target_reward = target_reward.view(cur_batch_size, -1)
            target_value = target_value.view(cur_batch_size, -1)

            # Transform scalar targets to their categorical representation.
            transformed_target_reward = scalar_transform(target_reward)
            transformed_target_value = scalar_transform(target_value)
            target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
            target_value_categorical = phi_transform(self.value_support, transformed_target_value)

            # Prepare the batch for the GPT-based world model.
            batch_for_gpt = {}
            if isinstance(self._cfg.model.observation_shape_list[task_id], int) or len(self._cfg.model.observation_shape_list[task_id]) == 1:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(cur_batch_size, -1, self._cfg.model.observation_shape_list[task_id])
            else:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(cur_batch_size, -1, *self._cfg.model.observation_shape_list[task_id])

            batch_for_gpt['actions'] = action_batch.squeeze(-1)
            batch_for_gpt['child_sampled_actions'] = torch.from_numpy(child_sampled_actions_batch).to(self._cfg.device)[:, :-1]
            batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
            batch_for_gpt['mask_padding'] = (mask_batch == 1.0)[:, :-1]  # 0 indicates invalid padding data.
            batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
            batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long, device=self._cfg.device)
            batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
            batch_for_gpt['target_policy'] = target_policy[:, :-1]

            # Compute target policy entropy for monitoring.
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
            average_target_policy_entropy = target_policy_entropy.mean().item()

            # Compute losses using the world model.
            losses = self._learn_model.world_model.compute_loss(
                batch_for_gpt, self._target_model.world_model.tokenizer, self.inverse_scalar_transform_handle, task_id=task_id
            )
            
            # Accumulate weighted total loss.
            current_task_weight = task_weights[task_id] if task_weights is not None else 1
            weighted_total_loss += losses.loss_total * current_task_weight
            losses_list.append(losses.loss_total * current_task_weight)
            task_weight_multi_task.append(current_task_weight)

            # Store intermediate losses for logging.
            for loss_name, loss_value in losses.intermediate_losses.items():
                self.intermediate_losses[f"{loss_name}"] = loss_value

            # Collect individual losses for the current task.
            obs_loss_multi_task.append(self.intermediate_losses.get('loss_obs', 0.0) or 0.0)
            reward_loss_multi_task.append(self.intermediate_losses.get('loss_rewards', 0.0) or 0.0)
            policy_loss_multi_task.append(self.intermediate_losses.get('loss_policy', 0.0) or 0.0)
            orig_policy_loss_multi_task.append(self.intermediate_losses.get('orig_policy_loss', 0.0) or 0.0)
            policy_entropy_multi_task.append(self.intermediate_losses.get('policy_entropy', 0.0) or 0.0)
            value_loss_multi_task.append(self.intermediate_losses.get('loss_value', 0.0) or 0.0)
            latent_recon_loss_multi_task.append(self.intermediate_losses.get('latent_recon_loss', 0.0) or 0.0)
            perceptual_loss_multi_task.append(self.intermediate_losses.get('perceptual_loss', 0.0) or 0.0)
            latent_state_l2_norms_multi_task.append(self.intermediate_losses.get('latent_state_l2_norms', 0.0) or 0.0)
            average_target_policy_entropy_multi_task.append(average_target_policy_entropy)
            value_priority = torch.tensor(0., device=self._cfg.device)  # Placeholder
            value_priority_multi_task.append(value_priority)
            value_priority_mean_multi_task.append(value_priority.mean().item())

        # --- Model Update Step ---
        self._optimizer_world_model.zero_grad()

        # Perform backward pass, either with or without gradient correction.
        if self._cfg.use_moco:
            # Use MoCo for gradient correction and backpropagation.
            lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)
        elif self._cfg.only_use_moco_stats:
            # Compute MoCo stats but perform standard backpropagation.
            lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)
            weighted_total_loss.backward()
        else:
            # Standard backpropagation without gradient correction.
            lambd = torch.tensor([0. for _ in range(self.task_num_for_current_rank)], device=self._cfg.device)
            weighted_total_loss.backward()

        # Clip gradients to prevent exploding gradients.
        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(self._learn_model.world_model.parameters(), self._cfg.grad_clip_value)

        # NOTE: If ignore_grad is True, zero out gradients. This is useful for DDP synchronization
        # when a GPU has finished all its tasks but still needs to participate in the training step.
        if ignore_grad:
            self._optimizer_world_model.zero_grad()

        # Synchronize gradients across GPUs in multi-GPU setup.
        if self._cfg.multi_gpu:
            if not self._cfg.use_moco:
                # TODO: Investigate if a barrier is needed here for synchronization.
                # dist.barrier()
                self.sync_gradients(self._learn_model)

        # Update model parameters.
        self._optimizer_world_model.step()

        # Step the learning rate scheduler.
        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Update the target model using a soft update rule.
        self._target_model.update(self._learn_model.state_dict())

        # Monitor GPU memory usage.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            current_memory_allocated_gb, max_memory_allocated_gb = 0., 0.

        # --- Logging and Return ---
        return_loss_dict = {
            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self._collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
        }

        # Generate and merge task-specific loss dictionaries.
        # The "noreduce_" prefix indicates these are per-rank values before DDP reduction.
        multi_task_loss_dicts = {
            **generate_task_loss_dict(task_weight_multi_task, 'noreduce_task_weight_task{}', self.task_id),
            **generate_task_loss_dict(obs_loss_multi_task, 'noreduce_obs_loss_task{}', self.task_id),
            **generate_task_loss_dict(latent_recon_loss_multi_task, 'noreduce_latent_recon_loss_task{}', self.task_id),
            **generate_task_loss_dict(perceptual_loss_multi_task, 'noreduce_perceptual_loss_task{}', self.task_id),
            **generate_task_loss_dict(latent_state_l2_norms_multi_task, 'noreduce_latent_state_l2_norms_task{}', self.task_id),
            **generate_task_loss_dict(policy_loss_multi_task, 'noreduce_policy_loss_task{}', self.task_id),
            **generate_task_loss_dict(orig_policy_loss_multi_task, 'noreduce_orig_policy_loss_task{}', self.task_id),
            **generate_task_loss_dict(policy_entropy_multi_task, 'noreduce_policy_entropy_task{}', self.task_id),
            **generate_task_loss_dict(reward_loss_multi_task, 'noreduce_reward_loss_task{}', self.task_id),
            **generate_task_loss_dict(value_loss_multi_task, 'noreduce_value_loss_task{}', self.task_id),
            **generate_task_loss_dict(average_target_policy_entropy_multi_task, 'noreduce_target_policy_entropy_task{}', self.task_id),
            **generate_task_loss_dict(lambd, 'noreduce_lambd_task{}', self.task_id),
            **generate_task_loss_dict(value_priority_multi_task, 'noreduce_value_priority_task{}', self.task_id),
            **generate_task_loss_dict(value_priority_mean_multi_task, 'noreduce_value_priority_mean_task{}', self.task_id),
        }
        return_loss_dict.update(multi_task_loss_dicts)

        # Log to wandb if enabled.
        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_loss_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_loss_dict

    def _monitor_vars_learn(self, num_tasks: int = 2) -> List[str]:
        """
        Overview:
            Specifies the variables to be monitored during training. These variables will be logged
            (e.g., to TensorBoard) based on the dictionary returned by `_forward_learn`.
        Arguments:
            - num_tasks (:obj:`int`): The number of tasks to generate monitored variables for. This argument is for API consistency and is overridden by `self.task_num_for_current_rank`.
        Returns:
            - (:obj:`List[str]`): A list of variable names to monitor.
        """
        # Basic monitored variables, independent of the number of tasks.
        monitored_vars = [
            'Current_GPU', 'Max_GPU', 'collect_epsilon', 'collect_mcts_temperature',
            'cur_lr_world_model', 'weighted_total_loss', 'total_grad_norm_before_clip_wm',
        ]

        # Task-specific variables.
        task_specific_vars = [
            'noreduce_task_weight', 'noreduce_obs_loss', 'noreduce_orig_policy_loss',
            'noreduce_policy_loss', 'noreduce_latent_recon_loss', 'noreduce_policy_entropy',
            'noreduce_target_policy_entropy', 'noreduce_reward_loss', 'noreduce_value_loss',
            'noreduce_perceptual_loss', 'noreduce_latent_state_l2_norms', 'noreduce_lambd',
            'noreduce_value_priority_mean',
        ]
        
        # The number of tasks handled by the current rank.
        num_tasks_on_rank = self.task_num_for_current_rank
        
        # Generate full variable names for each task on the current rank.
        if num_tasks_on_rank is not None:
            for var in task_specific_vars:
                for task_idx in range(num_tasks_on_rank):
                    # The task ID is offset by the base task ID for this rank.
                    monitored_vars.append(f'{var}_task{self.task_id + task_idx}')
        else:
            monitored_vars.extend(task_specific_vars)

        return monitored_vars
        
    def monitor_weights_and_grads(self, model: torch.nn.Module) -> None:
        """
        Overview:
            A utility function to monitor and print the statistics (mean, std) of model weights and their gradients.
        Arguments:
            - model (:obj:`torch.nn.Module`): The model to inspect.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Layer: {name} | "
                      f"Weight mean: {param.data.mean():.4f} | "
                      f"Weight std: {param.data.std():.4f} | "
                      f"Grad mean: {param.grad.mean():.4f} | "
                      f"Grad std: {param.grad.std():.4f}")

    def _init_collect(self) -> None:
        """
        Overview:
            Initializes the collection mode. This method sets up the collect model, MCTS utilities,
            and initial states for the collector environments.
        """
        self._collect_model = self._model

        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)
        self._collect_mcts_temperature = 1.
        self._task_weight_temperature = 10.
        self._collect_epsilon = 0.0
        self.collector_env_num = self._cfg.collector_env_num
        
        # Initialize placeholders for the last observation and action batches.
        if self._cfg.model.model_type == 'conv':
            obs_shape = [self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]
            self.last_batch_obs = torch.zeros(obs_shape, device=self._cfg.device)
        elif self._cfg.model.model_type == 'mlp':
            obs_shape = [self.collector_env_num, self._cfg.model.observation_shape_list[0]]
            self.last_batch_obs = torch.zeros(obs_shape, device=self._cfg.device)
        self.last_batch_action = [-1 for _ in range(self.collector_env_num)]

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: List = None,
            temperature: float = 1.0,
            to_play: List[int] = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.ndarray = None,
            timestep: List[int] = [0],
            task_id: int = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Overview:
            The forward pass for data collection. It uses MCTS to select actions for the current states.
        Arguments:
            - data (:obj:`torch.Tensor`): The current batch of observations.
            - action_mask (:obj:`List`): A list of action masks for each environment.
            - temperature (:obj:`float`): The temperature parameter for MCTS action selection.
            - to_play (:obj:`List[int]`): A list indicating the current player for each environment.
            - epsilon (:obj:`float`): The exploration noise parameter.
            - ready_env_id (:obj:`np.ndarray`): An array of environment IDs that are ready for action.
            - timestep (:obj:`List[int]`): The current timestep for each environment.
            - task_id (:obj:`int`): The ID of the task being executed.
        Returns:
            - (:obj:`Dict[int, Dict[str, Any]]`): A dictionary mapping environment IDs to action selection results.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            # 1. Initial inference to get root information.
            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            # 2. Prepare MCTS roots.
            if not self._cfg.model.continuous_action_space:
                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            else:
                legal_actions = [[-1] * self._cfg.model.world_model_cfg.num_of_sampled_actions for _ in range(active_collect_env_num)]

            noises = [np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.world_model_cfg.num_of_sampled_actions).astype(np.float32).tolist() for _ in range(active_collect_env_num)]

            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(active_collect_env_num, legal_actions, self._cfg.model.world_model_cfg.action_space_size, self._cfg.model.world_model_cfg.num_of_sampled_actions, self._cfg.model.continuous_action_space)
            else:
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
            
            # 3. MCTS search.
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, timestep=timestep, task_id=task_id)

            # 4. Get results from MCTS and select actions.
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            roots_sampled_actions = roots.get_sampled_actions()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                root_sampled_actions = np.array([getattr(action, 'value', action) for action in roots_sampled_actions[i]])

                # Select action based on visit counts, with temperature for exploration.
                action_idx, visit_count_distribution_entropy = select_action(distributions, temperature=self._collect_mcts_temperature, deterministic=False)
                action = root_sampled_actions[action_idx]
                if not self._cfg.model.continuous_action_space:
                    action = int(action.item())

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                batch_action.append(action)

            # 5. Update state for the next step.
            self.last_batch_obs = data
            self.last_batch_action = batch_action

            # Reset collector if the number of active environments is less than expected.
            if active_collect_env_num < self.collector_env_num:
                logging.warning(f'Number of active envs ({active_collect_env_num}) is less than collector_env_num ({self.collector_env_num}). Resetting collector.')
                self._reset_collect(reset_init_data=True, task_id=task_id)

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Initializes the evaluation mode. This method sets up the evaluation model, MCTS utilities,
            and initial states for the evaluator environments.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)
        self.evaluator_env_num = self._cfg.evaluator_env_num

        self.task_id_for_eval = self._cfg.task_id
        self.task_num_for_current_rank = self._cfg.task_num
        
        # Initialize placeholders for the last observation and action batches for evaluation.
        if self._cfg.model.model_type == 'conv':
            obs_shape = [self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]
            self.last_batch_obs_eval = torch.zeros(obs_shape, device=self._cfg.device)
        elif self._cfg.model.model_type == 'mlp':
            # TODO: Ensure observation_shape_list is correctly indexed for the evaluation task.
            obs_shape = [self.evaluator_env_num, self._cfg.model.observation_shape_list[self.task_id_for_eval]]
            self.last_batch_obs_eval = torch.zeros(obs_shape, device=self._cfg.device)
            print(f'rank {get_rank()} last_batch_obs_eval shape: {self.last_batch_obs_eval.shape}')
        self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1, ready_env_id: np.ndarray = None, timestep: List[int] = [0], task_id: int = None) -> Dict[int, Dict[str, Any]]:
        """
        Overview:
            The forward pass for evaluation. It uses MCTS to select actions deterministically.
        Arguments:
            - data (:obj:`torch.Tensor`): The current batch of observations.
            - action_mask (:obj:`List`): A list of action masks for each environment.
            - to_play (:obj:`int`): The current player.
            - ready_env_id (:obj:`np.ndarray`): An array of environment IDs that are ready for action.
            - timestep (:obj:`List[int]`): The current timestep for each environment.
            - task_id (:obj:`int`): The ID of the task being evaluated.
        Returns:
            - (:obj:`Dict[int, Dict[str, Any]]`): A dictionary mapping environment IDs to action selection results.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            # 1. Initial inference.
            network_output = self._eval_model.initial_inference(self.last_batch_obs_eval, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            # 2. Prepare MCTS roots without noise for deterministic evaluation.
            if not self._cfg.model.continuous_action_space:
                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            else:
                legal_actions = [[-1] * self._cfg.model.world_model_cfg.num_of_sampled_actions for _ in range(active_eval_env_num)]

            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions, self._cfg.model.world_model_cfg.action_space_size, self._cfg.model.world_model_cfg.num_of_sampled_actions, self._cfg.model.continuous_action_space)
            else:
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            
            # 3. MCTS search.
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, timestep=timestep, task_id=task_id)

            # 4. Get results and select actions deterministically.
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            roots_sampled_actions = roots.get_sampled_actions()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                root_sampled_actions = np.array([getattr(action, 'value', action) for action in roots_sampled_actions[i]])

                # Select action deterministically (greedy selection from visit counts).
                action_idx, visit_count_distribution_entropy = select_action(distributions, temperature=1, deterministic=True)
                action = root_sampled_actions[action_idx]
                if not self._cfg.model.continuous_action_space:
                    action = int(action.item())

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                batch_action.append(action)
            
            # 5. Update state for the next evaluation step.
            self.last_batch_obs_eval = data
            self.last_batch_action = batch_action

        return output

    def _reset_collect(self, env_id: int = None, current_steps: int = 0, reset_init_data: bool = True, task_id: int = None) -> None:
        """
        Overview:
            Resets the collector state. This can be a full reset of initial data or a periodic
            clearing of model caches to manage memory.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. If None, applies to all.
            - current_steps (:obj:`int`): The current number of steps, used for periodic cache clearing.
            - reset_init_data (:obj:`bool`): Whether to reset the initial observation and action batches.
            - task_id (:obj:`int`, optional): The task ID, used to determine observation shape.
        """
        if reset_init_data:
            obs_shape = self._cfg.model.observation_shape_list[task_id] if task_id is not None else self._cfg.model.observation_shape
            self.last_batch_obs = initialize_zeros_batch(obs_shape, self._cfg.collector_env_num, self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]
            logging.info(f'Collector: last_batch_obs and last_batch_action have been reset. Shape: {self.last_batch_obs.shape}')

        if env_id is None or isinstance(env_id, list):
            return

        # Periodically clear model caches to free up memory.
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200
        if current_steps > 0 and current_steps % clear_interval == 0:
            logging.info(f'Clearing model caches at step {current_steps}.')
            world_model = self._collect_model.world_model
            world_model.past_kv_cache_init_infer.clear()
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
            logging.info('Collector: collect_model caches cleared.')
            self._reset_target_model()

    def _reset_target_model(self) -> None:
        """
        Overview:
            Resets the caches of the target model to free up GPU memory.
        """
        world_model = self._target_model.world_model
        world_model.past_kv_cache_init_infer.clear()
        for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        world_model.past_kv_cache_recurrent_infer.clear()
        world_model.keys_values_wm_list.clear()
        torch.cuda.empty_cache()
        logging.info('Collector: target_model caches cleared.')

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Returns the state dictionary of the learning components.
        Returns:
            - (:obj:`Dict[str, Any]`): A dictionary containing the state of the model, target model, and optimizer.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_world_model': self._optimizer_world_model.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Loads the state dictionary into the learning components.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The state dictionary to load.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_world_model.load_state_dict(state_dict['optimizer_world_model'])

    # TODO: The following is a version for pretrain-finetune workflow, which only loads backbone parameters.
    # def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
    #     """
    #     Overview:
    #         Loads a state_dict into the policy's learn mode, but excludes parameters related to
    #         multi-task heads and task embeddings. This is useful for fine-tuning a pre-trained model
    #         on a new set of tasks.
    #     Arguments:
    #         - state_dict (:obj:`Dict[str, Any]`): The dict of the policy learn state saved previously.
    #     """
    #     # Define prefixes of parameters to exclude (e.g., multi-task heads, task embeddings).
    #     exclude_prefixes = [
    #         '_orig_mod.world_model.head_policy_multi_task.',
    #         '_orig_mod.world_model.head_value_multi_task.',
    #         '_orig_mod.world_model.head_rewards_multi_task.',
    #         '_orig_mod.world_model.head_observations_multi_task.',
    #         '_orig_mod.world_model.task_emb.'
    #     ]
        
    #     # Define specific keys to exclude if they don't fit a prefix pattern.
    #     exclude_keys = [
    #         '_orig_mod.world_model.task_emb.weight',
    #         '_orig_mod.world_model.task_emb.bias',
    #     ]
        
    #     def filter_state_dict(state_dict_loader: Dict[str, Any], exclude_prefixes: list, exclude_keys: list = []) -> Dict[str, Any]:
    #         """
    #         Filters out parameters that should not be loaded.
    #         """
    #         filtered = {}
    #         for k, v in state_dict_loader.items():
    #             if any(k.startswith(prefix) for prefix in exclude_prefixes) or k in exclude_keys:
    #                 print(f"Excluding parameter from loading: {k}")
    #                 continue
    #             filtered[k] = v
    #         return filtered

    #     # Filter and load state_dict for the main model.
    #     if 'model' in state_dict:
    #         model_state_dict = state_dict['model']
    #         filtered_model_state_dict = filter_state_dict(model_state_dict, exclude_prefixes, exclude_keys)
    #         missing, unexpected = self._learn_model.load_state_dict(filtered_model_state_dict, strict=False)
    #         if missing:
    #             print(f"Missing keys when loading _learn_model: {missing}")
    #         if unexpected:
    #             print(f"Unexpected keys when loading _learn_model: {unexpected}")
    #     else:
    #         print("Warning: 'model' key not found in the state_dict.")

    #     # Filter and load state_dict for the target model.
    #     if 'target_model' in state_dict:
    #         target_model_state_dict = state_dict['target_model']
    #         filtered_target_model_state_dict = filter_state_dict(target_model_state_dict, exclude_prefixes, exclude_keys)
    #         missing, unexpected = self._target_model.load_state_dict(filtered_target_model_state_dict, strict=False)
    #         if missing:
    #             print(f"Missing keys when loading _target_model: {missing}")
    #         if unexpected:
    #             print(f"Unexpected keys when loading _target_model: {unexpected}")
    #     else:
    #         print("Warning: 'target_model' key not found in the state_dict.")

    #     # Load optimizer state_dict. This is often skipped during fine-tuning, but included here for completeness.
    #     if 'optimizer_world_model' in state_dict:
    #         try:
    #             self._optimizer_world_model.load_state_dict(state_dict['optimizer_world_model'])
    #         except Exception as e:
    #             print(f"Could not load optimizer state_dict: {e}. This may be expected during fine-tuning.")
    #     else:
    #         print("Warning: 'optimizer_world_model' key not found in the state_dict.")