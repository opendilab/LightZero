import copy
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY

from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
from lzero.model import ImageTransforms
from lzero.model.utils import cal_dormant_ratio
from lzero.policy import (
    scalar_transform,
    InverseScalarTransform,
    cross_entropy_loss,
    phi_transform,
    DiscreteSupport,
    to_torch_float_tensor,
    mz_network_output_unpack,
    select_action,
    negative_cosine_similarity,
    prepare_obs,
)
from lzero.policy.muzero import MuZeroPolicy


def generate_task_loss_dict(multi_task_losses: List[float], task_name_template: str, task_id: int) -> Dict[str, float]:
    """
    Overview:
        Generates a dictionary for the losses of each task.
    Arguments:
        - multi_task_losses (:obj:`List[float]`): A list containing the loss for each task.
        - task_name_template (:obj:`str`): A template for the task name, e.g., 'loss_task{}'.
        - task_id (:obj:`int`): The starting ID for the tasks.
    Returns:
        - task_loss_dict (:obj:`Dict[str, float]`): A dictionary containing the loss for each task.
    """
    task_loss_dict = {}
    for task_idx, task_loss in enumerate(multi_task_losses):
        task_name = task_name_template.format(task_idx + task_id)
        try:
            # Ensure the loss is a scalar value for logging.
            task_loss_dict[task_name] = task_loss.item() if hasattr(task_loss, 'item') else task_loss
        except Exception:
            task_loss_dict[task_name] = task_loss
    return task_loss_dict

class WrappedModelV2:
    """
    Overview:
        A wrapper class to bundle different parts of a model (tokenizer, transformer, embeddings)
        for easier management of parameters and gradients.
    """
    def __init__(self, tokenizer, transformer, pos_emb, task_emb, act_embedding_table):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.pos_emb = pos_emb
        self.task_emb = task_emb
        self.act_embedding_table = act_embedding_table

    def parameters(self) -> List[torch.nn.Parameter]:
        """
        Overview:
            Returns a list of all parameters from the tokenizer, transformer, and all embedding layers.
        """
        return (
            list(self.tokenizer.parameters()) +
            list(self.transformer.parameters()) +
            list(self.pos_emb.parameters()) +
            list(self.task_emb.parameters()) +
            list(self.act_embedding_table.parameters())
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Overview:
            Sets the gradients of all parameters in the tokenizer, transformer, and embedding layers to zero.
        Arguments:
            - set_to_none (:obj:`bool`): Whether to set gradients to None instead of zero.
        """
        self.tokenizer.zero_grad(set_to_none=set_to_none)
        self.transformer.zero_grad(set_to_none=set_to_none)
        self.pos_emb.zero_grad(set_to_none=set_to_none)
        self.task_emb.zero_grad(set_to_none=set_to_none)
        self.act_embedding_table.zero_grad(set_to_none=set_to_none)

@POLICY_REGISTRY.register('muzero_multitask')
class MuZeroMTPolicy(MuZeroPolicy):
    """
    Overview:
        The multi-task policy for MuZero, extending MuZeroPolicy. It supports training multiple tasks
        simultaneously by separating the loss for each task and optimizing them jointly.
    """

    # Default configuration for MuZeroMTPolicy.
    config = dict(
        type='muzero_multitask',
        model=dict(
            model_type='conv',  # options={'mlp', 'conv'}
            continuous_action_space=False,
            observation_shape=(4, 96, 96),  # example shape
            self_supervised_learning_loss=False,
            categorical_distribution=True,
            image_channel=1,
            frame_stack_num=1,
            num_res_blocks=1,
            num_channels=64,
            support_scale=300,
            bias=True,
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN',
            analysis_sim_norm=False,
            analysis_dormant_ratio=False,
            harmony_balance=False,
        ),
        # ****** common ******
        use_rnd_model=False,
        multi_gpu=False,
        sampled_algo=False,
        gumbel_algo=False,
        mcts_ctree=True,
        cuda=True,
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        action_type='fixed_action_space',
        battle_mode='play_with_bot_mode',
        monitor_extra_statistics=True,
        game_segment_length=200,
        eval_offline=False,
        cal_dormant_ratio=False,
        analysis_sim_norm=False,
        analysis_dormant_ratio=False,

        # ****** observation ******
        transform2string=False,
        gray_scale=False,
        use_augmentation=False,
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        use_wandb=False,
        ignore_done=False,
        update_per_collect=None,
        replay_ratio=0.25,
        batch_size=256,
        optim_type='SGD',
        learning_rate=0.2,
        target_update_freq=100,
        target_update_freq_for_intrinsic_reward=1000,
        weight_decay=1e-4,
        momentum=0.9,
        grad_clip_value=10,
        n_episode=8,
        num_segments=8,
        num_simulations=50,
        discount_factor=0.997,
        td_steps=5,
        num_unroll_steps=5,
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        policy_entropy_weight=0,
        ssl_loss_weight=0,
        lr_piecewise_constant_decay=True,
        threshold_training_steps_for_final_lr=int(5e4),
        manual_temperature_decay=False,
        threshold_training_steps_for_final_temperature=int(1e5),
        fixed_temperature_value=0.25,
        use_ture_chance_label_in_chance_encoder=False,

        # ****** Priority ******
        use_priority=False,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,

        # ****** UCB ******
        root_dirichlet_alpha=0.3,
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            eps_greedy_exploration_in_collect=False,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),

        # ****** Multi-task related ******
        task_num=2,  # Number of tasks, adjust as needed.
        task_id=0,    # The starting ID of the current task.
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration for this algorithm.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): A tuple containing the model name and a list of import paths.
        """
        return 'MuZeroMTModel', ['lzero.model.muzero_model_multitask']

    def _init_learn(self) -> None:
        """
        Overview:
            Initializes the learning mode. This method sets up the learning model, optimizer, and MCTS utilities.
        """
        super()._init_learn()

        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        # NOTE: In board games, for a fixed learning rate of 0.003, 'Adam' performs better than 'SGD'.
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

        # Learning rate scheduler
        if self._cfg.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: 1, 0.1, 0.01 are decay rates, not the learning rate itself.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # Use model_wrapper for specialized demands of different modes.
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.target_update_freq}
        )
        self._learn_model = self._model

        # Image augmentation
        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        
        # Support for categorical distribution
        self.value_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.reward_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )
        
        # ==============================================================
        # HarmonyDream (learnable weights for different losses)
        # ==============================================================
        if self._cfg.model.harmony_balance:
            # List of parameter names.
            harmony_names = ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
            # Initialize and name each parameter.
            for name in harmony_names:
                param = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
                setattr(self, name, param)
        
        # RND model for intrinsic reward
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

        # ========= Logging for analysis =========
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        self.dormant_ratio_encoder = 0.
        self.dormant_ratio_dynamics = 0.
        
        # Initialize multi-task related parameters.
        self.task_num_for_current_rank = self._cfg.task_num
        self.task_id = self._cfg.task_id

    def _forward_learn(self, data: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning, which is the core of the learning process.
            Data is sampled from the replay buffer, and the loss is calculated and backpropagated
            to update the model.
        Arguments:
            - data (:obj:`List[Tuple[torch.Tensor, torch.Tensor, int]]`): A list of data tuples for each task,
              where each tuple contains (current_batch, target_batch, task_id).
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): A dictionary of information for logging,
              including the current learning loss and other learning statistics.
        """
        self._learn_model.train()
        self._target_model.train()

        # Initialize lists for multi-task losses.
        reward_loss_multi_task = []
        policy_loss_multi_task = []
        value_loss_multi_task = []
        consistency_loss_multi_task = []
        policy_entropy_multi_task = []
        lambd_multi_task = []
        value_priority_multi_task = []
        value_priority_mean_multi_task = []

        weighted_total_loss = 0.0  # Initialize to zero.
        losses_list = []  # To store the loss for each task.

        for task_idx, (current_batch, target_batch, task_id) in enumerate(data):
            obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
            target_reward, target_value, target_policy = target_batch

            obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

            # Data augmentation.
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

            # Prepare action batch and convert to tensor.
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
            data_list = [mask_batch, target_reward, target_value, target_policy, weights]
            mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(
                data_list, self._cfg.device
            )

            target_reward = target_reward.view(self._cfg.batch_size[task_idx], -1)
            target_value = target_value.view(self._cfg.batch_size[task_idx], -1)

            assert obs_batch.size(0) == self._cfg.batch_size[task_idx] == target_reward.size(0)

            # Transform rewards and values to scaled representation.
            transformed_target_reward = scalar_transform(target_reward)
            transformed_target_value = scalar_transform(target_value)

            # Convert to categorical distribution.
            target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
            target_value_categorical = phi_transform(self.value_support, transformed_target_value)

            # Initial inference.
            network_output = self._learn_model.initial_inference(obs_batch, task_id=task_id)

            latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

            # Log Dormant Ratio and L2 Norm.
            if self._cfg.cal_dormant_ratio:
                self.dormant_ratio_encoder = cal_dormant_ratio(
                    self._learn_model.representation_network, obs_batch.detach(),
                    percentage=self._cfg.dormant_threshold
                )
            latent_state_l2_norms = torch.norm(latent_state.view(latent_state.shape[0], -1), p=2, dim=1).mean()

            # Inverse transform value.
            original_value = self.inverse_scalar_transform_handle(value)

            # Initialize predicted values and policies.
            predicted_rewards = []
            if self._cfg.monitor_extra_statistics:
                predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                    policy_logits, dim=1
                ).detach().cpu()

            # Calculate priority.
            value_priority = torch.nn.L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
            value_priority = value_priority.data.cpu().numpy() + 1e-6

            # Calculate policy and value loss for the first step.
            policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])
            value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])

            prob = torch.softmax(policy_logits, dim=-1)
            entropy = -(prob * torch.log(prob + 1e-9)).sum(-1)
            policy_entropy_loss = -entropy

            reward_loss = torch.zeros(self._cfg.batch_size[task_idx], device=self._cfg.device)
            consistency_loss = torch.zeros(self._cfg.batch_size[task_idx], device=self._cfg.device)
            target_policy_entropy = 0

            # Unroll loop for multiple steps.
            for step_k in range(self._cfg.num_unroll_steps):
                # Recurrent inference using the dynamics function.
                network_output = self._learn_model.recurrent_inference(latent_state, action_batch[:, step_k])
                latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

                # Log Dormant Ratio for the dynamics network.
                if step_k == self._cfg.num_unroll_steps - 1 and self._cfg.cal_dormant_ratio:
                    action_tmp = action_batch[:, step_k]
                    if len(action_tmp.shape) == 1:
                        action_tmp = action_tmp.unsqueeze(-1)
                    # Convert action to one-hot encoding.
                    action_one_hot = torch.zeros(action_tmp.shape[0], policy_logits.shape[-1], device=action_tmp.device)
                    action_tmp = action_tmp.long()
                    action_one_hot.scatter_(1, action_tmp, 1)
                    action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
                    action_encoding = action_encoding_tmp.expand(
                        latent_state.shape[0], policy_logits.shape[-1], latent_state.shape[2], latent_state.shape[3]
                    )
                    state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)
                    self.dormant_ratio_dynamics = cal_dormant_ratio(
                        self._learn_model.dynamics_network,
                        state_action_encoding.detach(),
                        percentage=self._cfg.dormant_threshold
                    )

                # Inverse transform value.
                original_value = self.inverse_scalar_transform_handle(value)

                # Calculate consistency loss (self-supervised learning).
                if self._cfg.model.self_supervised_learning_loss and self._cfg.ssl_loss_weight > 0:
                    beg_index, end_index = self._get_target_obs_index_in_step_k(step_k)
                    network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index], task_id=task_id)

                    latent_state = to_tensor(latent_state)
                    representation_state = to_tensor(network_output.latent_state)

                    dynamic_proj = self._learn_model.project(latent_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_k]
                    consistency_loss += temp_loss

                # Calculate policy and value losses.
                policy_loss += cross_entropy_loss(policy_logits, target_policy[:, step_k + 1])
                value_loss += cross_entropy_loss(value, target_value_categorical[:, step_k + 1])
                reward_loss += cross_entropy_loss(reward, target_reward_categorical[:, step_k])

                # Calculate policy entropy loss.
                prob = torch.softmax(policy_logits, dim=-1)
                entropy = -(prob * torch.log(prob + 1e-9)).sum(-1)
                policy_entropy_loss += -entropy

                # Calculate target policy entropy (for debugging purposes only).
                target_normalized_visit_count = target_policy[:, step_k + 1]
                non_masked_indices = torch.nonzero(mask_batch[:, step_k + 1]).squeeze(-1)
                if len(non_masked_indices) > 0:
                    target_normalized_visit_count_masked = torch.index_select(
                        target_normalized_visit_count, 0, non_masked_indices
                    )
                    target_policy_entropy += -(
                        (target_normalized_visit_count_masked + 1e-6) *
                        torch.log(target_normalized_visit_count_masked + 1e-6)
                    ).sum(-1).mean()
                else:
                    target_policy_entropy += torch.log(
                        torch.tensor(target_normalized_visit_count.shape[-1], device=self._cfg.device)
                    )

                # Log predicted values and rewards if monitoring extra statistics.
                if self._cfg.monitor_extra_statistics:
                    original_rewards = self.inverse_scalar_transform_handle(reward)
                    original_rewards_cpu = original_rewards.detach().cpu()

                    predicted_values = torch.cat(
                        (predicted_values, self.inverse_scalar_transform_handle(value).detach().cpu())
                    )
                    predicted_rewards.append(original_rewards_cpu)
                    predicted_policies = torch.cat(
                        (predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu())
                    )

            # Core learning model update step.
            weighted_loss = self._cfg.policy_loss_weight * policy_loss + \
                            self._cfg.value_loss_weight * value_loss + \
                            self._cfg.reward_loss_weight * reward_loss + \
                            self._cfg.ssl_loss_weight * consistency_loss + \
                            self._cfg.policy_entropy_weight * policy_entropy_loss

            # Accumulate losses from multiple tasks.
            weighted_total_loss += weighted_loss.mean()

            # Store per-task losses for logging.
            reward_loss_multi_task.append(reward_loss.mean().item())
            policy_loss_multi_task.append(policy_loss.mean().item())
            value_loss_multi_task.append(value_loss.mean().item())
            consistency_loss_multi_task.append(consistency_loss.mean().item())
            policy_entropy_multi_task.append(policy_entropy_loss.mean().item())
            # TODO: Adjust if using gradient correction.
            lambd_multi_task.append(torch.tensor(0., device=self._cfg.device).item())
            value_priority_multi_task.append(value_priority.mean().item())
            value_priority_mean_multi_task.append(value_priority.mean().item())
            losses_list.append(weighted_loss.mean().item())

        # Zero the optimizer's gradients.
        self._optimizer.zero_grad()

        # Backward pass.
        weighted_total_loss.backward()

        # Gradient clipping.
        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(),
            self._cfg.grad_clip_value
        )

        # Sync gradients for multi-GPU training.
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)

        # Update optimizer.
        self._optimizer.step()
        if self._cfg.lr_piecewise_constant_decay:
            self.lr_scheduler.step()

        # Update target model.
        self._target_model.update(self._learn_model.state_dict())

        # Get GPU memory usage.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        else:
            current_memory_allocated_gb = 0.0
            max_memory_allocated_gb = 0.0

        # Build the return loss dictionary.
        return_loss_dict = {
            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self.collect_epsilon,
            'cur_lr_world_model': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
        }

        # Generate task-specific loss dictionaries, prefixing each with "noreduce_".
        multi_task_loss_dicts = {
            **generate_task_loss_dict(consistency_loss_multi_task, 'noreduce_consistency_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(reward_loss_multi_task, 'noreduce_reward_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(policy_loss_multi_task, 'noreduce_policy_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_loss_multi_task, 'noreduce_value_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(policy_entropy_multi_task, 'noreduce_policy_entropy_task{}', task_id=self.task_id),
            **generate_task_loss_dict(lambd_multi_task, 'noreduce_lambd_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_multi_task, 'noreduce_value_priority_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_mean_multi_task, 'noreduce_value_priority_mean_task{}', task_id=self.task_id),
        }

        # Merge the dictionaries.
        return_loss_dict.update(multi_task_loss_dicts)

        # Return the final loss dictionary.
        return return_loss_dict

    def _reset_collect(self, data_id: Optional[List[int]] = None, task_id: int = None) -> None:
        """
        Overview:
            Reset the observation and action for the collector environment.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
            - task_id (:obj:`int`): The ID of the task.
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

    def _reset_eval(self, data_id: Optional[List[int]] = None, task_id: int = None) -> None:
        """
        Overview:
            Reset the observation and action for the evaluator environment.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
            - task_id (:obj:`int`): The ID of the task.
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.evaluator_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.evaluator_env_num)]
    
 
    def _monitor_vars_learn(self, num_tasks: int = None) -> List[str]:
        """
        Overview:
            Registers variables to be monitored during the learning phase. The registered variables
            will be recorded to TensorBoard based on the return value of `_forward_learn`.
            If `num_tasks` is provided, it generates monitoring variables for each task.
        Arguments:
            - num_tasks (:obj:`int`, optional): The number of tasks.
        Returns:
            - monitored_vars (:obj:`List[str]`): A list of variable names to be monitored.
        """
        # Basic monitoring variables.
        monitored_vars = [
            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'weighted_total_loss',
            'total_grad_norm_before_clip_wm',
        ]

        # Task-specific monitoring variables.
        task_specific_vars = [
            'noreduce_consistency_loss',
            'noreduce_reward_loss',
            'noreduce_policy_loss',
            'noreduce_value_loss',
            'noreduce_policy_entropy',
            'noreduce_lambd',
            'noreduce_value_priority',
            'noreduce_value_priority_mean',
        ]
        
        # Use self.task_num_for_current_rank as the number of tasks for the current rank.
        num_tasks = self.task_num_for_current_rank
        print(f'self.task_num_for_current_rank: {self.task_num_for_current_rank}')
        if num_tasks is not None:
            for var in task_specific_vars:
                for task_idx in range(num_tasks):
                    monitored_vars.append(f'{var}_task{self.task_id + task_idx}')
        else:
            monitored_vars.extend(task_specific_vars)

        return monitored_vars
       
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
        if self._cfg.model.model_type == 'conv_context':
            self.last_batch_obs = torch.zeros([8, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(8)]

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            task_id: int = None,
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
            - task_id (:obj:`int`): The ID of the task.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._collect_model.initial_inference(data, task_id=task_id)
            elif self._cfg.model.model_type == "conv_context":
                network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action,
                                                                       data, task_id=task_id)

            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            if not self._cfg.collect_with_pure_policy:
                # The only difference between collect and eval is the dirichlet noise.
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                if self._cfg.mcts_ctree:
                    # C++ MCTS tree.
                    roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
                else:
                    # Python MCTS tree.
                    roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, task_id=task_id)

                # List of lists, shape: ``{list: batch_size} -> {list: action_space_size}``
                roots_visit_count_distributions = roots.get_distributions()
                roots_values = roots.get_values()  # shape: {list: batch_size}

                batch_action = []
                for i, env_id in enumerate(ready_env_id):
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    if self._cfg.eps.eps_greedy_exploration_in_collect:
                        # Epsilon-greedy exploration for collection.
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=True
                        )
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else:
                        # Normal collection.
                        # NOTE: Only legal actions possess visit counts, so ``action_index_in_legal_action_set`` represents
                        # the index within the legal action set, not the entire action set.
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=False
                        )
                        # NOTE: Convert ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
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
                    self.last_batch_obs = data
                    self.last_batch_action = batch_action
            else:
                # Pure policy collection (without MCTS).
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

    def _get_target_obs_index_in_step_k(self, step: int) -> Tuple[int, int]:
        """
        Overview:
            Get the begin and end indices of the target observation at step k.
        Arguments:
            - step (:obj:`int`): The current step k.
        Returns:
            - beg_index (:obj:`int`): The beginning index of the target observation.
            - end_index (:obj:`int`): The ending index of the target observation.
        """
        if self._cfg.model.model_type in ['conv', 'conv_context']:
            beg_index = self._cfg.model.image_channel * step
            end_index = self._cfg.model.image_channel * (step + self._cfg.model.frame_stack_num)
        elif self._cfg.model.model_type in ['mlp', 'mlp_context']:
            beg_index = self._cfg.model.observation_shape * step
            end_index = self._cfg.model.observation_shape * (step + self._cfg.model.frame_stack_num)
        return beg_index, end_index

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
        if self._cfg.model.model_type == 'conv_context':
            self.last_batch_obs = torch.zeros([3, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(3)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, task_id: int = None) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
            - task_id (:obj:`int`): The ID of the task.
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
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._eval_model.initial_inference(data, task_id=task_id)
            elif self._cfg.model.model_type == "conv_context":
                network_output = self._eval_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._eval_model.training:
                # If not in training, obtain the scalar values of the value/reward.
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape (B, 1)
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape (B, A)

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # C++ MCTS tree.
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # Python MCTS tree.
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, task_id=task_id)

            # List of lists, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, not the entire action set.
                # Setting deterministic=True implies choosing the action with the highest value (argmax)
                # rather than sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
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
                if self._cfg.model.model_type in ["conv_context"]:
                    batch_action.append(action)

            if self._cfg.model.model_type in ["conv_context"]:
                self.last_batch_obs = data
                self.last_batch_action = batch_action

        return output