# /Users/puyuan/code/LightZero/lzero/policy/sample_unizero_multitask.py

import copy
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import wandb
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY

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
    prepare_obs_stack4_for_unizero
)
from lzero.policy.unizero import UniZeroPolicy
from .utils import configure_optimizers_nanogpt
import torch.nn.functional as F
import torch.distributed as dist
from ding.utils import set_pkg_seed, get_rank, get_world_size

import sys
sys.path.append('/mnt/afs/niuyazhe/code/LibMTL/')
from LibMTL.weighting.MoCo_unizero import MoCo as GradCorrect
# from LibMTL.weighting.CAGrad_unizero import CAGrad as GradCorrect

def generate_task_loss_dict(multi_task_losses, task_name_template, task_id):
    """
    生成每个任务的损失字典
    :param multi_task_losses: 包含每个任务损失的列表
    :param task_name_template: 任务名称模板，例如 'obs_loss_task{}'
    :param task_id: 基础任务 ID
    :return: 一个字典，包含每个任务的损失
    """
    task_loss_dict = {}
    for task_idx, task_loss in enumerate(multi_task_losses):
        task_name = task_name_template.format(task_idx + task_id)
        try:
            task_loss_dict[task_name] = task_loss.item() if hasattr(task_loss, 'item') else float(task_loss)
        except Exception as e:
            task_loss_dict[task_name] = task_loss
    return task_loss_dict


class WrappedModelV2:
    def __init__(self, tokenizer, transformer, pos_emb, task_emb, act_embedding_table):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.pos_emb = pos_emb
        self.task_emb = task_emb
        self.act_embedding_table = act_embedding_table

    def parameters(self):
        # 返回 tokenizer, transformer 以及所有嵌入层的参数
        return (
            list(self.tokenizer.parameters()) +
            list(self.transformer.parameters()) +
            list(self.pos_emb.parameters()) +
            # list(self.task_emb.parameters()) +
            list(self.act_embedding_table.parameters())
        )

    def zero_grad(self, set_to_none=False):
        # 将 tokenizer, transformer 和所有嵌入层的梯度设为零
        self.tokenizer.zero_grad(set_to_none=set_to_none)
        self.transformer.zero_grad(set_to_none=set_to_none)
        self.pos_emb.zero_grad(set_to_none=set_to_none)
        # self.task_emb.zero_grad(set_to_none=set_to_none)
        self.act_embedding_table.zero_grad(set_to_none=set_to_none)
    
    def get_group_parameters(self):
        """
        返回一个字典，其中 key 为模块名或更细粒度的层，
        value 为对应的参数列表。注意返回顺序应与 parameters()方法中参数的排列顺序一致。
        """
        groups = {}
        groups['tokenizer'] = list(self.tokenizer.parameters())
        groups['transformer'] = list(self.transformer.parameters())
        groups['pos_emb'] = list(self.pos_emb.parameters())
        groups['act_embedding_table'] = list(self.act_embedding_table.parameters())

        # 如 transformer 内部分层（假设 transformer.blocks 是列表）
        if hasattr(self.transformer, 'blocks'):
            # 若要单独统计 transformer 内各层，保持原 transformer 参数在 parameters() 中顺序不变，
            # 可以在这里添加各层的切片，但需保证 parameters() 返回的顺序与此一致，
            # 此处仅作为示例：
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
                dormant_threshold=0.025,
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
        Return this algorithm's default model setting for demonstration.
        """
        return 'SampledUniZeroMTModel', ['lzero.model.sampled_unizero_model_multitask']

    def _init_learn(self) -> None:
        """
        Learn mode init method. Initialize the learn model, optimizer, and MCTS utils.
        """
        # Configure optimizer for world model
        self._optimizer_world_model = configure_optimizers_nanogpt(
            model=self._model.world_model,
            learning_rate=self._cfg.learning_rate,
            weight_decay=self._cfg.weight_decay,
            device_type=self._cfg.device,
            betas=(0.9, 0.95),
        )

        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

            if self._cfg.cos_lr_scheduler:
                self.lr_scheduler = CosineAnnealingLR(
                    self._optimizer_world_model, T_max=int(1e5), eta_min=0, last_epoch=-1
                )
            elif self._cfg.piecewise_decay_lr_scheduler:
                # Example step scheduler, adjust milestones and gamma as needed
                self.lr_scheduler = StepLR(
                    self._optimizer_world_model, step_size=int(5e4), gamma=0.1
                )

        if self._cfg.model.continuous_action_space:
            # Weight Init for the last output layer of gaussian policy head in prediction network.
            init_w = self._cfg.init_w
            self._model.world_model.fc_policy_head.mu.weight.data.uniform_(-init_w, init_w)
            self._model.world_model.fc_policy_head.mu.bias.data.uniform_(-init_w, init_w)
            try:
                self._model.world_model.fc_policy_head.log_sigma_layer.weight.data.uniform_(-init_w, init_w)
                self._model.world_model.fc_policy_head.log_sigma_layer.bias.data.uniform_(-init_w, init_w)
            except Exception as exception:
                logging.warning(exception)

        # Initialize target model
        self._target_model = copy.deepcopy(self._model)
        # Ensure torch version >= 2.0
        assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "We need torch version >= 2.0"
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)
        # Soft target update
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.target_update_theta}
        )
        self._learn_model = self._model

        # if self._cfg.use_augmentation:
        #     self.image_transforms = ImageTransforms(
        #         self._cfg.augmentation,
        #         image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
        #     )

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

        self.task_id = self._cfg.task_id
        self.task_num_for_current_rank = self._cfg.task_num
        print(f'self._cfg.only_use_moco_stats:{self._cfg.only_use_moco_stats}')
        if self._cfg.use_moco or self._cfg.only_use_moco_stats:
            # 创建 WrappedModel 实例，仅矫正部分参数，保持可扩展性
            # wrapped_model = WrappedModelV2(
            #     self._learn_model.world_model.tokenizer.encoder[0],  # 假设只有一个编码器
            #     self._learn_model.world_model.transformer,
            #     self._learn_model.world_model.pos_emb,
            #     self._learn_model.world_model.task_emb,
            #     self._learn_model.world_model.act_embedding_table,
            # )

            # head 没有矫正梯度
            wrapped_model = WrappedModelV2(
                self._learn_model.world_model.tokenizer.encoder,  # TODO: one or N encoder inside
                self._learn_model.world_model.transformer,
                self._learn_model.world_model.pos_emb,
                self._learn_model.world_model.task_emb,
                self._learn_model.world_model.act_embedding_table,
            )

            # TODO
            # 如果需要，可以在这里初始化梯度校正方法（如 MoCo, CAGrad）
            # self.grad_correct = GradCorrect(wrapped_model, self.task_num, self._cfg.device)
            # self.grad_correct = GradCorrect(wrapped_model, self._cfg.task_num, self._cfg.device, self._cfg.multi_gpu) # only compatiable with for 1GPU training
            self.grad_correct = GradCorrect(wrapped_model, self._cfg.total_task_num, self._cfg.device, self._cfg.multi_gpu) # only compatiable with for 1GPU training

            self.grad_correct.init_param()
            self.grad_correct.rep_grad = False


    def _forward_learn(self, data: Tuple[torch.Tensor], task_weights=None, ignore_grad=False) -> Dict[str, Union[float, int]]:
        """
        Forward function for learning policy in learn mode, handling multiple tasks.
        """
        self._learn_model.train()
        self._target_model.train()

        # Initialize multi-task loss lists
        task_weight_multi_task = []

        obs_loss_multi_task = []
        reward_loss_multi_task = []
        policy_loss_multi_task = []
        orig_policy_loss_multi_task = []
        policy_entropy_multi_task = []
        value_loss_multi_task = []
        latent_recon_loss_multi_task = []
        perceptual_loss_multi_task = []
        latent_state_l2_norms_multi_task = []
        average_target_policy_entropy_multi_task = []
        value_priority_multi_task = []
        value_priority_mean_multi_task = []

        weighted_total_loss = 0.0
        losses_list = []  # 存储每个任务的损失

        for task_id, data_one_task in enumerate(data):
            current_batch, target_batch, task_id = data_one_task
            obs_batch_ori, action_batch, child_sampled_actions_batch, target_action_batch, mask_batch, indices, weights, make_time = current_batch
            target_reward, target_value, target_policy = target_batch

            # Prepare observations based on frame stack number
            if self._cfg.model.frame_stack_num == 4:
                obs_batch, obs_target_batch = prepare_obs_stack4_for_unizero(obs_batch_ori, self._cfg)
            else:
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg, task_id)

            # Apply augmentations if needed
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

            # Prepare action batch and convert to torch tensor
            if self._cfg.model.continuous_action_space:
                action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1)
            else:
                action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
            data_list = [
                mask_batch,
                target_reward.astype('float32'),
                target_value.astype('float32'),
                target_policy,
                weights
            ]
            mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list, self._cfg.device)

            target_reward = target_reward.view(self._cfg.batch_size[task_id], -1)
            target_value = target_value.view(self._cfg.batch_size[task_id], -1)

            # Transform rewards and values to their scaled forms
            transformed_target_reward = scalar_transform(target_reward)
            transformed_target_value = scalar_transform(target_value)

            # Convert to categorical distributions
            target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
            target_value_categorical = phi_transform(self.value_support, transformed_target_value)

            # Prepare batch for GPT model
            batch_for_gpt = {}
            if isinstance(self._cfg.model.observation_shape_list[task_id], int) or len(self._cfg.model.observation_shape_list[task_id]) == 1:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    self._cfg.batch_size[task_id], -1, self._cfg.model.observation_shape_list[task_id])
            elif len(self._cfg.model.observation_shape_list[task_id]) == 3:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    self._cfg.batch_size[task_id], -1, *self._cfg.model.observation_shape_list[task_id])

            batch_for_gpt['actions'] = action_batch.squeeze(-1)
            batch_for_gpt['child_sampled_actions'] = torch.from_numpy(child_sampled_actions_batch).to(self._cfg.device)[:, :-1]
            batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
            batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data
            batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
            batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
            batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long, device=self._cfg.device)
            batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
            batch_for_gpt['target_policy'] = target_policy[:, :-1]

            # Extract valid target policy data and compute entropy
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
            average_target_policy_entropy = target_policy_entropy.mean().item()

            # Update world model
            losses = self._learn_model.world_model.compute_loss(
                batch_for_gpt,
                self._target_model.world_model.tokenizer,
                self.inverse_scalar_transform_handle,
                task_id=task_id
            )
            if task_weights is not None:
                weighted_total_loss += losses.loss_total * task_weights[task_id]
                losses_list.append(losses.loss_total * task_weights[task_id])

                task_weight_multi_task.append(task_weights[task_id])
            else:
                weighted_total_loss += losses.loss_total
                losses_list.append(losses.loss_total)

                task_weight_multi_task.append(1)


            for loss_name, loss_value in losses.intermediate_losses.items():
                self.intermediate_losses[f"{loss_name}"] = loss_value
                # print(f'{loss_name}: {loss_value.sum()}')
                # print(f'{loss_name}: {loss_value[0][0]}')

            # print(f"=== 全局任务权重 (按 task_id 排列): {task_weights}")
            # assert not torch.isnan(losses.loss_total).any(), f"Loss contains NaN values, losses.loss_total:{losses.loss_total}, losses:{losses}"
            # assert not torch.isinf(losses.loss_total).any(), f"Loss contains Inf values, losses.loss_total:{losses.loss_total}, losses:{losses}"

            # Collect losses per task
            obs_loss = self.intermediate_losses.get('loss_obs', 0.0) or 0.0
            reward_loss = self.intermediate_losses.get('loss_rewards', 0.0) or 0.0
            policy_loss = self.intermediate_losses.get('loss_policy', 0.0) or 0.0
            orig_policy_loss = self.intermediate_losses.get('orig_policy_loss', 0.0) or 0.0
            policy_entropy = self.intermediate_losses.get('policy_entropy', 0.0) or 0.0
            value_loss = self.intermediate_losses.get('loss_value', 0.0) or 0.0
            latent_recon_loss = self.intermediate_losses.get('latent_recon_loss', 0.0) or 0.0
            perceptual_loss = self.intermediate_losses.get('perceptual_loss', 0.0) or 0.0
            latent_state_l2_norms = self.intermediate_losses.get('latent_state_l2_norms', 0.0) or 0.0
            value_priority = torch.tensor(0., device=self._cfg.device)  # Placeholder, adjust as needed

            obs_loss_multi_task.append(obs_loss)
            reward_loss_multi_task.append(reward_loss)
            policy_loss_multi_task.append(policy_loss)
            orig_policy_loss_multi_task.append(orig_policy_loss)
            policy_entropy_multi_task.append(policy_entropy)
            value_loss_multi_task.append(value_loss)
            latent_recon_loss_multi_task.append(latent_recon_loss)
            perceptual_loss_multi_task.append(perceptual_loss)
            latent_state_l2_norms_multi_task.append(latent_state_l2_norms)
            average_target_policy_entropy_multi_task.append(average_target_policy_entropy)
            value_priority_multi_task.append(value_priority)
            value_priority_mean_multi_task.append(value_priority.mean().item())

        # Core learn model update step
        self._optimizer_world_model.zero_grad()

        # 假设每个进程计算出的 losses_list 为可求梯度的 tensor list，比如多个标量 loss 组成的列表
        # 例如 losses_list = [loss1, loss2, ...]，其中每个 loss_i 都是形如 (1,) 的 tensor 且 requires_grad=True
        if self._cfg.use_moco:
            # 调用 MoCo backward，由 grad_correct 中的 backward 实现梯度校正
            lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)
            # print(f'rank:{get_rank()}, after moco backword')
        elif self._cfg.only_use_moco_stats:
            lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)
            # 不使用梯度校正的情况，由各 rank 自己执行反向传播
            weighted_total_loss.backward()
        else:
            # 不使用梯度校正的情况，由各 rank 自己执行反向传播
            lambd = torch.tensor([0. for _ in range(self.task_num_for_current_rank)], device=self._cfg.device)
            weighted_total_loss.backward()

        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(self._learn_model.world_model.parameters(), self._cfg.grad_clip_value)

        if ignore_grad:
            #  =========== NOTE: 对于一个GPU上所有任务都解决了的情况，为了ddp同步仍然调用train但是grad应该清零 ===========
            self._optimizer_world_model.zero_grad()
            # print(f"ignore_grad")

        if self._cfg.multi_gpu:
            # if not self._cfg.use_moco or self._cfg.only_use_moco_stats:
            #     self.sync_gradients(self._learn_model)
            if not self._cfg.use_moco:
                # self.sync_gradients(self._learn_model)
                # dist.barrier() # ================== TODO: ==================
                self.sync_gradients(self._learn_model)
                # print(f'rank:{get_rank()}, after self.sync_gradients(self._learn_model)')

        self._optimizer_world_model.step()

        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Core target model update step
        self._target_model.update(self._learn_model.state_dict())

        # 获取GPU内存使用情况
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        else:
            current_memory_allocated_gb = 0.
            max_memory_allocated_gb = 0.

        # 构建损失字典
        return_loss_dict = {
            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self._collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
        }

        # if task_weights is None:
        #     task_weights = {self.task_id+i: 1 for i in range(self.task_num_for_current_rank)}
        # else:
        #     print(f'task_weights:{task_weights}')
        # from ding.utils import EasyTimer, set_pkg_seed, get_rank

        # print(f'rank:{get_rank()}, task_id:{self.task_id}')

        # 生成任务相关的损失字典，并为每个任务相关的 loss 添加前缀 "noreduce_"
        multi_task_loss_dicts = {
            **generate_task_loss_dict(task_weight_multi_task, 'noreduce_task_weight_task{}', task_id=self.task_id),
            **generate_task_loss_dict(obs_loss_multi_task, 'noreduce_obs_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(latent_recon_loss_multi_task, 'noreduce_latent_recon_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(perceptual_loss_multi_task, 'noreduce_perceptual_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(latent_state_l2_norms_multi_task, 'noreduce_latent_state_l2_norms_task{}', task_id=self.task_id),
            **generate_task_loss_dict(policy_loss_multi_task, 'noreduce_policy_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(orig_policy_loss_multi_task, 'noreduce_orig_policy_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(policy_entropy_multi_task, 'noreduce_policy_entropy_task{}', task_id=self.task_id),
            **generate_task_loss_dict(reward_loss_multi_task, 'noreduce_reward_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_loss_multi_task, 'noreduce_value_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(average_target_policy_entropy_multi_task, 'noreduce_target_policy_entropy_task{}', task_id=self.task_id),
            **generate_task_loss_dict(lambd, 'noreduce_lambd_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_multi_task, 'noreduce_value_priority_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_mean_multi_task, 'noreduce_value_priority_mean_task{}', task_id=self.task_id),
        }

        # print(f'multi_task_loss_dicts:{ multi_task_loss_dicts}')

        # 合并两个字典
        return_loss_dict.update(multi_task_loss_dicts)

        # 如果需要，可以将损失字典记录到日志或其他地方
        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_loss_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_loss_dict

    # TODO: num_tasks
    def _monitor_vars_learn(self, num_tasks=2) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
            If num_tasks is provided, generate monitored variables for each task.
        """
        # Basic monitored variables that do not depend on the number of tasks
        monitored_vars = [
            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'weighted_total_loss',
            'total_grad_norm_before_clip_wm',
        ]

        # rank = get_rank()
        task_specific_vars = [
            'noreduce_task_weight',
            'noreduce_obs_loss',
            'noreduce_orig_policy_loss',
            'noreduce_policy_loss',
            'noreduce_latent_recon_loss',
            'noreduce_policy_entropy',
            'noreduce_target_policy_entropy',
            'noreduce_reward_loss',
            'noreduce_value_loss',
            'noreduce_perceptual_loss',
            'noreduce_latent_state_l2_norms',
            'noreduce_lambd',
            'noreduce_value_priority_mean',
        ]
        # self.task_num_for_current_rank 作为当前rank的base_index
        num_tasks = self.task_num_for_current_rank
        # If the number of tasks is provided, extend the monitored variables list with task-specific variables
        if num_tasks is not None:
            for var in task_specific_vars:
                for task_idx in range(num_tasks):
                    # print(f"learner policy Rank {rank}, self.task_id: {self.task_id}")
                    monitored_vars.append(f'{var}_task{self.task_id+task_idx}')
        else:
            # If num_tasks is not provided, we assume there's only one task and keep the original variable names
            monitored_vars.extend(task_specific_vars)

        return monitored_vars
        
    def monitor_weights_and_grads(self, model):
        """
        Monitor and print the weights and gradients of the model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | "
                      f"Weight mean: {param.data.mean():.4f} | "
                      f"Weight std: {param.data.std():.4f} | "
                      f"Grad mean: {param.grad.mean():.4f} | "
                      f"Grad std: {param.grad.std():.4f}")

    def _init_collect(self) -> None:
        """
        Collect mode init method. Initialize the collect model and MCTS utils.
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
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros(
                [self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]
            ).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.collector_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros(
                [self.collector_env_num, self._cfg.model.observation_shape_list[0]]
            ).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.collector_env_num)]

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
        Forward function for collecting data in collect mode, handling multiple tasks.
        """
        self._collect_model.eval()

        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            network_output = self._collect_model.initial_inference(
                self.last_batch_obs,
                self.last_batch_action,
                data,
                task_id=task_id
            )
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
            ] if not self._cfg.model.continuous_action_space else [
                [-1 for _ in range(self._cfg.model.world_model_cfg.num_of_sampled_actions)]
                for _ in range(active_collect_env_num)
            ]

            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(self._cfg.model.world_model_cfg.num_of_sampled_actions))
                .astype(np.float32).tolist() for _ in range(active_collect_env_num)
            ]

            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(
                    active_collect_env_num,
                    legal_actions,
                    self._cfg.model.world_model_cfg.action_space_size,
                    self._cfg.model.world_model_cfg.num_of_sampled_actions,
                    self._cfg.model.continuous_action_space
                )
            else:
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
            
            # try:
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, task_id=task_id)
                # print("latent_state_roots.shape:", latent_state_roots.shape)
            # except Exception as e:
            #     print("="*20)
            #     print(e)
            #     print("roots:", roots, "latent_state_roots:", latent_state_roots)
            #     print("latent_state_roots.shape:", latent_state_roots.shape)
            #     print("="*20)
            #     import ipdb; ipdb.set_trace()


            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            roots_sampled_actions = roots.get_sampled_actions()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                root_sampled_actions = np.array([
                    getattr(action, 'value', action) for action in roots_sampled_actions[i]
                ])

                # 选择动作
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self._collect_mcts_temperature, deterministic=False
                )

                # 获取采样动作
                action = root_sampled_actions[action]
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

            self.last_batch_obs = data
            self.last_batch_action = batch_action

            # 检查并重置采集器
            if active_collect_env_num < self.collector_env_num:
                print('==========collect_forward============')
                print(f'len(self.last_batch_obs) < self.collector_env_num, {active_collect_env_num}<{self.collector_env_num}')
                self._reset_collect(reset_init_data=True, task_id=task_id)

        return output

    def _init_eval(self) -> None:
        """
        Evaluate mode init method. Initialize the eval model and MCTS utils.
        """
        from ding.utils import EasyTimer, set_pkg_seed, get_rank

        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)
        self.evaluator_env_num = self._cfg.evaluator_env_num

        self.task_id_for_eval = self._cfg.task_id
        self.task_num_for_current_rank = self._cfg.task_num
        
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs_eval = torch.zeros(
                [self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]
            ).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs_eval = torch.zeros(
                [self.evaluator_env_num, self._cfg.model.observation_shape_list[self.task_id_for_eval]] # TODO
            ).to(self._cfg.device)
            print(f'rank {get_rank()} last_batch_obs_eval:', self.last_batch_obs_eval.shape)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, task_id: int = None) -> Dict:
        """
        Forward function for evaluating the current policy in eval mode, handling multiple tasks.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            network_output = self._eval_model.initial_inference(
                self.last_batch_obs_eval,
                self.last_batch_action,
                data,
                task_id=task_id
            )
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            # TODO:========
            # self._eval_model.training = False
            # if not self._eval_model.training:
            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
            ] if not self._cfg.model.continuous_action_space else [
                [-1 for _ in range(self._cfg.model.world_model_cfg.num_of_sampled_actions)]
                for _ in range(active_eval_env_num)
            ]

            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(
                    active_eval_env_num,
                    legal_actions,
                    self._cfg.model.world_model_cfg.action_space_size,
                    self._cfg.model.world_model_cfg.num_of_sampled_actions,
                    self._cfg.model.continuous_action_space
                )
            else:
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)

            # print(f'type(policy_logits): {type(policy_logits)}')
            # print(f'policy_logits.shape: {policy_logits.shape}')
            # print(f'policy_logits: {policy_logits}')

            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, task_id=task_id)

            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            roots_sampled_actions = roots.get_sampled_actions()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                root_sampled_actions = np.array([
                    getattr(action, 'value', action) for action in roots_sampled_actions[i]
                ])

                # 选择动作（确定性）
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )

                # 获取采样动作
                action = root_sampled_actions[action]
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

            self.last_batch_obs_eval = data
            self.last_batch_action = batch_action

        return output

    def _reset_collect(self, env_id: int = None, current_steps: int = 0, reset_init_data: bool = True, task_id: int = None) -> None:
        """
        Reset the collection process for a specific environment.
        """
        if reset_init_data:
            if task_id is not None:
                self.last_batch_obs = initialize_zeros_batch(
                    self._cfg.model.observation_shape_list[task_id],
                    self._cfg.collector_env_num,
                    self._cfg.device
                )
            else:
                self.last_batch_obs = initialize_zeros_batch(
                    self._cfg.model.observation_shape,
                    self._cfg.collector_env_num,
                    self._cfg.device
                )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]
            logging.info(f'collector: last_batch_obs, last_batch_action reset() {self.last_batch_obs.shape}')

        if env_id is None or isinstance(env_id, list):
            return

        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200

        if current_steps % clear_interval == 0:
            logging.info(f'clear_interval: {clear_interval}')

            world_model = self._collect_model.world_model
            world_model.past_kv_cache_init_infer.clear()
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            torch.cuda.empty_cache()

            logging.info('collector: collect_model clear()')
            logging.info(f'eps_steps_lst[{env_id}]: {current_steps}')

            self._reset_target_model()

    def _reset_target_model(self) -> None:
        """
        Reset the target model's caches.
        """
        world_model = self._target_model.world_model
        world_model.past_kv_cache_init_infer.clear()
        for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        world_model.past_kv_cache_recurrent_infer.clear()
        world_model.keys_values_wm_list.clear()

        torch.cuda.empty_cache()
        logging.info('collector: target_model past_kv_cache.clear()')

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Return the state_dict of learn mode, including model, target_model, and optimizer.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_world_model': self._optimizer_world_model.state_dict(),
        }

    # ========== TODO: original version: load all parameters ==========
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

    # ========== TODO: pretrain-finetue version: only load encoder and transformer-backbone parameters  ==========
    # def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
    #     """
    #     Overview:
    #         Load the state_dict variable into policy learn mode, excluding multi-task related parameters.
    #     Arguments:
    #         - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved previously.
    #     """
    #     # 定义需要排除的参数前缀
    #     exclude_prefixes = [
    #         '_orig_mod.world_model.head_policy_multi_task.',
    #         '_orig_mod.world_model.head_value_multi_task.',
    #         '_orig_mod.world_model.head_rewards_multi_task.',
    #         '_orig_mod.world_model.head_observations_multi_task.',
    #         '_orig_mod.world_model.task_emb.'
    #     ]
        
    #     # 定义需要排除的具体参数（如果有特殊情况）
    #     exclude_keys = [
    #         '_orig_mod.world_model.task_emb.weight',
    #         '_orig_mod.world_model.task_emb.bias',  # 如果存在则添加
    #         # 添加其他需要排除的具体参数名
    #     ]
        
    #     def filter_state_dict(state_dict_loader: Dict[str, Any], exclude_prefixes: list, exclude_keys: list = []) -> Dict[str, Any]:
    #         """
    #         过滤掉需要排除的参数。
    #         """
    #         filtered = {}
    #         for k, v in state_dict_loader.items():
    #             if any(k.startswith(prefix) for prefix in exclude_prefixes):
    #                 print(f"Excluding parameter: {k}")  # 调试用，查看哪些参数被排除
    #                 continue
    #             if k in exclude_keys:
    #                 print(f"Excluding specific parameter: {k}")  # 调试用
    #                 continue
    #             filtered[k] = v
    #         return filtered

    #     # 过滤并加载 'model' 部分
    #     if 'model' in state_dict:
    #         model_state_dict = state_dict['model']
    #         filtered_model_state_dict = filter_state_dict(model_state_dict, exclude_prefixes, exclude_keys)
    #         missing_keys, unexpected_keys = self._learn_model.load_state_dict(filtered_model_state_dict, strict=False)
    #         if missing_keys:
    #             print(f"Missing keys when loading _learn_model: {missing_keys}")
    #         if unexpected_keys:
    #             print(f"Unexpected keys when loading _learn_model: {unexpected_keys}")
    #     else:
    #         print("No 'model' key found in the state_dict.")

    #     # 过滤并加载 'target_model' 部分
    #     if 'target_model' in state_dict:
    #         target_model_state_dict = state_dict['target_model']
    #         filtered_target_model_state_dict = filter_state_dict(target_model_state_dict, exclude_prefixes, exclude_keys)
    #         missing_keys, unexpected_keys = self._target_model.load_state_dict(filtered_target_model_state_dict, strict=False)
    #         if missing_keys:
    #             print(f"Missing keys when loading _target_model: {missing_keys}")
    #         if unexpected_keys:
    #             print(f"Unexpected keys when loading _target_model: {unexpected_keys}")
    #     else:
    #         print("No 'target_model' key found in the state_dict.")

    #     # 加载优化器的 state_dict，不需要过滤，因为优化器通常不包含模型参数
    #     if 'optimizer_world_model' in state_dict:
    #         optimizer_state_dict = state_dict['optimizer_world_model']
    #         try:
    #             self._optimizer_world_model.load_state_dict(optimizer_state_dict)
    #         except Exception as e:
    #             print(f"Error loading optimizer state_dict: {e}")
    #     else:
    #         print("No 'optimizer_world_model' key found in the state_dict.")

    #     # 如果需要，还可以加载其他部分，例如 scheduler 等