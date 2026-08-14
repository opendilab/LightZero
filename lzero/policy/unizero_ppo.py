import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from torch.distributions import Categorical

from ding.utils import POLICY_REGISTRY

from lzero.entry.utils import initialize_pad_batch
from lzero.policy import mz_network_output_unpack
from lzero.policy.unizero import UniZeroPolicy


@POLICY_REGISTRY.register('unizero_ppo')
class UniZeroPPOPolicy(UniZeroPolicy):
    """UniZero policy variant that replaces MCTS-based improvement with PPO updates."""

    config = copy.deepcopy(UniZeroPolicy.config)
    config.update(
        dict(
            type='unizero_ppo',
            ppo=dict(
                rollout_length=64,
                mini_batch_size=32,
                update_epochs=4,
                gamma=0.997,
                gae_lambda=0.95,
                clip_ratio=0.2,
                value_coef=0.25,
                entropy_coef=0.01,
                advantage_normalization=True,
            ),
        )
    )

    def _init_collect(self) -> None:
        """Initialize structures used during data collection."""
        self._collect_model = self._model
        env_num = self._cfg.collector_env_num
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros(
                [env_num, self._cfg.model.observation_shape[0], 64, 64],
                device=self._cfg.device,
            )
        else:
            self.last_batch_obs = torch.full(
                [env_num, self._cfg.model.observation_shape],
                fill_value=self.pad_token_id,
                device=self._cfg.device,
            )
        self.last_batch_action = [-1 for _ in range(env_num)]

    def _init_eval(self) -> None:
        """Evaluation reuses collect path (no MCTS search)."""
        self._eval_model = self._model
        env_num = self._cfg.evaluator_env_num
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros(
                [env_num, self._cfg.model.observation_shape[0], 64, 64],
                device=self._cfg.device,
            )
        else:
            self.last_batch_obs = torch.full(
                [env_num, self._cfg.model.observation_shape],
                fill_value=self.pad_token_id,
                device=self._cfg.device,
            )
        self.last_batch_action = [-1 for _ in range(env_num)]

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: List[np.ndarray],
            ready_env_id: Optional[np.ndarray] = None,
            timestep: Optional[List[int]] = None,
            deterministic: bool = False,
            **kwargs: Any,
    ) -> Dict[int, Dict[str, Any]]:
        """Sample actions directly from the policy head and expose statistics for PPO."""
        self._collect_model.eval()

        if ready_env_id is None:
            ready_env_id = np.arange(data.shape[0])
        elif isinstance(ready_env_id, (list, tuple)):
            ready_env_id = np.asarray(ready_env_id)

        if timestep is None:
            timestep = [0 for _ in ready_env_id]

        ready_env_list = ready_env_id.tolist()
        prev_obs_snapshot = torch.stack(
            [self.last_batch_obs[env_id] for env_id in ready_env_list]
        ).clone().to(self._cfg.device)
        prev_action_snapshot = [self.last_batch_action[env_id] for env_id in ready_env_list]

        with torch.no_grad():
            network_output = self._collect_model.initial_inference(
                prev_obs_snapshot, prev_action_snapshot, data, timestep
            )
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            del latent_state_roots, reward_roots
            pred_values = self.value_inverse_scalar_transform_handle(pred_values)

        outputs: Dict[int, Dict[str, Any]] = {}
        batch_action: List[int] = []
        for idx, env_id in enumerate(ready_env_list):
            logits = policy_logits[idx]
            mask = torch.tensor(action_mask[idx], dtype=torch.bool, device=logits.device)
            masked_logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=masked_logits)
            action = torch.argmax(masked_logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            action_int = int(action.item())
            batch_action.append(action_int)
            outputs[env_id] = dict(
                action=action_int,
                log_prob=float(log_prob.item()),
                entropy=float(entropy.item()),
                predicted_value=float(pred_values[idx].item()),
                policy_logits=logits.detach().cpu(),
                action_mask=np.asarray(action_mask[idx]).copy(),
                obs=data[idx].detach().cpu(),
                timestep=int(timestep[idx]),
            )

        for idx, env_id in enumerate(ready_env_list):
            self.last_batch_obs[env_id] = data[idx].detach().clone()
            self.last_batch_action[env_id] = batch_action[idx]

        return outputs

    def _forward_eval(
            self,
            data: torch.Tensor,
            action_mask: List[np.ndarray],
            to_play: Optional[List[int]] = None,
            ready_env_id: Optional[np.ndarray] = None,
            timestep: Optional[List[int]] = None,
            **kwargs: Any,
    ) -> Dict[int, Dict[str, Any]]:
        return self._forward_collect(
            data=data,
            action_mask=action_mask,
            ready_env_id=ready_env_id,
            timestep=timestep,
            deterministic=True,
        )

    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, float]:
        batch_dict, _, train_iter = data

        self._learn_model.train()
        self._target_model.train()
        device = next(self._learn_model.parameters()).device

        prev_obs = torch.as_tensor(batch_dict['prev_obs'], device=device)
        obs = torch.as_tensor(batch_dict['obs'], device=device)
        action_mask = torch.as_tensor(batch_dict['action_mask'], device=device).bool()
        actions = torch.as_tensor(batch_dict['action'], device=device).long()
        old_log_prob = torch.as_tensor(batch_dict['old_log_prob'], device=device).float()
        advantages = torch.as_tensor(batch_dict['advantage'], device=device).float()
        returns = torch.as_tensor(batch_dict['return_'], device=device).float()
        prev_actions = [int(a) for a in batch_dict['prev_action']]
        timesteps = batch_dict['timestep'].tolist()

        prev_obs = prev_obs.float() if prev_obs.is_floating_point() else prev_obs.long()
        obs = obs.float() if obs.is_floating_point() else obs.long()

        network_output = self._learn_model.initial_inference(prev_obs, prev_actions, obs, timesteps)
        _, _, pred_values, policy_logits = mz_network_output_unpack(network_output)

        loss_tensors = self._learn_model.world_model.compute_loss_ppo(
            dict(
                policy_logits=policy_logits,
                values=pred_values,
                action_mask=action_mask,
                actions=actions,
                old_log_prob=old_log_prob,
                advantages=advantages,
                returns=returns,
            ),
            inverse_scalar_transform_handle=self.value_inverse_scalar_transform_handle,
            clip_ratio=self._cfg.ppo.clip_ratio,
            value_coef=self._cfg.ppo.value_coef,
            entropy_coef=self._cfg.ppo.entropy_coef,
        )

        total_loss = loss_tensors['loss_total']

        if (train_iter % self.accumulation_steps) == 0:
            self._optimizer_world_model.zero_grad()

        (total_loss / self.accumulation_steps).backward()

        total_grad_norm_before_clip = torch.tensor(0., device=device)
        if (train_iter + 1) % self.accumulation_steps == 0:
            total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                self._learn_model.world_model.parameters(), self._cfg.grad_clip_value
            )
            if self._cfg.multi_gpu:
                self.sync_gradients(self._learn_model)
            self._optimizer_world_model.step()
            if self.accumulation_steps > 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
                self.lr_scheduler.step()

            self._target_model.update(self._learn_model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            current_memory_allocated = 0.0
            max_memory_allocated = 0.0

        log_dict = {
            'loss_policy': loss_tensors['loss_policy'].item(),
            'loss_value': loss_tensors['loss_value'].item(),
            'loss_entropy': loss_tensors['loss_entropy'].item(),
            'loss_total': total_loss.item(),
            'ratio_mean': loss_tensors['ratio_mean'].item(),
            'advantage_mean': loss_tensors['advantage_mean'].item(),
            'return_mean': loss_tensors['return_mean'].item(),
            'entropy_mean': loss_tensors['entropy_mean'].item(),
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            'Current_GPU': current_memory_allocated,
            'Max_GPU': max_memory_allocated,
            'train_iter': train_iter,
        }

        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in log_dict.items()}, step=self.env_step)
            wandb.log({'learner_iter_vs_env_step': self.train_iter}, step=self.env_step)

        return log_dict

    def reset(self, env_id: Optional[List[int]] = None) -> None:
        """Reset cached context for specified environments."""
        if env_id is None:
            self._reset_collect(reset_init_data=True)
            return

        if isinstance(env_id, int):
            env_id = [env_id]
        for e_id in env_id:
            self.last_batch_obs[e_id] = initialize_pad_batch(
                self._cfg.model.observation_shape, 1, self._cfg.device, pad_token_id=getattr(self, 'pad_token_id', 0)
            )[0]
            self.last_batch_action[e_id] = -1
