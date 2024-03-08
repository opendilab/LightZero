from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn

from .episode import Episode


# def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
#     """Credits to https://github.com/karpathy/minGPT"""
#     # separate out all parameters to those that will and won't experience regularizing weight decay
#     decay = set()
#     no_decay = set()
#     whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
#     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#     for mn, m in model.named_modules():
#         for pn, p in m.named_parameters():
#             fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
#             if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
#                 no_decay.add(fpn)
#             elif 'bias' in pn:
#                 # all biases will not be decayed
#                 no_decay.add(fpn)
#             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                 # weights of whitelist modules will be weight decayed
#                 decay.add(fpn)
#             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                 # weights of blacklist modules will NOT be weight decayed
#                 no_decay.add(fpn)

#     # validate that we considered every parameter
#     param_dict = {pn: p for pn, p in model.named_parameters()}
#     inter_params = decay & no_decay
#     union_params = decay | no_decay
#     assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
#     assert len(
#         param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

#     # create the pytorch optimizer object
#     optim_groups = [
#         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
#         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#     ]
#     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
#     return optimizer

from lzero.model.common import RepresentationNetwork

def init_weights(module):
    if not isinstance(module, RepresentationNetwork):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, latent_recon_loss_weight=0, perceptual_loss_weight=0, **kwargs):
        # self.loss_total = sum(kwargs.values())

        # Ensure that kwargs is not empty
        if not kwargs:
            raise ValueError("At least one loss must be provided")

        # Get a reference device from one of the provided losses
        device = next(iter(kwargs.values())).device

        # similar with ssl_loss in EZ
        # self.obs_loss_weight = 2.
        # self.reward_loss_weight = 1.
        # self.value_loss_weight = 0.25
        # self.policy_loss_weight = 1.
        # # self.ends_loss_weight = 1.
        # self.ends_loss_weight = 0.

        # self.obs_loss_weight = 0.1
        self.obs_loss_weight = 10
        # self.obs_loss_weight = 2


        self.reward_loss_weight = 1.
        self.value_loss_weight = 0.25
        self.policy_loss_weight = 1.
        # self.ends_loss_weight = 1.
        self.ends_loss_weight = 0.

        # self.latent_kl_loss_weight = 0.1 # for lunarlander
        self.latent_kl_loss_weight = 0. # for lunarlander

        self.latent_recon_loss_weight = latent_recon_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        # self.latent_recon_loss_weight = 0.1


        # Initialize the total loss tensor on the correct device
        self.loss_total = torch.tensor(0., device=device)
        for k, v in kwargs.items():
            if k == 'loss_obs':
                self.loss_total += self.obs_loss_weight * v
            elif k == 'loss_rewards':
                self.loss_total += self.reward_loss_weight * v
            elif k == 'loss_policy':
                self.loss_total += self.policy_loss_weight * v
            elif k == 'loss_value':
                self.loss_total += self.value_loss_weight * v
            elif k == 'loss_ends':
                self.loss_total += self.ends_loss_weight * v
            elif k == 'latent_kl_loss':
                self.loss_total += self.latent_kl_loss_weight * v
            elif k == 'latent_recon_loss':
                self.loss_total += self.latent_recon_loss_weight * v
            elif k == 'perceptual_loss':
                self.loss_total += self.perceptual_loss_weight * v
            # else:
            #     raise ValueError(f"Unknown loss type : {k}")

        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        # self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()
