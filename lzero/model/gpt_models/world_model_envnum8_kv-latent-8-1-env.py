import copy
from dataclasses import dataclass
import random
from typing import Any, Optional, Tuple
from typing import List, Optional, Union
import logging
# 设置日志记录级别为DEBUG
logging.getLogger().setLevel(logging.DEBUG)
from PIL import Image
from einops import rearrange
from einops import rearrange
import gym
from joblib import hash
import numpy as np
import torch
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .kv_caching import KeysValues
from .slicer import Embedder, Head, ActEmbedder
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights
from ding.torch_utils import to_device
# from memory_profiler import profile
from line_profiler import line_profiler

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor

    logits_policy: torch.FloatTensor
    logits_value: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig, tokenizer, representation_network=None) -> None:
        super().__init__()

        # config.max_tokens = int(2*50) # TODO

        self.tokenizer = tokenizer
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        self.transformer = Transformer(config)
        # self.num_observations_tokens = 16
        self.num_observations_tokens = config.tokens_per_block -1

        self.latent_recon_loss_weight = config.latent_recon_loss_weight
        self.perceptual_loss_weight = config.perceptual_loss_weight

        self.device = config.device
        self.support_size = config.support_size
        self.action_shape = config.action_shape
        self.max_cache_size = config.max_cache_size
        self.env_num = config.env_num


        all_but_last_latent_state_pattern = torch.ones(config.tokens_per_block)
        all_but_last_latent_state_pattern[-2] = 0 # 1,...,0,1

        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)  # 17
        act_tokens_pattern[-1] = 1   # 0,...,0,1
        latent_state_pattern = 1 - act_tokens_pattern  # 1,...,1,0

        # current latent state's policy value
        value_policy_tokens_pattern = torch.zeros(config.tokens_per_block)
        value_policy_tokens_pattern[-2] = 1  # [0,...,1,0]

        # next latent state's policy value
        # value_policy_tokens_pattern = torch.zeros(config.tokens_per_block)
        # value_policy_tokens_pattern[-1] = 1  # [0,...,0,1]

        obs_per_embdding_dim=config.embed_dim

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)


        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, latent_state_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        # self.act_embedder = ActEmbedder(
        #     max_blocks=config.max_blocks,
        #     block_masks=[act_tokens_pattern],
        #     embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim)])
        # )

        self.act_embedding_table = nn.Embedding(act_vocab_size, config.embed_dim)

        # self.head_observations = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=all_but_last_latent_state_pattern, # 1,...,0,1 # https://github.com/eloialonso/iris/issues/19
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, config.embed_dim),
        #         nn.ReLU(),
        #         nn.Linear(config.embed_dim, obs_vocab_size)
        #     )
        # )
        self.obs_per_embdding_dim = config.embed_dim # 16*64=1024
        self.head_observations = Head( # TODO
            max_blocks=config.max_blocks,
            block_mask=all_but_last_latent_state_pattern, # 1,...,0,1 # https://github.com/eloialonso/iris/issues/19
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.BatchNorm1d(config.embed_dim),
                # nn.ReLU(),
                # nn.Linear(config.embed_dim, obs_vocab_size)
                nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.Linear(config.embed_dim, self.obs_per_embdding_dim),
                # nn.Tanh(), # TODO
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,  # 0,...,0,1
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, self.support_size)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,  # 0,...,0,1
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.head_observations_for_root = Head( # TODO
            max_blocks=config.max_blocks,
            block_mask=latent_state_pattern,  # 1,...,1,0
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.BatchNorm1d(config.embed_dim),
                nn.ReLU(),
                # nn.Linear(config.embed_dim, obs_vocab_size)
                nn.Linear(config.embed_dim, obs_per_embdding_dim)
            )
        )
        self.head_policy = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,  # TODO: value_policy_tokens_pattern # [0,...,1,0]
            head_module=nn.Sequential( # （8, 5, 128）
                # nn.BatchNorm1d(config.embed_dim), # TODO: 1
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.Linear(config.embed_dim, self.action_shape)  # TODO(pu); action shape
            )
        )
        self.head_value = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,
            head_module=nn.Sequential(
                # nn.BatchNorm1d(config.embed_dim), # TODO: 1
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.Linear(config.embed_dim, self.support_size)  # TODO(pu): action shape
            )
        )

        self.apply(init_weights)

        last_linear_layer_init_zero = True  # TODO: is beneficial for convergence speed.
        if last_linear_layer_init_zero:
            # TODO: policy init : 3
            # Locate the last linear layer and initialize its weights and biases to 0.
            # for _, layer in enumerate(reversed(self.head_policy.head_module)):
            #     if isinstance(layer, nn.Linear):
            #         nn.init.zeros_(layer.weight)
            #         nn.init.zeros_(layer.bias)
            #         break
            for _, layer in enumerate(reversed(self.head_value.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break
            for _, layer in enumerate(reversed(self.head_rewards.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break
            for _, layer in enumerate(reversed(self.head_observations.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break


        import collections
        self.past_keys_values_cache = collections.OrderedDict()
        self.past_policy_value_cache = collections.OrderedDict()

        # TODO: Transformer更新后应该清除缓存
        # NOTE
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=8, max_tokens=self.config.max_tokens)

        if self.num_observations_tokens==16:  # k=16
            self.projection_input_dim = 128
        elif self.num_observations_tokens==1:  # K=1
            self.projection_input_dim =  self.obs_per_embdding_dim# for atari #TODO
            # self.projection_input_dim = 256 # for cartpole


        self.proj_hid = 1024
        self.proj_out = 1024
        self.pred_hid = 512
        self.pred_out = 1024
        activation = nn.ReLU()
        self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            activation,
            nn.Linear(self.pred_hid, self.pred_out),
        )
        self.hit_count = 0
        self.total_query_count = 0



    def __repr__(self) -> str:
        return "world_model"


    # def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None,
    #             is_root=False) -> WorldModelOutput:
    # def forward(self, obs_embeddings, act_tokens, past_keys_values: Optional[KeysValues] = None,
    #         is_root=False) -> WorldModelOutput:
    # @profile
    def forward(self, obs_embeddings_or_act_tokens, past_keys_values: Optional[KeysValues] = None,
            is_root=False) -> WorldModelOutput:
        
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        # print(f'prev_steps:{prev_steps}')

        # sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))
        if 'obs_embeddings' in obs_embeddings_or_act_tokens.keys():
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            num_steps = obs_embeddings.size(1)  # (B, T, E)
            # if prev_steps>0:
            #     prev_steps = prev_steps+1  # TODO: NOTE: 在collect的每一步，执行init_infer时，不reset kv_cache
            sequences = obs_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))
        elif 'act_tokens' in obs_embeddings_or_act_tokens.keys():
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            num_steps = act_tokens.size(1)  # (B, T)
            # act_embeddings = self.act_embedder(act_tokens, num_steps, prev_steps)
            act_embeddings = self.act_embedding_table(act_tokens)

            sequences = act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=act_tokens.device))
        else:
            obs_embeddings_and_act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
            # obs_embeddings: (B, L, K=16, E), act_tokens: (B, L, 1)
            obs_embeddings, act_tokens = obs_embeddings_and_act_tokens
            if len(obs_embeddings.shape)==3: # for batch compute loss
                obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens, -1)

            num_steps = int(obs_embeddings.size(1)*(obs_embeddings.size(2)+1)) # L(k+1)
            # assert num_steps <= self.config.max_tokens
            # Rearrange observation embeddings from (B, L, K, E) to (B, L*K, E)
            # obs_embeddings = rearrange(obs_embeddings, 'b l k e -> b (l k) e')

            # Generate action embeddings from action tokens
            # (B, L, 1) -> (B, L, 1, E) 
            act_embeddings = self.act_embedding_table(act_tokens)

            # 已知obs_embeddings的维度为 (B, L, K, E), act_embeddings的维度为(B, L, 1, E) 希望得到一个obs_act_embeddings向量的维度为 (B, L(K+1), E) 
            # 而且让得到的obs_act_embeddings的第2个维度的数据为：obs act, obs, act, ..., obs, act，即 L, 1, L,1 ... 这样的排列顺序。请给出高效的实现，用中文回答

            B, L, K, E = obs_embeddings.size()
            # _, _, _, _ = act_embeddings.size()

            # 初始化一个新的空tensor，用于存放最终的拼接结果
            obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=obs_embeddings.device)

            # 对每一个序列长度L进行循环
            for i in range(L):
                # 获取当前时刻的obs和act embeddings
                obs = obs_embeddings[:, i, :, :]  # Shape: (B, K, E)
                act = act_embeddings[:, i, 0, :].unsqueeze(1)  # Shape: (B, 1, E), 补充维度以便拼接

                # 交替拼接obs和act
                obs_act = torch.cat([obs, act], dim=1)  # Shape: (B, K + 1, E)

                # 将结果填充到最终的tensor中
                obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

            # 确保形状正确无误
            # assert obs_act_embeddings.shape == (B, L * (K + 1), E)

            # Add positional embeddings
            sequences = obs_act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))


        # print('transformer forward begin') 函数里面更新了update past_keys_values
        x = self.transformer(sequences, past_keys_values)
        # print('transformer forward done')

        if is_root:
            logits_observations = self.head_observations_for_root(x, num_steps=num_steps, prev_steps=prev_steps)
        else:
            # 1,...,0,1 https://github.com/eloialonso/iris/issues/19
            logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        # return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # TODO: root reward value
        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_policy, logits_value)

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_latent_state().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    # only foe inference now, now is invalid
    @torch.no_grad()
    def decode_latent_state(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.latent_state)  # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)  # (B, C, H, W)
        # TODO: for atari image
        return torch.clamp(rec, 0, 1)
        # for cartpole obs
        # return rec


    @torch.no_grad()
    def render(self):
        assert self.latent_state.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(
            0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)


    @torch.no_grad()
    # @profile
    def reset_from_initial_observations_v2(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(obs_act_dict, dict):
            observations = obs_act_dict['obs']
            buffer_action = obs_act_dict['action']
        else:
            observations = obs_act_dict
            buffer_action = None
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations, should_preprocess=True) # (B, C, H, W) -> (B, K, E)

        outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer_v2(obs_embeddings, buffer_action)
        self.latent_state = obs_embeddings

        return outputs_wm, self.latent_state

    @torch.no_grad()
    # @profile
    def refresh_keys_values_with_initial_latent_state_for_init_infer_v2(self, latent_state: torch.LongTensor, buffer_action=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape

        if n <= self.env_num:
            # MCTS root节点: 需要准确的估计 value, policy_logits 或许需要结合context的kv_cache进行更准确的估计，而不是当前的从0开始推理
            # self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)

            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
            self.total_query_count  += 1
            # Compute the hash of latest_state
            latest_state = latent_state.detach().cpu().numpy()

            # 假设 latest_state 是新的 latent_state，包含 ready_env_num 个环境的信息
            ready_env_num = latest_state.shape[0]

            if ready_env_num < self.env_num:
                keys_values_wm_list = []
                for i in range(ready_env_num):
                    state_single_env = latest_state[i]  # 获取单个环境的 latent state
                    hash_latest_state = hash(state_single_env)  # 计算哈希值
                    matched_value = self.past_keys_values_cache.get(hash_latest_state)  # 检索缓存值
                    if matched_value is not None:
                        self.hit_count += 1
                        # 如果找到匹配的值，将其添加到列表中
                        keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda')))
                    else:
                        # use zero
                        keys_values_wm_list.append(self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens))

                # self.keys_values_wm = keys_values_wm_list
                kv_cache_k_list = []
                kv_cache_v_list = []
                for keys_values in keys_values_wm_list:
                    kv_cache_k_list.append(keys_values[0]._k_cache._cache)
                    kv_cache_v_list.append(keys_values[0]._v_cache._cache)
                self.keys_values_wm[0]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
                self.keys_values_wm[0]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)
            elif ready_env_num == self.env_num:
                hash_latest_state = hash(latest_state)
                matched_value = self.past_keys_values_cache.get(hash_latest_state)
                if matched_value is not None:
                    self.keys_values_wm_find = copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda') )
                    self.hit_count += 1
                    # self.total_query_count  += 1
                    # print('recurrent_inference:find matched_value!')
                    # NOTE: very important, 相当于policy value由单步计算得到，往后的推理，基于context
                    # TODO: policy value也从缓存中找
                    self.keys_values_wm = self.keys_values_wm_find


        elif n == int(256): 
            # TODO: n=256 means train tokenizer, 不需要计算target value
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            # print('init inference: not find matched_value! reset!')
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
        elif n > self.env_num and n != int(256) and buffer_action is not None: 
            # TODO: n=256 means train tokenizer
            # [192, 16, 64] -> [32, 6, 16, 64]
            latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens, self.obs_per_embdding_dim) # (BL, K) for unroll_step=1

            # latent_state = latent_state.view(-1, self.config.max_blocks+1, num_observations_tokens) # (BL, K)
            latent_state = latent_state[:, :-1, :]
            # latent_state = latent_state.reshape(32*6, num_observations_tokens) # (BL, K)
            buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
            act_tokens = rearrange(buffer_action, 'b l -> b l 1')

            # # 选择每个样本的最后一步
            last_steps = act_tokens[:, -1:, :]  # 这将选择最后一列并保持维度不变, 最后一步的target policy/value本身就没有用到
            # 使用torch.cat在第二个维度上连接原始act_tokens和last_steps
            act_tokens = torch.cat((act_tokens, last_steps), dim=1)

            # print('init inference: unroll 5 steps!')  17*6=102  17*5=85
            obs_embeddings = latent_state
            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, is_root=False)

            # 选择每个样本的最后一步
            last_steps_value = outputs_wm.logits_value[:, -1:, :]  # 这将选择最后一列并保持维度不变
            # 使用torch.cat在第二个维度上连接原始act_tokens和last_steps
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]  # 这将选择最后一列并保持维度不变
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            #  outputs_wm.logits_value.shape (30,21) = (B*6, 21)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')

        return outputs_wm

    @torch.no_grad()
    # @profile
    def reset_from_initial_observations(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(obs_act_dict, dict):
            # obs_act_dict = {'obs':obs, 'action':action_batch}
            observations = obs_act_dict['obs']
            buffer_action = obs_act_dict['action']
        else:
            observations = obs_act_dict
            buffer_action = None

        # NOTE: should_preprocess=True is important
        # latent_state = self.tokenizer.encode(observations, should_preprocess=True).tokens  # (B, C, H, W) -> (B, K)
        # _, num_observations_tokens = latent_state.shape

        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations, should_preprocess=True) # (B, C, H, W) -> (B, K, E)
        # num_observations_tokens = obs_embeddings.shape[1]

        # if self.num_observations_tokens is None:
        #     self._num_observations_tokens = num_observations_tokens

        # outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer(latent_state, buffer_action)
        outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action)

        self.latent_state = obs_embeddings

        # return outputs_wm, self.decode_latent_state(), self.latent_state
        return outputs_wm, self.latent_state


    @torch.no_grad()
    # @profile
    def refresh_keys_values_with_initial_latent_state_for_init_infer(self, latent_state: torch.LongTensor, buffer_action=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        # assert num_observations_tokens == self.num_observations_tokens
        # self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)

        if n <= self.env_num:
            # Compute the hash of latent_state
            # cache_key = hash(latent_state.detach().cpu().numpy())
            # # Try to get the value associated with the hash of latest_state
            # matched_value = self.past_keys_values_cache.get(cache_key)
            # if matched_value is not None:
            #     # If a matching value is found, do something with it
            #     self.keys_values_wm = copy.deepcopy(matched_value)
            #     print('init inference: find matched_value!')
            # else:
            #     self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            #     # print('init inference: not find matched_value! reset!')
            
            # MCTS root节点: 需要准确的估计 value, policy_logits 或许需要结合context的kv_cache进行更准确的估计，而不是当前的从0开始推理
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
        elif n == int(256): 
            # TODO: n=256 means train tokenizer, 不需要计算target value
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            # print('init inference: not find matched_value! reset!')
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
        # elif n > self.env_num and n != int(256) and buffer_action is not None: 
        #     # transformer只能unroll 5步
        #     # TODO: n=256 means train tokenizer
        #     # TODO: for n=32*6=192 means 通过unroll 5 steps，计算target value 
        #     # [192, 16, 64] -> [32, 6, 16, 64]
        #     latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens, self.obs_per_embdding_dim) # (BL, K) for unroll_step=1
        #     buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
        #     act_tokens = rearrange(buffer_action, 'b l -> b l 1')
        #     # 将5步动作的最后一步，重复一次，以拼接为6步的动作
        #     act_tokens = torch.cat((act_tokens, act_tokens[:, -1:, :]), dim=1)
        #     obs_embeddings = latent_state
        #     outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, is_root=False)
        #     # Reshape your tensors
        #     #  outputs_wm.logits_value.shape (30,21) = (B*6, 21)
        #     outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
        #     outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')
        elif n > self.env_num and n != int(256) and buffer_action is not None: 
            # TODO: n=256 means train tokenizer
            # TODO: for n=32*6=192 means 通过unroll 5 steps，计算target value 
            # latent_state = latent_state.reshape(32, 6, num_observations_tokens) # (BL, K)
            # latent_state = latent_state.view(-1, 6, num_observations_tokens) # (BL, K)

            # [192, 16] -> [32, 6, 16]
            # latent_state = latent_state.view(buffer_action.shape[0], -1, num_observations_tokens) # (BL, K) for unroll_step=1

            # [192, 16, 64] -> [32, 6, 16, 64]
            latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens, self.obs_per_embdding_dim) # (BL, K) for unroll_step=1

            # latent_state = latent_state.view(-1, self.config.max_blocks+1, num_observations_tokens) # (BL, K)
            latent_state = latent_state[:, :-1, :]
            # latent_state = latent_state.reshape(32*6, num_observations_tokens) # (BL, K)
            buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
            act_tokens = rearrange(buffer_action, 'b l -> b l 1')

            # # 选择每个样本的最后一步
            last_steps = act_tokens[:, -1:, :]  # 这将选择最后一列并保持维度不变, 最后一步的target policy/value本身就没有用到
            # 使用torch.cat在第二个维度上连接原始act_tokens和last_steps
            act_tokens = torch.cat((act_tokens, last_steps), dim=1)

            # print('init inference: unroll 5 steps!')  17*6=102  17*5=85
            obs_embeddings = latent_state
            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, is_root=False)

            # 选择每个样本的最后一步
            last_steps_value = outputs_wm.logits_value[:, -1:, :]  # 这将选择最后一列并保持维度不变
            # 使用torch.cat在第二个维度上连接原始act_tokens和last_steps
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]  # 这将选择最后一列并保持维度不变
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            #  outputs_wm.logits_value.shape (30,21) = (B*6, 21)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')


        # return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_policy, logits_value)
        return outputs_wm


    @torch.no_grad()
    # @profile
    def refresh_keys_values_with_initial_latent_state(self, latent_state: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        assert num_observations_tokens == self.num_observations_tokens
        # self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
        obs_embeddings_or_act_tokens = {'obs_embeddings': latent_state}
        outputs_wm = self.forward(obs_embeddings_or_act_tokens, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False

        # return outputs_wm.output_sequence  # (B, K, E)
        return outputs_wm

    @torch.no_grad()
    # @profile
    def forward_initial_inference(self, obs_act_dict: torch.LongTensor, should_predict_next_obs: bool = True):

        if isinstance(obs_act_dict, dict):
            # obs_act_dict = {'obs':obs, 'action':action_batch}
            obs = obs_act_dict['obs']
        else:
            obs = obs_act_dict

        if len(obs[0].shape) == 3:
            # obs is a 3-dimensional image, for atari
            pass
        # elif len(obs[0].shape) == 1:
        #     # TODO(): for cartpole, 4 -> 4,64,64
        #     # obs is a 1-dimensional vector
        #     original_shape = list(obs.shape)
        #     desired_shape = original_shape + [64, 64]
        #     expanded_observations = obs.unsqueeze(-1).unsqueeze(-1)
        #     expanded_observations = expanded_observations.expand(*desired_shape)
        #     obs = expanded_observations

        #     obs_act_dict['obs'] = obs
            
            # for cartpole, 4 -> 3,64,64
            # obs is a 1-dimensional vector
            # original_shape = list(obs.shape)
            # desired_shape = original_shape[:-1] + [3, 64, 64]  # 修改最后一个维度为3，然后添加64和64
            # repeated_observations = obs.repeat(1, int(3*64*64/original_shape[-1]))  # 将最后一个维度复制到3,64,64
            # obs = repeated_observations.view(*desired_shape)  # 重新调整形状到3,64,64
        


        outputs_wm, latent_state = self.reset_from_initial_observations_v2(obs_act_dict) # TODO
        # outputs_wm, latent_state = self.reset_from_initial_observations(obs_act_dict) # 从零开始

        return outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value


    """
    past-kv-dict-batch envnum8 latest multi-step
    fix init infer
    把8个样本的self.keys_values_wm 看做一个整体来寻找

    TODO：很多时候都是执行的refresh_keys_values_with_initial_latent_state，导致没有充分利用序列建模能力？
    """


    @torch.no_grad()
    # @profile
    def forward_recurrent_inference(self, state_action_history, should_predict_next_obs: bool = True):
        # 一般来讲，在一次 MCTS search中，我们需要维护H长度的context来使用transformer进行推理。
        # 由于在一次search里面。agent最多访问sim个不同的节点，因此我们只需维护一个 {(state:kv_cache)}的列表。
        # 但如果假设环境是MDP的话，然后根据当前的 latest_state s_t 在这个列表中查找即可
        # TODO: 但如果假设环境是非MDP的话，需要维护一个 {(rootstate_action_history:kv_cache)}的列表？

        # if self.total_query_count>0:
        #     self.hit_freq = self.hit_count/self.total_query_count
        #     print('hit_freq:', self.hit_freq)
        #     print('hit_count:', self.hit_count)
        #     print('total_query_count:', self.total_query_count)

        self.total_query_count  += 1
        latest_state = state_action_history[-1][0]

        # 假设 latest_state 是新的 latent_state，包含 ready_env_num 个环境的信息
        ready_env_num = latest_state.shape[0]
        if ready_env_num < self.env_num:
            keys_values_wm_list = []
            for i in range(ready_env_num):
                state_single_env = latest_state[i]  # 获取单个环境的 latent state
                hash_latest_state = hash(state_single_env)  # 计算哈希值
                matched_value = self.past_keys_values_cache.get(hash_latest_state)  # 检索缓存值
                if matched_value is not None:
                    self.hit_count += 1
                    # 如果找到匹配的值，将其添加到列表中
                    keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda')))
                else:
                    # use zero
                    keys_values_wm_list.append(self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens))
            # self.keys_values_wm <- keys_values_wm_list
            kv_cache_k_list = []
            kv_cache_v_list = []
            for keys_values in keys_values_wm_list:
                kv_cache_k_list.append(keys_values[0]._k_cache._cache)
                kv_cache_v_list.append(keys_values[0]._v_cache._cache)
            self.keys_values_wm[0]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm[0]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)
        elif ready_env_num == self.env_num:
            hash_latest_state = hash(latest_state)
            matched_value = self.past_keys_values_cache.get(hash_latest_state)
            if matched_value is not None:
                self.keys_values_wm_find = copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda') )
                self.hit_count += 1
                # self.total_query_count  += 1
                # print('recurrent_inference:find matched_value!')
                # NOTE: very important, 相当于policy value由单步计算得到，往后的推理，基于context
                # TODO: policy value也从缓存中找
                self.keys_values_wm = self.keys_values_wm_find

        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, latent_state = [], []

        if self.keys_values_wm.size + num_passes > self.config.max_tokens:
            del self.keys_values_wm # TODO
            # TODO: the impact
            _ = self.refresh_keys_values_with_initial_latent_state(torch.tensor(latest_state, dtype=torch.float32).to(self.device))
            # Depending on the shape of latent_state, create a cache key and store a deep copy of keys_values_wm
            self.past_keys_values_cache[hash(latest_state)] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm, 'cpu'))

        # if self.keys_values_wm.size>5:
        #     print('debug self.keys_values_wm.size ')

        # TODO
        action = state_action_history[-1][-1]

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.
            # action_token obs_token, ..., obs_token  1+16

            # obs is in token level
            # act_token num_steps=1, prev_steps=16
            # obs_token_0 num_steps=1, prev_steps=17
            # obs_token_1 num_steps=1, prev_steps=18
            if k==0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}
            outputs_wm = self.forward(obs_embeddings_or_act_tokens, past_keys_values=self.keys_values_wm, is_root=False)
            # if k==0, action_token self.head_observations 1,...,0,1
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                # if k==0, token is action_token  outputs_wm.logits_rewards 是有值的
                # reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                # done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)  # (B,)
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                # 一共产生16个obs_token，每次产生一个
                # TODO： sample or argmax
                # token = Categorical(logits=outputs_wm.logits_observations).sample()
                # Use argmax to select the most likely token
                # token = outputs_wm.logits_observations.argmax(-1, keepdim=True)
                token = outputs_wm.logits_observations

                if len(token.shape) != 2:
                    token = token.squeeze(-1)  # Ensure the token tensor shape is (B, 1)
                latent_state.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)  # (B, 1 + K, E)
        # Before updating self.latent_state, delete the old one to free memory
        del self.latent_state
        self.latent_state = torch.cat(latent_state, dim=1)  # (B, K)
        latent_state = self.latent_state

        # cache_key = hash(latent_state.detach().cpu().numpy())
        # # TODO: 在计算结束后，是否需要更新最新的缓存. 是否需要deepcopy
        # self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm, 'cpu'))

        if latent_state.size(0)<self.env_num:
            for i in range(latent_state.size(0)):  # 遍历每个环境
                state_single_env = latent_state[i]   # 获取单个环境的 latent state
                cache_key = hash(state_single_env.detach().cpu().numpy())  # 计算哈希值
                # 复制单个环境对应的 keys_values_wm 并存储
                keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)
                keys_values_wm_single_env[0]._k_cache._cache = self.keys_values_wm[0]._k_cache._cache[i].unsqueeze(0) # shape torch.Size([2, 100, 512])
                keys_values_wm_single_env[0]._v_cache._cache = self.keys_values_wm[0]._v_cache._cache[i].unsqueeze(0)
                self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(keys_values_wm_single_env, 'cpu'))
        elif latent_state.size(0) == self.env_num:
            cache_key = hash(latent_state)
            self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm, 'cpu'))

        # outputs_wm.logits_policy, outputs_wm.logits_value
        if len(self.past_keys_values_cache) > self.max_cache_size:
            # TODO: lru_cache
            _, popped_kv_cache = self.past_keys_values_cache.popitem(last=False)
            del popped_kv_cache # 不要这一行

        # Example usage:
        # Assuming `past_keys_values_cache` is a populated instance of `KeysValues`
        # and `num_layers` is the number of transformer layers
        # cuda_memory_gb = self.calculate_cuda_memory_gb(self.past_keys_values_cache, num_layers=2)
        # print(f'len(self.past_keys_values_cache): {len(self.past_keys_values_cache)}, Memory used by past_keys_values_cache: {cuda_memory_gb:.2f} GB')

        return outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value


    def to_device_for_kvcache(self, keys_values: KeysValues, device: str) -> KeysValues:
        """
        Transfer all KVCache objects within the KeysValues object to a certain device.

        Arguments:
            - keys_values (KeysValues): The KeysValues object to be transferred.
            - device (str): The device to transfer to.

        Returns:
            - keys_values (KeysValues): The KeysValues object with its caches transferred to the specified device.
        """
        # Check if CUDA is available and select the first available CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for kv_cache in keys_values:
            kv_cache._k_cache._cache = kv_cache._k_cache._cache.to(device)
            kv_cache._v_cache._cache = kv_cache._v_cache._cache.to(device)
        return keys_values


    # 计算显存使用量的函数
    def calculate_cuda_memory_gb(self, past_keys_values_cache, num_layers: int):
        total_memory_bytes = 0
        
        # 遍历OrderedDict中所有的KeysValues实例
        for kv_instance in past_keys_values_cache.values():
            num_layers = len(kv_instance)  # 获取层数
            for layer in range(num_layers):
                kv_cache = kv_instance[layer]
                k_shape = kv_cache._k_cache.shape  # 获取keys缓存的形状
                v_shape = kv_cache._v_cache.shape  # 获取values缓存的形状

                # 计算元素个数并乘以每个元素的字节数
                k_memory = torch.prod(torch.tensor(k_shape)) * 4
                v_memory = torch.prod(torch.tensor(v_shape)) * 4
                
                # 累加keys和values缓存的内存
                layer_memory = k_memory + v_memory
                total_memory_bytes += layer_memory.item()  # .item()确保转换为Python标准数字
        
        # 将总内存从字节转换为吉字节
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        return total_memory_gb

    # @profile
    def compute_loss(self, batch, target_tokenizer: Tokenizer=None, **kwargs: Any) -> LossWithIntermediateLosses:

        # if len(batch['observations'][0, 0].shape) == 3:
        #     # obs is a 3-dimensional image
        #     pass

        # NOTE: 这里是需要梯度的
        #with torch.no_grad():  # TODO: 非常重要
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'], should_preprocess=False) # (B, C, H, W) -> (B, K, E)
        obs_embeddings.register_hook(lambda grad: grad * 1/5)  # TODO：只提供重建损失更新表征网络
        # obs_embeddings.register_hook(lambda grad: grad * 1)  # TODO：只提供重建损失更新表征网络

        # Assume that 'cont_embeddings' and 'original_images' are available from prior code
        # Decode the embeddings to reconstruct the images
        reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
        # Calculate the reconstruction loss
        # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 4, 64, 64), reconstructed_images) # TODO: for stack=4
        latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # TODO: for stack=1
        perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # TODO: for stack=1
        
        # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)
        # perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)

        latent_kl_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)


        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # TODO: 是否只用重建损失更新表征网络 非常重要
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, is_root=False)
        # outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens)}, is_root=False)

        with torch.no_grad(): 
            traget_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'], should_preprocess=False) # (B, C, H, W) -> (B, K, E)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(traget_obs_embeddings, batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])
        # labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_embeddings, batch['rewards'],
        #                                                                                    batch['ends'],
        #                                                                                    batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        # labels_observations = labels_observations.contiguous().view(-1, self.projection_input_dim)  # TODO:
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)  # TODO:

        
        loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations.detach(), reduction='none').mean(-1)
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1) # TODO:
        # mask_padding_expanded = batch['mask_padding'][:, 1:].reshape(-1)

        # 应用mask到loss_obs
        loss_obs = (loss_obs * mask_padding_expanded).mean(-1)
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')
        loss_policy = self.compute_cross_entropy_loss(outputs, labels_policy, batch, element='policy')
        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        return LossWithIntermediateLosses(latent_recon_loss_weight=self.latent_recon_loss_weight, perceptual_loss_weight=self.perceptual_loss_weight, loss_obs=loss_obs, loss_rewards=loss_rewards, loss_value=loss_value,
                                          loss_policy=loss_policy, latent_kl_loss=latent_kl_loss, latent_recon_loss=latent_recon_loss, perceptual_loss=perceptual_loss)

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs.logits_rewards and labels are your predictions and targets
        # And mask_padding is a boolean tensor with True at positions to keep and False at positions to ignore

        if element == 'rewards':
            logits = outputs.logits_rewards
        elif element == 'policy':
            logits = outputs.logits_policy
        elif element == 'value':
            logits = outputs.logits_value

        # Reshape your tensors
        logits_rewards = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assuming labels originally has shape [b, t, reward_dim]

        # Reshape your mask. True means valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        loss_rewards = -(torch.log_softmax(logits_rewards, dim=1) * labels).sum(1)
        # loss_rewards = (loss_rewards * mask_padding.squeeze(-1).float()).mean()
        loss_rewards = (loss_rewards * mask_padding).mean()


        return loss_rewards

    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # each sequence sample has at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:] # self.projection_input_dim


        # labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1} TODO(pu)

        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)

    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor]:

        mask_fill = torch.logical_not(mask_padding)
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)
        return labels_policy.reshape(-1, self.action_shape), labels_value.reshape(-1, self.support_size)  # TODO(pu)


    def negative_cosine_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            consistency loss function: the negative cosine similarity.
        Arguments:
            - x1 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
            - x2 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
        Returns:
            (x1 * x2).sum(dim=1) is the cosine similarity between vector x1 and x2.
            The cosine similarity always belongs to the interval [-1, 1].
            For example, two proportional vectors have a cosine similarity of 1,
            two orthogonal vectors have a similarity of 0,
            and two opposite vectors have a similarity of -1.
            -(x1 * x2).sum(dim=1) is consistency loss, i.e. the negative cosine similarity.
        Reference:
            https://en.wikipedia.org/wiki/Cosine_similarity
        """
        x1 = F.normalize(x1, p=2., dim=-1, eps=1e-5)
        x2 = F.normalize(x2, p=2., dim=-1, eps=1e-5)
        return -(x1 * x2).sum(dim=1)


    def render_img(self, obs: int, rec_img: int):
        import torch
        from PIL import Image
        import matplotlib.pyplot as plt

        # 假设batch是一个字典，其中包含了observations键，
        # 并且它的形状是torch.Size([B, N, C, H, W])
        # batch_observations = batch_for_gpt['observations']
        # batch_observations = batch['observations']
        batch_observations = obs.unsqueeze(0)
        # batch_observations = rec_img.unsqueeze(0)

        # batch_observations = observations.unsqueeze(0)
        # batch_observations = x.unsqueeze(0)
        # batch_observations = reconstructions.unsqueeze(0)



        B, N, C, H, W = batch_observations.shape  # 自动检测维度

        # 分隔条的宽度（可以根据需要调整）
        separator_width = 2

        # 遍历每个样本
        for i in range(B):
            # 提取当前样本中的所有帧
            frames = batch_observations[i]

            # 计算拼接图像的总宽度（包括分隔条）
            total_width = N * W + (N - 1) * separator_width

            # 创建一个新的图像，其中包含分隔条
            concat_image = Image.new('RGB', (total_width, H), color='black')

            # 拼接每一帧及分隔条
            for j in range(N):
                frame = frames[j].permute(1, 2, 0).cpu().numpy()  # 转换为(H, W, C)
                frame_image = Image.fromarray((frame * 255).astype('uint8'), 'RGB')

                # 计算当前帧在拼接图像中的位置
                x_position = j * (W + separator_width)
                concat_image.paste(frame_image, (x_position, 0))

            # 显示图像
            plt.imshow(concat_image)
            plt.title(f'Sample {i+1}')
            plt.axis('off')  # 关闭坐标轴显示
            plt.show()

            # 保存图像到文件
            concat_image.save(f'sample_{i+1}.png')