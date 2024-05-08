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
import os
from joblib import hash
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections

from .kv_caching import KeysValues
from .slicer import Embedder, Head, ActEmbedder
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights
from lzero.model.utils import cal_dormant_ratio
from ding.torch_utils import to_device

from line_profiler import line_profiler
import hashlib
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

class SimNorm(nn.Module):
    """
    简单单位向量归一化。
    改编自 https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        # 确保至少有一个单纯形用于归一化。
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.softmax(x, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

# 使用LRU缓存替换原有的字典缓存
from functools import lru_cache

@lru_cache(maxsize=5000)
# @lru_cache(maxsize=config.max_cache_size)
def quantize_state_with_lru_cache(state, num_buckets=15):
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    return tuple(quantized_state)

# 在retrieve_or_generate_kvcache方法中使用优化后的缓存函数
# cache_key = quantize_state_with_lru_cache(state_single_env)

# def quantize_state(state, num_buckets=15): # for atari
def quantize_state(state, num_buckets=100): # for memory NOTE:TODO
    """
    量化状态向量。
    参数:
        state: 要量化的状态向量。
        num_buckets: 量化的桶数。
    返回:
        量化后的状态向量的哈希值。
    """
    # 使用np.digitize将状态向量的每个维度值映射到num_buckets个桶中 
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    # 使用更稳定的哈希函数
    quantized_state_bytes = quantized_state.tobytes()
    hash_object = hashlib.sha256(quantized_state_bytes)
    return hash_object.hexdigest()

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor 
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_policy: torch.FloatTensor
    logits_value: torch.FloatTensor

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig, tokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.policy_entropy_weight = config.policy_entropy_weight
        self.predict_latent_loss_type = config.predict_latent_loss_type
        self.group_size = config.group_size
        self.num_groups = config.embed_dim // config.group_size
        self.obs_type = config.obs_type
        self.embed_dim = config.embed_dim 
        self.num_heads = config.num_heads
        self.gamma = config.gamma
        self.context_length = config.context_length  # TODO config.context_length
        # self.context_length_for_recurrent = config.context_length_for_recurrent
        # self.context_length = self.config.max_tokens  # TODO
        # self.context_length_for_recurrent = self.config.max_tokens  # TODO
        self.dormant_threshold = config.dormant_threshold
        self.analysis_dormant_ratio =  config.analysis_dormant_ratio

        self.transformer = Transformer(config)
        self.num_observations_tokens = config.tokens_per_block - 1

        self.latent_recon_loss_weight = config.latent_recon_loss_weight
        self.perceptual_loss_weight = config.perceptual_loss_weight

        self.device = config.device
        self.support_size = config.support_size
        self.action_shape = config.action_shape
        self.max_cache_size = config.max_cache_size
        self.env_num = config.env_num
        self.num_layers = config.num_layers
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)
        # self.recurrent_keep_deepth = config.recurrent_keep_deepth

        all_but_last_latent_state_pattern = torch.ones(config.tokens_per_block)
        all_but_last_latent_state_pattern[-2] = 0  # 1,...,0,1

        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)  # 17
        act_tokens_pattern[-1] = 1  # 0,...,0,1
        latent_state_pattern = 1 - act_tokens_pattern  # 1,...,1,0

        # 当前latent state的策略值
        value_policy_tokens_pattern = torch.zeros(config.tokens_per_block)
        value_policy_tokens_pattern[-2] = 1  # [0,...,1,0]

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        # 预先计算位置编码矩阵，只用于collect/eval 的推理阶段，不用于训练阶段
        # self.positional_embedding_k = [self.transformer.blocks[layer].attn.key(self.pos_emb.weight).view(1, config.max_tokens, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2).cuda().detach() for layer in range(config.num_layers)]  # (B, nh, T, hs)
        # self.positional_embedding_v = [self.transformer.blocks[layer].attn.value(self.pos_emb.weight).view(1, config.max_tokens, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2).cuda().detach() for layer in range(config.num_layers)]  # (B, nh, T, hs)
            
        # 预先计算位置编码矩阵，只用于collect/eval 的推理阶段，不用于训练阶段
        self.precompute_pos_emb_diff_kv()

        self.act_embedding_table = nn.Embedding(act_vocab_size, config.embed_dim)
         # NOTE: 对于离散动作，使用fixed_act_embedding，可能前期效率更高,但后期性能较差, 注意需要self.act_embedding_table.weight不是全零初始化的 ####
        # self.act_embedding_table.weight.requires_grad = False

        self.obs_per_embdding_dim = config.embed_dim  # 16*64=1024

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,  # 0,...,0,1
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, self.support_size)
            )
        )
        self.head_observations = Head(  # TODO  
            max_blocks=config.max_blocks,
            block_mask=all_but_last_latent_state_pattern,  # 1,...,0,1 # https://github.com/eloialonso/iris/issues/19
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, self.obs_per_embdding_dim),
                self.sim_norm, # TODO
                # nn.Sigmoid(),  # only for ablation
                # nn.Softmax(),  # only for ablation
            )
        )
        self.head_policy = Head(
            max_blocks=config.max_blocks, 
            block_mask=value_policy_tokens_pattern,  # [0,...,1,0]
            head_module=nn.Sequential(  # （8, 5, 128）
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, self.action_shape)  # TODO(pu); action shape
            )
        )
        self.head_value = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),  
                nn.GELU(),
                nn.Linear(config.embed_dim, self.support_size)  # TODO(pu): action shape
            )
        )

        self.apply(init_weights)

        last_linear_layer_init_zero = True  # TODO: 有利于收敛速度。
        if last_linear_layer_init_zero:
            # 将头部模块的最后一个线性层的权重和偏置初始化为零
            for _, layer in enumerate(reversed(self.head_value.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break
            for _, layer in enumerate(reversed(self.head_rewards.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break  
            for _, layer in enumerate(reversed(self.head_observations.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break

        # 使用collections.OrderedDict作为缓存结构，可以维持插入顺序
        self.past_keys_values_cache_recurrent_infer = collections.OrderedDict()
        self.past_keys_values_cache_init_infer = collections.OrderedDict()
        self.past_keys_values_cache_init_infer_envs = [collections.OrderedDict() for _ in range(self.env_num )]


        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []

        if self.num_observations_tokens == 16:  # k=16
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:  # K=1
            self.projection_input_dim = self.obs_per_embdding_dim  # for atari #TODO

        self.hit_count = 0
        self.total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
        # TODO: Transformer更新后应该清除缓存 
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num, max_tokens=self.context_length)


    def precompute_pos_emb_diff_kv(self):
        if self.context_length <= 2:
            # 即全部是单帧的，没有context
            return
        # 预先计算位置编码矩阵,只用于collect/eval的推理阶段,不用于训练阶段
        self.positional_embedding_k = [self.transformer.blocks[layer].attn.key(self.pos_emb.weight).view(1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2).cuda().detach() for layer in range(self.config.num_layers)]  # (B, nh, T, hs)
        self.positional_embedding_v = [self.transformer.blocks[layer].attn.value(self.pos_emb.weight).view(1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2).cuda().detach() for layer in range(self.config.num_layers)]  # (B, nh, T, hs)
        
        # 预先计算所有可能的位置编码差值
        self.pos_emb_diff_k = []
        self.pos_emb_diff_v = []
        for layer in range(self.config.num_layers):
            layer_pos_emb_diff_k = {}
            layer_pos_emb_diff_v = {}
            # for start in range(self.config.max_tokens):
            #     for end in range(start+1, self.config.max_tokens):
            for start in [2]:
                for end in [self.context_length-1]:
                    original_pos_emb_k = self.positional_embedding_k[layer][:,:, start:end, :]
                    new_pos_emb_k = self.positional_embedding_k[layer][:,:, :end-start, :]
                    layer_pos_emb_diff_k[(start, end)] = new_pos_emb_k - original_pos_emb_k
                    
                    original_pos_emb_v = self.positional_embedding_v[layer][:,:, start:end, :]
                    new_pos_emb_v = self.positional_embedding_v[layer][:,:, :end-start, :]
                    layer_pos_emb_diff_v[(start, end)] = new_pos_emb_v - original_pos_emb_v
            self.pos_emb_diff_k.append(layer_pos_emb_diff_k)
            self.pos_emb_diff_v.append(layer_pos_emb_diff_v)

    def forward(self, obs_embeddings_or_act_tokens, past_keys_values: Optional[KeysValues] = None, kvcache_independent=False, is_init_infer=True, valid_context_lengths=None) -> WorldModelOutput:
        if kvcache_independent:
            # 根据past_keys_values获取每个样本的步骤数
            prev_steps = 0 if past_keys_values is None else [past_kv.size for past_kv in past_keys_values]
            prev_steps = torch.tensor(prev_steps, device=self.device)
        else:
            prev_steps = 0 if past_keys_values is None else past_keys_values.size
        if is_init_infer: # TODO ===================
            valid_context_lengths=None
        if 'obs_embeddings' in obs_embeddings_or_act_tokens.keys():
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)  # (B, T, E)
            if kvcache_independent:
                # 生成每个样本的步骤indices
                steps_indices = prev_steps + torch.arange(num_steps, device=obs_embeddings.device)
                # 获取位置嵌入
                position_embeddings = self.pos_emb(steps_indices)
                # 将位置嵌入reshape回(batch_size, num_steps, embedding_dim)
                position_embeddings = position_embeddings.view(-1, num_steps, position_embeddings.shape[-1])
                # 将位置嵌入加到obs_embeddings上
                sequences = obs_embeddings + position_embeddings
            else:
                if is_init_infer:
                    sequences = obs_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))
                else:
                    # 获取每个样本的有效长度
                    valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)  # NOTE
                    # NOTE: 根据有效长度获取位置编码
                    position_embeddings = self.pos_emb(valid_context_lengths + torch.arange(num_steps, device=self.device)).unsqueeze(1)
                    sequences = obs_embeddings + position_embeddings
        elif 'act_tokens' in obs_embeddings_or_act_tokens.keys():
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            if len(act_tokens.shape) == 3:
                act_tokens = act_tokens.squeeze(1)
            num_steps = act_tokens.size(1)  # (B, T)
            act_embeddings = self.act_embedding_table(act_tokens)

            if kvcache_independent:
                # 生成每个样本的步骤indices
                steps_indices = prev_steps + torch.arange(num_steps, device=act_embeddings.device)
                # 获取位置嵌入
                position_embeddings = self.pos_emb(steps_indices)
                # 将位置嵌入reshape回(batch_size, num_steps, embedding_dim)
                position_embeddings = position_embeddings.view(-1, num_steps, position_embeddings.shape[-1])
                # 将位置嵌入加到obs_embeddings上  
                sequences = act_embeddings + position_embeddings
            else:
                if is_init_infer:
                    sequences = act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=act_tokens.device))
                else:
                    # 获取每个样本的有效长度
                    valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)
                    # NOTE: 根据有效长度获取位置编码
                    position_embeddings = self.pos_emb(valid_context_lengths + torch.arange(num_steps, device=self.device)).unsqueeze(1)
                    sequences = act_embeddings + position_embeddings
        else:
            # ============== for learn ==============
            obs_embeddings_and_act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
            # obs_embeddings: (B, L, K=16, E), act_tokens: (B, L, 1)
            obs_embeddings, act_tokens = obs_embeddings_and_act_tokens
            if len(obs_embeddings.shape) == 3:  # for batch compute loss
                obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens, -1)

            num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))  # L(k+1)

            # 根据动作tokens生成动作嵌入
            act_embeddings = self.act_embedding_table(act_tokens)  # (B, L, 1) -> (B, L, 1, E)

            # 已知obs_embeddings的维度为 (B, L, K, E), act_embeddings的维度为(B, L, 1, E) 
            # 希望得到一个obs_act_embeddings向量的维度为 (B, L(K+1), E)
            # 而且让得到的obs_act_embeddings的第2个维度的数据为：obs, act, obs, act, ..., 这样的排列顺序。

            B, L, K, E = obs_embeddings.size()
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

            # 添加位置嵌入  
            sequences = obs_act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))
            

        if kvcache_independent:
            x = []
            for k, past_kv in enumerate(past_keys_values):
                # x.append(self.transformer(sequences[k].unsqueeze(0), past_kv))
                x.append(self.transformer(sequences[k].unsqueeze(0), past_kv, valid_context_lengths=valid_context_lengths[k].unsqueeze(0)))
            x = torch.cat(x, dim=0)
        else: #
            # x = self.transformer(sequences, past_keys_values)
            x = self.transformer(sequences, past_keys_values, valid_context_lengths=valid_context_lengths)
            # ============ visualize_attention_map ================= 
            # TODO: only in train
            # if 'obs_embeddings_and_act_tokens' in obs_embeddings_or_act_tokens.keys():

            #     from lzero.model.gpt_models.attention_map import visualize_attention_map, visualize_attention_maps
            #     visualize_attention_maps(self.transformer, sequences, past_keys_values, valid_context_lengths)
            #     # past_keys_values = None
            #     # for layer_id in range(8):
            #     #     for head_id in range(8):
            #     #         visualize_attention_map(self.transformer, sequences, past_keys_values, valid_context_lengths, layer_id=layer_id, head_id=head_id)
            #     # import sys
            #     # sys.exit(0)
                # ========== for visualize ==========
        
        # 1,...,0,1 https://github.com/eloialonso/iris/issues/19
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)

        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # TODO: root reward value
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    @torch.no_grad()
    def reset_from_initial_observations(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(obs_act_dict, dict):
            observations = obs_act_dict['obs']
            buffer_action = obs_act_dict['action']
            current_obs = obs_act_dict['current_obs']
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations, should_preprocess=True)  # (B, C, H, W) -> (B, K, E)

        if current_obs is not None:
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(current_obs, should_preprocess=True)  # (B, C, H, W) -> (B, K, E)
            self.latent_state = current_obs_embeddings
            outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action, current_obs_embeddings)
        else:
            self.latent_state = obs_embeddings
            outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action, None)

        return outputs_wm, self.latent_state

    @torch.no_grad()
    def refresh_keys_values_with_initial_latent_state_for_init_infer(self, latent_state: torch.LongTensor, buffer_action=None, current_obs_embeddings=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        if n <= self.env_num:
            if current_obs_embeddings is not None:
                if max(buffer_action) == -1:
                    # 一集的第一步
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.context_length)
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # 复制单个环境对应的 keys_values_wm 并存储
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True) # TODO
                else:
                    # 假设 latest_state 是新的 latent_state，包含 ready_env_num 个环境的信息
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []
                    for i in range(ready_env_num):
                        state_single_env = latent_state[i]  # 获取单个环境的 latent state
                        quantized_state = state_single_env.detach().cpu().numpy()
                        cache_key = quantize_state(quantized_state)  # 使用量化后的状态计算哈希值
                        # matched_value = self.past_keys_values_cache_init_infer.get(cache_key)  # 检索缓存值
                        matched_value = self.past_keys_values_cache_init_infer_envs[i].get(cache_key)  # 检索缓存值

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # 如果找到匹配的值，将其添加到列表中
                            self.root_hit_cnt += 1
                            # if self.root_total_query_cnt > 0 and self.root_total_query_cnt % 1000 == 0:
                            #     root_hit_ratio = self.root_hit_cnt / self.root_total_query_cnt
                            #     print('root_total_query_cnt:', self.root_total_query_cnt)
                            #     print(f'root_hit_ratio:{root_hit_ratio}')
                            #     print(f'root_hit find size {self.past_keys_values_cache_init_infer_envs[i][cache_key].size}') # TODO env
                            #     if self.past_keys_values_cache_init_infer[cache_key].size >= self.config.max_tokens - 3:
                            #         print(f'==' * 20)
                            #         print(f'NOTE: root_hit find size >= self.config.max_tokens - 3')
                            #         print(f'==' * 20)
                            # 这里需要deepcopy因为在transformer的forward中会原地修改matched_value
                            self.keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda')))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # 使用零值重置
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)}, past_keys_values=self.keys_values_wm_single_env, is_init_infer=True)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # 输入self.keys_values_wm_list，输出为self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    buffer_action = buffer_action[:ready_env_num]
                    # if ready_env_num<self.env_num:
                    #     print(f'init inference ready_env_num: {ready_env_num} < env_num: {self.env_num}')
                    buffer_action = torch.from_numpy(np.array(buffer_action)).to(latent_state.device)
                    act_tokens = buffer_action.unsqueeze(-1)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm, is_init_infer=True)

                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # 复制单个环境对应的 keys_values_wm 并存储
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True) # TODO


        elif n > self.env_num and buffer_action is not None:
            # 训练时计算 target value/
            # [192, 16, 64] -> [32, 6, 16, 64]
            latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens, self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

            latent_state = latent_state[:, :-1, :]
            buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
            act_tokens = rearrange(buffer_action, 'b l -> b l 1')

            # 选择每个样本的最后一步
            ###### 这将选择最后一列并保持维度不变, 最后一步的target policy/value本身就没有用到 ######
            last_steps = act_tokens[:, -1:, :]  
            # 使用torch.cat在第二个维度上连接原始act_tokens和last_steps
            act_tokens = torch.cat((act_tokens, last_steps), dim=1)

            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (latent_state, act_tokens)})

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
    def forward_initial_inference(self, obs_act_dict):
        # self.past_keys_values_cache_init_infer_envs[i]
        outputs_wm, latent_state = self.reset_from_initial_observations(obs_act_dict)  # root节点也有context
        # self.past_keys_values_cache_recurrent_infer
        # TODO: 每次search后清空，可以保证存储的 kv_cache_recurrent 都是从根节点的最长context来的
        # 或者每次search后不清空，但每次都重新存储，这样便可以保证 存储的kv_cache_recurrent, latent_state是预测部分的限制在一次search内
        self.past_keys_values_cache_recurrent_infer.clear()
        # print('=='*20)
        # print('self.past_keys_values_cache_recurrent_infer.clear() after init_inference')

        # self.latent_state_index_in_search_path = [[] for i in range(latent_state.shape[0])]
        # self.next_latent_state_depth = [[] for i in range(latent_state.shape[0])]
        # self.last_depth = [0 for i in range(latent_state.shape[0])]
        # 维护一个全局的depth_map字典,用于存储已计算过的深度信息
        # self.depth_map =  [{0: 1} for i in range(latent_state.shape[0])]  # 根节点处的深度映射
        
        return outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

    """
    假设env_num=8  
    8个环境的kv_cache单独存储与寻找,都存储在一个dict中,在recurrent_inference时, 
    由于不同环境找到的kv_cache的size不同,先根据最大size对kv_cache在前部补零,然后组成batch_size的kv_cache
    其内部也是通过batch执行transformer forward的推理 
    """

    def convert_to_depth(self, search_path, depth_map, last_depth):
        # 获取新加入的元素
        new_index = search_path[-1]
        
        # 如果新加入的元素对应的深度没有被计算过,则基于父节点的深度计算它
        if new_index not in depth_map:
            if search_path[new_index] not in depth_map:
                depth_map[search_path[new_index]] = max(list(depth_map.values())) + 1
            else:
                depth_map[new_index] = depth_map[search_path[new_index]] + 1
        
        # 将新加入元素的深度添加到last_depth的末尾
        last_depth.append(depth_map[new_index])
        
        return last_depth


    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0, latent_state_index_in_search_path=[]):
        # 一般来讲,在一次 MCTS search中,我们需要维护H长度的context来使用transformer进行推理。
        # 由于在一次search里面。agent最多访问sim个不同的节点,因此我们只需维护一个 {(state:kv_cache)}的列表。
        # 但如果假设环境是MDP的话,然后根据当前的 latest_state s_t 在这个列表中查找即可
        # TODO: 但如果假设环境是非MDP的话,需要维护一个 {(rootstate_action_history:kv_cache)}的列表?
        # print(f'latent_state_index_in_search_path:{latent_state_index_in_search_path}')
        latest_state, action = state_action_history[-1]
        # print(f'action:{action}')

        # 假设 latest_state 是新的 latent_state,包含 ready_env_num 个环境的信息
        ready_env_num = latest_state.shape[0]
        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index)  

        latent_state_list = []
        token = action.reshape(-1, 1)

        # 获取self.keys_values_wm_size_list的最小值min_size
        min_size = min(self.keys_values_wm_size_list)
        if min_size >= self.config.max_tokens - 5:
            self.length_largethan_maxminus5_context_cnt += len(self.keys_values_wm_size_list)
        if min_size >= self.config.max_tokens - 7:
            self.length_largethan_maxminus7_context_cnt += len(self.keys_values_wm_size_list)
        # 打印统计信息
        if self.total_query_count > 0 and self.total_query_count % 10000 == 0:
            self.hit_freq = self.hit_count / (self.total_query_count)
            print('total_query_count:', self.total_query_count)
            # 如果总查询次数大于0,计算并打印cnt的比率
            length_largethan_maxminus5_context_cnt_ratio = self.length_largethan_maxminus5_context_cnt / self.total_query_count
            print('recurrent largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
            print('recurrent largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
            length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.total_query_count
            print('recurrent largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
            print('recurrent largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)

        # 输入self.keys_values_wm_list,输出为self.keys_values_wm
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False) # 与上面self.retrieve_or_generate_kvcache返回的一致
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list
        for k in range(2):  # 假设每次只有一个动作token。
            # action_token obs_token, ..., obs_token  1+1
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}
            # self.keys_values_wm 会被原地改动 ===============
            outputs_wm = self.forward(obs_embeddings_or_act_tokens, past_keys_values=self.keys_values_wm, kvcache_independent=False, is_init_infer=False)
            # print('keys_values_wm_size_list_current:', self.keys_values_wm_size_list_current)
            self.keys_values_wm_size_list_current = [i+1 for i in self.keys_values_wm_size_list_current] # NOTE: +1 ===============
            if k == 0:
                # 如果k==0,token是action_token,outputs_wm.logits_rewards 是有值的
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                # 一共产生16个obs_token,每次产生一个
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        # 删除旧的self.latent_state以释放内存 ===========
        del self.latent_state
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(self.latent_state, is_init_infer=False, simulation_index=simulation_index, latent_state_index_in_search_path=latent_state_index_in_search_path) # TODO


        return outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value

    def trim_and_pad_kv_cache(self, is_init_infer=True):
        # =========== TODO： is_init_infer=True 在episode快结束时，batch里面不同env的context的处理=========
        # if is_init_infer:
        #     print('='*20)
        #     print(f'self.keys_values_wm_size_list: {self.keys_values_wm_size_list}')
        # print(f'is_init_infer: {is_init_infer}')
        # print(f'self.keys_values_wm_size_list: {self.keys_values_wm_size_list}')
        # NOTE: self.keys_values_wm_size_list会传递到world_model.forward()中
        # 找到所有key-value尺寸中的最大尺寸,用于填充
        max_size = max(self.keys_values_wm_size_list)

        # 遍历transformer的每一层
        for layer in range(self.num_layers):
            # 初始化列表来存储修剪和填充后的k和v缓存
            kv_cache_k_list = []
            kv_cache_v_list = []

            # 枚举key-value对列表
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                # 检索当前层的key和value缓存
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                # 获取当前缓存的有效尺寸
                effective_size = self.keys_values_wm_size_list[idx]
                # 计算需要填充的尺寸差异
                pad_size = max_size - effective_size

                # 如果需要填充,在缓存的开头添加'pad_size'个零 ====================
                if pad_size > 0:
                    # NOTE: 先去掉后面pad_size个无效的零 kv, 再在缓存的开头添加'pad_size'个零 ，注意位置编码的正确性
                    k_cache_trimmed = k_cache[:, :, :-pad_size, :]
                    v_cache_trimmed = v_cache[:, :, :-pad_size, :]
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)  # 在缓存的开头添加'pad_size'个零 
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache  
                    v_cache_padded = v_cache

                # 将处理后的缓存添加到列表中 
                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # 沿新维度堆叠缓存,并用squeeze()移除额外维度
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1) 
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # 填充后,将缓存尺寸更新为最大尺寸
            self.keys_values_wm._keys_values[layer]._k_cache._size = max_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = max_size
        
        return self.keys_values_wm_size_list


    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0, latent_state_index_in_search_path=[], valid_context_lengths=None):
        if self.context_length <= 2:
            # 即全部是单帧的，没有context
            return
        for i in range(latent_state.size(0)):  # 遍历每个环境
            state_single_env = latent_state[i]  # 获取单个环境的潜在状态
            quantized_state = state_single_env.detach().cpu().numpy()  # 分离并将状态移至CPU
            cache_key = quantize_state(quantized_state)  # 量化状态并将其哈希值计算为缓存键

            context_length = self.context_length

            # if is_init_infer:
            #     context_length = self.context_length
            # else:
            #     context_length = self.context_length_for_recurrent

            if not is_init_infer: # NOTE: check 在recurrent_inference时去掉前面填充的0 ============
                # 从全局缓存复制keys和values到单个环境缓存
                current_max_context_length = max(self.keys_values_wm_size_list_current)
                trim_size = current_max_context_length - self.keys_values_wm_size_list_current[i]
                for layer in range(self.num_layers):
                    # 裁剪和填充逻辑
                    # 假设cache的维度是 [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i] # [num_heads, sequence_length, features]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                    if trim_size > 0:
                        # 根据有效长度裁剪 TODO=======================================
                        # NOTE: 先去掉前面pad_size/trim_size个无效的零kv, 注意位置编码的正确性
                        k_cache_trimmed = k_cache_current[:, trim_size:, :]
                        v_cache_trimmed = v_cache_current[:, trim_size:, :]
                        # 如果有效长度<current_max_context_length, 需要在缓存的后面补充'trim_size'个零 ====================
                        k_cache_padded = F.pad(k_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)  # 在缓存的后面补充'trim_size'个零 
                        v_cache_padded = F.pad(v_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)
                    else:
                        k_cache_padded = k_cache_current  
                        v_cache_padded = v_cache_current

                    # 更新单环境cache
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm_size_list_current[i] # TODO
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm_size_list_current[i]

                    # NOTE: check 非常重要 ============
                    if self.keys_values_wm_single_env._keys_values[layer]._k_cache._size >= context_length-1: 
                        # 固定只保留最近self.context_length-3个timestep的context 
                        # ===============对于memory环境，训练时是H步，recurrent_inference时可能超出H步 =================
                        # print(f'self.keys_values_wm_size_list_current[i]:{self.keys_values_wm_size_list_current[i]}')
                        # 需要对self.keys_values_wm_single_env进行处理，而不是self.keys_values_wm
                        # 裁剪和填充逻辑
                        # 假设cache的维度是 [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache
                        
                        # 移除前2步并保留最近的self.context_length-3步
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length-1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length-1, :].squeeze(0)
                        
                        # 索引预先计算的位置编码差值
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length-1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length-1)]
                        #=========== 对k和v应用位置编码矫正 ==================
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # 沿第3维，用0填充后3步
                        padding_size = (0, 0, 0, 3)  # F.pad的参数(0, 0, 0, 3)指定了在每个维度上的填充量。这些参数是按(左, 右, 上, 下)的顺序给出的,对于三维张量来说,分别对应于(维度2左侧, 维度2右侧, 维度1左侧, 维度1右侧)的填充。
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # 更新单环境cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length-3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length-3


            else: # init_inference
                # 从全局缓存复制keys和values到单个环境缓存
                for layer in range(self.num_layers):
                    if self.keys_values_wm._keys_values[layer]._k_cache._size < context_length-1:  # 固定只保留最近5个timestep的context
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(0)  # Shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm._keys_values[layer]._k_cache._size
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm._keys_values[layer]._v_cache._size
                    else:
                        # if is_init_infer:
                            # return # TODO: reset kv_cache。如果上下文信息是滑动窗口（Sliding Window），需要修复position_embedding的kv_cache，增加了计算开销。
                        # else:
                        # 裁剪和填充逻辑
                        # 假设cache的维度是 [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                        v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]
                        
                        # 移除前2步并保留最近的self.context_length-3步
                        # k_cache_trimmed = k_cache_current[:, 2:self.config.max_tokens - 1, :]
                        # v_cache_trimmed = v_cache_current[:, 2:self.config.max_tokens - 1, :]
                        k_cache_trimmed = k_cache_current[:, 2:context_length-1, :]
                        v_cache_trimmed = v_cache_current[:, 2:context_length-1, :]
                        
                        # 索引预先计算的位置编码差值
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length-1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length-1)]
                        #=========== 对k和v应用位置编码矫正 ==================
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # 沿第3维，用0填充后3步
                        padding_size = (0, 0, 0, 3)  # F.pad的参数(0, 0, 0, 3)指定了在每个维度上的填充量。这些参数是按(左, 右, 上, 下)的顺序给出的,对于三维张量来说,分别对应于(维度2左侧, 维度2右侧, 维度1左侧, 维度1右侧)的填充。
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # 更新单环境cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length-3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length-3

            if is_init_infer:
                # TODO：每次都存储最新的    
                # self.past_keys_values_cache_init_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
                self.past_keys_values_cache_init_infer_envs[i][cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
            
            else:
                self.past_keys_values_cache_recurrent_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))


    def retrieve_or_generate_kvcache(self, latent_state, ready_env_num, simulation_index=0):
        """
        This method iterates over the environments, retrieves a matching cache if available,
        or generates a new one otherwise. It updates the lists with the keys_values caches and their sizes.
        """
        # self.root_latent_state = [False for i in range(ready_env_num)]
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # 获取单个环境的潜在状态
            cache_key = quantize_state(state_single_env)  # 使用量化后的状态计算哈希值
            # 如果存在,检索缓存值
            # 先在self.past_keys_values_cache_init_infer中寻找
            # matched_value = self.past_keys_values_cache_init_infer.get(cache_key)
            matched_value = self.past_keys_values_cache_init_infer_envs[i].get(cache_key)

            # 再在self.past_keys_values_cache中寻找 TODO
            if matched_value is None:
                matched_value = self.past_keys_values_cache_recurrent_infer.get(cache_key)
            if matched_value is not None:
                # 如果找到匹配值,将其添加到列表中
                self.hit_count += 1
                # 需要深度拷贝,因为transformer的forward可能会就地修改matched_value
                self.keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, self.device)))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # 如果没有找到匹配值,使用零重置
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                self.forward({'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)}, past_keys_values=self.keys_values_wm_single_env, is_init_infer=True)
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)
        return self.keys_values_wm_size_list

    def to_device_for_kvcache(self, keys_values: KeysValues, device: str) -> KeysValues:
        """
        Transfer all KVCache objects within the KeysValues object to a certain device.

        Arguments:
            - keys_values (KeysValues): The KeysValues object to be transferred.
            - device (str): The device to transfer to.

        Returns:
            - keys_values (KeysValues): The KeysValues object with its caches transferred to the specified device.
        """
        # 检查CUDA是否可用并选择第一个可用的CUDA设备
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

    def compute_loss(self, batch, target_tokenizer: Tokenizer=None, inverse_scalar_transform_handle=None, **kwargs: Any) -> LossWithIntermediateLosses:
        # 将观察编码为潜在状态表示
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'], should_preprocess=False)
        
        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # calculate dormant ratio of encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:]) # (32,5,3,64,64) -> (160,3,64,64)
            dormant_ratio_encoder = cal_dormant_ratio(self.tokenizer.representation_network, inputs.detach(), percentage=self.dormant_threshold)
            self.past_keys_values_cache_init_infer.clear()
            self.past_keys_values_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)
            
        # 假设latent_state_roots是一个tensor
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()  # 计算L2范数
        # print("L2 Norms:", l2_norms)

        # 注册梯度钩子,用于梯度缩放。这里的作用是将梯度缩小为原来的1/5,有助于训练的稳定性。
        # 但是否必要取决于具体问题,需要通过实验来验证。
        # obs_embeddings.register_hook(lambda grad: grad * 1/5)
        
        if self.obs_type == 'image':
            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            
            # 计算重建损失和感知损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1

            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 4, 64, 64), reconstructed_images) # NOTE: for stack=4
            # perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4
        
        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4

            # TODO: no decoder
            # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4
            
            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))
            # 计算重建损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25), reconstructed_images) # NOTE: for stack=1
        elif self.obs_type == 'image_memory':
            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            original_images, reconstructed_images = batch['observations'], reconstructed_images

            #  ========== for debugging ==========
            # batch['observations'].shape torch.Size([2, 17, 3, 5, 5]) 
            # reconstructed_images.shape torch.Size([34, 3, 5, 5])
            # self.visualize_reconstruction_v1(original_images, reconstructed_images)

            #  ========== for visualize ==========
            # batch['target_policy'].shape torch.Size([2, 17, 4])
            # batch['target_value'].shape torch.Size([2, 17, 101])
            # batch['rewards'].shape torch.Size([2, 17, 101])

            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1,101)).reshape(batch['observations'].shape[0],batch['observations'].shape[1],1) # torch.Size([2, 17, 1])
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1,101)).reshape(batch['observations'].shape[0],batch['observations'].shape[1],1) # torch.Size([2, 17, 1])


            # 计算重建损失和感知损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5), reconstructed_images)  # NOTE: for stack=1 TODO
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 4, 5, 5), reconstructed_images)  # NOTE: for stack=1
            # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4
            perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4

        # 动作tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')


        # 前向传播,得到预测的观察、奖励和策略等
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # calculate dormant ratio of world_model
            dormant_ratio_world_model = cal_dormant_ratio(self, {'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())}, percentage=self.dormant_threshold)
            self.past_keys_values_cache_init_infer.clear()
            self.past_keys_values_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_world_model =  torch.tensor(0.)
        
        #  ========== for visualize ==========
        # outputs.logits_policy.shape torch.Size([2, 17, 4])
        # outputs.logits_value.shape torch.Size([2, 17, 101])
        # outputs.logits_rewards.shape torch.Size([2, 17, 101])

        # predict_policy = outputs.logits_policy
        # # 使用 softmax 对最后一个维度（dim=-1）进行处理
        # predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        # predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1,101)).reshape(batch['observations'].shape[0],batch['observations'].shape[1],1) # predict_value: torch.Size([2, 17, 1])
        # predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1,101)).reshape(batch['observations'].shape[0],batch['observations'].shape[1],1) # predict_rewards: torch.Size([2, 17, 1])
        # # import pdb; pdb_set_trace()
        # self.visualize_reconstruction_v2(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, suffix='visual_match_memlen1-0-15_v2') # TODO
        # import sys
        # sys.exit(0)
        #  ========== for visualize ==========

        # 为了训练稳定性,使用target_tokenizer计算真实的下一个潜在状态表示
        with torch.no_grad():
            traget_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'], should_preprocess=False)

        # 计算观察、奖励和结束标签  
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(traget_obs_embeddings, batch['rewards'],
                                                                                        batch['ends'],
                                                                                        batch['mask_padding'])
        
        # 重塑观察的logits和labels                                                                              
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        # 计算观察的预测损失。这里提供了两种选择:MSE和Group KL  
        if self.predict_latent_loss_type == 'mse':
            # MSE损失,直接比较logits和labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(-1) # labels_observations.detach()是冗余的，因为前面是在with torch.no_grad()中计算的
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL损失,将特征分组,然后计算组内的KL散度
            batch_size, num_features = logits_observations.shape

            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size)
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups,self. group_size)

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)
        
        # 应用mask到loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # 计算策略和价值的标签
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                batch['target_policy'],
                                                                                batch['mask_padding'])

        # 计算奖励、策略和价值的损失                                                                    
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')
        loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy, batch, element='policy')
        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')
        
        # return LossWithIntermediateLosses(
        #     latent_recon_loss_weight=self.latent_recon_loss_weight,
        #     perceptual_loss_weight=self.perceptual_loss_weight,
        #     loss_obs=loss_obs.mean(),
        #     loss_rewards=loss_rewards.mean(),
        #     loss_value=loss_value.mean(),
        #     loss_policy=loss_policy.mean(),
        #     latent_recon_loss=latent_recon_loss,
        #     perceptual_loss=perceptual_loss,
        #     orig_policy_loss=orig_policy_loss.mean(),
        #     policy_entropy=policy_entropy.mean()
        # )
        # 计算时间步
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # 计算每个时间步的折扣系数
        discounts = self.gamma ** timesteps

        # 将损失分为第一步、中间步和最后一步
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        #  batch['mask_padding'] 为后面H步的mask情况，如果mask为False则把对应的loss从统计中去掉，以维持平均统计值的准确性
        # 对每个损失项进行分组计算
        for loss_name, loss_tmp in zip(
            ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
            [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # 获取对应的 mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # 获取对应的 mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]
            
            # 将损失调整为 (batch_size, seq_len) 的形状
            loss_tmp = loss_tmp.view(-1, seq_len)
            
            # 第一步的损失
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()
            
            # 中间步的损失
            middle_step_index = seq_len // 2
            middle_step_mask = mask_padding[:, middle_step_index]
            middle_step_losses[loss_name] = loss_tmp[:, middle_step_index][middle_step_mask].mean()
            
            # 最后一步的损失
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # 对重构损失和感知损失进行折扣
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

        # 计算整体的折扣损失
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).mean()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).mean()

        return LossWithIntermediateLosses(
            latent_recon_loss_weight=self.latent_recon_loss_weight,
            perceptual_loss_weight=self.perceptual_loss_weight,
            loss_obs=discounted_loss_obs,
            loss_rewards=discounted_loss_rewards,
            loss_value=discounted_loss_value,
            loss_policy=discounted_loss_policy,
            latent_recon_loss=discounted_latent_recon_loss,
            perceptual_loss=discounted_perceptual_loss,
            orig_policy_loss=discounted_orig_policy_loss,
            policy_entropy=discounted_policy_entropy,
            first_step_losses=first_step_losses,
            middle_step_losses=middle_step_losses,
            last_step_losses=last_step_losses,
            dormant_ratio_encoder=dormant_ratio_encoder,
            dormant_ratio_world_model=dormant_ratio_world_model,
            latent_state_l2_norms=latent_state_l2_norms,
        )


    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # 假设outputs是一个具有'rewards'、'policy'和'value'的logits属性的对象
        # labels是一个与之比较的目标张量。batch是一个带有指示有效时间步的mask的字典。

        logits = getattr(outputs, f'logits_{element}')

        # 重塑你的张量
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # 假设labels最初的shape是 [batch, time, dim]

        # 重塑你的mask。True表示有效数据。
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # 计算交叉熵损失
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if element == 'policy':
            # 计算策略熵损失
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # 用指定的权重组合损失
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    def compute_policy_entropy_loss(self, logits, mask):
        # 计算策略的熵
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # 应用mask并返回平均熵损失
        entropy_loss = (entropy * mask)
        return entropy_loss

    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # 每个序列样本最多只有1个done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)

    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mask_fill = torch.logical_not(mask_padding)
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)
        return labels_policy.reshape(-1, self.action_shape), labels_value.reshape(-1, self.support_size)  # TODO(pu)

    def visualize_reconstruction_v1(self, original_images, reconstructed_images, suffix='pong', width=64):
        # 确保输入张量的维度匹配
        assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
        assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
        assert original_images.shape[2:] == reconstructed_images.shape[1:]

        batch_size = original_images.shape[0]
        num_timesteps = original_images.shape[1]

        for batch_idx in range(batch_size):
            # 创建一个白色背景的大图像
            big_image = torch.ones(3, (width + 1) * 2 + 1, (width + 1) * num_timesteps + 1)

            # 将原始图像和重建图像复制到大图像中
            for i in range(num_timesteps):
                original_image = original_images[batch_idx, i, :, :, :]
                reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

                big_image[:, 1:1+width, (width + 1) * i + 1:(width + 1) * (i + 1)] = original_image
                big_image[:, 2+width:2+2*width, (width + 1) * i + 1:(width + 1) * (i + 1)] = reconstructed_image

            # 转换张量为PIL图像
            image = torchvision.transforms.ToPILImage()(big_image)

            # 绘制图像
            plt.figure(figsize=(20, 4))
            plt.imshow(image)
            plt.axis('off')

            # 添加时间步标签
            for i in range(num_timesteps):
                plt.text((width + 1) * i + width/2, -10, str(i + 1), ha='center', va='top', fontsize=12)

            # 添加行标签
            plt.text(-0.5, 3, 'Original', ha='right', va='center', fontsize=12)
            plt.text(-0.5, 3+width+1, 'Reconstructed', ha='right', va='center', fontsize=12)

            plt.tight_layout()
            # plt.savefig(f'./render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v1.png')
            plt.savefig(f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v1.png')
            plt.close()

    def visualize_reconstruction_v2(self, original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, suffix='pong', width=64):
        # 确保输入张量的维度匹配
        assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
        assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
        assert original_images.shape[2:] == reconstructed_images.shape[1:]

        batch_size = original_images.shape[0]
        num_timesteps = original_images.shape[1]
        num_actions = predict_policy.shape[2]

        # 根据动作空间大小自适应颜色
        colors = plt.cm.viridis(np.linspace(0, 1, num_actions))
        # colors = ['r', 'g', 'b', 'y']

        for batch_idx in range(batch_size):
            fig, ax = plt.subplots(5, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

            # 绘制rewards和value的折线图
            timesteps = range(1, num_timesteps + 1)
            ax[0].plot(timesteps, true_rewards[batch_idx, :, 0].cpu().detach().numpy(), 'g-', label='True Rewards')
            ax[0].plot(timesteps, predict_rewards[batch_idx, :, 0].cpu().detach().numpy(), 'g--', label='Predict Rewards')
            ax[0].set_xticks(timesteps)
            ax[0].set_xticklabels([])
            ax[0].legend(loc='upper left')
            ax[0].set_ylabel('Rewards')

            ax0_twin = ax[0].twinx()
            # ax0_twin.plot(timesteps, target_predict_value[batch_idx, :, 0].cpu().detach().numpy(), 'b-', label='Target Predict Value')
            ax0_twin.plot(timesteps, predict_value[batch_idx, :, 0].cpu().detach().numpy(), 'b--', label='Predict Value')
            ax0_twin.legend(loc='upper right')
            ax0_twin.set_ylabel('Value')

            # 绘制原始图像和重建图像
            image_width = 1.0
            image_height = original_images.shape[3] / original_images.shape[4] * image_width
            gap_width = 0.2
            for i in range(num_timesteps):
                original_image = original_images[batch_idx, i, :, :, :]
                reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

                left = i * (image_width + gap_width)
                right = left + image_width
                bottom = 0.5 - image_height / 2
                top = 0.5 + image_height / 2

                ax[1].imshow(torchvision.transforms.ToPILImage()(original_image), extent=[left, right, bottom, top], aspect='auto')
                ax[2].imshow(torchvision.transforms.ToPILImage()(reconstructed_image), extent=[left, right, bottom, top], aspect='auto')

            ax[1].set_xlim(0, num_timesteps * (image_width + gap_width) - gap_width)
            ax[1].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(num_timesteps)])
            ax[1].set_xticklabels([])
            ax[1].set_yticks([])
            ax[1].set_ylabel('Original', rotation=0, labelpad=30)

            ax[2].set_xlim(0, num_timesteps * (image_width + gap_width) - gap_width)
            ax[2].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(num_timesteps)])
            ax[2].set_xticklabels([])
            ax[2].set_yticks([])
            ax[2].set_ylabel('Reconstructed', rotation=0, labelpad=30)

            # # 绘制predict_policy和target_policy的概率分布柱状图
            # 计算柱状图的宽度和偏移量，确保它们不会重叠
            bar_width = 0.8 / num_actions
            offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)
            # 绘制predict_policy和target_policy的概率分布柱状图
            for i in range(num_timesteps):
                for j in range(num_actions):
                    ax[3].bar(i + offset[j], predict_policy[batch_idx, i, j].item(), width=bar_width, color=colors[j], alpha=0.5)
                    ax[4].bar(i + offset[j], target_policy[batch_idx, i, j].item(), width=bar_width, color=colors[j], alpha=0.5)
            ax[3].set_xticks(timesteps)
            ax[3].set_ylabel('Predict Policy')
            ax[4].set_xticks(timesteps)
            ax[4].set_xlabel('Timestep')
            ax[4].set_ylabel('Target Policy')
            # 添加图例
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
            labels = [f'Action {i}' for i in range(num_actions)]
            ax[4].legend(handles, labels, loc='upper right', ncol=num_actions)

            plt.tight_layout()
            directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
            # 检查路径是否存在，不存在则创建
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f'{directory}/reconstruction_visualization_batch_{batch_idx}_v3.png')
            # plt.savefig(f'./render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v2.png')
            plt.close()


    def save_as_image(self, batch_tensor):
        # batch_tensor 的形状应该是 [batch_size, sequence_length, channels, height, width]
        # 在这里 channels = 4, height = 5, width = 5
        batch_size, sequence_length, channels, height, width = batch_tensor.shape

        # 为了将所有帧组合成一张图，我们设置每行显示 sequence_length 个图像
        rows = batch_size
        cols = sequence_length

        # 创建一个足够大的空白图像来容纳所有的帧
        # 每个RGB图像的大小是 height x width，总图像的大小是 (rows * height) x (cols * width)
        final_image = Image.new('RGB', (cols * width, rows * height))

        # 遍历每一帧，将其转换为PIL图像，并粘贴到正确的位置
        for i in range(rows):
            for j in range(cols):
                # 提取当前帧的前三个通道（假设前三个通道是RGB）
                frame = batch_tensor[i, j, :3, :, :]
                # 转换为numpy数组，并调整数据范围为0-255
                frame = frame.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)
                # 创建一个PIL图像
                img = Image.fromarray(frame)
                # 粘贴到最终图像的相应位置
                final_image.paste(img, (j * width, i * height))

        # 保存图像
        final_image.save('batch_image.png')

    # # 假设 batch['observations'] 是一个满足条件的tensor
    # # 示例tensor，实际使用中应替换为实际的tensor数据
    # batch = {'observations': torch.randn(3, 16, 4, 5, 5)}

    # # 调用函数
    # save_as_image(batch['observations'])

    def render_img(self, obs: int, rec_img: int):
        import torch
        from PIL import Image
        import matplotlib.pyplot as plt

        # 假设batch是一个字典,其中包含了observations键,
        # 并且它的形状是torch.Size([B, N, C, H, W])
        # batch_observations = batch_for_gpt['observations']
        # batch_observations = batch['observations']
        batch_observations = obs.unsqueeze(0)
        # batch_observations = rec_img.unsqueeze(0)

        # batch_observations = observations.unsqueeze(0)
        # batch_observations = x.unsqueeze(0)
        # batch_observations = reconstructions.unsqueeze(0)

        B, N, C, H, W = batch_observations.shape  # 自动检测维度

        # 分隔条的宽度(可以根据需要调整)
        separator_width = 2

        # 遍历每个样本
        for i in range(B):
            # 提取当前样本中的所有帧
            frames = batch_observations[i]

            # 计算拼接图像的总宽度(包括分隔条)
            total_width = N * W + (N - 1) * separator_width

            # 创建一个新的图像,其中包含分隔条
            concat_image = Image.new('RGB', (total_width, H), color='black')

            # 拼接每一帧及分隔条
            for j in range(N):
                frame = frames[j].permute(1, 2, 0).cpu().detach().numpy()  # 转换为(H, W, C)
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
            concat_image.save(f'render/sample_{i+1}.png')



    def __repr__(self) -> str:
        return "world_model"