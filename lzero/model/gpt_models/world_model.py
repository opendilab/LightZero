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
import collections

from .kv_caching import KeysValues
from .slicer import Embedder, Head, ActEmbedder
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights
from ding.torch_utils import to_device

from line_profiler import line_profiler
import hashlib

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

def quantize_state(state, num_buckets=15): # for atari
# def quantize_state(state, num_buckets=100): # for memory NOTE:TODO
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
        self.gamma = config.gamma

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


        all_but_last_latent_state_pattern = torch.ones(config.tokens_per_block)
        all_but_last_latent_state_pattern[-2] = 0  # 1,...,0,1

        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)  # 17
        act_tokens_pattern[-1] = 1  # 0,...,0,1
        latent_state_pattern = 1 - act_tokens_pattern  # 1,...,1,0

        # 当前latent state的策略值
        value_policy_tokens_pattern = torch.zeros(config.tokens_per_block)
        value_policy_tokens_pattern[-2] = 1  # [0,...,1,0]

        # TODO: delete
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, latent_state_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.act_embedding_table = nn.Embedding(act_vocab_size, config.embed_dim)
         # NOTE: 对于离散动作，使用fixed_act_embedding，可能前期效率更高, 注意需要self.act_embedding_table.weight不是全零初始化的 ####
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
                self.sim_norm,
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
        self.context_length = config.context_length  # TODO config.context_length
        # self.context_length = self.config.max_tokens  # TODO
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
        # TODO: Transformer更新后应该清除缓存 
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num, max_tokens=self.context_length)


    def forward(self, obs_embeddings_or_act_tokens, past_keys_values: Optional[KeysValues] = None, kvcache_independent=False) -> WorldModelOutput:
        if kvcache_independent:
            # 根据past_keys_values获取每个样本的步骤数
            prev_steps = 0 if past_keys_values is None else [past_kv.size for past_kv in past_keys_values]
            prev_steps = torch.tensor(prev_steps, device=self.device)
        else:
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

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
                sequences = obs_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))
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
                sequences = act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=act_tokens.device))
        else:
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
                x.append(self.transformer(sequences[k].unsqueeze(0), past_kv))
            x = torch.cat(x, dim=0)
        else:
            x = self.transformer(sequences, past_keys_values)

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
        else:
            observations = obs_act_dict
            buffer_action = None
            current_obs = None
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
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm)

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
                        matched_value = self.past_keys_values_cache_init_infer.get(cache_key)  # 检索缓存值
                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # 如果找到匹配的值，将其添加到列表中
                            self.root_hit_cnt += 1
                            if self.root_total_query_cnt > 0 and self.root_total_query_cnt % 1000 == 0:
                                root_hit_ratio = self.root_hit_cnt / self.root_total_query_cnt
                                print('root_total_query_cnt:', self.root_total_query_cnt)
                                print(f'root_hit_ratio:{root_hit_ratio}')
                                print(f'root_hit find size {self.past_keys_values_cache_init_infer[cache_key].size}')
                                if self.past_keys_values_cache_init_infer[cache_key].size >= self.config.max_tokens - 5:
                                    print(f'==' * 20)
                                    print(f'NOTE: root_hit find size >= self.config.max_tokens - 5')
                                    print(f'==' * 20)
                            # 这里需要deepcopy因为在transformer的forward中会原地修改matched_value
                            self.keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda')))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # 使用零值重置
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)}, past_keys_values=self.keys_values_wm_single_env)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # 输入self.keys_values_wm_list，输出为self.keys_values_wm
                    self.trim_and_pad_kv_cache()

                    buffer_action = buffer_action[:ready_env_num]
                    buffer_action = torch.from_numpy(np.array(buffer_action)).to(latent_state.device)
                    act_tokens = buffer_action.unsqueeze(-1)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm)

                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm)

                    # 复制单个环境对应的 keys_values_wm 并存储
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True) # TODO


        elif n > self.env_num and buffer_action is not None:
            # 训练时计算 target value
            # [192, 16, 64] -> [32, 6, 16, 64]
            latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens, self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

            latent_state = latent_state[:, :-1, :]
            buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
            act_tokens = rearrange(buffer_action, 'b l -> b l 1')

            # 选择每个样本的最后一步
            last_steps = act_tokens[:, -1:, :]  # 这将选择最后一列并保持维度不变, 最后一步的target policy/value本身就没有用到
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
        outputs_wm, latent_state = self.reset_from_initial_observations(obs_act_dict)  # root节点也有context
        return outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

    """
    假设env_num=8  
    8个环境的kv_cache单独存储与寻找,都存储在一个dict中,在recurrent_inference时, 
    由于不同环境找到的kv_cache的size不同,先根据最大size对kv_cache在前部补零,然后组成batch_size的kv_cache
    其内部也是通过batch执行transformer forward的推理 
    """

    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history):
        # 一般来讲,在一次 MCTS search中,我们需要维护H长度的context来使用transformer进行推理。
        # 由于在一次search里面。agent最多访问sim个不同的节点,因此我们只需维护一个 {(state:kv_cache)}的列表。
        # 但如果假设环境是MDP的话,然后根据当前的 latest_state s_t 在这个列表中查找即可
        # TODO: 但如果假设环境是非MDP的话,需要维护一个 {(rootstate_action_history:kv_cache)}的列表?

        latest_state, action = state_action_history[-1]

        # 假设 latest_state 是新的 latent_state,包含 ready_env_num 个环境的信息
        ready_env_num = latest_state.shape[0]
        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.retrieve_or_generate_kvcache(latest_state, ready_env_num)  

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
            print('largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
            print('largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
            length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.total_query_count
            print('largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
            print('largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)

        # 输入self.keys_values_wm_list,输出为self.keys_values_wm
        self.trim_and_pad_kv_cache()

        for k in range(2):  # 假设每次只有一个动作token。
            # action_token obs_token, ..., obs_token  1+1
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            outputs_wm = self.forward(obs_embeddings_or_act_tokens, past_keys_values=self.keys_values_wm, kvcache_independent=False)

            if k == 0:
                # 如果k==0,token是action_token,outputs_wm.logits_rewards 是有值的
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                # 一共产生16个obs_token,每次产生一个
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        # 删除旧的self.latent_state以释放内存
        del self.latent_state
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(self.latent_state, is_init_infer=False) # TODO


        return outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value

    def trim_and_pad_kv_cache(self):
        """
        This method trims and pads the key and value caches of the attention mechanism
        to a consistent size across all items in the batch, determined by the smallest cache size.
        """

        # 找到所有key-value尺寸中的最小尺寸,用于填充/修剪
        min_size = min(self.keys_values_wm_size_list)

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
                # 计算需要修剪的尺寸差异
                trim_size = effective_size - min_size if effective_size > min_size else 0

                # 如果需要修剪,从缓存的开头移除'trim_size'
                if trim_size > 0:
                    k_cache_trimmed = k_cache[:, :, trim_size:, :]
                    v_cache_trimmed = v_cache[:, :, trim_size:, :]
                    # 在第三维上用零填充修剪后的缓存
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, trim_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, trim_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                # 将处理后的缓存添加到列表中
                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # 沿新维度堆叠缓存,并用squeeze()移除额外维度
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # 修剪和填充后,将缓存尺寸更新为最小尺寸
            self.keys_values_wm._keys_values[layer]._k_cache._size = min_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = min_size

    def update_cache_context(self, latent_state, is_init_infer=True):
        for i in range(latent_state.size(0)):  # 遍历每个环境
            state_single_env = latent_state[i]  # 获取单个环境的潜在状态
            quantized_state = state_single_env.detach().cpu().numpy()  # 分离并将状态移至CPU
            cache_key = quantize_state(quantized_state)  # 量化状态并将其哈希值计算为缓存键

            # 从全局缓存复制keys和values到单个环境缓存
            for layer in range(self.num_layers):
                # if self.keys_values_wm._keys_values[layer]._k_cache._size < self.config.max_tokens - 1:
                if self.keys_values_wm._keys_values[layer]._k_cache._size < self.context_length-1:  # 固定只保留最近5个timestep的context
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(0)  # Shape torch.Size([2, 100, 512])
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm._keys_values[layer]._k_cache._size
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm._keys_values[layer]._v_cache._size
                # elif self.keys_values_wm._keys_values[layer]._k_cache._size == self.config.max_tokens - 1:
                else:
                    # 裁剪和填充逻辑
                    # 假设cache的维度是 [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]
                    
                    # 移除前2步并保留最近的self.context_length-3步
                    # k_cache_trimmed = k_cache_current[:, 2:self.config.max_tokens - 1, :]
                    # v_cache_trimmed = v_cache_current[:, 2:self.config.max_tokens - 1, :]
                    k_cache_trimmed = k_cache_current[:, 2:self.context_length-1, :]
                    v_cache_trimmed = v_cache_current[:, 2:self.context_length-1, :]
                    
                    # 沿第3维填充后2步
                    padding_size = (0, 0, 0, 3)  # F.pad的参数(0, 0, 0, 2)指定了在每个维度上的填充量。这些参数是按(左, 右, 上, 下)的顺序给出的,对于三维张量来说,分别对应于(维度2左侧, 维度2右侧, 维度1左侧, 维度1右侧)的填充。
                    k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                    v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                    # 更新单环境cache
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    
                    # self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.config.max_tokens - 3
                    # self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.config.max_tokens - 3
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.context_length-3
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.context_length-3

            # TODO: memory_env;每次都存最新的
            if is_init_infer:
                # init_infer: laten_state是encoder编码得到的，没有误差
                # 比较并存储较大的缓存
                if cache_key in self.past_keys_values_cache_init_infer:
                    existing_kvcache = self.past_keys_values_cache_init_infer[cache_key]
                    # 检查现有缓存和新缓存之间是否存在大小差异
                    # if self.keys_values_wm_single_env.size > existing_kvcache.size and self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                    if self.keys_values_wm_single_env.size > existing_kvcache.size and self.keys_values_wm_single_env.size < self.context_length-1:
                        # 仅在大小小于 max_tokens - 1 时存储,以避免重置
                        self.past_keys_values_cache_init_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
                # elif self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                elif self.keys_values_wm_single_env.size < self.context_length-1:
                    # 仅在大小小于 max_tokens - 1 时存储,以避免重置
                    self.past_keys_values_cache_init_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
            else:
                # recurrent_infer: laten_state是预测的，会有误差
                # 比较并存储较大的缓存
                if cache_key in self.past_keys_values_cache_recurrent_infer:
                    existing_kvcache = self.past_keys_values_cache_recurrent_infer[cache_key]
                    # 检查现有缓存和新缓存之间是否存在大小差异
                    # if self.keys_values_wm_single_env.size > existing_kvcache.size and self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                    if self.keys_values_wm_single_env.size > existing_kvcache.size and self.keys_values_wm_single_env.size < self.context_length-1:
                        # 仅在大小小于 max_tokens - 1 时存储,以避免重置
                        self.past_keys_values_cache_recurrent_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
                # elif self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                elif self.keys_values_wm_single_env.size < self.context_length-1:
                    # 仅在大小小于 max_tokens - 1 时存储,以避免重置
                    self.past_keys_values_cache_recurrent_infer[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))

    def retrieve_or_generate_kvcache(self, latent_state, ready_env_num):
        """
        This method iterates over the environments, retrieves a matching cache if available,
        or generates a new one otherwise. It updates the lists with the keys_values caches and their sizes.
        """
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # 获取单个环境的潜在状态
            cache_key = quantize_state(state_single_env)  # 使用量化后的状态计算哈希值
            # 如果存在,检索缓存值
            # 先在self.past_keys_values_cache_init_infer中寻找
            matched_value = self.past_keys_values_cache_init_infer.get(cache_key)
            # 再在self.past_keys_values_cache中寻找
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
                self.forward({'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)}, past_keys_values=self.keys_values_wm_single_env)
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

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

    def compute_loss(self, batch, target_tokenizer: Tokenizer=None, **kwargs: Any) -> LossWithIntermediateLosses:
        # 将观察编码为潜在状态表示
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'], should_preprocess=False)
        
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

            # 计算重建损失和感知损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
                                                                   reconstructed_images)  # NOTE: for stack=1
            # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4
            perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4

        # 动作tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # 前向传播,得到预测的观察、奖励和策略等
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})
        
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
        #     loss_obs=loss_obs,
        #     loss_rewards=loss_rewards,
        #     loss_value=loss_value,
        #     loss_policy=loss_policy,
        #     latent_recon_loss=latent_recon_loss,
        #     perceptual_loss=perceptual_loss,
        #     orig_policy_loss=orig_policy_loss,
        #     policy_entropy=policy_entropy
        # )

        # 计算时间步
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # 计算每个时间步的折扣系数
        discounts = self.gamma ** timesteps

        # 将折扣系数应用到每个损失项
        # discounted_latent_recon_loss = (latent_recon_loss.view(-1, batch['actions'].shape[1]) * discounts).mean()
        # discounted_perceptual_loss = (perceptual_loss.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

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
            policy_entropy=discounted_policy_entropy
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

    def __repr__(self) -> str:
        return "world_model"