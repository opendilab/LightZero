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
import hashlib
# def quantize_state(state, num_buckets=1000):
def quantize_state(state, num_buckets=15):
# def quantize_state(state, num_buckets=10):
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
        self.num_layers = config.num_layers


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

        self.act_embedding_table = nn.Embedding(act_vocab_size, config.embed_dim)

        self.obs_per_embdding_dim = config.embed_dim # 16*64=1024

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
                nn.Linear(config.embed_dim, obs_per_embdding_dim)
            )
        )
        
        ###### TODO: 2层的性能, LeakyReLU->GELU ######
        self.head_observations = Head( # TODO
            max_blocks=config.max_blocks,
            block_mask=all_but_last_latent_state_pattern, # 1,...,0,1 # https://github.com/eloialonso/iris/issues/19
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.GELU(),
                nn.Linear(config.embed_dim, self.obs_per_embdding_dim),
                # nn.Tanh(), # TODO
                nn.Sigmoid(),  # 这里添加Sigmoid函数 TODO
            )
        )
        self.head_policy = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,  # TODO: value_policy_tokens_pattern # [0,...,1,0]
            head_module=nn.Sequential( # （8, 5, 128）
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.GELU(),
                nn.Linear(config.embed_dim, self.action_shape)  # TODO(pu); action shape
            )
        )
        self.head_value = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                # nn.LeakyReLU(negative_slope=0.01), # TODO: 2
                nn.GELU(),
                nn.Linear(config.embed_dim, self.support_size)  # TODO(pu): action shape
            )
        )

        ###### TODO: 单层的性能 ######
        # self.head_observations = Head( # TODO
        #     max_blocks=config.max_blocks,
        #     block_mask=all_but_last_latent_state_pattern, # 1,...,0,1 # https://github.com/eloialonso/iris/issues/19
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, self.obs_per_embdding_dim),
        #         nn.Sigmoid(),  # 这里添加Sigmoid函数 TODO
        #     )
        # )
        # self.head_policy = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=value_policy_tokens_pattern,  # TODO: value_policy_tokens_pattern # [0,...,1,0]
        #     head_module=nn.Sequential( # （8, 5, 128）
        #         nn.Linear(config.embed_dim, self.action_shape)  # TODO(pu); action shape
        #     )
        # )
        # self.head_value = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=value_policy_tokens_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, self.support_size)  # TODO(pu): action shape
        #     )
        # )

        self.apply(init_weights)

        last_linear_layer_init_zero = True  # TODO: is beneficial for convergence speed.
        if last_linear_layer_init_zero:
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
                    # layer.weight.data.fill_(0.5) # TODO:bug
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break


        import collections
        self.past_keys_values_cache = collections.OrderedDict()
        self.past_policy_value_cache = collections.OrderedDict()

        # TODO: Transformer更新后应该清除缓存
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num, max_tokens=self.config.max_tokens)

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []


        if self.num_observations_tokens==16:  # k=16
            self.projection_input_dim = 128
        elif self.num_observations_tokens==1:  # K=1
            self.projection_input_dim =  self.obs_per_embdding_dim# for atari #TODO
            # self.projection_input_dim = 256 # for cartpole


        # self.proj_hid = 1024
        # self.proj_out = 1024
        # self.pred_hid = 512
        # self.pred_out = 1024
        # activation = nn.ReLU()
        # self.projection = nn.Sequential(
        #         nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
        #         nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
        #         nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
        #     )
        # self.prediction_head = nn.Sequential(
        #     nn.Linear(self.proj_out, self.pred_hid),
        #     nn.BatchNorm1d(self.pred_hid),
        #     activation,
        #     nn.Linear(self.pred_hid, self.pred_out),
        # )
        self.hit_count = 0
        self.total_query_count = 0
        self.length3_context_cnt = 0
        self.length2_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)


    def __repr__(self) -> str:
        return "world_model"

    # @profile
    def forward(self, obs_embeddings_or_act_tokens, past_keys_values: Optional[KeysValues] = None,
            is_root=False, kvcache_independent=False) -> WorldModelOutput:
        
        if kvcache_independent:
            prev_steps = 0 if past_keys_values is None else [past_kv.size for past_kv in past_keys_values]
            prev_steps = torch.tensor(prev_steps, device=self.device)
            # 我们需要为每个样本生成一个序列的步骤indices，然后获取它们的位置嵌入
            # 首先扩展prev_steps至(num_steps, batch_size)，这里num_steps=1
            # prev_steps = prev_steps.unsqueeze(0)

        else:
            prev_steps = 0 if past_keys_values is None else past_keys_values.size
            # print(f'prev_steps:{prev_steps}')

        if 'obs_embeddings' in obs_embeddings_or_act_tokens.keys():
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            if len(obs_embeddings.shape)==2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)  # (B, T, E)
            if kvcache_independent:
                # 生成每个样本的步骤indices
                # steps_indices = prev_steps + torch.arange(num_steps, device=obs_embeddings.device).unsqueeze(1)
                steps_indices = prev_steps + torch.arange(num_steps, device=obs_embeddings.device)

                # 步骤indices需要被reshape成一维，以便于embedding操作
                # steps_indices = steps_indices.view(-1)
                # 获取位置嵌入
                position_embeddings = self.pos_emb(steps_indices)
                # 由于我们要将它们加到obs_embeddings上，需要将位置嵌入reshape回(batch_size, num_steps, embedding_dim)
                position_embeddings = position_embeddings.view(-1, num_steps, position_embeddings.shape[-1])
                # 现在我们可以将位置嵌入加到obs_embeddings上了
                sequences = obs_embeddings + position_embeddings
            else:
                sequences = obs_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device))
        elif 'act_tokens' in obs_embeddings_or_act_tokens.keys():
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            if len(act_tokens.shape)==3:
                act_tokens = act_tokens.squeeze(1)
            num_steps = act_tokens.size(1)  # (B, T)
            act_embeddings = self.act_embedding_table(act_tokens)

            if kvcache_independent:
                # 生成每个样本的步骤indices
                # steps_indices = prev_steps + torch.arange(num_steps, device=act_embeddings.device).unsqueeze(1)
                steps_indices = prev_steps + torch.arange(num_steps, device=act_embeddings.device)
                # 步骤indices需要被reshape成一维，以便于embedding操作
                # steps_indices = steps_indices.view(-1)
                # 获取位置嵌入
                position_embeddings = self.pos_emb(steps_indices)
                # 由于我们要将它们加到obs_embeddings上，需要将位置嵌入reshape回(batch_size, num_steps, embedding_dim)
                position_embeddings = position_embeddings.view(-1, num_steps, position_embeddings.shape[-1])
                # 现在我们可以将位置嵌入加到obs_embeddings上了
                sequences = act_embeddings + position_embeddings
            else:
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
        if kvcache_independent:
            x = []
            for k, past_kv in enumerate(past_keys_values):
                x.append(self.transformer(sequences[k].unsqueeze(0), past_kv))
            x =  torch.cat(x, dim=0)

            # TODO: 在collect时，是一步一步的 obs act 传入的
            # prev_steps = prev_steps//1

        else:
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
    # @profile
    def reset_from_initial_observations_v2(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(obs_act_dict, dict):
            observations = obs_act_dict['obs']
            buffer_action = obs_act_dict['action']
            current_obs = obs_act_dict['current_obs']
        else:
            observations = obs_act_dict
            buffer_action = None
            current_obs = None
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations, should_preprocess=True) # (B, C, H, W) -> (B, K, E)

        if current_obs is not None:
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(current_obs, should_preprocess=True) # (B, C, H, W) -> (B, K, E)
            self.latent_state = current_obs_embeddings
            outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer_v2(obs_embeddings, buffer_action, current_obs_embeddings)
        else:
            self.latent_state = obs_embeddings
            outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer_v2(obs_embeddings, buffer_action, None)


        return outputs_wm, self.latent_state

    @torch.no_grad()
    # @profile
    def refresh_keys_values_with_initial_latent_state_for_init_infer_v2(self, latent_state: torch.LongTensor, buffer_action=None, current_obs_embeddings=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        if n <= self.env_num:
            if buffer_action is None:
                # MCTS root节点: 需要准确的估计 value, policy_logits, 或许需要结合context的kv_cache进行更准确的估计，而不是当前的从0开始推理
                self.keys_values_wm = self.transformer.generate_empty_keys_values(n, max_tokens=self.config.max_tokens)
                outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False, kvcache_independent=False)
                self.keys_values_wm_size_list = [1 for i in range(n)]

                # 复制单个环境对应的 keys_values_wm 并存储
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)
                for i in range(latent_state.size(0)):  # 遍历每个环境
                    state_single_env = latent_state[i]   # 获取单个环境的 latent state
                    # cache_key = hash(state_single_env.detach().cpu().numpy())  # 计算哈希值
                    quantized_state = state_single_env.detach().cpu().numpy()
                    cache_key = quantize_state(quantized_state)  # 使用量化后的状态计算哈希值
                    for layer in range(self.num_layers):
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(0) # shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm._keys_values[layer]._k_cache._size 
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm._keys_values[layer]._v_cache._size
                        # keys_values_wm_single_env[layer].update(self.keys_values_wm[layer]._k_cache._cache[i].unsqueeze(0), self.keys_values_wm[layer]._v_cache._cache[i].unsqueeze(0))
                    self.root_total_query_cnt += 1
                    if cache_key not in self.past_keys_values_cache:
                        self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
                    else:
                        self.root_hit_cnt += 1
                        root_hit_ratio = self.root_hit_cnt / self.root_total_query_cnt
                        print('root_total_query_cnt:', self.root_total_query_cnt)
                        print(f'root_hit_ratio:{root_hit_ratio}')
                        print(f'root_hit_cnt:{self.root_hit_cnt}')
                        print(f'root_hit find size {self.past_keys_values_cache[cache_key].size}')
                        if self.past_keys_values_cache[cache_key].size>1:
                            print(f'=='*20)
                            print(f'NOTE: root_hit find size > 1')
                            print(f'=='*20)
            elif current_obs_embeddings is not None:

                if max(buffer_action) == -1:
                    # first step in one episode
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=8, max_tokens=self.config.max_tokens)
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm, is_root=False)

                    # 复制单个环境对应的 keys_values_wm 并存储
                    self.update_cache(current_obs_embeddings)
                else:
                    # self.retrieve_or_generate_kvcache(latent_state, current_obs.shape[0])
                    # 假设 latest_state 是新的 latent_state，包含 ready_env_num 个环境的信息
                    # ready_env_num = latent_state.shape[0]
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []
                    for i in range(ready_env_num):
                        state_single_env = latent_state[i]  # 获取单个环境的 latent state
                        quantized_state = state_single_env.detach().cpu().numpy()
                        cache_key = quantize_state(quantized_state)  # 使用量化后的状态计算哈希值
                        matched_value = self.past_keys_values_cache.get(cache_key)  # 检索缓存值
                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # 如果找到匹配的值，将其添加到列表中
                            self.root_hit_cnt += 1
                            if self.root_total_query_cnt>0 and self.root_total_query_cnt%1000==0:
                                root_hit_ratio = self.root_hit_cnt / self.root_total_query_cnt
                                print('root_total_query_cnt:', self.root_total_query_cnt)
                                print(f'root_hit_ratio:{root_hit_ratio}')
                                print(f'root_hit find size {self.past_keys_values_cache[cache_key].size}')
                                if self.past_keys_values_cache[cache_key].size>=7:
                                    print(f'=='*20)
                                    print(f'NOTE: root_hit find size >= 7')
                                    print(f'=='*20)
                            # 这里需要deepcopy因为在transformer的forward中会原地修改matched_value
                            self.keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, 'cuda')))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # use zero reset
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)
                            # outputs_wm = self.forward({'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)}, past_keys_values=self.keys_values_wm_single_env, is_root=False)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)}, past_keys_values=self.keys_values_wm_single_env, is_root=False)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)
                    
                    # print(f'NOTE: root {self.keys_values_wm_size_list}')
                    # print(f'=='*20)
                    # 输入self.keys_values_wm_list，输出为self.keys_values_wm
                    self.trim_and_pad_kv_cache()

                    buffer_action = buffer_action[:ready_env_num]
                    buffer_action = torch.from_numpy(np.array(buffer_action)).to(latent_state.device)
                    act_tokens = buffer_action.unsqueeze(-1)
                    # outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (latent_state, act_tokens)}, past_keys_values=self.keys_values_wm, is_root=False)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm, is_root=False)
                    
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, past_keys_values=self.keys_values_wm, is_root=False)

                    # 复制单个环境对应的 keys_values_wm 并存储
                    self.update_cache(current_obs_embeddings)

        elif n == int(256): 
            # TODO: n=256 means train tokenizer, 不需要计算target value
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            # print('init inference: not find matched_value! reset!')
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)
        elif n > self.env_num and n != int(256) and buffer_action is not None: 
            # train时计算target value
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

        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations, should_preprocess=True) # (B, C, H, W) -> (B, K, E)
        outputs_wm = self.refresh_keys_values_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action)
        self.latent_state = obs_embeddings

        return outputs_wm, self.latent_state


    @torch.no_grad()
    # @profile
    def refresh_keys_values_with_initial_latent_state_for_init_infer(self, latent_state: torch.LongTensor, buffer_action=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        if n <= self.env_num:
            # MCTS root节点: 需要准确的估计 value, policy_logits 或许需要结合context的kv_cache进行更准确的估计，而不是当前的从0开始推理
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
        elif n == int(256): 
            # TODO: n=256 means train tokenizer, 不需要计算target value
            self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
            # print('init inference: not find matched_value! reset!')
            outputs_wm = self.forward({'obs_embeddings': latent_state}, past_keys_values=self.keys_values_wm, is_root=False)  # Note: is_root=False
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
    def refresh_keys_values_with_initial_latent_state(self, latent_state: torch.LongTensor, reset_indices=None) -> torch.FloatTensor:
        n, num_observations_tokens, _ = latent_state.shape
        assert num_observations_tokens == self.num_observations_tokens
        # self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
        if reset_indices is None:
            self.keys_values_wm_list = [self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens) for i in range(n)]
        else:
            for i in reset_indices:
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)
                outputs_wm = self.forward({'obs_embeddings': latent_state[i].unsqueeze(0).to(self.device)}, past_keys_values=self.keys_values_wm_single_env, is_root=False, kvcache_independent=False)
                self.keys_values_wm_list[i] = self.keys_values_wm_single_env
                self.keys_values_wm_size_list[i] = 1
        return None

    @torch.no_grad()
    # @profile
    def forward_initial_inference(self, obs_act_dict: torch.LongTensor):
        if isinstance(obs_act_dict, dict):
            obs = obs_act_dict['obs']
        else:
            obs = obs_act_dict
        outputs_wm, latent_state = self.reset_from_initial_observations_v2(obs_act_dict) # root节点也有context

        return outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

    """
    假设env_num=8
    8个环境的kv_cache单独存储与寻找，都存储在一个dict中，在recurrent_inference时，
    由于不同环境找到的kv_cache的size不同，先根据最大size对kv_cache在前部补零，然后组成batch_size的kv_cache
    其内部也是通过batch执行transformer forward的推理
    """

    @torch.no_grad()
    # @profile
    def forward_recurrent_inference(self, state_action_history):
        # 一般来讲，在一次 MCTS search中，我们需要维护H长度的context来使用transformer进行推理。
        # 由于在一次search里面。agent最多访问sim个不同的节点，因此我们只需维护一个 {(state:kv_cache)}的列表。
        # 但如果假设环境是MDP的话，然后根据当前的 latest_state s_t 在这个列表中查找即可
        # TODO: 但如果假设环境是非MDP的话，需要维护一个 {(rootstate_action_history:kv_cache)}的列表？

        # latest_state = state_action_history[-1][0]
        # action = state_action_history[-1][-1]

        latest_state, action = state_action_history[-1]

        # 假设 latest_state 是新的 latent_state，包含 ready_env_num 个环境的信息
        ready_env_num = latest_state.shape[0]
        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.retrieve_or_generate_kvcache(latest_state, ready_env_num)

        latent_state_list = []
        # output_sequence_list, latent_state_list = [], []

        # reset_indices = [index for index, value in enumerate(self.keys_values_wm_size_list) if value + num_passes > self.config.max_tokens]
        # self.refresh_keys_values_with_initial_latent_state(torch.tensor(latest_state, dtype=torch.float32).to(self.device), reset_indices)

        # token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        # token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        token = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device) 


        # print(self.keys_values_wm_size_list)
        # 获取self.keys_values_wm_size_list的最小值min_size
        min_size = min(self.keys_values_wm_size_list)
        if min_size >= self.config.max_tokens - 5:
            self.length3_context_cnt += len(self.keys_values_wm_size_list)
        if min_size >= 3:
            self.length2_context_cnt += len(self.keys_values_wm_size_list)
        # if self.total_query_count>0 and self.total_query_count%1==0:
        if self.total_query_count>0 and self.total_query_count%10000==0:
            self.hit_freq = self.hit_count/(self.total_query_count)
            # print('hit_freq:', self.hit_freq)
            # print('hit_count:', self.hit_count)
            print('total_query_count:', self.total_query_count)
            # 如果总查询次数大于0，计算并打印cnt的比率
            length3_context_cnt_ratio = self.length3_context_cnt / self.total_query_count
            print('>=3 node context_cnt:', self.length3_context_cnt)
            print('>=3 node context_cnt_ratio:', length3_context_cnt_ratio)
            length2_context_cnt_ratio = self.length2_context_cnt / self.total_query_count
            print('>=2 node context_cnt_ratio:', length2_context_cnt_ratio)
            print('>=2 node context_cnt:', self.length2_context_cnt)
            # print(self.keys_values_wm_size_list)

        # 输入self.keys_values_wm_list，输出为self.keys_values_wm
        self.trim_and_pad_kv_cache()
        # print(f'NOTE: in search node {self.keys_values_wm_size_list}')

        # for k in range(num_passes):  # assumption that there is only one action token.
        # num_passes = 1 + self.num_observations_tokens
        for k in range(2):  # assumption that there is only one action token.
            # action_token obs_token, ..., obs_token  1+1
            if k==0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            outputs_wm = self.forward(obs_embeddings_or_act_tokens, past_keys_values=self.keys_values_wm, is_root=False, kvcache_independent=False)
            # if k==0, action_token self.head_observations 1,...,0,1
            # output_sequence_list.append(outputs_wm.output_sequence)

            if k == 0:
                # if k==0, token is action_token  outputs_wm.logits_rewards 是有值的
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                # 一共产生16个obs_token，每次产生一个
                # TODO： sample or argmax
                # token = Categorical(logits=outputs_wm.logits_observations).sample()
                # Use argmax to select the most likely token
                # token = outputs_wm.logits_observations.argmax(-1, keepdim=True)
                token = outputs_wm.logits_observations
                # if len(token.shape) != 2:
                #     token = token.squeeze(-1)  # Ensure the token tensor shape is (B, 1)
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        # output_sequence = torch.cat(output_sequence_list, dim=1)  # (B, 1 + K, E)
        # Before updating self.latent_state, delete the old one to free memory
        del self.latent_state
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache(self.latent_state)
        # TODO: 在计算结束后，是否需要更新最新的缓存. 是否需要deepcopy

        # if len(self.past_keys_values_cache) > self.max_cache_size:
        #     # TODO: lru_cache
        #     _, popped_kv_cache = self.past_keys_values_cache.popitem(last=False)
        #     del popped_kv_cache # 测试不要这一行的显存情况

        # Example usage:
        # Assuming `past_keys_values_cache` is a populated instance of `KeysValues`
        # and `num_layers` is the number of transformer layers
        # cuda_memory_gb = self.calculate_cuda_memory_gb(self.past_keys_values_cache, num_layers=2)
        # print(f'len(self.past_keys_values_cache): {len(self.past_keys_values_cache)}, Memory used by past_keys_values_cache: {cuda_memory_gb:.2f} GB')

        return outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value

    def trim_and_pad_kv_cache(self):
        """
        This method trims and pads the key and value caches of the attention mechanism 
        to a consistent size across all items in the batch, determined by the smallest cache size.
        """

        # Find the minimum size across all key-value sizes for padding/trimming
        min_size = min(self.keys_values_wm_size_list)

        # Iterate over each layer of the transformer
        for layer in range(self.num_layers):
            # Initialize lists to hold the trimmed and padded k and v caches
            kv_cache_k_list = []
            kv_cache_v_list = []

            # Enumerate over the key-value pairs list
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                # Retrieve the current layer's key and value caches
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                # Get the effective size of the current cache
                effective_size = self.keys_values_wm_size_list[idx]
                # Calculate the size difference to trim
                trim_size = effective_size - min_size if effective_size > min_size else 0

                # If trimming is needed, remove 'trim_size' from the beginning of the cache
                if trim_size > 0:
                    k_cache_trimmed = k_cache[:, :, trim_size:, :]
                    v_cache_trimmed = v_cache[:, :, trim_size:, :]
                    # Pad the trimmed cache with zeros on the third dimension
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, trim_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, trim_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                # Add the processed caches to the lists
                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # Stack the caches along the new dimension, and remove the extra dimension with squeeze()
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # Update the cache size to the minimum size after trimming and padding
            self.keys_values_wm._keys_values[layer]._k_cache._size = min_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = min_size

    def update_cache(self, latent_state):
        for i in range(latent_state.size(0)):  # Iterate over each environment
            state_single_env = latent_state[i]  # Get the latent state for a single environment
            quantized_state = state_single_env.detach().cpu().numpy()  # Detach and move the state to CPU
            cache_key = quantize_state(quantized_state)  # Quantize state and compute its hash value as cache key

            # Copy keys and values from the global cache to a single environment cache
            for layer in range(self.num_layers):
                if self.keys_values_wm._keys_values[layer]._k_cache._size < self.config.max_tokens - 1:
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(0)  # Shape torch.Size([2, 100, 512])
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm._keys_values[layer]._k_cache._size
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm._keys_values[layer]._v_cache._size
                elif self.keys_values_wm._keys_values[layer]._k_cache._size == self.config.max_tokens - 1:
                    # 裁剪和填充逻辑
                    # 假设cache的维度是 [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]
                    
                    # 移除前2步并保留最近的max_tokens - 3步
                    k_cache_trimmed = k_cache_current[:, 2:self.config.max_tokens - 1, :]
                    v_cache_trimmed = v_cache_current[:, 2:self.config.max_tokens - 1, :]
                    
                    # 沿第3维填充后2步
                    padding_size = (0, 0, 0, 3)  #F.pad的参数(0, 0, 0, 2)指定了在每个维度上的填充量。这些参数是按(左, 右, 上, 下)的顺序给出的，对于三维张量来说，分别对应于(维度2左侧, 维度2右侧, 维度1左侧, 维度1右侧)的填充。
                    k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                    v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                    # 更新单环境cache
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.config.max_tokens - 3
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.config.max_tokens - 3


            # Compare and store the larger cache
            if cache_key in self.past_keys_values_cache:
                existing_kvcache = self.past_keys_values_cache[cache_key]
                # Check if there is a size difference between existing cache and new cache
                if self.keys_values_wm_single_env.size > existing_kvcache.size and self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                    # Only store if size is less than max_tokens - 1 to avoid reset
                    self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
            elif self.keys_values_wm_single_env.size < self.config.max_tokens - 1:
                # Only store if size is less than max_tokens - 1 to avoid reset
                self.past_keys_values_cache[cache_key] = copy.deepcopy(self.to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))

    def retrieve_or_generate_kvcache(self, latent_state, ready_env_num):
        """
        This method iterates over the environments, retrieves a matching cache if available,
        or generates a new one otherwise. It updates the lists with the keys_values caches and their sizes.
        """
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # Get the latent state for a single environment
            cache_key = quantize_state(state_single_env)  # Compute the hash value using the quantized state
            # Retrieve the cached value if it exists
            matched_value = self.past_keys_values_cache.get(cache_key)
            if matched_value is not None:
                # If a matching value is found, add it to the list
                self.hit_count += 1
                # Deepcopy is needed because the transformer's forward may modify matched_value in place
                self.keys_values_wm_list.append(copy.deepcopy(self.to_device_for_kvcache(matched_value, self.device)))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # If no match is found, use a zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.config.max_tokens)
                self.forward({'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)}, past_keys_values=self.keys_values_wm_single_env, is_root=False)
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
        # perceptual_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # for stack=4 gray obs

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