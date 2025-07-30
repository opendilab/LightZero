
"""
Modified from https://github.com/karpathy/nanoGPT

在原 transformer.py 基础上增加 LoRA 微调相关代码，
并通过传入配置参数控制 LoRA 微调的模块（默认是 attention 中的 k, q, v, proj 和 feed_forward）
保持原有代码的可扩展性。
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from ding.torch_utils.network import GRUGatingUnit
from einops import rearrange
from torch.nn import functional as F

from .kv_caching import KeysValues

from line_profiler import line_profiler
from lzero.model.common import SimNorm
import logging

# class LearnableScale(nn.Module):
#     """
#     可学习且有界的标量参数:
#       s = s_max * sigmoid(ŝ)         (0, s_max)
#     """
#     def __init__(self, init=1.0, s_max=1.2):
#         super().__init__()
#         # 反推初始值
#         inv_sig = math.log(init / (s_max - init + 1e-9))
#         self.logit = nn.Parameter(torch.tensor(inv_sig))
#         self.logit.requires_grad = True # TODO
#         self.s_max = s_max

#     def forward(self):
#         return self.s_max * torch.sigmoid(self.logit)

class LearnableScale(nn.Module):
    """
    一个被约束在特定范围内的可学习标量参数。
    
    s = offset + scale * tanh(ŝ)
    
    这将无界的 logit ŝ 映射到 (offset - scale, offset + scale) 范围内。
    使用 tanh 有时比 sigmoid 能提供更稳定的梯度。
    
    例如: 要获得 (0.8, 1.2) 的范围，使用 init=1.0, s_range=0.2。
    """
    def __init__(self, init: float = 1.0, s_range: float = 0.2):
        super().__init__()
        assert s_range > 0, "缩放范围必须为正。"
        self.offset = init
        self.scale = s_range

        # 将 logit 初始化为 0，使初始输出恰好为 `init`。
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.logit.requires_grad = False  # TODO 初始时冻结，由 CurriculumController 激活
        # self.logit.requires_grad = True # TODO


    def forward(self) -> torch.Tensor:
        return self.offset + self.scale * torch.tanh(self.logit)
    
##############################################
# CurriculumLoRALinear 实现
##############################################

class CurriculumLoRALinear(nn.Module):
    """
    CurriculumLoRALinear 对标准的线性映射进行了扩展：
    
    - 内部保存了基础的 W 和 bias 参数（基础 transformer 部分）。
    - 同时初始化了多个 LoRA adapter 参数（数量 = curriculum_stage_num - 1）。
    - 前向计算：
        如果 curriculum_stage == 0：
            输出 = F.linear(x, W, bias)
        如果 curriculum_stage >= 1：
            输出 = 基础输出 + sum_{i=0}^{curriculum_stage-1} scaling * adapter_i(x)
             其中，仅当前阶段 adapter（即 index == curriculum_stage - 1）参与更新，其它 adapter 使用 detach() 保证前向贡献但不传递梯度。
    
    注意：
        - 外部在阶段切换时调用 set_curriculum_stage(stage) 来更新状态。
        - 每次调用时，通过 log 信息展示当前模块的维度信息以及冻结/激活状态。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
                 curriculum_stage_num: int = 1, lora_scale_init=1.0):
        """
        如果 curriculum_stage_num > 1，则初始化 (curriculum_stage_num - 1) 个 LoRA adapter。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.curriculum_stage_num = curriculum_stage_num  # 总阶段数
        self.curriculum_stage = 0  # 初始阶段 0

        # 初始化基础权重（基础 transformer 部分），默认参与训练
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # 初始化 LoRA adapter，只有在 r > 0 且 curriculum_stage_num > 1 时才存在
        self.adapters = nn.ModuleList()
        # self.adapter_scales = nn.ParameterList()
        self.adapter_scales = nn.ModuleList()

        if r > 0 and (curriculum_stage_num - 1) > 0:
            for i in range(curriculum_stage_num - 1):
                adapter = nn.ParameterDict({
                    'lora_A': nn.Parameter(torch.randn(r, in_features) * 0.01),
                    'lora_B': nn.Parameter(torch.zeros(out_features, r))
                })
                self.adapters.append(adapter)

                self.adapter_scales.append(LearnableScale(lora_scale_init, s_max=1.2))
                
                # self.adapter_scales.append(  #  ← 新增
                #     nn.Parameter(torch.tensor(lora_scale_init, dtype=torch.float32))
                # )

            # --- CurriculumLoRALinear.__init__() ------------
            # for p in self.adapter_scales:
            #     p.requires_grad = True   # 统一设 True，避免遗漏
        else:
            self.adapters = None

        # 初始时：stage==0，基础层参与更新，adapter 均冻结
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
        if self.adapters is not None:
            for adapter in self.adapters:
                adapter['lora_A'].requires_grad = False
                adapter['lora_B'].requires_grad = False

    def set_curriculum_stage(self, stage: int):
        """
        设置当前阶段 stage，取值范围 [0, curriculum_stage_num-1]，并同步冻结/激活各部分参数。
        
        - stage == 0：基础层参与前向和更新，所有 adapter 均冻结；
        - stage >= 1：冻结基础层（只用于前向），仅当前 adapter（index == stage - 1）参与更新，
          前面 adapter 虽然前向贡献，但通过 detach() 不传导梯度。
          
        同时将 log 出模块信息和状态变化。
        """
        assert 0 <= stage < self.curriculum_stage_num, f"stage 必须在 [0, {self.curriculum_stage_num-1}] 范围内"
        self.curriculum_stage = stage

        # 输出 log 信息，展示当前模块（可结合 in_features, out_features 标识）
        module_id = f"({self.in_features}x{self.out_features})"
        if stage == 0:
            self.weight.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = True
            if self.adapters is not None:
                for idx, adapter in enumerate(self.adapters):
                    adapter['lora_A'].requires_grad = False
                    adapter['lora_B'].requires_grad = False
                    # self.adapter_scales[idx].requires_grad = True   #  ← 新增
            logging.info(f"[CurriculumLoRALinear {module_id}] Stage 0: 基础层可训练，所有 adapter 均冻结。")
            logging.info(f"[self.adapter_scales:] {self.adapter_scales}")
            logging.info(f"self.adapter_scales[0].item(): {self.adapter_scales[0]().item()}")

        else:
            # 阶段大于 0，冻结基础层
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            for idx, adapter in enumerate(self.adapters):
                logging.info(f"[self.adapter_scales:] {self.adapter_scales}")
                logging.info(f"self.adapter_scales[0].item(): {self.adapter_scales[0]().item()}")

                if idx == stage - 1:
                    adapter['lora_A'].requires_grad = True
                    adapter['lora_B'].requires_grad = True
                    logging.info(f"[CurriculumLoRALinear {module_id}] Stage {stage}: 激活 adapter {idx} (可训练)。")
                else:
                    adapter['lora_A'].requires_grad = False
                    adapter['lora_B'].requires_grad = False
                    logging.info(f"[CurriculumLoRALinear {module_id}] Stage {stage}: 冻结 adapter {idx} (仅前向不更新)。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline_out = F.linear(x, self.weight, self.bias)
        if self.curriculum_stage == 0 or self.adapters is None:
            return baseline_out

        adapter_out = 0
        # 对于前 curriculum_stage 个 adapter，只有最后一个正常反向传播，其它用 detach() 保证仅前向效果
        for idx in range(self.curriculum_stage):
            if idx >= len(self.adapters):
                break
            adapter = self.adapters[idx]
            out = F.linear(self.lora_dropout(x), adapter['lora_A'])
            out = F.linear(out, adapter['lora_B'])
            scale = self.adapter_scales[idx]() # TODO: 所有adapter  对应的scale都参与训练
            if idx == self.curriculum_stage - 1:
                adapter_out = adapter_out + self.scaling * out * scale  # 仅当前 adapter 参与更新
            else:
                adapter_out = adapter_out + self.scaling * out.detach() * scale
        return baseline_out + adapter_out

##############################################
# 修改 _maybe_wrap_linear 辅助函数
##############################################

def _maybe_wrap_linear(linear: nn.Linear, config, module_label: str) -> nn.Module:
    """
    辅助函数：当满足以下条件时，将传入的 nn.Linear 层替换为
    CurriculumLoRALinear：
      - config.lora_r > 0
      - module_label 在 config.lora_target_modules 中
      - 并且 config 中配置了 curriculum_stage_num > 1
    否则，若仅满足基础 LoRA 条件，则返回原有 LoRALinear；否则返回原始的线性层。
    """
    if config.lora_r > 0 and (module_label in config.lora_target_modules) and getattr(config, "curriculum_stage_num", 1) > 1:
        new_linear = CurriculumLoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            curriculum_stage_num=config.curriculum_stage_num,
             lora_scale_init        = config.lora_scale_init # todo
        )
        new_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.data.copy_(linear.bias.data)
        return new_linear
    # elif config.lora_r > 0 and (module_label in config.lora_target_modules):
    #     # 若不使用课程学习，则调用原有 LoRALinear 实现（未展示，此处假设其已定义）
    #     new_linear = LoRALinear(
    #         in_features=linear.in_features,
    #         out_features=linear.out_features,
    #         bias=(linear.bias is not None),
    #         r=config.lora_r,
    #         lora_alpha=config.lora_alpha,
    #         lora_dropout=config.lora_dropout
    #     )
    #     new_linear.weight.data.copy_(linear.weight.data)
    #     if linear.bias is not None:
    #         new_linear.bias.data.copy_(linear.bias.data)
    #     return new_linear
    else:
        return linear

##############################################
# 辅助函数：在 transformer 内部遍历所有 CurriculumLoRALinear 模块，并设置阶段
##############################################

def set_curriculum_stage_for_transformer(transformer: nn.Module, stage: int):
    """
    遍历 transformer 内的所有子模块，找到所有 CurriculumLoRALinear 的实例，
    并调用其 set_curriculum_stage(stage) 方法，同时记录 log 信息。
    """
    count = 0
    for module in transformer.modules():
        # logging.info(f"[Transformer] module {module}.")

        if isinstance(module, CurriculumLoRALinear):
            module.set_curriculum_stage(stage)
            count += 1
    logging.info(f"[Transformer] 共更新 {count} 个 CurriculumLoRALinear 模块为 curriculum stage {stage}.")


##############################################
# TransformerConfig 示例（增加 curriculum_stage_num）
##############################################
@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    # LoRA 参数：
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_target_modules: list = None

    # 课程学习相关参数：
    # curriculum_stage_num 表示总阶段数（例如 3 表示阶段 0,1,2）
    curriculum_stage_num: int = 5        # 1 + 可用的 LoRA adapter 数
    min_stage0_iters:   int = 10_000     # stage0 最少迭代
    max_stage_iters:    int = 20_000     # 每个 stage 最多迭代
    lora_scale_init:    float = 1.0      # 每个 adapter 的可学习初值

    # 其它配置项（略）
    task_embed_option: str = "none"
    register_token_num: int = 4
    register_token_shared: bool = True

    gru_gating: bool = False
    moe_in_transformer: bool = False
    multiplication_moe_in_transformer: bool = False
    num_experts_of_moe_in_transformer: int = 1

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    """
    Transformer model class.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration for the Transformer model.

    Attributes:
        - config (:obj:`TransformerConfig`): Configuration object.
        - drop (:obj:`nn.Dropout`): Dropout layer for embedding dropout.
        - blocks (:obj:`nn.ModuleList`): List of Transformer blocks.
        - ln_f (:obj:`nn.LayerNorm`): Layer normalization applied to the final output.
    """

    def __init__(self, config: TransformerConfig, task_embed=None) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.task_embed = task_embed
        self.task_embed_option = self.config.task_embed_option  # Strategy for task embeddings
        self.register_token_shared = True

        # TODO: 共享模式下，所有任务使用同一参数

        if self.task_embed_option == "register_task_embed":
            self.use_register_token = True # TODO
            # Register token setup
            self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4

            # 判断是否采用共享模式
            self.register_token_shared = getattr(config, "register_token_shared", True)
            if self.register_token_shared:
                # print(f'self.register_token_shared:{self.register_token_shared}')
                # print(f'='*20)
                # 共享模式：所有任务使用同一个 register_tokens 参数，形状为 (register_token_num, embed_dim)
                self.register_tokens = nn.Parameter(torch.empty(self.register_token_num, config.embed_dim))
                nn.init.xavier_uniform_(self.register_tokens)
            else:
                # 非共享模式：依赖外部传入的 task_embed 模块来生成 task embedding，
                # 并通过 SimNorm 归一化后复制出 register token
                self.task_embed = task_embed  # 外部传入的模块，如 nn.Embedding
                self.sim_norm = SimNorm(simnorm_dim=config.embed_dim) # Normalization for task embeddings

        else:
            self.use_register_token = False # TODO
   

    def add_register_tokens(self, sequences: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        将 register_token_num 个 Register Token 拼接到序列最前面。

        Arguments:
            - sequences (:obj:`torch.Tensor`): (B, T, C)
            - task_id (:obj:`int`): 当前任务的 ID

        Returns:
            - new_sequences (:obj:`torch.Tensor`): (B, T + register_token_num, C)
        """
        B = sequences.size(0)
        device = sequences.device

        if self.register_token_shared:
            # 共享模式：直接使用同一组 register_tokens 参数
            # register_tokens 形状为 (register_token_num, embed_dim)
            register_tokens = self.register_tokens  
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)  # 形状 (B, register_token_num, embed_dim)
        else:
            # 非共享模式：依靠 task_embed 动态生成 task embedding，然后复制出 register tokens
            task_embedding = self.task_embed(torch.tensor([task_id], device=device))  # (1, embed_dim)
            task_embedding = self.sim_norm(task_embedding.view(1, -1)).view(-1)         # (embed_dim,)
            register_tokens = task_embedding.unsqueeze(0).expand(self.register_token_num, -1)  # (register_token_num, embed_dim)
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, register_token_num, embed_dim)

        new_sequences = torch.cat([sequences, register_tokens], dim=1)  # 在序列末尾拼接 register tokens (B, register_token_num + T, C)
        return new_sequences

    def remove_register_tokens_from_kv(self, past_keys_values: KeysValues) -> None:
        """
        移除所有层 KV 中最前面的 register_token_num 个 token，用于在 forward() 结束时调用。
        """
        if past_keys_values is None:
            return
        past_keys_values.remove_register_tokens(self.register_token_num)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        Generate a placeholder for keys and values.

        Arguments:
            - n (:obj:`int`): Batch size.
            - max_tokens (:obj:`int`): Maximum number of tokens in the sequence.

        Returns:
            - KeysValues: An object containing empty keys and values.
        """
        device = self.ln_f.weight.device  # Assumption: All submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)


    #@profile
    def forward(
        self,
        sequences: torch.Tensor,         # (B, T, C)
        past_keys_values: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        task_id: int = 0,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Arguments:
            - sequences (:obj:`torch.Tensor`): (B, T, C)
            - past_keys_values (:obj:`Optional[KeysValues]`): 缓存，用于推理时加速
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): 某些场景下可用的有效上下文长度
            - task_id (:obj:`int`): 任务 ID

        Returns:
            - 输出张量 (B, T + register_token_num, C) 或 (B, T, C)，视是否添加 Register Token 而定
        """
        # 若使用 Register Token，则将其拼到序列最前面
        # 训练阶段和推理阶段都统一处理
        if self.use_register_token:
            sequences = self.add_register_tokens(sequences, task_id)

        # 接入 dropout
        x = self.drop(sequences)

        # 逐层调用
        for i, block in enumerate(self.blocks):
            x = block(x,
                      None if past_keys_values is None else past_keys_values[i],
                      valid_context_lengths)

        # 最后层 LN
        x = self.ln_f(x)

        # 如果 past_keys_values 不为 None，说明是推理阶段，此时我们需要把 KV 缓存中
        # 尾部多加的 Register Token 移除，以保证外键信息一致，不用修改外部逻辑
        # if self.use_register_token and (past_keys_values is not None):
        if self.use_register_token:
            self.remove_register_tokens_from_kv(past_keys_values)

        # TODO
        if self.use_register_token:
            # import ipdb; ipdb.set_trace()
            x = x[:, :-self.register_token_num, :]

        return x




class Block(nn.Module):
    """
    Transformer block class.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration for the Transformer block.

    Attributes:
        - gru_gating (:obj:`bool`): Flag to use GRU gating mechanism.
        - gru_bias (:obj:`float`): Bias for the GRU gating mechanism.
        - gate1 (:obj:`Optional[GRUGatingUnit]`): First GRU gating unit (if GRU gating is enabled).
        - gate2 (:obj:`Optional[GRUGatingUnit]`): Second GRU gating unit (if GRU gating is enabled).
        - ln1 (:obj:`nn.LayerNorm`): Layer normalization before the attention layer.
        - ln2 (:obj:`nn.LayerNorm`): Layer normalization before the MLP.
        - attn (:obj:`SelfAttention`): Self-attention mechanism.
        - mlp (:obj:`nn.Sequential`): Multi-layer perceptron.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # NOTE: GRU gating as in GTrXL
        self.gru_gating = config.gru_gating
        self.gru_bias = 2.0
        if self.gru_gating:
            self.gate1 = GRUGatingUnit(config.embed_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(config.embed_dim, self.gru_bias)

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)


        if config.moe_in_transformer:
            from .moe import MoELayer, MultiplicationFeedForward
            # 创Create multiple independent MLP instances
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.embed_dim, 4 * config.embed_dim),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(4 * config.embed_dim, config.embed_dim),
                    nn.Dropout(config.resid_pdrop),
                ) for _ in range(config.num_experts_of_moe_in_transformer)
            ])
            self.feed_forward = MoELayer(
                config,
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=config.num_experts_per_tok,
            )
            
            print("="*20)
            print(f'use moe in feed_forward of transformer, num of expert: {config.num_experts_of_moe_in_transformer}')
            print("="*20)
        elif config.multiplication_moe_in_transformer:
            # TODO: deepseek-v3
            # from .moe import MoeConfig,MoELayer
            # moe_cfg = MoeConfig(
            #     embed_dim=config.embed_dim,
            #     num_experts_total=config.num_experts_of_moe_in_transformer,
            #     num_experts_per_tok=1,
            # )
            # self.feed_forward = MoELayer(moe_cfg)
            # print("=" * 20)
            # print(f"Use MoE feed_forward, num_experts={moe_cfg.num_experts_total}")
            # print("=" * 20)

            from .moe import MoELayer, MultiplicationFeedForward
            # Create multiple FeedForward instances for multiplication-based MoE
            self.experts = nn.ModuleList([
                MultiplicationFeedForward(config) for _ in range(config.num_experts_of_moe_in_transformer)
            ])
            self.feed_forward = MoELayer(
                config,
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=config.num_experts_per_tok,
            )
            print("="*20)
            print(f'use multiplication moe in feed_forward of transformer, num of expert: {config.num_experts_of_moe_in_transformer}')
            print("="*20)
        else:
            # self.feed_forward = nn.Sequential(
            #     nn.Linear(config.embed_dim, 4 * config.embed_dim),
            #     nn.GELU(approximate='tanh'),
            #     nn.Linear(4 * config.embed_dim, config.embed_dim),
            #     nn.Dropout(config.resid_pdrop),
            # )
            # 普通的 MLP，若在 feed_forward 上启用 LoRA，则对其中线性层进行包装
            self.feed_forward = nn.Sequential(
                _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim), config, "feed_forward"),
                nn.GELU(approximate='tanh'),
                _maybe_wrap_linear(nn.Linear(4 * config.embed_dim, config.embed_dim), config, "feed_forward"),
                nn.Dropout(config.resid_pdrop),
            )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation (default: None).
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking (default: None).

        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        x_attn = self.attn(self.ln1(x), past_keys_values, valid_context_lengths)
        if self.gru_gating:
            x = self.gate1(x, x_attn)
            x = self.gate2(x, self.feed_forward(self.ln2(x)))
        else:
            x = x + x_attn
            x = x + self.feed_forward(self.ln2(x))

        return x


class SelfAttention(nn.Module):
    """
    Implements self-attention mechanism for transformers.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration object containing hyperparameters.

    Attributes:
        - config (:obj:`TransformerConfig`): Stores the configuration for the self-attention module.
        - num_heads (:obj:`int`): Number of attention heads.
        - key (:obj:`nn.Linear`): Linear layer to project input to key vectors.
        - query (:obj:`nn.Linear`): Linear layer to project input to query vectors.
        - value (:obj:`nn.Linear`): Linear layer to project input to value vectors.
        - attn_drop (:obj:`nn.Dropout`): Dropout layer for attention weights.
        - resid_drop (:obj:`nn.Dropout`): Dropout layer for residual connection.
        - proj (:obj:`nn.Linear`): Final linear layer for projection.
        - mask (:obj:`torch.Tensor`): Mask tensor for causal or block-causal attention.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.config = config

        self.task_embed_option = self.config.task_embed_option
        if self.task_embed_option == "register_task_embed":
            self.use_register_token = True # TODO
            # Register token setup
            self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4
        else:
            self.use_register_token = False # TODO

        self.num_heads = config.num_heads

        if config.lora_r > 0 and ("attn" in config.lora_target_modules):
            self.key = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            # print("key type:", type(self.key))  # 期望返回 CurriculumLoRALinear
            self.query = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            self.value = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            self.proj = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        else:
            self.key = nn.Linear(config.embed_dim, config.embed_dim)
            self.query = nn.Linear(config.embed_dim, config.embed_dim)
            self.value = nn.Linear(config.embed_dim, config.embed_dim)
            self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        if self.use_register_token: # ======= TODO ========
            causal_mask = torch.tril(torch.ones(config.max_tokens+self.register_token_num*5, config.max_tokens+self.register_token_num*5))
        else:
            causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))

        self.register_buffer('mask', causal_mask)

    #@profile
    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        """
        Forward pass for the self-attention mechanism.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (B, T, C) where B is batch size,
                                        T is sequence length, and C is embedding dimension.
            - kv_cache (:obj:`Optional[KeysValues]`): Optional key-value cache for faster inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Optional tensor containing valid context lengths.

        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            # try:
            assert nh == self.num_heads and b == B and c * nh == C, "Cache dimensions do not match input dimensions."
            # except Exception as e:
            #     print('debug')
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)

        if kv_cache is not None:
            # import ipdb; ipdb.set_trace()
            kv_cache.update(k, v) # time occupancy 21%
            k, v = kv_cache.get() # time occupancy 5%

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if valid_context_lengths is not None:
            # Final mask.shape: (B, T, L + T)
            # L is the context length, T is the current input length,
            # valid_context_lengths is the valid length at the end of the context.
            mask = torch.zeros(B, T, L + T, device=att.device)
            # For each sample, set the invalid parts to 0 based on its valid length.
            for i in range(B):
                mask[i] = self.mask[L:L + T, :L + T].clone()
                mask[i, :, :(L - valid_context_lengths[i])] = 0  # Set invalid parts to 0.
            # Adjust mask dimensions to match the last two dimensions of att.
            # (B, T, L + T) -> (B, 1, T, L + T) -> (B, num_heads, T, L + T)
                mask = mask.unsqueeze(1).expand(-1, att.size(1), -1, -1)
        else:
            # mask.shape: (T, L + T)
            mask = self.mask[L:L + T, :L + T]

        # import ipdb; ipdb.set_trace()

        # Adjust mask for register tokens if applicable
        if self.use_register_token and self.register_token_num > 0:
            # Allow all positions to attend to the last `register_token_num` tokens
            register_mask = mask.clone()  # (T, L + T)
            register_mask[-self.register_token_num:, :] = 1  # Allow register tokens to see all positions
            register_mask[:, -self.register_token_num:] = 1  # Allow all positions to see register tokens
            mask = register_mask

            if kv_cache is not None:
                # =============TODO=============
                # import ipdb; ipdb.set_trace()
                b, nh, new_L, c = kv_cache.shape # new_L可能小于L + T
                mask = mask[:,-new_L:]
            # else:
            #     import ipdb; ipdb.set_trace()

        # att.shape: (B, num_heads, T, L + T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # import ipdb; ipdb.set_trace()
        y = att @ v  # (B, num_heads, T, L + T) x (B, num_heads, L + T, head_size) -> (B, num_heads, T, head_size)

        y = rearrange(y, 'b h t e -> b t (h e)')  # Combine the heads back together (B, T, embed_dim)
        y = self.resid_drop(self.proj(y))

        return y

    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the attention map for the input sequence. This is useful for visualization purposes.
        More details can be found in visualizing_utils.py.

        Arguments:
            - x (:obj:`torch.Tensor`): Input sequence with shape (B, T, C).
            - kv_cache (:obj:`Optional[KeysValues]`): Cached keys and values for supporting long sequence inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for handling variable-length contexts.

        Returns:
            - torch.Tensor: Attention map with shape (B, nh, T, L + T), representing the distribution of attention.
        """
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C, "Cache dimensions are inconsistent with input dimensions."
        else:
            L = 0

        # Compute query, key, and value projections
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            # Update the kv_cache with the new keys and values
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # Compute the attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if valid_context_lengths is not None:
            mask = torch.zeros(B, T, L + T, device=att.device)
            for i in range(B):
                # Create attention mask for each batch
                mask[i] = self.mask[L:L + T, :L + T].clone()
                mask[i, :, :(L - valid_context_lengths[i])] = 0
            mask = mask.unsqueeze(1).expand(-1, att.size(1), -1, -1)
        else:
            mask = self.mask[L:L + T, :L + T]

        # Apply the attention mask
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att