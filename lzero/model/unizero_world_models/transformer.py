
import math
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

# Assuming these are part of your project structure
from ding.torch_utils.network import GRUGatingUnit
from .kv_caching import KeysValues
from lzero.model.common import SimNorm


@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer model.

    Arguments:
        - tokens_per_block (int): The number of tokens in a single block.
        - max_blocks (int): The maximum number of blocks.
        - attention (str): The type of attention mechanism to use.
        - num_layers (int): The number of transformer layers.
        - num_heads (int): The number of attention heads.
        - embed_dim (int): The embedding dimension.
        - embed_pdrop (float): Dropout probability for embeddings.
        - resid_pdrop (float): Dropout probability for residual connections.
        - attn_pdrop (float): Dropout probability for attention weights.
        - lora_r (int): The rank for LoRA decomposition. If 0, LoRA is disabled. Defaults to 0.
        - lora_alpha (int): The alpha parameter for LoRA scaling. Defaults to 1.
        - lora_dropout (float): Dropout probability for LoRA layers. Defaults to 0.0.
        - lora_target_modules (list): A list of module names to apply LoRA to. Defaults to None.
        - curriculum_stage_num (int): The total number of curriculum stages. (e.g., 3 means stages 0, 1, 2). It equals 1 + the number of available LoRA adapters. Defaults to 5.
        - min_stage0_iters (int): The minimum number of iterations for stage 0. Defaults to 10,000.
        - max_stage_iters (int): The maximum number of iterations per stage. Defaults to 20,000.
        - lora_scale_init (float): The initial value for the learnable scale of each LoRA adapter. Defaults to 1.0.
        - task_embed_option (str): Strategy for task embeddings. Defaults to "none".
        - register_token_num (int): The number of register tokens to use. Defaults to 4.
        - register_token_shared (bool): Whether to use shared register tokens across all tasks. Defaults to True.
        - gru_gating (bool): Whether to use GRU gating. Defaults to False.
        - moe_in_transformer (bool): Whether to use Mixture of Experts in the transformer feed-forward layers. Defaults to False.
        - multiplication_moe_in_transformer (bool): Whether to use multiplication-based MoE. Defaults to False.
        - num_experts_of_moe_in_transformer (int): The number of experts for MoE. Defaults to 1.
    """
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    # LoRA parameters
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None

    # Curriculum Learning related parameters
    curriculum_stage_num: int = 5
    min_stage0_iters: int = 10_000
    max_stage_iters: int = 20_000
    lora_scale_init: float = 1.0

    # Other configurations
    task_embed_option: str = "none"
    register_token_num: int = 4
    register_token_shared: bool = True
    gru_gating: bool = False
    moe_in_transformer: bool = False
    multiplication_moe_in_transformer: bool = False
    num_experts_of_moe_in_transformer: int = 1

    @property
    def max_tokens(self) -> int:
        """
        Calculates the maximum number of tokens.
        """
        return self.tokens_per_block * self.max_blocks


class LearnableScale(nn.Module):
    """
    A learnable scalar parameter constrained within a specific range.

    The transformation is defined as:
        s = offset + scale * tanh(ŝ)
    This maps an unbounded logit `ŝ` to the range (offset - scale, offset + scale).
    Using tanh can sometimes provide more stable gradients than sigmoid.

    Example:
        To get a range of (0.8, 1.2), use init=1.0 and s_range=0.2.

    Arguments:
        - init (float): The initial and center value of the learnable scale. Defaults to 1.0.
        - s_range (float): The range of scaling, determining the bounds. Must be positive. Defaults to 0.2.
    """

    def __init__(self, init: float = 1.0, s_range: float = 0.2):
        super().__init__()
        assert s_range > 0, "The scaling range must be positive."
        self.offset = init
        self.scale = s_range

        # Initialize the logit to 0, so the initial output is exactly `init`.
        # This parameter is intended to be frozen initially and activated by a curriculum controller.
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.logit.requires_grad = False

    def forward(self) -> torch.Tensor:
        """
        Computes the scaled value.
        """
        return self.offset + self.scale * torch.tanh(self.logit)


class CurriculumLoRALinear(nn.Module):
    """
    An extension of a standard linear layer for curriculum-based LoRA fine-tuning.

    This module maintains a base weight and bias, and initializes multiple LoRA adapters
    (number of adapters = curriculum_stage_num - 1). The forward pass behavior depends
    on the current curriculum stage:

    - If `curriculum_stage == 0`:
        output = F.linear(x, W, bias)
    - If `curriculum_stage >= 1`:
        output = base_output + sum_{i=0}^{curriculum_stage-1} scaling * adapter_i(x)

    During training, only the adapter corresponding to the current stage
    (`index == curriculum_stage - 1`) is updated. Previous adapters contribute to the
    forward pass but their gradients are detached.

    Note:
        The curriculum stage is controlled externally by calling `set_curriculum_stage(stage)`.

    Arguments:
        - in_features (int): Size of each input sample.
        - out_features (int): Size of each output sample.
        - bias (bool): If set to False, the layer will not learn an additive bias. Defaults to True.
        - r (int): The rank for LoRA decomposition. Defaults to 0.
        - lora_alpha (int): The alpha parameter for LoRA scaling. Defaults to 1.
        - lora_dropout (float): Dropout probability for LoRA layers. Defaults to 0.0.
        - curriculum_stage_num (int): The total number of curriculum stages.
        - lora_scale_init (float): The initial value for the learnable scale of each adapter.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
                 curriculum_stage_num: int = 1, lora_scale_init: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.curriculum_stage_num = curriculum_stage_num
        self.curriculum_stage = 0  # Initial stage is 0

        # Initialize base weights (part of the base transformer), trainable by default.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize LoRA adapters if r > 0 and more than one curriculum stage exists.
        self.adapters = nn.ModuleList()
        self.adapter_scales = nn.ModuleList()
        if r > 0 and (curriculum_stage_num - 1) > 0:
            for _ in range(curriculum_stage_num - 1):
                adapter = nn.ParameterDict({
                    'lora_A': nn.Parameter(torch.randn(r, in_features) * 0.01),
                    'lora_B': nn.Parameter(torch.zeros(out_features, r))
                })
                self.adapters.append(adapter)
                self.adapter_scales.append(LearnableScale(lora_scale_init, s_range=0.2))
        else:
            self.adapters = None

        # At initialization (stage 0), base layer is trainable, all adapters are frozen.
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
        if self.adapters is not None:
            for adapter in self.adapters:
                adapter['lora_A'].requires_grad = False
                adapter['lora_B'].requires_grad = False

    def set_curriculum_stage(self, stage: int) -> None:
        """
        Sets the current curriculum stage and adjusts parameter trainability accordingly.

        - stage == 0: The base layer is trainable, and all adapters are frozen.
        - stage >= 1: The base layer is frozen. Only the current adapter (`index == stage - 1`)
                      is trainable. Previous adapters contribute to the forward pass but
                      do not receive gradients.

        Arguments:
            - stage (int): The curriculum stage, must be in [0, curriculum_stage_num - 1].
        """
        assert 0 <= stage < self.curriculum_stage_num, f"Stage must be in [0, {self.curriculum_stage_num - 1}]"
        self.curriculum_stage = stage

        module_id = f"({self.in_features}x{self.out_features})"
        if stage == 0:
            self.weight.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = True
            if self.adapters is not None:
                for adapter in self.adapters:
                    adapter['lora_A'].requires_grad = False
                    adapter['lora_B'].requires_grad = False
            logging.info(f"[CurriculumLoRALinear {module_id}] Stage 0: Base layer is trainable, all adapters are frozen.")
        else:
            # Freeze the base layer for stages > 0.
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            for idx, adapter in enumerate(self.adapters):
                is_current_adapter = (idx == stage - 1)
                adapter['lora_A'].requires_grad = is_current_adapter
                adapter['lora_B'].requires_grad = is_current_adapter
                status = "activated (trainable)" if is_current_adapter else "frozen (forward-only)"
                logging.info(f"[CurriculumLoRALinear {module_id}] Stage {stage}: Adapter {idx} is {status}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        """
        baseline_out = F.linear(x, self.weight, self.bias)
        if self.curriculum_stage == 0 or self.adapters is None:
            return baseline_out

        adapter_out = 0
        # Accumulate outputs from adapters up to the current stage.
        # Only the current adapter's output will propagate gradients.
        for idx in range(self.curriculum_stage):
            if idx >= len(self.adapters):
                break
            adapter = self.adapters[idx]
            out = F.linear(self.lora_dropout(x), adapter['lora_A'])
            out = F.linear(out, adapter['lora_B'])
            scale = self.adapter_scales[idx]()

            if idx == self.curriculum_stage - 1:
                # Current adapter's output contributes to the gradient computation.
                adapter_out = adapter_out + self.scaling * out * scale
            else:
                # Previous adapters' outputs are detached to prevent gradient flow.
                adapter_out = adapter_out + self.scaling * out.detach() * scale
        return baseline_out + adapter_out


def _maybe_wrap_linear(linear: nn.Linear, config: TransformerConfig, module_label: str) -> nn.Module:
    """
    A helper function to conditionally wrap an nn.Linear layer with CurriculumLoRALinear.

    The wrapping occurs if:
      - LoRA is enabled (config.lora_r > 0).
      - The module_label is in the target modules list (config.lora_target_modules).
      - Curriculum learning is enabled (config.curriculum_stage_num > 1).

    Otherwise, it returns the original linear layer.

    Arguments:
        - linear (nn.Linear): The original linear layer to be potentially wrapped.
        - config (TransformerConfig): The model configuration.
        - module_label (str): A label identifying the module type (e.g., "attn", "feed_forward").

    Returns:
        - nn.Module: The wrapped or original linear layer.
    """
    use_curriculum_lora = (
        config.lora_r > 0 and
        config.lora_target_modules and
        module_label in config.lora_target_modules and
        getattr(config, "curriculum_stage_num", 1) > 1
    )

    if use_curriculum_lora:
        new_linear = CurriculumLoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            curriculum_stage_num=config.curriculum_stage_num,
            lora_scale_init=config.lora_scale_init
        )
        # Copy original weights and bias
        new_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.data.copy_(linear.bias.data)
        return new_linear
    else:
        return linear


def set_curriculum_stage(model: nn.Module, stage: int) -> None:
    """
    Recursively traverses a model and sets the curriculum stage for all CurriculumLoRALinear instances.

    This function is generic and can be applied to any model containing CurriculumLoRALinear modules.

    Arguments:
        - model (nn.Module): The model to traverse (e.g., a Transformer).
        - stage (int): The curriculum stage to set.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, CurriculumLoRALinear):
            module.set_curriculum_stage(stage)
            count += 1
    if count > 0:
        logging.info(f"[Curriculum] Updated {count} CurriculumLoRALinear modules in {type(model).__name__} to stage {stage}.")

# Backward compatibility
set_curriculum_stage_for_transformer = set_curriculum_stage


class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism for a Transformer.

    This module computes query, key, and value projections and applies scaled dot-product attention.
    It supports LoRA customization for its linear layers and includes logic for handling register tokens.

    Arguments:
        - config (TransformerConfig): Configuration object with hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.config = config
        self.num_heads = config.num_heads

        # Flag to enable register token mechanism
        self.use_register_token = (config.task_embed_option == "register_task_embed")
        self.register_token_num = config.register_token_num if self.use_register_token else 0

        # Conditionally wrap linear layers with LoRA wrappers
        self.key = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.query = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.value = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.proj = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # Create a causal mask, expanded to accommodate register tokens if used.
        # The buffer is made larger to avoid out-of-bounds errors during long sequence generation.
        mask_size = config.max_tokens + self.register_token_num * 5
        causal_mask = torch.tril(torch.ones(mask_size, mask_size))
        self.register_buffer('mask', causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the self-attention mechanism.

        Arguments:
            - x (torch.Tensor): Input tensor of shape (B, T, C).
            - kv_cache (Optional[KeysValues]): Optional key-value cache for efficient inference.
            - valid_context_lengths (Optional[torch.Tensor]): Tensor containing valid context lengths for masking.

        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        L = kv_cache.shape[2] if kv_cache is not None else 0

        # Project and reshape Q, K, V for multi-head attention
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Get the appropriate mask slice
        current_mask = self.mask[L:L + T, :L + T]

        # Adjust mask for register tokens if they are in use
        if self.use_register_token and self.register_token_num > 0:
            # This modification allows register tokens to attend to all other tokens,
            # and all other tokens to attend to them, breaking causality for these specific tokens.
            register_mask = current_mask.clone()
            # This logic assumes register tokens are at the end of the sequence.
            register_mask[-self.register_token_num:, :] = 1  # Register tokens can see all positions.
            register_mask[:, -self.register_token_num:] = 1  # All positions can see register tokens.
            current_mask = register_mask

            if kv_cache is not None:
                # Adjust mask size if cache length differs from expected L+T
                new_L = kv_cache.shape[2]
                current_mask = current_mask[:, -new_L:]

        att = att.masked_fill(current_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, L+T) x (B, nh, L+T, hs) -> (B, nh, T, hs)
        y = rearrange(y, 'b h t e -> b t (h e)')  # Combine heads
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

class Block(nn.Module):
    """
    A single Transformer block, composed of self-attention and a feed-forward network.

    Arguments:
        - config (TransformerConfig): Configuration for the Transformer block.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        # Optional GRU gating, as in GTrXL
        self.gru_gating = config.gru_gating
        if self.gru_gating:
            self.gate1 = GRUGatingUnit(config.embed_dim, bias=2.0)
            self.gate2 = GRUGatingUnit(config.embed_dim, bias=2.0)

        # Define the feed-forward network (MLP)
        # This can be a standard MLP, a Mixture of Experts (MoE), or other variants.
        if config.moe_in_transformer:
            # Implementation for MoE would go here
            raise NotImplementedError("MoE is not fully implemented in this refactored code.")
        else:
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
            - x (torch.Tensor): Input tensor of shape (B, T, C).
            - past_keys_values (Optional[KeysValues]): Precomputed keys and values for faster inference.
            - valid_context_lengths (Optional[torch.Tensor]): Valid lengths of context for masking.

        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """
        attn_output = self.attn(self.ln1(x), past_keys_values, valid_context_lengths)
        if self.gru_gating:
            x = self.gate1(x, attn_output)
            x = self.gate2(x, self.feed_forward(self.ln2(x)))
        else:
            x = x + attn_output
            x = x + self.feed_forward(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    A Transformer model composed of multiple Blocks.

    This class orchestrates the overall architecture, including embedding dropout,
    a stack of transformer blocks, and final layer normalization. It also manages
    register tokens and task-specific embeddings.

    Arguments:
        - config (TransformerConfig): Configuration for the Transformer model.
        - task_embed (Optional[nn.Module]): An optional module for generating task embeddings.
    """

    def __init__(self, config: TransformerConfig, task_embed: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Configure register token and task embedding strategy
        self.use_register_token = (config.task_embed_option == "register_task_embed")
        if self.use_register_token:
            self.register_token_num = config.register_token_num
            self.register_token_shared = config.register_token_shared
            if self.register_token_shared:
                # Shared mode: a single set of register tokens for all tasks.
                self.register_tokens = nn.Parameter(torch.empty(self.register_token_num, config.embed_dim))
                nn.init.xavier_uniform_(self.register_tokens)
            else:
                # Non-shared mode: generate tokens from a task-specific embedding.
                assert task_embed is not None, "task_embed module must be provided for non-shared register tokens."
                self.task_embed = task_embed
                self.sim_norm = SimNorm(simnorm_dim=config.embed_dim)

    def add_register_tokens(self, sequences: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Appends register tokens to the end of the input sequences.

        Arguments:
            - sequences (torch.Tensor): Input sequences of shape (B, T, C).
            - task_id (int): The ID of the current task.

        Returns:
            - torch.Tensor: Sequences with register tokens appended, shape (B, T + register_token_num, C).
        """
        B = sequences.size(0)
        device = sequences.device

        if self.register_token_shared:
            # Use the same set of register tokens for all samples in the batch.
            register_tokens = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            # Generate task-specific register tokens.
            task_embedding = self.task_embed(torch.tensor([task_id], device=device))
            task_embedding = self.sim_norm(task_embedding.view(1, -1)).view(-1)
            register_tokens = task_embedding.unsqueeze(0).expand(self.register_token_num, -1)
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)

        return torch.cat([sequences, register_tokens], dim=1)

    def remove_register_tokens_from_kv(self, past_keys_values: Optional[KeysValues]) -> None:
        """
        Removes register tokens from the key-value cache in-place.
        This is called at the end of the forward pass during inference to maintain consistency.
        """
        if past_keys_values is not None and self.use_register_token:
            past_keys_values.remove_register_tokens(self.register_token_num)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        Generates a placeholder for keys and values for inference.

        Arguments:
            - n (int): Batch size.
            - max_tokens (int): Maximum number of tokens in the sequence.

        Returns:
            - KeysValues: An object containing empty keys and values.
        """
        device = self.ln_f.weight.device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(
        self,
        sequences: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        task_id: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Arguments:
            - sequences (torch.Tensor): Input tensor of shape (B, T, C).
            - past_keys_values (Optional[KeysValues]): Cache for efficient inference.
            - valid_context_lengths (Optional[torch.Tensor]): Valid context lengths for masking.
            - task_id (int): The ID of the current task.

        Returns:
            - torch.Tensor: The output tensor of shape (B, T, C).
        """
        # Add register tokens if enabled. They are handled internally and removed from the final output.
        if self.use_register_token:
            sequences = self.add_register_tokens(sequences, task_id)

        x = self.drop(sequences)

        for i, block in enumerate(self.blocks):
            kv_cache_for_block = None if past_keys_values is None else past_keys_values[i]
            x = block(x, kv_cache_for_block, valid_context_lengths)

        x = self.ln_f(x)

        # During inference, remove the register tokens from the KV cache to keep it clean for the next step.
        self.remove_register_tokens_from_kv(past_keys_values)

        # Remove register tokens from the final output sequence before returning.
        if self.use_register_token:
            x = x[:, :-self.register_token_num, :]

        return x

