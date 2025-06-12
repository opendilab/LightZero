# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from ding.torch_utils.network import GRUGatingUnit # Keep if GRU gating is used outside Block
from einops import rearrange
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

class Mamba(nn.Module):
    """
    Mamba-based model potentially for UniZero architecture.
    Replaces the Transformer backbone.

    Arguments:
        - config (:obj:`MambaConfig`): Configuration for the Mamba model.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim 
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList()

        for i in range(config.num_layers):
            mamba_block = Mamba2(
                d_model=config.embed_dim,
                d_state=128,
                d_conv=4,
                expand=2,
                headdim=64,
                ngroups=1,
                bias=False,
                conv_bias=True,
                chunk_size=256,
                use_mem_eff_path=True,
                layer_idx=i,  
            )
            self.blocks.append(mamba_block)

        self.ln_f = nn.LayerNorm(config.embed_dim)

    def _get_device(self):
        return self.ln_f.weight.device

    def _get_dtype(self):
        return self.ln_f.weight.dtype
    
    def generate_empty_state(self,
                             batch_size: int,
                             max_seq_len: Optional[int] = None, 
                             ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        为所有 Mamba 层分配零初始化的状态张量 (conv_state, ssm_state)，用于推理。
        """
        _device =  self._get_device()
        _dtype = self._get_dtype()
        _max_seq_len = max_tokens if max_seq_len is not None else getattr(self.config, 'max_seq_len', 2048)

        all_layer_states = []
        for mamba_layer in self.blocks:
            conv_state, ssm_state = mamba_layer.allocate_inference_cache(
                    batch_size=batch_size,
                    max_seqlen=_max_seq_len,
                    dtype=_dtype
                )
            all_layer_states.append((conv_state.to(_device), ssm_state.to(_device)))
        return all_layer_states


    def forward(self, sequences: torch.Tensor, past_mamba_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
                seqlen_offset: Optional[int] = 0) -> torch.Tensor:
        """
        Forward pass for training or full sequence processing.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input tensor of shape (B, L, D) or (B*L, D) if seqlen is provided.
            - seqlen (:obj:`Optional[int]`): Sequence length if input is flattened (B*L, D).
            - inference_params (:obj:`Optional[Any]`): If provided, indicates potential step-by-step inference mode
                                                       (though `step` is preferred for that). Mamba2 forward might use it.

        Returns:
            - torch.Tensor: Output tensor, same shape principles as input `sequences`.
        """
        x = self.drop(sequences)
        current_inference_params = None
        if past_mamba_states is not None:
            batch_size, cur_seq_len, _ = sequences.shape
            current_inference_params = InferenceParams(
                max_seqlen=cur_seq_len + seqlen_offset,
                max_batch_size=batch_size,
                seqlen_offset=seqlen_offset
            )
            for i in range(self.config.num_layers):
                current_inference_params.key_value_memory_dict[i] = past_mamba_states[i]
                
        for i, block in enumerate(self.blocks):
            x = block(x, inference_params=current_inference_params)        

        x = self.ln_f(x)

        updated_layer_states_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if current_inference_params is not None:
            updated_layer_states_list = []
            for i in range(self.config.num_layers):
                updated_conv_state, updated_ssm_state = current_inference_params.key_value_memory_dict[i]
                updated_layer_states_list.append((updated_conv_state, updated_ssm_state))
                
        return x, updated_layer_states_list


