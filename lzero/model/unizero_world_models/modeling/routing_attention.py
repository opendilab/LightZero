"""
Routing Attention module for Transformers.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange
from .attention import Attention
from .kv_caching import KeysValues
from .transformer import apply_rotary_emb
from .transformer_config import TransformerConfig

class RoutingAttention(Attention):
    """

    Content-based sparse attention mechanism via Routing.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration object containing hyperparameters.
        use_local (:obj:`bool`): Flag to indicate whether to combine with local attention.

    Attributes:
        # TODO: Add docstring for attributes.
    """

    def __init__(self, config : TransformerConfig, use_local : bool = False):
        super.__init__(config)
        self.use_local = use_local

        # Initialize attributes

    def forward(self, x, kv_cache=None, valid_ctx_len=None, freqs_cis=None):
        pass