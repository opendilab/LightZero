"""
Attention module factory class.
"""

import torch.nn as nn
from .transformer_config import TransformerConfig

class Attention(nn.Module):
    """
    Base interface for all attention modules.
    All subclasses must implement forward(x, kv_cache=None, valid_ctx_len=None, freqs_cis=None)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def forward(self, x, kv_cache=None, valid_ctx_len=None, freqs_cis=None):
        raise NotImplementedError("Attention subclasses must implement forward()")

def build_attention(config: TransformerConfig) -> Attention:
    """
    Factory function to build the attention module based on the configuration.

    config.attention must be one of the following:
        - 'causal' : standard dense self-attention in vanilla UniZero
        - 'routing' : content-based sparse routing only
        - 'local+routing' : sliding-window local attention + routing
        - 'local' : local attention only
    """
    attention_mode = config.attention.lower()
    if attention_mode == 'causal':
        from .causal_attention import CausalAttention
        return CausalAttention(config)
    elif attention_mode == 'routing':
        from .routing_attention import RoutingAttention
        return RoutingAttention(
            config,
            use_local=False
        )
    elif attention_mode == 'local+routing':
        # Combines routing and local (sliding-window) attention
        from .routing_attention import RoutingAttention
        return RoutingAttention(
            config,
            use_local=True
        )
    elif attention_mode == 'local':
        from .local_attention import LocalAttention
        return LocalAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {config.attention}")
