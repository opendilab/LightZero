"""
Config Dataclass for the Transformer backbone.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str  # 'causal', 'local', 'local+routing', 'routing', 'adaptive'

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    # for RoPE
    rope_theta: float
    max_seq_len: int
    rotary_emb: bool = False

    # Routing Attention Params
    # n : number of clusters
    routing_num_clusters: Optional[int] = None
    # m : recompute centroids every m layers or at forward
    routing_update_interval: Optional[int] = None
    # d : sliding-window size for pure local attention (if use_local_attention = True)
    local_window_size: Optional[int] = None
    # k : attend top-k keys in that cluster for that query
    routing_topk: Optional[int] = None

    # Adaptive Hybrid Params
    init_adaptive_span: Optional[float] = 64.0
    max_adaptive_span: Optional[int] = None
    adaptive_span_regularization: float = 0.0 # regularization weight for adaptive span
    aha : bool = False # Whether to combine adaptive span with local attention
    gru_gating : Optional[bool] = True
    hybrid_local_layers: Optional[int] = 4
    interleave_local_causal : bool = False
    adaptive_regularization : Optional[str] = "l1"

    # GAAM Params
    init_adaptive_mu: Optional[float] = 4.0  # where to initialize each head’s mean offset
    init_adaptive_sigma: Optional[float] = 1.0  # where to initialize each head’s variance (before softplus)
    gaam_span_diversity_coeff: float = 0.0 # diversity regularization for GAAM

    # MGK Params
    num_mixtures: Optional[int] = 2
    init_mgk_sigma: Optional[float] = 1.0 # intializes sigma before softplus
    mgk_pi_entropy_coeff : float = 0.0 # entropy regularization for mixture weights

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks