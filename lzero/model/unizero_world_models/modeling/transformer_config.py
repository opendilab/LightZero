"""
Config Dataclass for the Transformer backbone.
"""
from typing import Optional
from attr import dataclass


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str  # 'global', 'causal', 'routing'

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

    # TODO: Set default values for these parameters
    # Routing Attention Params
    # n : number of clusters
    routing_num_clusters: Optional[int] = None
    # m : recompute centroids every m layers or at forward
    routing_update_interval: Optional[int] = None
    # whether to combine routing with local sliding-window attention
    use_local_attention: bool = False
    # d : sliding-window size for pure local attention (if use_local_attention = True)
    local_window_size: Optional[int] = None
    # k : attend top-k keys in that cluster for that query
    routing_topk: Optional[int] = None

    # how many extra memory KV slots per cluster
    routing_num_mem_kv: int = 0
    # decay for centroid updates
    routing_decay: float = 0.999
    # commitment loss weight for KMeans
    routing_commitment: float = 1e-4
    # context window size when receives_context=True
    routing_context_window_size: Optional[int] = None

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks