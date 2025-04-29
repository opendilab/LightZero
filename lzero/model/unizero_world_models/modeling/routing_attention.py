"""
Routing Attention module for Transformers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange
from .attention import Attention
from .transformer import apply_rotary_emb
from .transformer_config import TransformerConfig

# constants
KMEAN_INIT_ITERS = 5
TOKEN_SELF_ATTN_VALUE = -5e4

# general helper functions
def exists(x):
    return x is not None

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def is_empty(x):
    return x.nelement() == 0

def eps(x, y, decay):
    if not exists(x):
        return y
    return x * decay + y * (1 - decay)

def eps_inplace(x, y, decay):
    if is_empty(x):
        x.data.copy_(y)
        return
    x.data.mul_(decay).add_(y, alpha=(1-decay))

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def expand_dim(x, dim, n):
    x = x.unsqueeze(dim)
    expand_shape = [-1] * len(x.shape)
    expand_shape[dim] = n
    return x.expand(*expand_shape)

def batched_index_select(x, indices):
    last_dim = x.shape[-1]
    return x.gather(2, expand_dim(indices, -1, last_dim))

def batched_bincount(index, num_classes, dim = - 1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def scatter_mean(x, t, index, dim, eps = 1e-5):
    numer = x.scatter_add(dim, index, t)
    denom = x.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)

# k-means helpers

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)

def kmeans_iter(x, means, buckets=None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        # compute distances
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True),dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)

    return means

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def distribution(dists, window_size):
    _, topk_indices = dists.topk(k = window_size, dim=-2)
    indices = topk_indices.transpose(-2, -1)
    return indices.reshape(*indices.shape()[:2], -1)

class RoutingAttention(Attention):
    """
    Content-based sparse attention mechanism via Routing.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration object containing hyperparameters.
        use_local (:obj:`bool`): Flag to indicate whether to combine with local attention.
    """

    def __init__(self, config : TransformerConfig, use_local : bool = False):
        super.__init__(config)
        self.use_local = use_local

        # Initialize attributes

        # head dims
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        # routing (k-means) params
        default_clusters = int(math.sqrt(config.max_tokens))
        self.num_clusters = config.routing_num_clusters or default_clusters
        self.update_interval = config.routing_update_interval or 1 # recompute centroids every forward
        self.top_k = config.routing_topk
        self.window_size = config.local_window_size if self.use_local else default_clusters
        self.causal = True
        self.shared_qk = False
        self.receives_context = False

        # centroids
        self.centroids = nn.Parameter(
            # centroids : (heads, clusters, head_dim)
            torch.randn(self.num_heads, self.num_clusters, self.head_dim)
        )

        # kv_cache
        self.num_mem_kv = getattr(config, 'routing_num_mem_kv', 0) or 0
        if self.num_mem_kv > 0:
            self.mem_key = nn.Parameter(torch.randn(self.num_heads, self.num_clusters, self.num_mem_kv , self.head_dim))
            self.mem_value = nn.Parameter(torch.randn(self.num_heads, self.num_clusters, self.num_mem_kv , self.head_dim))

        # projections
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.mem_value = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # init k-means
        self.kmeans = KMeans(
            self.num_heads,
            self.head_dim,
            self.num_clusters,
            eps_decay = config.routing_decay,
            commitment = config.routing_commitment
        )


    def forward(self, x, kv_cache=None, valid_ctx_len=None, freqs_cis=None):
        B, T, C = x.shape

        # project Q, K, V reshape to (B, H, T, D)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # apply embeddings
        if self.config.rotary_emb:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # build attention masks
        query_mask = key_mask = None
        if valid_ctx_len is not None:
            device = x.device
            idx = torch.arange(T, device=device)
            query_mask = idx[None, :] < valid_ctx_len[:, None]
            key_mask = query_mask

        concat_qk = torch.cat((q, k), dim=2) # (B, H, 2T, D)
        dists, aux_loss = self.kmeans(concat_qk, update=self.training)
        # split distances into q and k
        q_dists, k_dists = split_at_index(dists, index = T, dim=2)

        wsz = min(self.window_size, T)
        c_wsz = min(self.context_window_size, k.shape[2])

        if not self.shared_qk or self.receives_context:
            indices = distribution(q_dists, wsz)
            kv_indices = distribution(k_dists, c_wsz)
        else:
            indices = distribution(dists, wsz)
            kv_indices = indices

        # gather the routing
        q_sel = batched_index_select(q, indices) # (B, H, T, wsz, D)
        k_sel = batched_index_select(k, kv_indices) # (B, H, T, c_wsz, D)
        v_sel = batched_index_select(v, kv_indices)

        reshape = lambda x : x.reshape(B, self.num_heads, self.num_clusters, -1, self.head_dim)
        qh, kh, vh = map(reshape, (q_sel, k_sel, v_sel))

        # memory slots in kv
        if self.num_mem_kv > 0:
            mk = expand_dim(self.mem_key, 0, B).to(q)
            mv = expand_dim(self.mem_value, 0, B).to(q)
            kh = torch.cat((mk, kh), dim=3)
            vh = torch.cat((mv, vh), dim=3)

        # compute sparse QK attention per cluster
        dots = torch.einsum('bhkqd,bhkrd->bhkqr', qh, kh) * (self.head_dim ** -0.5)
        mask_value = max_neg_value(dots)

        if exists(query_mask) or exists(key_mask):
            query_mask = default(query_mask,  lambda: torch.ones((B, T), device=x.device).bool())
            key_mask = default(key_mask, lambda: torch.ones((B, k.shape[2]), device=x.device).bool())

            qm = expand_dim(query_mask, 1, self.num_heads).gather(2, indices)
            km = expand_dim(key_mask, 1, self.num_heads).gather(2, kv_indices)
            qm.reshape(B, self.num_heads, self.num_clusters, -1)
            km.reshape(B, self.num_heads, self.num_clusters, -1)
            m = qm[:, :, :, :, None] * km[:, :, :, None, :]
            m = F.pad(m, (self.num_mem_kv, 0), value = True)
            dots = dots.masked_fill(~m, mask_value)
            del m

        if self.causal:
            qm = indices.reshape(B, self.num_heads, self.num_clusters, -1)
            km = kv_indices.reshape(B, self.num_heads, self.num_clusters, -1)
            m = qm[:, :, :, :, None] >= km[:, :, :, None, :]
            m = F.pad(m, (self.num_mem_kv, 0), value = True)
            dots.masked_fill_(~m, mask_value)
            del m

        if self.shared_qk:
            qm = indices.reshape(B, self.num_heads, self.num_clusters, -1)
            km = kv_indices.reshape(B, self.num_heads, self.num_clusters, -1)
            m = qm[:, :, :, :, None] == km[:, :, :, None, :]
            m = F.pad(m, (self.num_mem_kv, 0), value = True)
            dots.masked_fill_(~m, TOKEN_SELF_ATTN_VALUE)
            del m

        # softmax + dropout
        att = F.softmax(dots, dim=-1)
        att = self.attn_drop(att)

        # agg values
        bo = torch.einsum('bhkqr,bhkrd->bhkqd', att, vh)
        so = rearrange(bo, 'b h k t d -> b h (k t) d')

        # scatter back and mean over clusters
        out = torch.zeros_like(q)
        out = scatter_mean(out, so, indices.unsqueeze(-1).expand_as(so), dim=2)

        # combine heads + project
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(out)), aux_loss



class KMeans(nn.Module):
    """
    KMeans clustering module.
    """

    def __init__(self, num_heads, head_dim, num_clusters, eps_decay = 0.999, commitment = 1e-4):
        super().__init__()
        self.commitment = commitment
        self.eps_decay = eps_decay

        self.register_buffer('means', torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.innitted:
            return

        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        num_clusters = self.means.shape[1]
        means = x.tranpose(0, 1).contiguous().view(h, -1, d)
        num_samples = means.shape[1]

        if num_samples >= num_clusters:
            indices = torch.randperm(num_samples, device=device)[:num_clusters]
        else:
            indices = torch.randint(num_samples, (num_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means = None):
        new_means = default(new_means, self.new_means)

        assert exists(new_means), 'no new means to update'

        eps_inplace(self.means, new_means, self.eps_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means = False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim = -1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = eps(self.new_means, means, self.num_new_means / (self.num_new_means + 1))
            self.num_new_means += 1

        return dists, loss