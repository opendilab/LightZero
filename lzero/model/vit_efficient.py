# vit_efficient.py
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from packaging import version

# ---------------- 工具 ----------------
def pair(x): return x if isinstance(x, tuple) else (x, x)
def trunc_normal_(t, std=.02): nn.init.trunc_normal_(t, std=std, a=-2*std, b=2*std)

# ---------------- 归一化 ----------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add_(self.eps).sqrt_()
        return self.scale * x / rms

def get_norm(norm_type:str, dim:int):
    return nn.LayerNorm(dim) if norm_type=="LN" else RMSNorm(dim)

# ---------------- Patch Embedding ----------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_type):
        super().__init__()
        img_h,img_w = pair(img_size); p_h,p_w = pair(patch_size)
        assert img_h%p_h==0 and img_w%p_w==0
        self.num_patches = (img_h//p_h)*(img_w//p_w)
        self.proj = nn.Conv2d(in_chans, embed_dim, (p_h,p_w), (p_h,p_w))
        self.norm = get_norm(norm_type, embed_dim)
    def forward(self,x):
        x = self.proj(x)                # B,C,H',W'
        x = rearrange(x,'b c h w -> b (h w) c')
        return self.norm(x)

# ---------------- Attention ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.h = heads; self.d = dim//heads
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.o   = nn.Linear(dim, dim)
        self.attn_drop = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.use_sdpa = version.parse(torch.__version__)>=version.parse("2.0.0")

    def forward(self,x):
        B,N,C = x.shape
        q,k,v = self.qkv(x).chunk(3,-1)
        q,k,v = (t.view(B,N,self.h,self.d).transpose(1,2) for t in (q,k,v))
        if self.use_sdpa:
            o = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop if self.training else 0.
                )
        else:
            q = q * self.d**-0.5
            attn = (q@k.transpose(-2,-1)).softmax(-1)
            attn = F.dropout(attn, self.attn_drop, self.training)
            o = attn@v
        o = o.transpose(1,2).reshape(B,N,C)
        o = self.o(o)
        return self.proj_drop(o)

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden,dim), nn.Dropout(dropout)
        )
    def forward(self,x): return self.net(x)

# ---------------- Block ----------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout, norm_type):
        super().__init__()
        self.n1 = get_norm(norm_type, dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.n2 = get_norm(norm_type, dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), dropout)
    def forward(self,x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

# ---------------- Config & ViT ----------------
@dataclass
class ViTConfig:
    img_size:Tuple[int,int]=(64,64)
    patch_size:Tuple[int,int]=(8,8)
    in_ch:int=3
    num_classes:int=768
    dim:int=768
    depth:int=12
    heads:int=12
    mlp_ratio:float=4.
    dropout:float=.1
    emb_dropout:float=.1
    norm_type:Literal["LN","RMS"]="LN"   # 新增
    pool:Literal["cls","mean"]="cls"

class VisionTransformer(nn.Module):
    def __init__(self,cfg:ViTConfig, final_norm="LayerNorm"):
        super().__init__()
        self.cfg=cfg
        self.patch = PatchEmbed(cfg.img_size, cfg.patch_size,
                                cfg.in_ch, cfg.dim, cfg.norm_type)
        self.cls = nn.Parameter(torch.zeros(1,1,cfg.dim))
        self.pos = nn.Parameter(torch.zeros(1,1+self.patch.num_patches,cfg.dim))
        trunc_normal_(self.pos); trunc_normal_(self.cls)
        self.drop = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList([Block(cfg.dim,cfg.heads,cfg.mlp_ratio,cfg.dropout,cfg.norm_type)
                                     for _ in range(cfg.depth)])
        self.norm = get_norm(cfg.norm_type, cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        if final_norm=="LayerNorm":
            self.final_norm = nn.LayerNorm(cfg.num_classes, eps=1e-6)
        elif final_norm=="SimNorm":
            self.final_norm = SimNorm(simnorm_dim=8)
        else:
            self.final_norm = nn.Identity()

    def forward(self,x):
        B = x.size(0)
        x = self.patch(x)
        x = torch.cat((self.cls.expand(B,-1,-1),x),1)+self.pos
        x = self.drop(x)
        for blk in self.blocks: x=blk(x)
        x = self.norm(x)
        x = x.mean(1) if self.cfg.pool=="mean" else x[:,0]
        x = self.head(x)
        return self.final_norm(x)

# --------------------------- 测试代码 --------------------------- #
if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    random.seed(42)
    # cfg = ViTConfig(num_classes=768, norm_type="RMS")
    cfg = ViTConfig(num_classes=768, norm_type="LN")

    model = VisionTransformer(cfg, final_norm="LayerNorm").cuda() if torch.cuda.is_available() else VisionTransformer(cfg)
    dummy = torch.randn(256,3,*cfg.img_size).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)      # => (10, 768)
    print("output[0]", out[0][:50])      # => (1, 50)

    # 简单基准
    import time, contextlib
    warm, rep = 5, 20
    for _ in range(warm): out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0=time.time()
    for _ in range(rep):
        out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"Average latency: {(time.time()-t0)/rep*1000:.2f} ms")