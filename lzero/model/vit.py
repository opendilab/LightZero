import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from lzero.model.common import SimNorm

# ==================== 新增/修改部分 开始 ====================

# 从您的 transformer.py 中导入核心组件
# 假设 vit.py 和 transformer.py 在同一个目录下
# 如果不在，请调整导入路径
try:
    from .transformer import _maybe_wrap_linear, TransformerConfig
except ImportError:
    # 提供一个备用路径或占位符，以防直接运行此文件
    print("无法导入 LoRA 组件，将使用标准 nn.Linear。")
    _maybe_wrap_linear = lambda linear, config, label: linear
    class TransformerConfig: pass

# ==================== 新增/修改部分 结束 ====================

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    # <--- 修改：__init__ 需要接收 config
    def __init__(self, dim, hidden_dim, dropout = 0., config: TransformerConfig = None):
        super().__init__()
        # <--- 修改：使用 _maybe_wrap_linear 包装线性层
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            _maybe_wrap_linear(nn.Linear(dim, hidden_dim), config, "feed_forward"),
            nn.GELU(),
            nn.Dropout(dropout),
            _maybe_wrap_linear(nn.Linear(hidden_dim, dim), config, "feed_forward"),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # <--- 修改：__init__ 需要接收 config
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., config: TransformerConfig = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # <--- 修改：使用 _maybe_wrap_linear 包装 to_qkv
        self.to_qkv = _maybe_wrap_linear(nn.Linear(dim, inner_dim * 3, bias = False), config, "attn")

        # <--- 修改：使用 _maybe_wrap_linear 包装 to_out
        self.to_out = _maybe_wrap_linear(nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ), config, "attn") if project_out else nn.Identity()
        # 注意：这里的包装方式可能需要根据 _maybe_wrap_linear 的实现进行调整。
        # 如果它只接受 nn.Linear，你需要像下面这样单独包装：
        if project_out:
            wrapped_linear = _maybe_wrap_linear(nn.Linear(inner_dim, dim), config, "attn")
            self.to_out = nn.Sequential(
                wrapped_linear,
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Identity()


    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    # <--- 修改：__init__ 需要接收 config
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., config: TransformerConfig = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # <--- 修改：将 config 传递下去
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, config=config),
                FeedForward(dim, mlp_dim, dropout = dropout, config=config)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    # <--- 修改：__init__ 增加一个 config 参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., final_norm_option_in_encoder='SimNorm', config: TransformerConfig = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # <--- 修改：将 config 传递给内部的 Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, config=config)

        self.pool = pool
        self.last_linear = nn.Linear(dim, num_classes)

        group_size = 8

        if final_norm_option_in_encoder == 'LayerNorm':
            self.final_norm = nn.LayerNorm(num_classes, eps=1e-5)
        elif final_norm_option_in_encoder == 'SimNorm':
            self.final_norm = SimNorm(simnorm_dim=group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {final_norm_option_in_encoder}")


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.last_linear(x)
        x = self.final_norm(x)

        return x



# --------------------------- 测试代码 --------------------------- #
if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    random.seed(42)
    model = ViT(
        image_size = 64,
        patch_size = 8,
        num_classes =768,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1,
        final_norm_option_in_encoder="LayerNorm"
    )
    model = model.cuda() if torch.cuda.is_available() else model
    dummy = torch.randn(256,3,64,64).to(next(model.parameters()).device)
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
