import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from lzero.model.common import SimNorm

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., lora_config=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # LoRA 配置
        if lora_config is None:
            lora_config = {}
        
        self.lora_r = lora_config.get('r', 0)
        self.lora_alpha = lora_config.get('alpha', 1)
        self.lora_dropout_p = lora_config.get('dropout', 0.0)
        self.use_lora = self.lora_r > 0
        
        # LoRA 参数（如果启用）
        if self.use_lora:
            self.scaling = self.lora_alpha / self.lora_r
            self.lora_dropout = nn.Dropout(self.lora_dropout_p)
            
            # 为 q、k、v 分别创建 LoRA 参数
            self.lora_A_q = nn.Parameter(torch.randn(self.lora_r, dim) * 0.01)
            self.lora_B_q = nn.Parameter(torch.zeros(inner_dim, self.lora_r))
            
            self.lora_A_k = nn.Parameter(torch.randn(self.lora_r, dim) * 0.01)
            self.lora_B_k = nn.Parameter(torch.zeros(inner_dim, self.lora_r))
            
            self.lora_A_v = nn.Parameter(torch.randn(self.lora_r, dim) * 0.01)
            self.lora_B_v = nn.Parameter(torch.zeros(inner_dim, self.lora_r))

    def forward(self, x):
        x = self.norm(x)

        # 原有的预训练路径：获得 q、k、v
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 如果启用了 LoRA，添加 LoRA 贡献
        if self.use_lora:
            x_dropped = self.lora_dropout(x)
            
            # 计算每个分量的 LoRA 贡献
            lora_q = (x_dropped @ self.lora_A_q.T) @ self.lora_B_q.T  # (b, n, inner_dim)
            lora_k = (x_dropped @ self.lora_A_k.T) @ self.lora_B_k.T  # (b, n, inner_dim)
            lora_v = (x_dropped @ self.lora_A_v.T) @ self.lora_B_v.T  # (b, n, inner_dim)
            
            # 重排成多头格式：(b, n, inner_dim) -> (b, h, n, d)
            lora_q = rearrange(lora_q, 'b n (h d) -> b h n d', h = self.heads)
            lora_k = rearrange(lora_k, 'b n (h d) -> b h n d', h = self.heads)
            lora_v = rearrange(lora_v, 'b n (h d) -> b h n d', h = self.heads)
            
            # 加到对应的 q、k、v 上
            q = q + self.scaling * lora_q
            k = k + self.scaling * lora_k
            v = v + self.scaling * lora_v

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., lora_config=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, lora_config=lora_config),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., final_norm_option_in_encoder='SimNorm', lora_config=None):
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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, lora_config=lora_config)

        self.pool = pool
        self.last_linear = nn.Linear(dim, num_classes)

        group_size = 8

        # 最后归一化层，根据 final_norm_option_in_encoder 进行选择
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
    
    # 创建一个带 LoRA 的模型
    print("=== 创建 ViT with LoRA ===")
    lora_config = {
        'r': 16,
        'alpha': 32, 
        'dropout': 0.1
    }
    model = ViT(
        image_size = 64,
        patch_size = 8,
        num_classes = 768,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1,
        final_norm_option_in_encoder="LayerNorm",
        lora_config=lora_config
    )
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    dummy = torch.randn(256,3,64,64).to(next(model.parameters()).device)
    
    print("Total param count:", sum(p.numel() for p in model.parameters()))
    
    # 统计 LoRA 参数数量
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
    print("LoRA-only param count:", lora_params)
    
    # 测试关闭 LoRA (use_lora=False)
    print("\n=== 测试关闭 LoRA ===")
    for module in model.modules():
        if hasattr(module, 'use_lora'):
            module.use_lora = False
    
    with torch.no_grad():
        out_no_lora = model(dummy)
    print("No LoRA output shape:", out_no_lora.shape)
    
    # 测试开启 LoRA (use_lora=True)
    print("\n=== 测试开启 LoRA ===")
    for module in model.modules():
        if hasattr(module, 'use_lora'):
            module.use_lora = True
    
    with torch.no_grad():
        out_with_lora = model(dummy)
    print("With LoRA output shape:", out_with_lora.shape)
    
    # 验证初始时两个输出相近（因为 LoRA 的 B 矩阵初始化为 0）
    print("\nOutput difference (should be very small initially):")
    print("Max diff:", torch.max(torch.abs(out_no_lora - out_with_lora)).item())
    print("Mean diff:", torch.mean(torch.abs(out_no_lora - out_with_lora)).item())
    
    # 简单基准测试
    print("\n=== 性能测试 ===")
    import time
    warm, rep = 5, 20
    
    # 关闭 LoRA 测试
    for module in model.modules():
        if hasattr(module, 'use_lora'):
            module.use_lora = False
    
    for _ in range(warm): 
        with torch.no_grad(): out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0=time.time()
    for _ in range(rep):
        with torch.no_grad(): out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    no_lora_time = (time.time()-t0)/rep*1000
    print(f"No LoRA latency: {no_lora_time:.2f} ms")
    
    # 开启 LoRA 测试
    for module in model.modules():
        if hasattr(module, 'use_lora'):
            module.use_lora = True
    
    for _ in range(warm): 
        with torch.no_grad(): out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0=time.time()
    for _ in range(rep):
        with torch.no_grad(): out = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lora_time = (time.time()-t0)/rep*1000
    print(f"With LoRA latency: {lora_time:.2f} ms")
    print(f"LoRA Overhead: {((lora_time-no_lora_time)/no_lora_time*100):.1f}%")
