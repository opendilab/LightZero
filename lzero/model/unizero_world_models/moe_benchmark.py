# benchmark_moe.py
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------- 1. 原/新 MoE 实现（请提前 import）-------------
from moe import MoELayer, MoELayerOptimized

# ------------- 2. 辅助组件 -------------

@dataclass
class DummyCfg:
    embed_dim: int = 4096
    n_shared_experts: int = 1      # =0 代表不开 shared expert
    moe_use_lora: bool = False     # 只是占位，无实际作用

def make_experts(cfg: DummyCfg, num_experts: int) -> List[nn.Module]:
    """这里直接用一个两层 MLP 做 expert；也可以换成 MultiplicationFeedForward。"""
    return nn.ModuleList([
        nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim, bias=False),
        )
        for _ in range(num_experts)
    ])

class SimpleGate(nn.Module):
    """最朴素的门控：线性映射到 num_experts 维度。"""
    def __init__(self, cfg: DummyCfg, num_experts: int):
        super().__init__()
        self.proj = nn.Linear(cfg.embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

# ------------- 3. Benchmark / Correctness -------------

@torch.inference_mode()
def compare_outputs(layer1: nn.Module, layer2: nn.Module, x: torch.Tensor):
    """返回两层网络输出的平均|最大绝对误差"""
    y1 = layer1(x)
    y2 = layer2(x)
    diff = (y1 - y2).abs()
    return diff.mean().item(), diff.max().item()

@torch.inference_mode()
def measure_speed(layer: nn.Module, x: torch.Tensor, repeat: int = 20, warmup: int = 5):
    """返回每次 forward 的平均耗时（ms）"""
    device = x.device
    # ---- warm-up ----
    for _ in range(warmup):
        layer(x); torch.cuda.synchronize(device) if device.type == "cuda" else None
    # ---- timing ----
    t0 = time.perf_counter()
    for _ in range(repeat):
        layer(x)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / repeat  # ms

def main():
    # ----- 可根据显卡尺寸调整 B、T、E -----
    B, T = 8, 1024                # batch_size, sequence_len
    num_experts = 8
    k = 1                         # num_experts_per_tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    torch.manual_seed(42)

    cfg = DummyCfg()
    experts = make_experts(cfg, num_experts).to(device, dtype)
    gate    = SimpleGate(cfg, num_experts).to(device, dtype)

    original = MoELayer(cfg, experts, gate, num_experts_per_tok=k).to(device, dtype)
    optimized = MoELayerOptimized(cfg, experts, gate, num_experts_per_tok=k).to(device, dtype)

    # 随机输入
    x = torch.randn(B, T, cfg.embed_dim, device=device, dtype=dtype)

    # ---- 1) 检查数值一致性 ----
    mean_err, max_err = compare_outputs(original, optimized, x)
    print(f"[Correctness] mean_abs_err={mean_err:.3e}, max_abs_err={max_err:.3e}")

    # ---- 2) 速度对比 ----
    t_org = measure_speed(original, x)
    t_opt = measure_speed(optimized, x)
    print(f"[Speed] original={t_org:.2f} ms | optimized={t_opt:.2f} ms | speed-up x{t_org/t_opt:.2f}")

if __name__ == "__main__":
    main()