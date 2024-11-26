import triton
import torch
import numpy as np
import time
from typing import Tuple
import pytest

from lzero.mcts.trition_kernel.alphazero import ucb_score_kernel

def ucb_score_pytorch(
    parent_visit_count: torch.Tensor,
    child_visit_count: torch.Tensor,
    child_prior_p: torch.Tensor,
    child_value: torch.Tensor,
    pb_c_base: float,
    pb_c_init: float
) -> torch.Tensor:
    """PyTorch implementation for comparison"""
    pb_c = torch.log((parent_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c * torch.sqrt(parent_visit_count) / (child_visit_count + 1)
    prior_score = pb_c * child_prior_p
    return prior_score + child_value

def test_ucb_score():
    """Test the correctness and performance of UCB score implementations"""
    # Test parameters
    BATCH_SIZES = [1024, 4096, 16384, 65536]
    PB_C_BASE = 19652
    PB_C_INIT = 1.25
    BLOCK_SIZE = 1024
    NUM_WARMUP = 10
    NUM_REPEATS = 100

    for N in BATCH_SIZES:
        # Generate random test data
        torch.manual_seed(42)
        parent_visits = torch.randint(1, 100, (N,), device='cuda').float()
        child_visits = torch.randint(0, 50, (N,), device='cuda').float()
        prior_p = torch.rand(N, dtype=torch.float32, device='cuda')
        value = torch.randn(N, dtype=torch.float32, device='cuda')
        output_triton = torch.empty_like(prior_p)

        # CPU reference implementation
        parent_visits_cpu = parent_visits.cpu()
        child_visits_cpu = child_visits.cpu()
        prior_p_cpu = prior_p.cpu()
        value_cpu = value.cpu()
        
        # Compute reference result on CPU
        reference = ucb_score_pytorch(
            parent_visits_cpu, child_visits_cpu, prior_p_cpu, value_cpu,
            PB_C_BASE, PB_C_INIT
        )

        # Compute result using Triton
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        ucb_score_kernel[grid](
            parent_visits,
            child_visits,
            prior_p,
            value,
            output_triton,
            PB_C_BASE,
            PB_C_INIT,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Verify correctness
        torch.testing.assert_close(output_triton.cpu(), reference, rtol=1e-2, atol=1e-3)
        
        # Benchmark PyTorch CPU implementation
        torch.cuda.synchronize()
        cpu_times = []
        for _ in range(NUM_WARMUP):
            _ = ucb_score_pytorch(
                parent_visits_cpu, child_visits_cpu, prior_p_cpu, value_cpu,
                PB_C_BASE, PB_C_INIT
            )
        for _ in range(NUM_REPEATS):
            start = time.perf_counter()
            _ = ucb_score_pytorch(
                parent_visits_cpu, child_visits_cpu, prior_p_cpu, value_cpu,
                PB_C_BASE, PB_C_INIT
            )
            cpu_times.append(time.perf_counter() - start)
        
        # Benchmark PyTorch CUDA implementation
        cuda_times = []
        for _ in range(NUM_WARMUP):
            _ = ucb_score_pytorch(
                parent_visits, child_visits, prior_p, value,
                PB_C_BASE, PB_C_INIT
            )
        torch.cuda.synchronize()
        for _ in range(NUM_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = ucb_score_pytorch(
                parent_visits, child_visits, prior_p, value,
                PB_C_BASE, PB_C_INIT
            )
            end.record()
            torch.cuda.synchronize()
            cuda_times.append(start.elapsed_time(end) / 1000)  # Convert to seconds
        
        # Benchmark Triton implementation
        triton_times = []
        for _ in range(NUM_WARMUP):
            ucb_score_kernel[grid](
                parent_visits,
                child_visits,
                prior_p,
                value,
                output_triton,
                PB_C_BASE,
                PB_C_INIT,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        torch.cuda.synchronize()
        for _ in range(NUM_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ucb_score_kernel[grid](
                parent_visits,
                child_visits,
                prior_p,
                value,
                output_triton,
                PB_C_BASE,
                PB_C_INIT,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end) / 1000)  # Convert to seconds

        # Print performance results
        cpu_avg = np.mean(cpu_times) * 1000  # Convert to ms
        cuda_avg = np.mean(cuda_times) * 1000
        triton_avg = np.mean(triton_times) * 1000
        
        print(f"\nBatch size: {N}")
        print(f"{'Implementation':<15} {'Time (ms)':<10} {'Speedup vs CPU':<15}")
        print("-" * 40)
        print(f"{'CPU':<15} {cpu_avg:>10.3f} {1.0:>15.2f}x")
        print(f"{'CUDA':<15} {cuda_avg:>10.3f} {cpu_avg/cuda_avg:>15.2f}x")
        print(f"{'Triton':<15} {triton_avg:>10.3f} {cpu_avg/triton_avg:>15.2f}x")

if __name__ == "__main__":
    test_ucb_score()