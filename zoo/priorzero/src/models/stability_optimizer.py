import logging
from collections import deque
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


class AdaptiveValueNormalizer:
    """
    作用：把 value/return/advantage 变成稳定尺度（近似零均值、单位方差），并支持 soft(log-sym)/hard(percentile) 抑制极端值。
    核心：batch 统计（只看当前） + EMA 运行统计（全局追踪非平稳） + 可选裁剪/压缩。
    """

    def __init__(
        self,
        init_momentum: float = 0.9,
        final_momentum: float = 0.99,
        warmup_steps: int = 100,
        clip_method: str = "soft",          # "soft" | "hard" | "none"
        clip_percentile: float = 0.95,      # hard clip 中间保留比例，如 0.95 => 保留 [2.5%, 97.5%]
        min_std: float = 1e-6,
        hard_clip_start_updates: int = 10,  # hard clip 前几次不启用
        history_size: int = 1000,
    ):
        self.init_momentum = init_momentum
        self.final_momentum = final_momentum
        self.warmup_steps = warmup_steps
        self.clip_method = clip_method
        self.clip_percentile = clip_percentile
        self.min_std = min_std
        self.hard_clip_start_updates = hard_clip_start_updates

        self.running_mean = 0.0
        self.running_std = 1.0
        self.update_count = 0

        self.value_history = deque(maxlen=history_size)

    def _momentum(self) -> float:
        if self.update_count >= self.warmup_steps:
            return self.final_momentum
        p = self.update_count / max(self.warmup_steps, 1)
        return self.init_momentum + (self.final_momentum - self.init_momentum) * p

    @staticmethod
    def _log_sym(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # f(x)=sign(x)*log(1+|x|)
        significant = int((x.abs() > 10).sum())
        y = torch.sign(x) * torch.log1p(torch.abs(x))
        return y, significant

    def _hard_percentile_clip(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.update_count < self.hard_clip_start_updates:
            return x, 0
        q = self.clip_percentile
        lo = (1 - q) / 2
        hi = 1 - lo

        xf = x.flatten()
        lb = torch.quantile(xf, lo)
        ub = torch.quantile(xf, hi)
        y = torch.clamp(x, lb, ub)

        clipped = int((y != x).sum())
        return y, clipped

    def _batch_mean_std(self, x: torch.Tensor) -> Tuple[float, float]:
        xf = x.flatten()
        n = xf.numel()
        if n == 0:
            return 0.0, 1.0
        if n == 1:
            mean = float(xf.item())
            return mean, self.min_std

        xf64 = xf.to(torch.float64)
        mean = float(xf64.mean().item())
        var = float(xf64.var(unbiased=True).item())
        std = max(var ** 0.5, self.min_std)
        return mean, std

    def normalize(
        self,
        values: torch.Tensor,
        clip_values: bool = True,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        x = values.detach()

        clipped_count = 0
        if clip_values:
            if self.clip_method == "soft":
                x, clipped_count = self._log_sym(x)
            elif self.clip_method == "hard":
                x, clipped_count = self._hard_percentile_clip(x)
            else:
                raise ValueError(f"Unknown clip_method: {self.clip_method}")

        batch_mean, batch_std = self._batch_mean_std(x)

        m = self._momentum()
        if self.update_count == 0:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            self.running_mean = m * self.running_mean + (1 - m) * batch_mean
            self.running_std = m * self.running_std + (1 - m) * batch_std

        self.update_count += 1
        self.value_history.extend(x.flatten().float().cpu().tolist())


        y = (x.to(values.dtype) - self.running_mean) / (self.running_std + self.min_std)

        if not return_stats:
            return y

        stats = {
            "batch_mean": batch_mean,
            "batch_std": batch_std,
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "momentum": m,
            "clip_method": self.clip_method,
            "clipped_count": clipped_count,
            "total_count": int(x.numel()),
        }
        return y, stats

    def summary(self) -> Dict:
        if self.update_count == 0:
            return {}
        recent = list(self.value_history)[-min(100, len(self.value_history)) :]
        return {
            "total_updates": self.update_count,
            "current_mean": float(self.running_mean),
            "current_std": float(self.running_std),
            "recent_mean": float(np.mean(recent)) if recent else 0.0,
            "recent_std": float(np.std(recent)) if recent else 1.0,
            "recent_min": float(np.min(recent)) if recent else 0.0,
            "recent_max": float(np.max(recent)) if recent else 0.0,
            "clip_method": self.clip_method,
        }
    
    def clear(self):
        """
        Reset all running statistics so the normalizer behaves like a fresh instance.
        Useful when starting a new experiment or episode.
        """
        self.running_mean = 0.0
        self.running_std = 1.0
        self.update_count = 0

        self.value_history.clear()

