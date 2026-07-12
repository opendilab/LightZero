import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union, Optional
from transformers import AutoTokenizer
from dataclasses import is_dataclass
import os
import logging
import inspect
import textwrap


# ============================================================================
# Structured Logging Setup
# ============================================================================

def setup_priorzero_logging(exp_name: str, rank: int = 0) -> Dict[str, logging.Logger]:
    """
    Create structured loggers for PriorZero training.
    Only rank 0 gets console output and file handlers.
    Other ranks get NullHandler (silent).

    Returns dict with keys: 'main', 'train', 'eval'
    """
    log_dir = os.path.join(exp_name, "run_logs")
    os.makedirs(log_dir, exist_ok=True)

    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_fmt = logging.Formatter("[%(levelname).1s] %(message)s")

    loggers = {}
    for name, filename in [("main", "main.log"), ("train", "train.log"), ("eval", "eval.log")]:
        lg = logging.getLogger(f"priorzero.{name}")
        lg.setLevel(logging.DEBUG)
        lg.handlers.clear()
        lg.propagate = False

        if rank == 0:
            fh = logging.FileHandler(os.path.join(log_dir, filename), mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(file_fmt)
            lg.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(console_fmt)
            lg.addHandler(ch)
        else:
            lg.addHandler(logging.NullHandler())

        loggers[name] = lg

    # Error log: captures WARNING+ from all priorzero loggers
    if rank == 0:
        err_handler = logging.FileHandler(os.path.join(log_dir, "error.log"), mode="a")
        err_handler.setLevel(logging.WARNING)
        err_handler.setFormatter(file_fmt)
        for lg in loggers.values():
            lg.addHandler(err_handler)

    return loggers

def dump_dataclass_cfg_py(cfg, path: str) -> str:
    if not is_dataclass(cfg):
        raise TypeError(type(cfg))

    def norm(x):
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if hasattr(x, "__class__") and x.__class__.__name__ == "EasyDict":
            return {k: norm(v) for k, v in dict(x).items()}
        if isinstance(x, (list, tuple)):
            t = [norm(v) for v in x]
            return tuple(t) if isinstance(x, tuple) else t
        return x
    cls = type(cfg)
    fields = cls.__dataclass_fields__.keys()
    lines = [f"{k} = {repr(norm(getattr(cfg, k)))}" for k in fields] + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return 

def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels



import time
from contextlib import contextmanager
from collections import defaultdict

class Profiler:
    def __init__(self, log_interval: int = 10, stats_file: str = None, enable_profile: bool = False):
        self.log_interval = max(1, int(log_interval))
        self.stats_file = stats_file
        self.stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
        self._inited = False
        self.enable_profile = enable_profile
        
    def _init_once(self):
        if self._inited:
            return
        with open(self.stats_file, "a", encoding="utf-8") as f:
            f.write("ts\tname\tcount\ttotal_s\tavg_s\tmax_s\n")
        self._inited = True

    def _record(self, name: str, elapsed: float):
        s = self.stats[name]
        s["count"] += 1
        s["total"] += elapsed
        s["max"] = max(s["max"], elapsed)
        if s["count"] % self.log_interval == 0:
            avg = s["total"] / s["count"]
            with open(self.stats_file, "a", encoding="utf-8") as f:
                f.write(f"{time.time():.3f}\t{name}\t{s['count']}\t{s['total']:.6f}\t{avg:.6f}\t{s['max']:.6f}\n")

    @contextmanager
    def block(self, name: str, rank: int = 0):
        if not self.enable_profile or rank != 0:
            yield None
            return
        self._init_once()
        t0 = time.perf_counter()
        try:
            yield None
        finally:
            self._record(name, time.perf_counter() - t0)