from typing import Union
import torch


class DiscreteSupport(object):

    def __init__(self, start: float, stop: float, step: float = 1., device: Union[str, torch.device] = 'cpu') -> None:
        assert start < stop
        self.arange = torch.arange(start, stop, step, dtype=torch.float32).unsqueeze(0).to(device)
        self.size = self.arange.shape[1]
        assert self.size > 0, "DiscreteSupport size must be greater than 0"
        self.step = step


def scalar_transform(x: torch.Tensor, epsilon: float = 0.001, delta: float = 1.) -> torch.Tensor:
    """
    Overview:
        Transform the original value to the scaled value, i.e. the h(.) function
        in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero: Appendix F: Network Architecture
        - https://arxiv.org/pdf/1805.11593.pdf (Page-11) Appendix A : Proposition A.2
    """
    # h(.) function
    if delta == 1:  # for speed up
        output = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
    else:
        # delta != 1
        output = torch.sign(x) * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
    return output


def inverse_scalar_transform(
        logits: torch.Tensor,
        scalar_support: DiscreteSupport,
        epsilon: float = 0.001,
        categorical_distribution: bool = True
) -> torch.Tensor:
    """
    Overview:
        transform the scaled value or its categorical representation to the original value,
        i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero Appendix F: Network Architecture.
        - https://arxiv.org/pdf/1805.11593.pdf Appendix A: Proposition A.2
    """
    if categorical_distribution:
        value_probs = torch.softmax(logits, dim=1)
        value_support = scalar_support.arange

        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True)
    else:
        value = logits

    # h^(-1)(.) function
    output = torch.sign(value) * (
        ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
    )

    return output


class InverseScalarTransform:
    """
    Overview:
        transform the the scaled value or its categorical representation to the original value,
        i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero Appendix F: Network Architecture.
        - https://arxiv.org/pdf/1805.11593.pdf Appendix A: Proposition A.2
    """

    def __init__(
            self,
            scalar_support: DiscreteSupport,
            categorical_distribution: bool = True
    ) -> None:
        self.value_support = scalar_support.arange
        self.categorical_distribution = categorical_distribution

    def __call__(self, logits: torch.Tensor, epsilon: float = 0.001) -> torch.Tensor:
        if self.categorical_distribution:
            value_probs = torch.softmax(logits, dim=1)
            value = value_probs.mul_(self.value_support).sum(1, keepdim=True)
        else:
            value = logits
        tmp = ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon))
        # t * t is faster than t ** 2
        output = torch.sign(value) * (tmp * tmp - 1)

        return output


def visit_count_temperature(
        manual_temperature_decay: bool, fixed_temperature_value: float,
        threshold_training_steps_for_final_lr_temperature: int, trained_steps: int
) -> float:
    if manual_temperature_decay:
        if trained_steps < 0.5 * threshold_training_steps_for_final_lr_temperature:
            return 1.0
        elif trained_steps < 0.75 * threshold_training_steps_for_final_lr_temperature:
            return 0.5
        else:
            return 0.25
    else:
        return fixed_temperature_value


def phi_transform(
    discrete_support: DiscreteSupport,
    x: torch.Tensor,
    label_smoothing_eps: float = 0.  # <--- 新增平滑参数
) -> torch.Tensor:
    """
    Overview:
        Map a real-valued scalar to a categorical distribution over a discrete support using linear interpolation (a.k.a. “soft” one-hot).

        For each scalar value the probability mass is split between the two
        nearest support atoms so that their weighted sum equals the original
        value (MuZero, Appendix F).

    Arguments:
        - discrete_support : DiscreteSupport
            Container with the support values (must be evenly spaced).
        - x : torch.Tensor
            Input tensor of arbitrary shape ``(...,)`` containing real numbers.

    Returns:
        - torch.Tensor
            Tensor of shape ``(*x.shape, N)`` where ``N = discrete_support.size``.
            The last dimension is a probability distribution (sums to 1).

    Notes
    -----
    • No in-place ops on the input are used, improving autograd safety.  
    • Only one `scatter_add_` kernel is launched for efficiency.  
    """
    # --- constants ----------------------------------------------------------
    min_bound = discrete_support.arange[0, 0]
    max_bound = discrete_support.arange[0, -1]
    step      = discrete_support.step
    size      = discrete_support.size

    # --- 1. clip to the valid range ----------------------------------------
    x = x.clamp(min_bound, max_bound)

    # --- 2. locate neighbouring indices ------------------------------------
    pos             = (x - min_bound) / step    # continuous position
    low_idx_float   = torch.floor(pos)          # lower index
    low_idx_long    = low_idx_float.long()      # lower index
    high_idx        = low_idx_long + 1          # upper index (may overflow)

    # --- 3. linear interpolation weights -----------------------------------
    p_high = pos - low_idx_float                # distance to lower atom
    p_low  = 1.0 - p_high                       # complementary mass

    # --- 4. stack indices / probs and scatter ------------------------------
    idx   = torch.stack([low_idx_long,
                         torch.clamp(high_idx, max=size - 1)], dim=-1)  # (*x, 2)
    prob  = torch.stack([p_low, p_high], dim=-1)                        # (*x, 2)

    target = torch.zeros(*x.shape, size,
                         dtype=x.dtype, device=x.device)

    target.scatter_add_(-1, idx, prob)

    # --- 5. 应用标签平滑 ---
    if label_smoothing_eps > 0:
        # 将原始的 two-hot 目标与一个均匀分布混合
        smooth_target = (1.0 - label_smoothing_eps) * target + (label_smoothing_eps / size)
        return smooth_target
    else:
        return target
    
    # return target


def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(torch.log_softmax(prediction, dim=1) * target).sum(1)
