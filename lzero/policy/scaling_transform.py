from typing import Union
import numpy as np
import torch


class DiscreteSupport(object):

    def __init__(self, start: float, stop: float, step: float = 1., device: Union[str, torch.device] = 'cpu') -> None:
        assert start < stop
        self.arange = torch.arange(start, stop, step).unsqueeze(0).to(device)
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


def ensure_softmax(logits, dim=1):
    """
    Overview:
        Ensure that the input tensor is normalized along the specified dimension.
    Arguments:
         - logits (:obj:`torch.Tensor`): The input tensor.
        - dim (:obj:`int`): The dimension along which to normalize the input tensor.
    Returns:
        - output (:obj:`torch.Tensor`): The normalized tensor.
    """
    # Calculate the sum along the specified dimension (dim=1 in this case)
    sum_along_dim = logits.sum(dim=dim, keepdim=True)
    
    # Create a tensor of ones with the same shape as sum_along_dim
    ones_like_sum = torch.ones_like(sum_along_dim)
    
    # Check if the logits are already normalized (i.e., if the sum along the dimension is approximately 1)
    # torch.allclose checks if all elements of two tensors are close within a tolerance
    # atol (absolute tolerance) is set to a small value to allow for numerical precision issues
    is_normalized = torch.allclose(sum_along_dim, ones_like_sum, atol=1e-5)
    
    # If logits are not normalized, apply softmax along the specified dimension
    if not is_normalized:
        return torch.softmax(logits, dim=dim)
    else:
        # If logits are already normalized, return them as they are
        return logits


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
        value_probs = ensure_softmax(logits, dim=1)
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
            value_probs = ensure_softmax(logits, dim=1)
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


def phi_transform(discrete_support: DiscreteSupport, x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        We then apply a transformation ``phi`` to the scalar in order to obtain equivalent categorical representations.
         After this transformation, each scalar is represented as the linear combination of its two adjacent supports.
    Reference:
        - MuZero paper Appendix F: Network Architecture.
    """
    arange = discrete_support.arange
    min_bound = arange[0, 0]
    max_bound = arange[0, -1]
    size = discrete_support.size
    step = discrete_support.step

    x.clamp_(min_bound, max_bound)
    x_low_idx = ((x - min_bound)/step).floor().long()
    x_high_idx = x_low_idx + 1
    x_high_idx = x_high_idx.clamp(0, size - 1)
    x_low_val = torch.take_along_dim(arange, x_low_idx, dim=1)
    p_high = (x - x_low_val)/step
    p_low = 1 - p_high

    target = torch.zeros(x.shape[0], x.shape[1], size).to(x.device)
    target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))

    return target


def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(torch.log_softmax(prediction, dim=1) * target).sum(1)
