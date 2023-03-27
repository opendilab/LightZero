import numpy as np
import torch


class DiscreteSupport(object):

    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.set_size = len(self.range)
        self.delta = delta


def scalar_transform(x, epsilon=0.001, delta=1):
    """
    Overview:
        transform the original value to the scaled value, i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero: Appendix F: Network Architecture
        - https://arxiv.org/pdf/1805.11593.pdf (Page-11) Appendix A : Proposition A.2
    """
    # h(.) function
    if delta == 1:
        output = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
    else:
        # delta != 1
        output = torch.sign(x) * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
    return output


def inverse_scalar_transform(logits, support_size, epsilon=0.001, categorical_distribution=True):
    """
    Overview:
        transform the the scaled value or its categorical representation to the original value,
        i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero Appendix F: Network Architecture.
        - https://arxiv.org/pdf/1805.11593.pdf Appendix A: Proposition A.2
    """
    if categorical_distribution:
        scalar_support = DiscreteSupport(-support_size, support_size, delta=1)
        value_probs = torch.softmax(logits, dim=1)

        value_support = torch.from_numpy(scalar_support.range).unsqueeze(0)

        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True)
    else:
        value = logits

    # h^(-1)(.) function
    output = torch.sign(value) * (
        ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
    )

    # TODO(pu): save time
    # output[torch.abs(output) < epsilon] = 0.

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
    def __init__(self, support_size, device='cpu', categorical_distribution=True):
        scalar_support = DiscreteSupport(-support_size, support_size, delta=1)
        self.value_support = torch.from_numpy(scalar_support.range).unsqueeze(0)
        self.value_support = self.value_support.to(device)
        self.categorical_distribution = categorical_distribution

    def __call__(self, logits, epsilon=0.001):
        if self.categorical_distribution:
            value_probs = torch.softmax(logits, dim=1)
            value = value_probs.mul_(self.value_support).sum(1, keepdim=True)
        else:
            value = logits
        tmp = ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon))
        # t * t is faster than t ** 2
        output = torch.sign(value) * (tmp * tmp - 1)

        return output


def visit_count_temperature(manual_temperature_decay, fixed_temperature_value, threshold_training_steps_for_final_lr_temperature, trained_steps):
    if manual_temperature_decay:
        if trained_steps < 0.5 * threshold_training_steps_for_final_lr_temperature:
            return 1.0
        elif trained_steps < 0.75 * threshold_training_steps_for_final_lr_temperature:
            return 0.5
        else:
            return 0.25
    else:
        return fixed_temperature_value


def phi_transform(discrete_support, x):
    """
    Overview:
        We then apply a transformation ``phi`` to the scalar in order to obtain equivalent categorical representations.
         After this transformation, each scalar is represented as the linear combination of its two adjacent supports.
    Reference:
        - MuZero paper Appendix F: Network Architecture.
    """
    min = discrete_support.min
    max = discrete_support.max
    set_size = discrete_support.set_size
    delta = discrete_support.delta

    x.clamp_(min, max)
    x_low = x.floor()
    x_high = x.ceil()
    p_high = x - x_low
    p_low = 1 - p_high

    target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
    x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
    target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))

    return target


def cross_entropy_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=1) * target).sum(1)
