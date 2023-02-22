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
        h(.) function
        Reference:
            MuZero: Appendix F: Network Architecture
            https://arxiv.org/pdf/1805.11593.pdf (Page-11) Appendix A : Proposition A.2
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
        h^(-1)(.) function
        Reference:
            MuZero: Appendix F: Network Architecture
            https://arxiv.org/pdf/1805.11593.pdf (Page-11) Appendix A : Proposition A.2 (iii)
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


def visit_count_temperature(auto_temperature, fixed_temperature_value, max_training_steps, trained_steps):
    if auto_temperature:
        if trained_steps < 0.5 * max_training_steps:
            return 1.0
        elif trained_steps < 0.75 * max_training_steps:
            return 0.5
        else:
            return 0.25
    else:
        return fixed_temperature_value


def modified_cross_entropy_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


def value_phi(value_support, x):
    return _phi(value_support, x)


def reward_phi(reward_support, x):
    return _phi(reward_support, x)


def _phi(discrete_support, x):
    """
    Overview:
        Under this transformation, each scalar is represented as the linear combination of its two adjacent supports,
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


if __name__ == "__main__":
    import time
    logit = torch.randn(16, 601)
    start = time.time()
    inverse_scalar_transform(logit, 300)
    print('t1', time.time() - start)
    logit = torch.randn(16, 601)
    handle = InverseScalarTransform(300)
    start = time.time()
    handle(logit)
    print('t2', time.time() - start)
