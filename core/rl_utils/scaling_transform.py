import numpy as np
import torch


class DiscreteSupport(object):

    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.delta = delta


def scalar_transform(x, support_size, epsilon=0.001):
    """
    Overview:
        Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    scalar_support = DiscreteSupport(-support_size, support_size, delta=1)
    assert scalar_support.delta == 1
    # sign = torch.ones(x.shape).float().to(x.device)
    # sign[x < 0] = -1.0
    # output = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x

    output = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x

    # delta !=1
    # output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
    return output


def inverse_scalar_transform(logits, support_size, epsilon=0.001):
    """
    Overview:
        Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    scalar_support = DiscreteSupport(-support_size, support_size, delta=1)
    value_probs = torch.softmax(logits, dim=1)

    value_support = torch.from_numpy(scalar_support.range).unsqueeze(0)

    value_support = value_support.to(device=value_probs.device)
    value = (value_support * value_probs).sum(1, keepdim=True)

    output = torch.sign(value) * (
        ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
    )

    output[torch.abs(output) < epsilon] = 0.

    return output


def inverse_scalar_transform_old(logits, support_size, epsilon=0.001):
    """
    Overview:
        Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    scalar_support = DiscreteSupport(-support_size, support_size, delta=1)

    delta = scalar_support.delta
    value_probs = torch.softmax(logits, dim=1)
    value_support = torch.ones(value_probs.shape)
    value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))

    value_support = value_support.to(device=value_probs.device)
    value = (value_support * value_probs).sum(1, keepdim=True) / delta

    sign = torch.ones_like(value)
    sign[value < 0] = -1.0

    output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
    output = sign * output * delta

    output[torch.abs(output) < epsilon] = 0.

    return output


# import time
#
# new_time = 0
# old_time = 0
# for i in range(1000):
#     support_size = 300
#     logits = torch.randn([1024, 601])
#     st = time.time()
#     inverse_scalar_transform(logits, support_size)
#     et = time.time()
#
#     new_time += (et - st)
#     print(et - st)
#
#     st = time.time()
#     inverse_scalar_transform_old(logits, support_size)
#     et = time.time()
#     old_time += (et - st)
#     print(et - st)
#
#     print('----')
#
# print('new_time:', new_time)
# print('old_time:', old_time)
