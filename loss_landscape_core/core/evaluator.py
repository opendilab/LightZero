"""
Model evaluation utilities for loss and accuracy computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


def eval_loss(net, criterion, loader, use_cuda=False):
    """Evaluate loss and accuracy on a dataset.

    Args:
        net: PyTorch model
        criterion: Loss function (CrossEntropyLoss or MSELoss)
        loader: DataLoader
        use_cuda: Whether to use CUDA

    Returns:
        Tuple of (average_loss, accuracy)
    """
    correct = 0
    total_loss = 0.0
    total = 0

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size

                inputs = Variable(inputs)
                targets = Variable(targets)

                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size

                inputs = Variable(inputs)

                # Convert targets to one-hot encoding
                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)

                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()

                outputs = F.softmax(net(inputs), dim=1)
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
