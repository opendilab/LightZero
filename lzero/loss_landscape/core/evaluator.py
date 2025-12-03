"""
Overview:
    Model evaluation utilities for computing loss and accuracy metrics during landscape exploration.
    Provides unified interface for evaluating models with different loss functions.

This module provides:
    - Loss and accuracy computation on datasets
    - Support for both CrossEntropyLoss and MSELoss
    - GPU acceleration via CUDA
    - Batch-wise evaluation with no gradient tracking

Key Functions:
    - eval_loss: Evaluate model on entire dataset, return loss and accuracy metrics

Notes:
    - Used during landscape computation to evaluate loss at each point
    - Evaluation is performed without gradient tracking (torch.no_grad) for efficiency
    - Batch-wise loss is accumulated and averaged over full dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from typing import Tuple


def eval_loss(net: nn.Module, criterion: nn.Module, loader: DataLoader, use_cuda: bool = False) -> Tuple[float, float]:
    """
    Overview:
        Evaluate model loss and accuracy on a complete dataset.
        Performs inference without gradient tracking for efficiency.

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model to evaluate (will be set to eval mode)
        - criterion (:obj:`nn.Module`): Loss function for evaluation
            - Supports CrossEntropyLoss (for classification)
            - Supports MSELoss (for regression with one-hot encoding)
        - loader (:obj:`DataLoader`): DataLoader providing (inputs, targets) batches
        - use_cuda (:obj:`bool`, optional): Transfer model and data to GPU. Default is False.

    Returns:
        - avg_loss (:obj:`float`): Average loss across all samples in dataset
        - accuracy (:obj:`float`): Top-1 accuracy as percentage (0-100) for classification

    Notes:
        - Model is set to eval mode, disabling dropout and batch norm updates
        - No gradient computation (torch.no_grad) for memory/speed efficiency
        - For CrossEntropyLoss: computes accuracy from argmax predictions
        - For MSELoss: converts targets to one-hot encoding (assumes 10 classes)
        - Loss is properly weighted by batch size to handle variable-size batches
        - GPU memory freed after evaluation if CUDA is used

    Shapes:
        - inputs (:obj:`torch.Tensor`): :math:`(B, C, H, W)` for images or :math:`(B, D)` for vectors
        - targets (:obj:`torch.Tensor`): :math:`(B,)` class indices or one-hot encoded

    Examples::
        >>> model = torchvision.models.resnet18()
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>> loss, acc = eval_loss(model, criterion, dataloader, use_cuda=True)
        >>> print(f"Loss: {loss:.4f}, Accuracy: {acc:.1f}%")
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
