"""
Example: Computing and plotting 1D loss landscape

This example demonstrates how to:
1. Load a pre-trained model
2. Create a dataloader
3. Compute 1D loss landscape along a random direction
4. Visualize the results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape_core import LossLandscape


def main():
    # ============================================================
    # Setup: Device and DataLoader
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a simple transform (no augmentation for loss evaluation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Load CIFAR-10
    print("\nDownloading CIFAR-10...")
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # Create dataloader (use subset for faster computation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # ============================================================
    # Setup: Model
    # ============================================================
    print("\nLoading model...")
    # Use a small model for demo (ResNet18)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.to(device)

    # ============================================================
    # Setup: Loss Function
    # ============================================================
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # Compute 1D Loss Landscape
    # ============================================================
    print("\n" + "=" * 60)
    print("Computing 1D Loss Landscape")
    print("=" * 60)

    landscape = LossLandscape(
        model,
        train_loader,
        criterion=criterion,
        use_cuda=torch.cuda.is_available(),
        surf_file='loss_1d.h5'
    )

    # Compute landscape along a random direction
    result = landscape.compute_1d(
        directions='random',
        xrange=(-1, 1, 51),  # Sample 51 points from -1 to 1
        dir_type='weights',
        normalize='filter',
        ignore='biasbn',
        save=True
    )

    print("\n1D Loss Landscape Computation Complete!")
    print(f"  Min loss: {result['losses'].min():.4f}")
    print(f"  Max loss: {result['losses'].max():.4f}")
    print(f"  Min accuracy: {result['accuracies'].min():.1f}%")
    print(f"  Max accuracy: {result['accuracies'].max():.1f}%")

    # ============================================================
    # Visualize Results
    # ============================================================
    print("\n" + "=" * 60)
    print("Plotting Results")
    print("=" * 60)

    try:
        landscape.plot_1d(
            xmin=-1,
            xmax=1,
            loss_max=5,
            log=False,
            show=False
        )
        print("\nPlots saved!")
        print("  loss_1d.h5_1d_loss_acc.pdf")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # ============================================================
    # Save and Display Results
    # ============================================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Surface file: {landscape.surf_file}")
    print(f"  Losses shape: {result['losses'].shape}")
    print(f"  Accuracies shape: {result['accuracies'].shape}")


if __name__ == '__main__':
    main()
