"""
Example: Computing and plotting 2D loss landscape with ParaView export

This example demonstrates how to:
1. Load a pre-trained model
2. Create a dataloader
3. Compute 2D loss landscape
4. Visualize with matplotlib
5. Export to ParaView for high-quality rendering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

    # Use subset for faster computation (2D landscape is slower!)
    # For full dataset, remove this
    subset_indices = list(range(0, len(train_dataset), 10))  # Use every 10th sample
    subset_dataset = Subset(train_dataset, subset_indices)

    # Create dataloader
    train_loader = DataLoader(
        subset_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    print(f"Using {len(subset_dataset)} samples for faster computation")

    # ============================================================
    # Setup: Model
    # ============================================================
    print("\nLoading model...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.to(device)

    # ============================================================
    # Setup: Loss Function
    # ============================================================
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # Compute 2D Loss Landscape
    # ============================================================
    print("\n" + "=" * 60)
    print("Computing 2D Loss Landscape")
    print("=" * 60)
    print("NOTE: This may take a while (resolution: 21x21 for demo)")

    landscape = LossLandscape(
        model,
        train_loader,
        criterion=criterion,
        use_cuda=torch.cuda.is_available(),
        surf_file='loss_2d.h5'
    )

    # Compute 2D landscape
    # Use smaller resolution for demo (21x21 instead of 51x51)
    result = landscape.compute_2d(
        xrange=(-1, 1, 21),
        yrange=(-1, 1, 21),
        dir_type='weights',
        normalize='filter',
        ignore='biasbn',
        save=True
    )

    print("\n2D Loss Landscape Computation Complete!")
    print(f"  Shape: {result['losses'].shape}")
    print(f"  Min loss: {result['losses'].min():.4f}")
    print(f"  Max loss: {result['losses'].max():.4f}")
    print(f"  Min accuracy: {result['accuracies'].min():.1f}%")
    print(f"  Max accuracy: {result['accuracies'].max():.1f}%")

    # ============================================================
    # Matplotlib Visualization
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating Matplotlib Plots")
    print("=" * 60)

    try:
        landscape.plot_2d_contour(
            vmin=0.1,
            vmax=5,
            vlevel=0.5,
            show=False
        )
        print("✓ Contour plots saved")

        landscape.plot_2d_surface(show=False)
        print("✓ 3D surface plot saved")

    except Exception as e:
        print(f"Warning: Could not generate matplotlib plots: {e}")

    # ============================================================
    # ParaView Export (High-Quality Rendering)
    # ============================================================
    print("\n" + "=" * 60)
    print("Exporting to ParaView Format")
    print("=" * 60)

    try:
        vtp_file = landscape.export_paraview(
            surf_name='train_loss',
            log=False,
            zmax=-1,  # No clipping
            interp=-1  # No interpolation for now
        )
        print(f"\nParaView file ready: {vtp_file}")
        print("\nTo visualize in ParaView:")
        print(f"  1. Download ParaView: https://www.paraview.org/download/")
        print(f"  2. Open {vtp_file} in ParaView")
        print(f"  3. Use 'Surface' representation for best results")

    except Exception as e:
        print(f"Warning: Could not export to ParaView: {e}")

    # ============================================================
    # Results Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Surface file: {landscape.surf_file}")
    print(f"  Losses shape: {result['losses'].shape}")
    print(f"  Accuracies shape: {result['accuracies'].shape}")
    print(f"\nGenerated files:")
    print(f"  - loss_2d.h5 (HDF5 data)")
    print(f"  - loss_2d.h5_train_loss_2dcontour.pdf")
    print(f"  - loss_2d.h5_train_loss_2dcontourf.pdf")
    print(f"  - loss_2d.h5_train_loss_2dheat.pdf")
    print(f"  - loss_2d.h5_train_loss_3dsurface.pdf")
    print(f"  - loss_2d.h5_train_loss.vtp (for ParaView)")


if __name__ == '__main__':
    main()
