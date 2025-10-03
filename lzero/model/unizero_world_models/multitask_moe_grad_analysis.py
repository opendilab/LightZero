"""
Multi-task Mixture of Experts (MoE) with Gradient Conflict Analysis

OVERVIEW:
This file implements a comprehensive analysis framework for comparing multi-task learning approaches,
specifically Sparse Gating Networks (Mixture of Experts) versus traditional Multi-Layer Perceptrons,
with a focus on gradient conflict analysis and expert utilization patterns.

KEY FEATURES:
1. Toy Dataset Generation: Creates synthetic 2D multi-task problems with controllable complexity
2. Gradient Analysis Tools: Computes gradient steepness, direction, and conflict metrics
3. Sparse Gating Network: Implements MoE with load balancing and expert selection tracking
4. Gradient Conflict Metrics: Measures inter-task gradient conflicts using cosine similarity
5. Expert Behavior Analysis: Tracks expert specialization, usage patterns, and spatial preferences
6. Comprehensive Visualization: Generates plots for gradient landscapes, expert patterns, and conflicts
7. Comparative Evaluation: Side-by-side comparison of MoE vs MLP performance and behaviors

PURPOSE:
- Research tool for understanding multi-task learning dynamics in MoE architectures
- Investigate how sparse gating helps mitigate gradient conflicts between tasks
- Analyze expert specialization patterns and load balancing effectiveness
- Provide insights into when and why MoE outperforms traditional approaches

MAIN COMPONENTS:
- ToyTaskDataset: Synthetic multi-task dataset with two related but conflicting objectives
- SparseGatingNetwork: MoE implementation with configurable experts and top-k routing
- Gradient conflict computation functions for both model-level and expert-level analysis
- Visualization tools for gradient landscapes, expert selection patterns, and training dynamics
- Experimental framework with comprehensive metrics and comparative analysis

USAGE:
Run the main experiment with: python multitask_moe_grad_analysis.py
This will train both MoE and MLP models, analyze their behaviors, and generate comparison plots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


LOWER = 0.000005  # Numerical stability constant to prevent log(0) errors in loss computation

# Global visualization hyperparameter - change this to adjust all visualizations
VISUALIZATION_RESOLUTION = 16  # Grid resolution for 2D plots (16x16 = 256 points)

class ToyTaskDataset:
    """
    Synthetic 2D multi-task dataset generator for gradient conflict analysis.
    
    Creates two related but potentially conflicting tasks over a 2D input space.
    The tasks are designed to have different optimal solutions in different regions,
    making them ideal for studying gradient conflicts in multi-task learning.
    
    Args:
        num_samples (int): Number of data points to generate. Defaults to 10000.
        x_range (tuple): Input range for both dimensions (min, max). Defaults to (-10, 10).
    
    Attributes:
        num_samples (int): Number of samples to generate.
        x_range (tuple): Input space range for x1 and x2 dimensions.
    """
    def __init__(self, num_samples=10000, x_range=(-10, 10)):
        self.num_samples = num_samples
        self.x_range = x_range
        
    def generate_data(self):
        """
        Generate random 2D input points and compute corresponding multi-task targets.
        
        Returns:
            tuple: (X, Y) where:
                - X (torch.Tensor): Input data of shape [num_samples, 2]
                - Y (torch.Tensor): Target data of shape [num_samples, 2] (task1, task2)
        """
        # Generate random 2D points
        x1 = torch.FloatTensor(self.num_samples).uniform_(*self.x_range)
        x2 = torch.FloatTensor(self.num_samples).uniform_(*self.x_range)
        X = torch.stack([x1, x2], dim=1)
        
        # Compute target values using toy problem functions
        Y = self._compute_targets(X)
        return X, Y
    
    def _compute_targets(self, X):
        """
        Compute target values for two conflicting tasks based on input coordinates.
        
        Args:
            X (torch.Tensor): Input coordinates of shape [batch_size, 2]
            
        Returns:
            torch.Tensor: Target values of shape [batch_size, 2] for tasks 1 and 2
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        # Task 1: f1 computation
        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)
        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)
        f1 = f1 * c1 + f1_sq * c2
        
        # Task 2: f2 computation  
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2 = f2 * c1 + f2_sq * c2
        
        return torch.stack([f1, f2], dim=1)


def compute_gradient_steepness_map(x_range=(-10, 10), resolution=VISUALIZATION_RESOLUTION):
    """
    Compute gradient magnitude (steepness) maps for both tasks over 2D input space.
    
    Args:
        x_range (tuple): Input range (min, max). Defaults to (-10, 10).
        resolution (int): Grid resolution. Defaults to VISUALIZATION_RESOLUTION.
    
    Returns:
        tuple: (steepness_task1, steepness_task2, x1_grid, x2_grid)
    """
    # Create coordinate grids
    x1_coords = np.linspace(x_range[0], x_range[1], resolution)
    x2_coords = np.linspace(x_range[0], x_range[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1_coords, x2_coords)
    
    # Flatten for computation
    x1_flat = x1_grid.flatten()
    x2_flat = x2_grid.flatten()
    
    # Convert to torch tensors and enable gradient computation
    x1_tensor = torch.tensor(x1_flat, dtype=torch.float32, requires_grad=True)
    x2_tensor = torch.tensor(x2_flat, dtype=torch.float32, requires_grad=True)
    X = torch.stack([x1_tensor, x2_tensor], dim=1)
    
    # Create dataset instance to use _compute_targets method
    dataset = ToyTaskDataset()
    
    # Compute target values
    Y = dataset._compute_targets(X)  # [N, 2] where N = resolution^2
    
    # Initialize steepness arrays
    steepness_task1 = np.zeros(resolution * resolution)
    steepness_task2 = np.zeros(resolution * resolution)
    
    # Compute gradients for each point
    for i in range(resolution * resolution):
        # Clear gradients
        if x1_tensor.grad is not None:
            x1_tensor.grad.zero_()
        if x2_tensor.grad is not None:
            x2_tensor.grad.zero_()
        
        # Task 1 gradient
        task1_output = Y[i, 0]
        task1_output.backward(retain_graph=True)
        
        # Gradient extraction with None handling - POTENTIAL BIAS WARNING:
        # Current approach: Set gradient to 0 if None to prevent crashes
        # Rationale: Ensures computational stability and avoids NoneType errors
        # Potential bias: This zero-filling may artificially inflate cosine similarity between tasks
        # - If both tasks have None gradients at the same point, they appear "similar" (both zeros)
        # - Zero-filling reduces gradient magnitude, potentially skewing angle statistics
        # - May lead to overestimated gradient alignment (cos_sim closer to 1.0 than reality)
        # Alternative approaches: Skip points with None gradients or use gradient estimation
        grad_x1_task1 = x1_tensor.grad[i].item() if x1_tensor.grad is not None else 0
        grad_x2_task1 = x2_tensor.grad[i].item() if x2_tensor.grad is not None else 0
        steepness_task1[i] = np.sqrt(grad_x1_task1**2 + grad_x2_task1**2)
        
        # Clear gradients for task 2
        x1_tensor.grad.zero_()
        x2_tensor.grad.zero_()
        
        # Task 2 gradient
        task2_output = Y[i, 1]
        task2_output.backward(retain_graph=True)
        
        # Same bias concerns apply to task 2 gradient extraction
        grad_x1_task2 = x1_tensor.grad[i].item() if x1_tensor.grad is not None else 0
        grad_x2_task2 = x2_tensor.grad[i].item() if x2_tensor.grad is not None else 0
        steepness_task2[i] = np.sqrt(grad_x1_task2**2 + grad_x2_task2**2)
    
    # Reshape back to 2D grids
    steepness_task1 = steepness_task1.reshape(resolution, resolution)
    steepness_task2 = steepness_task2.reshape(resolution, resolution)
    
    return steepness_task1, steepness_task2, x1_grid, x2_grid


def compute_gradient_direction_cosine_map(x_range=(-10, 10), resolution=VISUALIZATION_RESOLUTION):
    """
    Compute gradient direction maps using cosine similarity with x1 axis.
    
    Args:
        x_range (tuple): Input range. Defaults to (-10, 10).
        resolution (int): Grid resolution. Defaults to VISUALIZATION_RESOLUTION.
    
    Returns:
        tuple: (cosine_task1, cosine_task2, cosine_combined, x1_grid, x2_grid)
    """
    # Create coordinate grids
    x1_coords = np.linspace(x_range[0], x_range[1], resolution)
    x2_coords = np.linspace(x_range[0], x_range[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1_coords, x2_coords)
    
    # Flatten for computation
    x1_flat = x1_grid.flatten()
    x2_flat = x2_grid.flatten()
    
    # Convert to torch tensors and enable gradient computation
    x1_tensor = torch.tensor(x1_flat, dtype=torch.float32, requires_grad=True)
    x2_tensor = torch.tensor(x2_flat, dtype=torch.float32, requires_grad=True)
    X = torch.stack([x1_tensor, x2_tensor], dim=1)
    
    # Create dataset instance to use _compute_targets method
    dataset = ToyTaskDataset()
    
    # Compute target values
    Y = dataset._compute_targets(X)  # [N, 2] where N = resolution^2
    
    # Initialize cosine similarity arrays
    cosine_task1 = np.zeros(resolution * resolution)
    cosine_task2 = np.zeros(resolution * resolution)
    cosine_combined = np.zeros(resolution * resolution)
    
    # Compute gradients for each point
    for i in range(resolution * resolution):
        # Clear gradients
        if x1_tensor.grad is not None:
            x1_tensor.grad.zero_()
        if x2_tensor.grad is not None:
            x2_tensor.grad.zero_()
        
        # Task 1 gradient
        task1_output = Y[i, 0]
        task1_output.backward(retain_graph=True)
        
        # Gradient extraction with None handling - POTENTIAL BIAS WARNING:
        # Current approach: Set gradient to 0 if None to prevent crashes
        # Rationale: Ensures computational stability and avoids NoneType errors
        # Potential bias: This zero-filling may artificially inflate cosine similarity between tasks
        # - If both tasks have None gradients at the same point, they appear "similar" (both zeros)
        # - Zero-filling reduces gradient magnitude, potentially skewing angle statistics
        # - May lead to overestimated gradient alignment (cos_sim closer to 1.0 than reality)
        # Alternative approaches: Skip points with None gradients or use gradient estimation
        grad_x1_task1 = x1_tensor.grad[i].item() if x1_tensor.grad is not None else 0
        grad_x2_task1 = x2_tensor.grad[i].item() if x2_tensor.grad is not None else 0
        
        # Cosine similarity with x1 axis: cos(θ) = grad_x1 / ||grad||
        grad_magnitude_task1 = np.sqrt(grad_x1_task1**2 + grad_x2_task1**2)
        if grad_magnitude_task1 > 1e-8:
            cosine_task1[i] = grad_x1_task1 / grad_magnitude_task1
        else:
            cosine_task1[i] = 0  # undefined gradient direction
        
        # Clear gradients for task 2
        x1_tensor.grad.zero_()
        x2_tensor.grad.zero_()
        
        # Task 2 gradient
        task2_output = Y[i, 1]
        task2_output.backward(retain_graph=True)
        
        # Same bias concerns apply to task 2 gradient extraction
        grad_x1_task2 = x1_tensor.grad[i].item() if x1_tensor.grad is not None else 0
        grad_x2_task2 = x2_tensor.grad[i].item() if x2_tensor.grad is not None else 0
        
        # Cosine similarity with x1 axis for task 2
        grad_magnitude_task2 = np.sqrt(grad_x1_task2**2 + grad_x2_task2**2)
        if grad_magnitude_task2 > 1e-8:
            cosine_task2[i] = grad_x1_task2 / grad_magnitude_task2
        else:
            cosine_task2[i] = 0
        
        # Clear gradients for combined task
        x1_tensor.grad.zero_()
        x2_tensor.grad.zero_()
        
        # Combined task gradient (sum of both tasks)
        combined_output = Y[i, 0] + Y[i, 1]
        combined_output.backward(retain_graph=True)
        
        # Same bias concerns apply to combined task gradient extraction
        grad_x1_combined = x1_tensor.grad[i].item() if x1_tensor.grad is not None else 0
        grad_x2_combined = x2_tensor.grad[i].item() if x2_tensor.grad is not None else 0
        
        # Cosine similarity with x1 axis for combined task
        grad_magnitude_combined = np.sqrt(grad_x1_combined**2 + grad_x2_combined**2)
        if grad_magnitude_combined > 1e-8:
            cosine_combined[i] = grad_x1_combined / grad_magnitude_combined
        else:
            cosine_combined[i] = 0
    
    # Reshape back to 2D grids
    cosine_task1 = cosine_task1.reshape(resolution, resolution)
    cosine_task2 = cosine_task2.reshape(resolution, resolution)
    cosine_combined = cosine_combined.reshape(resolution, resolution)
    
    return cosine_task1, cosine_task2, cosine_combined, x1_grid, x2_grid


def plot_gradient_steepness_analysis(save_path='gradient_steepness_analysis.png'):
    """Plot gradient steepness maps for both tasks."""
    steepness_task1, steepness_task2, x1_grid, x2_grid = compute_gradient_steepness_map(resolution=VISUALIZATION_RESOLUTION)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Task 1 steepness
    im1 = axes[0].imshow(steepness_task1, cmap='viridis', aspect='auto', 
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest')
    axes[0].set_title('Task 1 Gradient Steepness')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_xticks([-10, -5, 0, 5, 10])
    axes[0].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im1, ax=axes[0], label='Gradient Magnitude')
    
    # Task 2 steepness
    im2 = axes[1].imshow(steepness_task2, cmap='viridis', aspect='auto',
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest')
    axes[1].set_title('Task 2 Gradient Steepness')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_xticks([-10, -5, 0, 5, 10])
    axes[1].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im2, ax=axes[1], label='Gradient Magnitude')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gradient steepness analysis saved to {save_path}")


def plot_gradient_direction_analysis(save_path='gradient_direction_analysis.png'):
    """Plot gradient direction analysis for all tasks."""
    cosine_task1, cosine_task2, cosine_combined, x1_grid, x2_grid = compute_gradient_direction_cosine_map(resolution=VISUALIZATION_RESOLUTION)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Task 1 direction
    im1 = axes[0].imshow(cosine_task1, cmap='RdBu_r', aspect='auto', 
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest', vmin=-1, vmax=1)
    axes[0].set_title('Task 1 Gradient Direction\n(Cosine with X1 axis)')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_xticks([-10, -5, 0, 5, 10])
    axes[0].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')
    
    # Task 2 direction
    im2 = axes[1].imshow(cosine_task2, cmap='RdBu_r', aspect='auto',
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest', vmin=-1, vmax=1)
    axes[1].set_title('Task 2 Gradient Direction\n(Cosine with X1 axis)')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_xticks([-10, -5, 0, 5, 10])
    axes[1].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im2, ax=axes[1], label='Cosine Similarity')
    
    # Combined tasks direction
    im3 = axes[2].imshow(cosine_combined, cmap='RdBu_r', aspect='auto',
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest', vmin=-1, vmax=1)
    axes[2].set_title('Combined Tasks Gradient Direction\n(Cosine with X1 axis)')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].set_xticks([-10, -5, 0, 5, 10])
    axes[2].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im3, ax=axes[2], label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gradient direction analysis saved to {save_path}")


def compute_target_function_map(x_range=(-10, 10), resolution=VISUALIZATION_RESOLUTION):
    """Compute target function value maps for visualization."""
    # Create coordinate grids
    x1_coords = np.linspace(x_range[0], x_range[1], resolution)
    x2_coords = np.linspace(x_range[0], x_range[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1_coords, x2_coords)
    
    # Flatten for computation
    x1_flat = x1_grid.flatten()
    x2_flat = x2_grid.flatten()
    
    # Convert to torch tensors
    x1_tensor = torch.tensor(x1_flat, dtype=torch.float32)
    x2_tensor = torch.tensor(x2_flat, dtype=torch.float32)
    X = torch.stack([x1_tensor, x2_tensor], dim=1)
    
    # Create dataset instance to use _compute_targets method
    dataset = ToyTaskDataset()
    
    # Compute target values
    with torch.no_grad():
        Y = dataset._compute_targets(X)  # [N, 2] where N = resolution^2
    
    # Extract task values
    task1_values = Y[:, 0].numpy().reshape(resolution, resolution)
    task2_values = Y[:, 1].numpy().reshape(resolution, resolution)
    combined_values = (Y[:, 0] + Y[:, 1]).numpy().reshape(resolution, resolution)
    
    return task1_values, task2_values, combined_values, x1_grid, x2_grid


def plot_target_function_analysis(save_path='target_function_analysis.png'):
    """Plot target function landscapes for both tasks."""
    task1_values, task2_values, combined_values, x1_grid, x2_grid = compute_target_function_map(resolution=VISUALIZATION_RESOLUTION)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Task 1 values
    im1 = axes[0].imshow(task1_values, cmap='plasma', aspect='auto', 
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest')
    axes[0].set_title('Task 1 Target Function')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_xticks([-10, -5, 0, 5, 10])
    axes[0].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im1, ax=axes[0], label='Function Value')
    
    # Task 2 values
    im2 = axes[1].imshow(task2_values, cmap='plasma', aspect='auto',
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest')
    axes[1].set_title('Task 2 Target Function')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_xticks([-10, -5, 0, 5, 10])
    axes[1].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im2, ax=axes[1], label='Function Value')
    
    # Combined tasks values
    im3 = axes[2].imshow(combined_values, cmap='plasma', aspect='auto',
                         extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
                         origin='lower', interpolation='nearest')
    axes[2].set_title('Combined Tasks Target Function\n(Task1 + Task2)')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].set_xticks([-10, -5, 0, 5, 10])
    axes[2].set_yticks([-10, -5, 0, 5, 10])
    plt.colorbar(im3, ax=axes[2], label='Function Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Target function analysis saved to {save_path}")


class SparseGatingNetwork(nn.Module):
    """
    Sparse Mixture of Experts (MoE) network with load balancing.
    
    Implements a sparse gating mechanism that routes each input to only the top-k
    most relevant experts. This architecture aims to reduce gradient conflicts
    in multi-task learning by allowing different experts to specialize on different
    tasks or input regions.
    
    Args:
        input_dim (int): Input feature dimension. Defaults to 2.
        hidden_dim (int): Hidden layer dimension for experts and gate. Defaults to 5.
        output_dim (int): Output dimension (number of tasks). Defaults to 2.
        num_experts (int): Number of expert networks. Defaults to 2.
        top_k (int): Number of experts to activate per input. Defaults to 1.
    
    Attributes:
        num_experts (int): Number of expert networks.
        top_k (int): Number of active experts per input (clamped to num_experts).
        experts (nn.ModuleList): List of expert networks.
        gate (nn.Sequential): Gating network for expert selection.
    """
    def __init__(self, input_dim=2, hidden_dim=5, output_dim=2, num_experts=2, top_k=1):
        super(SparseGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Expert networks - simple MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                # nn.Linear(hidden_dim, hidden_dim//2),
                # nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_experts)
        )
        
    def forward(self, x):
        """
        Forward pass through the sparse mixture of experts.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
        
        Returns:
            tuple: (output, gate_weights, load_balance_loss) where:
                - output (torch.Tensor): Final prediction [batch_size, output_dim]
                - gate_weights (torch.Tensor): Expert selection weights [batch_size, num_experts] 
                - load_balance_loss (torch.Tensor): Scalar loss for load balancing
        """
        batch_size = x.size(0)
        
        # Compute gating weights
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        gate_weights = F.softmax(gate_logits, dim=1)
        
        # Apply sparsity: keep only top-k experts
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=1)
        
        # Renormalize the top-k weights
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # Weighted combination using only top-k experts
        output = torch.zeros(batch_size, expert_outputs.size(-1), device=x.device)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # [batch_size]
            weights = top_k_weights[:, i:i+1]  # [batch_size, 1]
            
            # Select expert outputs for each sample in batch
            selected_outputs = expert_outputs[torch.arange(batch_size), expert_idx]  # [batch_size, output_dim]
            output += weights * selected_outputs
        
        # Compute load balancing loss
        load_balance_loss = compute_load_balancing_loss(gate_weights, self.num_experts)
            
        return output, gate_weights, load_balance_loss


class PureMLP(nn.Module):
    """
    Pure Multi-Layer Perceptron baseline model for multi-task learning.
    
    A traditional feedforward neural network that serves as a baseline
    to compare against the Mixture of Experts approach. All tasks share
    the same parameters, potentially leading to gradient conflicts.
    
    Args:
        input_dim (int): Input feature dimension. Defaults to 2.
        hidden_dim (int): Hidden layer dimension. Defaults to 5.
        output_dim (int): Output dimension (number of tasks). Defaults to 2.
    """
    def __init__(self, input_dim=2, hidden_dim=5, output_dim=2):
        super(PureMLP, self).__init__()
        
        # Make the network comparable in size to the gating network
        # Roughly same number of parameters as SparseGatingNetwork
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim * 2, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim//2),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Output predictions of shape [batch_size, output_dim]
        """
        return self.network(x)


def compute_load_balancing_loss(gate_weights, num_experts):
    """
    Compute load balancing loss to encourage even expert utilization
    
    Load Balancing Method Selection Rationale:
    1. Prevent expert network degeneration: Avoid routing all inputs to only a few experts, 
       leaving other experts underutilized
    2. Improve model capacity utilization: Ensure all experts participate in learning to 
       maximize the model's representational capacity
    3. Enhance generalization: Uniform expert usage prevents overfitting to specific expert features
    4. Stabilize training process: Balanced expert selection helps maintain gradient update stability
    
    Args:
        gate_weights: [batch_size, num_experts] softmax gate weights
        num_experts: number of experts
    
    Returns:
        load_balancing_loss: scalar loss encouraging uniform expert usage
    """
    # Compute the average weight proportion routed to each expert
    # Rationale: Using mean of gate_weights rather than hard selection better reflects actual expert load
    expert_fractions = gate_weights.mean(dim=0)  # [num_experts]
    
    # Compute frequency of each expert being the top choice (for analysis, not used in loss)
    # Rationale: Provides additional statistics for monitoring expert selection distribution
    top_expert_mask = torch.argmax(gate_weights, dim=1)  # [batch_size]
    expert_usage = torch.zeros(num_experts, device=gate_weights.device)
    for i in range(num_experts):
        expert_usage[i] = (top_expert_mask == i).float().mean()
    
    # Load balancing loss encourages uniform distribution (ideally 1/num_experts usage per expert)
    # Method 1: Using coefficient of variation to measure distribution imbalance (commented out)
    # Rationale: CV method is intuitive but less sensitive to extreme imbalances
    # target_fraction = 1.0 / num_experts
    # cv_loss = (expert_fractions - target_fraction).pow(2).sum()
    
    # Method 2: Entropy-based loss function to encourage uniform distribution (current choice)
    # Selection Rationale:
    # - Entropy loss naturally measures distribution uniformity, maximum entropy corresponds to perfect uniformity
    # - More sensitive to small distribution changes, better guides optimization process
    # - Mathematically elegant, avoids need to manually set target distribution
    # - Normalized entropy loss in [0,1] range facilitates combination with main loss function
    entropy_loss = -(expert_fractions * torch.log(expert_fractions + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float, device=gate_weights.device))
    normalized_entropy_loss = 1.0 - entropy_loss / max_entropy
    
    return normalized_entropy_loss


def analyze_expert_selection_patterns(expert_selection_history, num_experts=4):
    """
    Analyze how expert usage and specialization evolve during training.
    
    Args:
        expert_selection_history (list): Expert selection data per epoch
        num_experts (int): Number of experts in the model. Defaults to 4.
    
    Returns:
        dict: Analysis containing expert usage trends, specialization metrics,
              task-expert correlations, and spatial usage patterns.
    """
    if not expert_selection_history:
        return {}
    
    analysis = {
        'expert_usage_over_time': [],
        'expert_specialization': [],
        'task_expert_correlation': [],
        'spatial_expert_patterns': []
    }
    
    for epoch_data in expert_selection_history:
        epoch = epoch_data['epoch']
        
        # Aggregate all selections for this epoch
        all_expert_choices = []
        all_inputs = []
        all_targets = []
        all_gate_weights = []
        
        for batch_data in epoch_data['selections']:
            all_expert_choices.extend(batch_data['expert_choices'])
            all_inputs.extend(batch_data['inputs'])
            all_targets.extend(batch_data['targets'])
            all_gate_weights.extend(batch_data['gate_weights'])
        
        if not all_expert_choices:
            continue
            
        all_expert_choices = np.array(all_expert_choices)
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)
        all_gate_weights = np.array(all_gate_weights)
        
        # 1. Expert usage distribution
        expert_counts = np.bincount(all_expert_choices, minlength=num_experts)
        expert_usage = expert_counts / len(all_expert_choices) if len(all_expert_choices) > 0 else np.zeros(num_experts)
        analysis['expert_usage_over_time'].append({
            'epoch': epoch,
            'usage': expert_usage,
            'entropy': -np.sum(expert_usage * np.log(expert_usage + 1e-8))
        })
        
        # 2. Task-expert correlation
        # Analyze which experts are chosen for which target values
        task_expert_corr = {}
        for task_idx in range(2):  # Assuming 2 tasks
            task_values = all_targets[:, task_idx]
            
            # Divide task values into bins to see patterns
            task_bins = np.digitize(task_values, bins=np.linspace(task_values.min(), task_values.max(), 5))
            
            expert_by_task_bin = {}
            for bin_idx in range(1, 6):
                mask = task_bins == bin_idx
                if np.sum(mask) > 0:
                    bin_expert_choices = all_expert_choices[mask]
                    bin_expert_counts = np.bincount(bin_expert_choices, minlength=num_experts)
                    bin_expert_usage = bin_expert_counts / len(bin_expert_choices)
                    expert_by_task_bin[bin_idx] = bin_expert_usage
            
            task_expert_corr[f'task_{task_idx}'] = expert_by_task_bin
        
        analysis['task_expert_correlation'].append({
            'epoch': epoch,
            'correlation': task_expert_corr
        })
        
        # 3. Spatial patterns (input space regions)
        # Divide input space into grid for higher resolution
        x1_bins = np.digitize(all_inputs[:, 0], bins=np.linspace(-10, 10, VISUALIZATION_RESOLUTION + 1))  # +1 bins to get VISUALIZATION_RESOLUTION regions
        x2_bins = np.digitize(all_inputs[:, 1], bins=np.linspace(-10, 10, VISUALIZATION_RESOLUTION + 1))
        
        spatial_patterns = {}
        for x1_bin in range(1, VISUALIZATION_RESOLUTION + 1):
            for x2_bin in range(1, VISUALIZATION_RESOLUTION + 1):
                region_mask = (x1_bins == x1_bin) & (x2_bins == x2_bin)
                if np.sum(region_mask) > 0:
                    region_experts = all_expert_choices[region_mask]
                    region_expert_counts = np.bincount(region_experts, minlength=num_experts)
                    region_expert_usage = region_expert_counts / len(region_experts)
                    spatial_patterns[f'region_{x1_bin}_{x2_bin}'] = region_expert_usage
        
        analysis['spatial_expert_patterns'].append({
            'epoch': epoch,
            'patterns': spatial_patterns
        })
        
        # 4. Expert specialization (how concentrated is each expert's usage)
        expert_specialization = []
        for expert_idx in range(num_experts):
            expert_weights = all_gate_weights[:, expert_idx]
            # Use coefficient of variation as specialization measure
            if np.std(expert_weights) > 0:
                specialization = np.std(expert_weights) / (np.mean(expert_weights) + 1e-8)
            else:
                specialization = 0
            expert_specialization.append(specialization)
        
        analysis['expert_specialization'].append({
            'epoch': epoch,
            'specialization': expert_specialization
        })
    
    return analysis


def count_parameters(model):
    """
    Count total trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_gradient_conflict(model, batch_x, batch_y, criterion):
    """
    Efficient gradient conflict computation using online accumulation
    
    This is the main gradient conflict function - optimized for memory efficiency.
    For the original implementation, see compute_gradient_conflict_verbose().
    
    Key differences from verbose version:
    1. Memory efficient: Uses online accumulation instead of storing full gradient vectors
    2. Single-pass: Computes dot product and norms in one loop without vector concatenation
    3. Computational efficiency: Avoids creating large intermediate tensors
    4. Same output: Returns identical metrics but with better performance
    
    Args:
        model: Neural network model (SparseGatingNetwork or regular MLP)
        batch_x: Input batch [batch_size, input_dim]
        batch_y: Target batch [batch_size, output_dim]  
        criterion: Loss function
    
    Returns:
        Dict with gradient conflict metrics: cosine_similarity, conflict_angle, 
        is_conflicting, task1_grad_norm, task2_grad_norm, task1_loss, task2_loss
    """
    model.train()
    
    # Forward pass
    outputs = model(batch_x)[0] if isinstance(model, SparseGatingNetwork) else model(batch_x)

    # Compute task losses
    task1_loss = criterion(outputs[:, 0], batch_y[:, 0])
    task2_loss = criterion(outputs[:, 1], batch_y[:, 1])

    # Get parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    # Compute gradients for both tasks
    task1_grads = torch.autograd.grad(task1_loss, params, retain_graph=True, allow_unused=True)
    task2_grads = torch.autograd.grad(task2_loss, params, allow_unused=True)

    # Online accumulation of dot product and norms
    dot_product = 0.0
    task1_norm_sq = 0.0
    task2_norm_sq = 0.0
    
    for g1, g2 in zip(task1_grads, task2_grads):
        # Handle None gradients by treating as zeros
        g1 = torch.zeros_like(params[0]).flatten() if g1 is None else g1.flatten()
        g2 = torch.zeros_like(params[0]).flatten() if g2 is None else g2.flatten()
        
        # Accumulate statistics
        dot_product += torch.sum(g1 * g2).item()
        task1_norm_sq += torch.sum(g1 * g1).item()
        task2_norm_sq += torch.sum(g2 * g2).item()

    # Compute final metrics
    task1_norm = np.sqrt(task1_norm_sq)
    task2_norm = np.sqrt(task2_norm_sq)
    cosine_sim = dot_product / (task1_norm * task2_norm + 1e-12)
    conflict_angle = np.degrees(np.arccos(np.clip(cosine_sim, -1, 1)))

    return {
        'cosine_similarity': cosine_sim,
        'conflict_angle': conflict_angle,
        'is_conflicting': cosine_sim < 0,
        'task1_grad_norm': task1_norm,
        'task2_grad_norm': task2_norm,
        'task1_loss': task1_loss.item(),
        'task2_loss': task2_loss.item()
    }


def compute_gradient_conflict_verbose(model, batch_x, batch_y, criterion):
    """
    Original verbose gradient conflict computation (preserved for reference)
    
    This is the original implementation that explicitly constructs full gradient vectors.
    Kept for educational purposes and debugging. For production use, prefer the main
    compute_gradient_conflict() function which is more memory efficient.
    
    Key characteristics:
    1. Explicit: Creates full gradient vectors for each task
    2. Memory intensive: Stores complete gradients before computing similarity
    3. Step-by-step: Clear separation of gradient computation and similarity calculation
    4. Educational: Easier to understand the underlying math
    
    Args:
        model: Neural network model
        batch_x: Input batch
        batch_y: Target batch
        criterion: Loss function
        
    Returns:
        Same format as compute_gradient_conflict()
    """
    model.train()
    
    # Forward pass
    if isinstance(model, SparseGatingNetwork):
        outputs, _, _ = model(batch_x)
    else:
        outputs = model(batch_x)
    
    # Compute individual task losses
    task1_loss = criterion(outputs[:, 0], batch_y[:, 0])
    task2_loss = criterion(outputs[:, 1], batch_y[:, 1])
    
    # Clear gradients
    model.zero_grad()
    
    # Compute gradients for task 1
    task1_loss.backward(retain_graph=True)
    task1_grads = []
    
    for param in model.parameters():
        if param.grad is not None:
            task1_grads.append(param.grad.clone().flatten())
        else:
            task1_grads.append(torch.zeros_like(param).flatten())

    task1_grad_vector = torch.cat(task1_grads)
    
    # Clear gradients and compute gradients for task 2
    model.zero_grad()
    task2_loss.backward(retain_graph=True)
    task2_grads = []
    for param in model.parameters():
        if param.grad is not None:
            task2_grads.append(param.grad.clone().flatten())
    task2_grad_vector = torch.cat(task2_grads)
    
    # Clear gradients after computation
    model.zero_grad()
    
    # Compute cosine similarity between gradients
    cosine_sim = F.cosine_similarity(task1_grad_vector.unsqueeze(0), 
                                   task2_grad_vector.unsqueeze(0)).item()
    
    # Compute gradient norms
    task1_norm = torch.norm(task1_grad_vector).item()
    task2_norm = torch.norm(task2_grad_vector).item()
    
    # Conflict metrics
    conflict_angle = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi  # in degrees
    is_conflicting = cosine_sim < 0  # negative cosine means conflict
    
    return {
        'cosine_similarity': cosine_sim,
        'conflict_angle': conflict_angle,
        'is_conflicting': is_conflicting,
        'task1_grad_norm': task1_norm,
        'task2_grad_norm': task2_norm,
        'task1_loss': task1_loss.item(),
        'task2_loss': task2_loss.item()
    }


def compute_expert_gradient_conflicts(model, batch_x, batch_y, criterion):
    """
    Analyze gradient conflicts at the individual expert level in MoE models.
    
    This function computes task gradient conflicts for each expert separately,
    providing insights into how different experts handle multi-task conflicts.
    Some experts may specialize in reducing conflicts while others may focus
    on specific tasks.
    
    Args:
        model (SparseGatingNetwork): MoE model to analyze
        batch_x (torch.Tensor): Input batch [batch_size, input_dim]
        batch_y (torch.Tensor): Target batch [batch_size, output_dim]
        criterion (nn.Module): Loss function
    
    Returns:
        dict: Expert-wise conflict metrics in format:
            {'expert_0': {...}, 'expert_1': {...}} where each contains:
            - cosine_similarity, conflict_angle, is_conflicting
            - task1_grad_norm, task2_grad_norm, task1_loss, task2_loss
    """
    if not isinstance(model, SparseGatingNetwork):
        return {}
    
    model.train()
    expert_conflicts = {}
    
    # For each expert, compute the gradient conflicts between tasks
    for expert_idx in range(model.num_experts):
        expert = model.experts[expert_idx]
        
        # Forward pass through this specific expert
        expert_outputs = expert(batch_x)  # [batch_size, output_dim]
        
        # Compute individual task losses for this expert
        task1_loss = criterion(expert_outputs[:, 0], batch_y[:, 0])
        task2_loss = criterion(expert_outputs[:, 1], batch_y[:, 1])
        
        # Clear gradients
        expert.zero_grad()
        
        # Compute gradients for task 1
        task1_loss.backward(retain_graph=True)
        task1_grads = []
        
        for param in expert.parameters():
            if param.grad is not None:
                task1_grads.append(param.grad.clone().flatten())
            else:
                task1_grads.append(torch.zeros_like(param).flatten())
        
        if task1_grads:
            task1_grad_vector = torch.cat(task1_grads)
        else:
            continue
        
        # Clear gradients and compute gradients for task 2
        expert.zero_grad()
        task2_loss.backward(retain_graph=True)
        task2_grads = []
        
        for param in expert.parameters():
            if param.grad is not None:
                task2_grads.append(param.grad.clone().flatten())
            else:
                task2_grads.append(torch.zeros_like(param).flatten())
        
        if task2_grads:
            task2_grad_vector = torch.cat(task2_grads)
        else:
            continue
        
        # Clear gradients after computation
        expert.zero_grad()
        
        # Compute cosine similarity between gradients
        if torch.norm(task1_grad_vector) > 1e-8 and torch.norm(task2_grad_vector) > 1e-8:
            cosine_sim = F.cosine_similarity(task1_grad_vector.unsqueeze(0), 
                                           task2_grad_vector.unsqueeze(0)).item()
            
            # Compute gradient norms
            task1_norm = torch.norm(task1_grad_vector).item()
            task2_norm = torch.norm(task2_grad_vector).item()
            
            # Conflict metrics
            conflict_angle = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi  # in degrees
            is_conflicting = cosine_sim < 0  # negative cosine means conflict
            
            expert_conflicts[f'expert_{expert_idx}'] = {
                'cosine_similarity': cosine_sim,
                'conflict_angle': conflict_angle,
                'is_conflicting': is_conflicting,
                'task1_grad_norm': task1_norm,
                'task2_grad_norm': task2_norm,
                'task1_loss': task1_loss.item(),
                'task2_loss': task2_loss.item()
            }
    
    return expert_conflicts


def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, track_conflicts=False, 
                load_balance_weight=0.01, track_expert_selection=False, track_expert_conflicts=False):
    """
    Comprehensive training loop with multi-task conflict analysis and expert tracking.
    
    This function trains either MoE or MLP models while optionally tracking various
    metrics including gradient conflicts, expert selection patterns, and load balancing.
    The tracking capabilities make this ideal for analyzing multi-task learning dynamics.
    
    Args:
        model (nn.Module): Model to train (SparseGatingNetwork or PureMLP)
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader  
        num_epochs (int): Number of training epochs. Defaults to 30.
        lr (float): Learning rate for Adam optimizer. Defaults to 0.001.
        track_conflicts (bool): Whether to track gradient conflicts. Defaults to False.
        load_balance_weight (float): Weight for load balancing loss. Defaults to 0.01.
        track_expert_selection (bool): Whether to track expert selection patterns. Defaults to False.
        track_expert_conflicts (bool): Whether to track per-expert conflicts. Defaults to False.
    
    Returns:
        tuple: (train_losses, val_losses, conflict_history, expert_selection_history, expert_conflict_history)
            - train_losses (list): Training loss per epoch
            - val_losses (list): Validation loss per epoch  
            - conflict_history (list): Gradient conflict metrics per epoch (if tracked)
            - expert_selection_history (list): Expert selection data per epoch (if tracked)
            - expert_conflict_history (list): Expert-level conflict data per epoch (if tracked)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    conflict_history = []
    expert_selection_history = []
    expert_conflict_history = []  # New: track expert-specific conflicts
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training
        model.train()
        train_loss = 0.0
        epoch_conflicts = []
        epoch_expert_conflicts = []  # New: store expert conflicts for this epoch
        
        epoch_expert_selections = []
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Track gradient conflicts every 10 batches if requested
            if track_conflicts and batch_idx % 10 == 0:
                conflict_metrics = compute_gradient_conflict(model, batch_x, batch_y, criterion)
                epoch_conflicts.append(conflict_metrics)
            
            # Track expert gradient conflicts every 10 batches if requested
            if track_expert_conflicts and batch_idx % 10 == 0:
                expert_conflict_metrics = compute_expert_gradient_conflicts(model, batch_x, batch_y, criterion)
                if expert_conflict_metrics:  # Only add if we have expert conflicts (i.e., for gating model)
                    epoch_expert_conflicts.append(expert_conflict_metrics)
            
            optimizer.zero_grad()
            
            if isinstance(model, SparseGatingNetwork):
                outputs, gate_weights, load_balance_loss = model(batch_x)
                
                # Track expert selection every 20 batches if requested
                if track_expert_selection and batch_idx % 20 == 0:
                    expert_choices = torch.argmax(gate_weights, dim=1)  # [batch_size]
                    epoch_expert_selections.append({
                        'batch_idx': batch_idx,
                        'expert_choices': expert_choices.cpu().numpy(),
                        'gate_weights': gate_weights.detach().cpu().numpy(),
                        'inputs': batch_x.cpu().numpy(),
                        'targets': batch_y.cpu().numpy()
                    })
                
                # Combine main loss with load balancing loss
                main_loss = criterion(outputs, batch_y)
                loss = main_loss + load_balance_weight * load_balance_loss
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Store conflict metrics for this epoch
        if track_conflicts and epoch_conflicts:
            # Average conflict metrics across batches in this epoch
            avg_conflict = {
                'cosine_similarity': np.mean([c['cosine_similarity'] for c in epoch_conflicts]),
                'conflict_angle': np.mean([c['conflict_angle'] for c in epoch_conflicts]),
                'is_conflicting': np.mean([c['is_conflicting'] for c in epoch_conflicts]),
                'task1_grad_norm': np.mean([c['task1_grad_norm'] for c in epoch_conflicts]),
                'task2_grad_norm': np.mean([c['task2_grad_norm'] for c in epoch_conflicts])
            }
            conflict_history.append(avg_conflict)
        
        # Store expert conflict metrics for this epoch
        if track_expert_conflicts and epoch_expert_conflicts:
            # Average expert conflict metrics across batches in this epoch
            expert_names = list(epoch_expert_conflicts[0].keys()) if epoch_expert_conflicts else []
            epoch_expert_avg = {'epoch': epoch}
            
            for expert_name in expert_names:
                expert_conflicts_for_epoch = [batch_data[expert_name] for batch_data in epoch_expert_conflicts if expert_name in batch_data]
                if expert_conflicts_for_epoch:
                    epoch_expert_avg[expert_name] = {
                        'cosine_similarity': np.mean([c['cosine_similarity'] for c in expert_conflicts_for_epoch]),
                        'conflict_angle': np.mean([c['conflict_angle'] for c in expert_conflicts_for_epoch]),
                        'is_conflicting': np.mean([c['is_conflicting'] for c in expert_conflicts_for_epoch]),
                        'task1_grad_norm': np.mean([c['task1_grad_norm'] for c in expert_conflicts_for_epoch]),
                        'task2_grad_norm': np.mean([c['task2_grad_norm'] for c in expert_conflicts_for_epoch])
                    }
            
            expert_conflict_history.append(epoch_expert_avg)
        
        # Store expert selection data for this epoch
        if track_expert_selection and epoch_expert_selections:
            expert_selection_history.append({
                'epoch': epoch,
                'selections': epoch_expert_selections
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(model, SparseGatingNetwork):
                    outputs, _, _ = model(batch_x)
                else:
                    outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
            if track_conflicts and conflict_history:
                latest_conflict = conflict_history[-1]
                print(f"  Gradient Conflict: Angle = {latest_conflict['conflict_angle']:.1f}°, "
                      f"Cosine Sim = {latest_conflict['cosine_similarity']:.3f}")
            if track_expert_conflicts and expert_conflict_history:
                latest_expert_conflicts = expert_conflict_history[-1]
                print("  Expert Conflicts:")
                for expert_name, conflicts in latest_expert_conflicts.items():
                    if expert_name != 'epoch':
                        print(f"    {expert_name}: {conflicts['conflict_angle']:.1f}°")
    
    return train_losses, val_losses, conflict_history, expert_selection_history, expert_conflict_history


def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test data with per-task metrics.
    
    Args:
        model (nn.Module): Trained model to evaluate
        test_loader (DataLoader): Test data loader
    
    Returns:
        dict: Evaluation metrics containing:
            - total_loss (float): Overall test loss
            - task1_loss (float): Task 1 specific loss  
            - task2_loss (float): Task 2 specific loss
            - gate_weights (torch.Tensor): Final batch gate weights (MoE only)
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    task1_loss = 0.0
    task2_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if isinstance(model, SparseGatingNetwork):
                outputs, gate_weights, _ = model(batch_x)
            else:
                outputs = model(batch_x)
                gate_weights = None
                
            # Overall loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Per-task losses
            task1_loss += criterion(outputs[:, 0], batch_y[:, 0]).item()
            task2_loss += criterion(outputs[:, 1], batch_y[:, 1]).item()
            
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'task1_loss': task1_loss / num_batches,
        'task2_loss': task2_loss / num_batches,
        'gate_weights': gate_weights
    }


def compute_rolling_expert_conflicts(expert_conflict_history, window_size=5):
    """
    Smooth expert conflict metrics using rolling window averages.
    
    Args:
        expert_conflict_history (list): Expert conflict data per epoch
        window_size (int): Rolling window size. Defaults to 5.
    
    Returns:
        dict: Smoothed conflict statistics for each expert over time.
    """
    if not expert_conflict_history or len(expert_conflict_history) == 0:
        return {}
    
    rolling_stats = {}
    
    # Get expert names from the first epoch that has data
    expert_names = []
    for epoch_data in expert_conflict_history:
        if len(epoch_data) > 1:  # More than just 'epoch' key
            expert_names = [k for k in epoch_data.keys() if k != 'epoch']
            break
    
    if not expert_names:
        return {}
    
    for expert_name in expert_names:
        rolling_stats[expert_name] = {
            'epochs': [],
            'rolling_conflict_angle': [],
            'rolling_cosine_similarity': [],
            'rolling_conflicting_rate': [],
            'rolling_task1_norm': [],
            'rolling_task2_norm': []
        }
    
    # Compute rolling statistics for each epoch
    for i, epoch_data in enumerate(expert_conflict_history):
        epoch = epoch_data.get('epoch', i)
        
        # Determine the window for this epoch (recent 5 epochs)
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window_data = expert_conflict_history[start_idx:end_idx]
        
        # For each expert, compute rolling statistics
        for expert_name in expert_names:
            if expert_name in epoch_data:
                # Collect data from the window
                window_conflicts = []
                for window_epoch in window_data:
                    if expert_name in window_epoch:
                        window_conflicts.append(window_epoch[expert_name])
                
                if window_conflicts:
                    # Compute rolling averages
                    rolling_conflict_angle = np.mean([c['conflict_angle'] for c in window_conflicts])
                    rolling_cosine_sim = np.mean([c['cosine_similarity'] for c in window_conflicts])
                    rolling_conflicting_rate = np.mean([c['is_conflicting'] for c in window_conflicts])
                    rolling_task1_norm = np.mean([c['task1_grad_norm'] for c in window_conflicts])
                    rolling_task2_norm = np.mean([c['task2_grad_norm'] for c in window_conflicts])
                    
                    # Store results
                    rolling_stats[expert_name]['epochs'].append(epoch)
                    rolling_stats[expert_name]['rolling_conflict_angle'].append(rolling_conflict_angle)
                    rolling_stats[expert_name]['rolling_cosine_similarity'].append(rolling_cosine_sim)
                    rolling_stats[expert_name]['rolling_conflicting_rate'].append(rolling_conflicting_rate)
                    rolling_stats[expert_name]['rolling_task1_norm'].append(rolling_task1_norm)
                    rolling_stats[expert_name]['rolling_task2_norm'].append(rolling_task2_norm)
    
    return rolling_stats


def plot_expert_gradient_conflicts(expert_conflict_history, save_path='expert_gradient_conflicts.png', window_size=5):
    """Plot expert-level gradient conflict analysis over training."""
    if not expert_conflict_history:
        print("No expert conflict data to plot")
        return
    
    # Compute rolling statistics
    rolling_stats = compute_rolling_expert_conflicts(expert_conflict_history, window_size)
    
    if not rolling_stats:
        print("No valid expert conflict data found")
        return
    
    expert_names = list(rolling_stats.keys())
    num_experts = len(expert_names)
    
    # Create subplots: 2 rows, multiple columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Conflict angles over time (rolling average)
    ax1 = axes[0, 0]
    for expert_name in expert_names:
        data = rolling_stats[expert_name]
        if data['epochs'] and data['rolling_conflict_angle']:
            ax1.plot(data['epochs'], data['rolling_conflict_angle'], 
                    label=expert_name.replace('_', ' ').title(), marker='o', markersize=4)
    
    ax1.set_title(f'Expert Gradient Conflict Angles (Rolling {window_size}-Epoch Average)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Conflict Angle (degrees)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='No conflict (90°)')
    
    # Plot 2: Cosine similarity over time (rolling average)
    ax2 = axes[0, 1]
    for expert_name in expert_names:
        data = rolling_stats[expert_name]
        if data['epochs'] and data['rolling_cosine_similarity']:
            ax2.plot(data['epochs'], data['rolling_cosine_similarity'], 
                    label=expert_name.replace('_', ' ').title(), marker='o', markersize=4)
    
    ax2.set_title(f'Expert Gradient Cosine Similarity (Rolling {window_size}-Epoch Average)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='No correlation (0)')
    
    # Plot 3: Conflicting rate over time (rolling average)
    ax3 = axes[1, 0]
    for expert_name in expert_names:
        data = rolling_stats[expert_name]
        if data['epochs'] and data['rolling_conflicting_rate']:
            conflicting_rate_percent = [x * 100 for x in data['rolling_conflicting_rate']]  # Convert to percentage
            ax3.plot(data['epochs'], conflicting_rate_percent, 
                    label=expert_name.replace('_', ' ').title(), marker='o', markersize=4)
    
    ax3.set_title(f'Expert Gradient Conflicting Rate (Rolling {window_size}-Epoch Average)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Conflicting Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gradient norms comparison (rolling average)
    ax4 = axes[1, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, expert_name in enumerate(expert_names):
        data = rolling_stats[expert_name]
        if data['epochs'] and data['rolling_task1_norm'] and data['rolling_task2_norm']:
            color = colors[i % len(colors)]
            ax4.plot(data['epochs'], data['rolling_task1_norm'], 
                    label=f'{expert_name.replace("_", " ").title()} - Task 1', 
                    color=color, linestyle='-', marker='o', markersize=3)
            ax4.plot(data['epochs'], data['rolling_task2_norm'], 
                    label=f'{expert_name.replace("_", " ").title()} - Task 2', 
                    color=color, linestyle='--', marker='s', markersize=3)
    
    ax4.set_title(f'Expert Gradient Norms (Rolling {window_size}-Epoch Average)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gradient Norm')
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Expert gradient conflict analysis saved to {save_path}")
    
    # Print summary statistics
    print(f"\nExpert Gradient Conflict Summary (Last {window_size} epochs):")
    print("=" * 60)
    for expert_name in expert_names:
        data = rolling_stats[expert_name]
        if data['rolling_conflict_angle']:
            latest_angle = data['rolling_conflict_angle'][-1]
            latest_cosine = data['rolling_cosine_similarity'][-1]
            latest_conflicting_rate = data['rolling_conflicting_rate'][-1] * 100
            print(f"{expert_name.replace('_', ' ').title()}:")
            print(f"  Average Conflict Angle: {latest_angle:.1f}°")
            print(f"  Average Cosine Similarity: {latest_cosine:.3f}")
            print(f"  Conflicting Rate: {latest_conflicting_rate:.1f}%")


def plot_expert_selection_analysis(expert_analysis, save_path='expert_selection_analysis.png'):
    """Plot expert usage and specialization patterns over training."""
    if not expert_analysis:
        print("No expert selection data to plot")
        return
    
    # Get number of experts from the data
    num_experts = len(expert_analysis['expert_usage_over_time'][0]['usage'])
    
    # Create subplot grid: top row has 3 plots, bottom row has up to num_experts plots
    fig, axes = plt.subplots(2, max(3, num_experts), figsize=(18, 12))
    
    # 1. Expert usage over time
    epochs = [data['epoch'] for data in expert_analysis['expert_usage_over_time']]
    num_experts = len(expert_analysis['expert_usage_over_time'][0]['usage'])
    
    for expert_idx in range(num_experts):
        usage_over_time = [data['usage'][expert_idx] for data in expert_analysis['expert_usage_over_time']]
        axes[0, 0].plot(epochs, usage_over_time, label=f'Expert {expert_idx}', marker='o')
    
    axes[0, 0].set_title('Expert Usage Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Usage Probability')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0/num_experts, color='gray', linestyle='--', alpha=0.7, label='Uniform')
    
    # 2. Expert selection entropy (diversity measure)
    entropies = [data['entropy'] for data in expert_analysis['expert_usage_over_time']]
    max_entropy = np.log(num_experts)
    
    axes[0, 1].plot(epochs, entropies, 'b-', marker='o', label='Selection Entropy')
    axes[0, 1].axhline(y=max_entropy, color='red', linestyle='--', alpha=0.7, label='Max Entropy (Uniform)')
    axes[0, 1].set_title('Expert Selection Diversity')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Expert specialization over time
    for expert_idx in range(num_experts):
        specialization_over_time = [data['specialization'][expert_idx] for data in expert_analysis['expert_specialization']]
        axes[0, 2].plot(epochs, specialization_over_time, label=f'Expert {expert_idx}', marker='o')
    
    axes[0, 2].set_title('Expert Specialization Over Time')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Specialization (CV)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
   
    # 4. Final spatial patterns (last epoch)
    if expert_analysis['spatial_expert_patterns']:
        final_spatial = expert_analysis['spatial_expert_patterns'][-1]['patterns']
        regions = list(final_spatial.keys())
        
        # Create heatmap for each expert
        for expert_idx in range(num_experts):  # Show all experts
            region_usage = [final_spatial[region][expert_idx] if region in final_spatial else 0 
                           for region in regions]
            
            if expert_idx < axes.shape[1]:  # Check if we have enough columns
                ax = axes[1, expert_idx]
                
                # Reshape for grid visualization
                grid_data = np.zeros((VISUALIZATION_RESOLUTION, VISUALIZATION_RESOLUTION))
                for i, region in enumerate(regions):
                    if len(region.split('_')) >= 3:
                        x_idx = int(region.split('_')[1]) - 1
                        y_idx = int(region.split('_')[2]) - 1
                        if 0 <= x_idx < VISUALIZATION_RESOLUTION and 0 <= y_idx < VISUALIZATION_RESOLUTION:
                            grid_data[y_idx, x_idx] = final_spatial[region][expert_idx]
                
                # Set extent to match the actual coordinate system (-10 to 10)
                im = ax.imshow(grid_data, cmap='Blues', aspect='auto', interpolation='nearest',
                              extent=[-10, 10, -10, 10], origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Expert {expert_idx} Spatial Pattern (Final)')
                ax.set_xlabel('X1')
                ax.set_ylabel('X2')
                
                # Set ticks to match coordinate system
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.set_yticks([-10, -5, 0, 5, 10])
                
                plt.colorbar(im, ax=ax)
    
    # If we have more subplots than experts, hide the empty ones
    if axes.shape[1] > num_experts:
        for idx in range(num_experts, axes.shape[1]):
            axes[1, idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Expert selection analysis saved to {save_path}")


def plot_results(gating_results, mlp_results):
    """Plot comprehensive comparison between MoE and MLP models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training curves
    axes[0, 0].plot(gating_results['train_losses'], label='Sparse Gating', color='red')
    axes[0, 0].plot(mlp_results['train_losses'], label='Pure MLP', color='blue')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation curves
    axes[0, 1].plot(gating_results['val_losses'], label='Sparse Gating', color='red')
    axes[0, 1].plot(mlp_results['val_losses'], label='Pure MLP', color='blue')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Gradient conflict over time
    if gating_results.get('conflict_history') and mlp_results.get('conflict_history'):
        gating_conflicts = [c['conflict_angle'] for c in gating_results['conflict_history']]
        mlp_conflicts = [c['conflict_angle'] for c in mlp_results['conflict_history']]
        
        epochs = range(len(gating_conflicts))
        axes[0, 2].plot(epochs, gating_conflicts, label='Sparse Gating', color='red')
        axes[0, 2].plot(epochs, mlp_conflicts, label='Pure MLP', color='blue')
        axes[0, 2].set_title('Gradient Conflict Angle')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Angle (degrees)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[0, 2].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='No conflict')
    else:
        axes[0, 2].text(0.5, 0.5, 'No conflict data\navailable', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Gradient Conflict Angle')
    
    # Per-task performance comparison
    methods = ['Sparse Gating', 'Pure MLP']
    task1_losses = [gating_results['test_eval']['task1_loss'], mlp_results['test_eval']['task1_loss']]
    task2_losses = [gating_results['test_eval']['task2_loss'], mlp_results['test_eval']['task2_loss']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, task1_losses, width, label='Task 1', alpha=0.8)
    axes[1, 0].bar(x + width/2, task2_losses, width, label='Task 2', alpha=0.8)
    axes[1, 0].set_title('Per-Task Test Loss')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Parameter count comparison
    param_counts = [gating_results['param_count'], mlp_results['param_count']]
    axes[1, 1].bar(methods, param_counts, alpha=0.8, color=['red', 'blue'])
    axes[1, 1].set_title('Parameter Count')
    axes[1, 1].set_ylabel('Number of Parameters')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Average gradient conflict comparison
    if gating_results.get('conflict_history') and mlp_results.get('conflict_history'):
        gating_avg_conflict = np.mean([c['conflict_angle'] for c in gating_results['conflict_history']])
        mlp_avg_conflict = np.mean([c['conflict_angle'] for c in mlp_results['conflict_history']])
        
        conflict_angles = [gating_avg_conflict, mlp_avg_conflict]
        bars = axes[1, 2].bar(methods, conflict_angles, alpha=0.8, color=['red', 'blue'])
        axes[1, 2].set_title('Average Gradient Conflict')
        axes[1, 2].set_ylabel('Angle (degrees)')
        axes[1, 2].axhline(y=90, color='gray', linestyle='--', alpha=0.7)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, conflict_angles):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}°', ha='center', va='bottom')
    else:
        axes[1, 2].text(0.5, 0.5, 'No conflict data\navailable', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Average Gradient Conflict')
    
    plt.tight_layout()
    plt.savefig('multitask_gating_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_experiment():
    """
    Main experimental pipeline comparing MoE vs MLP for multi-task learning.
    
    This function orchestrates the complete experiment including:
    1. Dataset generation and splitting
    2. Model initialization and training  
    3. Comprehensive analysis of gradient conflicts and expert behaviors
    4. Visualization of results and expert patterns
    5. Performance comparison and statistical analysis
    
    Returns:
        tuple: (gating_results, mlp_results) containing complete experimental results
            including training curves, conflict metrics, expert analyses, and evaluations.
    """
    print("Starting Multi-task Learning Experiment: Sparse Gating vs Pure MLP")
    print("=" * 60)
    
    # Generate dataset
    dataset = ToyTaskDataset(num_samples=20000)
    X, Y = dataset.generate_data()
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    train_X, train_Y = X[:train_size], Y[:train_size]
    val_X, val_Y = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
    test_X, test_Y = X[train_size+val_size:], Y[train_size+val_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False)
    
    print(f"Data split: Train={len(train_X)}, Val={len(val_X)}, Test={len(test_X)}")
    
    # Initialize models
    gating_model = SparseGatingNetwork(input_dim=2, hidden_dim=32, output_dim=2, num_experts=4, top_k=1)
    mlp_model = PureMLP(input_dim=2, hidden_dim=32, output_dim=2)
    
    print(f"Sparse Gating Model Parameters: {count_parameters(gating_model):,}")
    print(f"Pure MLP Model Parameters: {count_parameters(mlp_model):,}")
    print()
    
    # Train models with gradient conflict tracking and expert selection tracking
    print("Training Sparse Gating Network...")
    start_time = time.time()
    gating_train_losses, gating_val_losses, gating_conflicts, gating_expert_history, gating_expert_conflicts = train_model(
        gating_model, train_loader, val_loader, num_epochs=100, track_conflicts=True, 
        track_expert_selection=True, track_expert_conflicts=True)
    gating_training_time = time.time() - start_time
    
    print("\nTraining Pure MLP...")
    start_time = time.time()
    mlp_train_losses, mlp_val_losses, mlp_conflicts, mlp_expert_history, mlp_expert_conflicts = train_model(
        mlp_model, train_loader, val_loader, num_epochs=100, track_conflicts=True)
    mlp_training_time = time.time() - start_time
    
    # Evaluate models
    print("\nEvaluating models...")
    gating_eval = evaluate_model(gating_model, test_loader)
    mlp_eval = evaluate_model(mlp_model, test_loader)
    
    # Analyze expert selection patterns for gating model
    expert_analysis = None
    if gating_expert_history:
        expert_analysis = analyze_expert_selection_patterns(gating_expert_history, num_experts=4)
    
    # Prepare results
    gating_results = {
        'train_losses': gating_train_losses,
        'val_losses': gating_val_losses,
        'test_eval': gating_eval,
        'param_count': count_parameters(gating_model),
        'training_time': gating_training_time,
        'conflict_history': gating_conflicts,
        'expert_selection_history': gating_expert_history,
        'expert_analysis': expert_analysis,
        'expert_conflict_history': gating_expert_conflicts
    }
    
    mlp_results = {
        'train_losses': mlp_train_losses,
        'val_losses': mlp_val_losses,
        'test_eval': mlp_eval,
        'param_count': count_parameters(mlp_model),
        'training_time': mlp_training_time,
        'conflict_history': mlp_conflicts,
        'expert_conflict_history': mlp_expert_conflicts
    }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Sparse Gating':<15} {'Pure MLP':<15} {'Winner'}")
    print("-" * 80)
    print(f"{'Total Test Loss':<25} {gating_eval['total_loss']:<15.4f} {mlp_eval['total_loss']:<15.4f} {'Gating' if gating_eval['total_loss'] < mlp_eval['total_loss'] else 'MLP'}")
    print(f"{'Task 1 Test Loss':<25} {gating_eval['task1_loss']:<15.4f} {mlp_eval['task1_loss']:<15.4f} {'Gating' if gating_eval['task1_loss'] < mlp_eval['task1_loss'] else 'MLP'}")
    print(f"{'Task 2 Test Loss':<25} {gating_eval['task2_loss']:<15.4f} {mlp_eval['task2_loss']:<15.4f} {'Gating' if gating_eval['task2_loss'] < mlp_eval['task2_loss'] else 'MLP'}")
    print(f"{'Parameters':<25} {count_parameters(gating_model):<15,} {count_parameters(mlp_model):<15,} {'Gating' if count_parameters(gating_model) < count_parameters(mlp_model) else 'MLP'}")
    print(f"{'Training Time (s)':<25} {gating_training_time:<15.2f} {mlp_training_time:<15.2f} {'Gating' if gating_training_time < mlp_training_time else 'MLP'}")
    
    # Gradient conflict analysis
    if gating_conflicts and mlp_conflicts:
        gating_avg_conflict = np.mean([c['conflict_angle'] for c in gating_conflicts])
        mlp_avg_conflict = np.mean([c['conflict_angle'] for c in mlp_conflicts])
        gating_conflicting_rate = np.mean([c['is_conflicting'] for c in gating_conflicts])
        mlp_conflicting_rate = np.mean([c['is_conflicting'] for c in mlp_conflicts])
        
        print("\n" + "="*80)
        print("GRADIENT CONFLICT ANALYSIS")
        print("="*80)
        print(f"{'Avg Conflict Angle (°)':<25} {gating_avg_conflict:<15.1f} {mlp_avg_conflict:<15.1f} {'Gating' if gating_avg_conflict < mlp_avg_conflict else 'MLP'}")
        print(f"{'Conflicting Rate (%)':<25} {gating_conflicting_rate*100:<15.1f} {mlp_conflicting_rate*100:<15.1f} {'Gating' if gating_conflicting_rate < mlp_conflicting_rate else 'MLP'}")
        
        # Final gradient conflict on test data
        test_batch = next(iter(test_loader))
        test_x, test_y = test_batch
        gating_final_conflict = compute_gradient_conflict(gating_model, test_x, test_y, nn.MSELoss())
        mlp_final_conflict = compute_gradient_conflict(mlp_model, test_x, test_y, nn.MSELoss())
        
        print(f"{'Final Test Conflict (°)':<25} {gating_final_conflict['conflict_angle']:<15.1f} {mlp_final_conflict['conflict_angle']:<15.1f} {'Gating' if gating_final_conflict['conflict_angle'] < mlp_final_conflict['conflict_angle'] else 'MLP'}")
        
        # Print detailed analysis
        print(f"\nDETAILED CONFLICT ANALYSIS:")
        print(f"Gating - Training avg vs Final test: {gating_avg_conflict:.1f}° vs {gating_final_conflict['conflict_angle']:.1f}° (diff: {abs(gating_avg_conflict - gating_final_conflict['conflict_angle']):.1f}°)")
        print(f"MLP - Training avg vs Final test: {mlp_avg_conflict:.1f}° vs {mlp_final_conflict['conflict_angle']:.1f}° (diff: {abs(mlp_avg_conflict - mlp_final_conflict['conflict_angle']):.1f}°)")
        
        print("\nNote: Lower conflict angle indicates better alignment between task gradients")
        print("Angles < 90° indicate cooperative gradients, > 90° indicate conflicting gradients")
        print("Large difference between training avg and final test may indicate:")
        print("- Different data distributions (train vs test)")
        print("- Model still learning during training (vs converged at end)")
        print("- Load balancing effects during training")
    
    # Analyze expert selection patterns (only for gating model)
    if expert_analysis:
        print("\nAnalyzing expert selection patterns...")
        plot_expert_selection_analysis(expert_analysis)
        
        # Print summary of expert selection
        print("\nEXPERT SELECTION SUMMARY:")
        print("="*50)
        
        # Final expert usage
        final_usage = expert_analysis['expert_usage_over_time'][-1]['usage']
        print(f"Final Expert Usage Distribution:")
        for i, usage in enumerate(final_usage):
            print(f"  Expert {i}: {usage:.3f} ({usage*100:.1f}%)")
        
        # Expert usage entropy over time
        initial_entropy = expert_analysis['expert_usage_over_time'][0]['entropy']
        final_entropy = expert_analysis['expert_usage_over_time'][-1]['entropy']
        max_entropy = np.log(4)  # 4 experts
        
        print(f"\nExpert Selection Diversity:")
        print(f"  Initial Entropy: {initial_entropy:.3f} (Normalized: {initial_entropy/max_entropy:.3f})")
        print(f"  Final Entropy: {final_entropy:.3f} (Normalized: {final_entropy/max_entropy:.3f})")
        print(f"  Max Possible Entropy: {max_entropy:.3f}")
        
        # Most specialized expert
        final_specialization = expert_analysis['expert_specialization'][-1]['specialization']
        most_specialized_expert = np.argmax(final_specialization)
        print(f"\nMost Specialized Expert: Expert {most_specialized_expert} (Specialization: {final_specialization[most_specialized_expert]:.3f})")
    
    # Analyze expert gradient conflicts (only for gating model)
    if gating_expert_conflicts:
        print("\nAnalyzing expert gradient conflicts...")
        plot_expert_gradient_conflicts(gating_expert_conflicts, window_size=5)
    
    # Plot results
    plot_results(gating_results, mlp_results)
    
    # Plot gradient steepness analysis for the toy tasks
    print("\nGenerating gradient steepness analysis...")
    plot_gradient_steepness_analysis()
    
    # Plot gradient direction analysis for the toy tasks
    print("Generating gradient direction analysis...")
    plot_gradient_direction_analysis()
    
    # Plot target function analysis
    print("Generating target function analysis...")
    plot_target_function_analysis()
    
    return gating_results, mlp_results


if __name__ == "__main__":
    gating_results, mlp_results = run_experiment()