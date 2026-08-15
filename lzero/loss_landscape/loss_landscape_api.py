"""
Overview:
    High-level API for computing and visualizing neural network loss landscapes.
    This module provides a unified interface for 1D and 2D loss landscape analysis,
    supporting both standard PyTorch loss functions and custom metric functions.

This module provides:
    - Loss landscape computation in 1D and 2D
    - Multiple visualization formats (contour, heatmap, 3D surface)
    - ParaView export for professional-quality rendering
    - Support for custom metrics and multi-metric analysis
    - Direction generation with various normalization schemes

Key Classes:
    - LossLandscape: Main class for landscape computation and visualization

Key Functions:
    - compute_1d: Compute 1D loss landscape
    - compute_2d: Compute 2D loss landscape
    - plot_2d_contour: Generate contour plots
    - export_paraview: Export to VTP format
"""

import torch.nn as nn
import numpy as np
import h5py
from typing import Dict, Tuple, Optional, Union, Callable
from torch.utils.data import DataLoader

from .core import direction, evaluator, perturbation
from . import utils as viz  # Visualization functions are now in utils module


class LossLandscape:
    """
    Overview:
        Main class for computing and visualizing neural network loss landscapes.
        Supports both 1D and 2D loss surface analysis with multiple visualization formats.
        Can work with standard PyTorch loss functions or custom metric functions.

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch neural network model to analyze
        - dataloader (:obj:`DataLoader`): DataLoader for evaluating loss and metrics
        - criterion (:obj:`nn.Module` or :obj:`Callable`, optional): Loss or metrics function
            - Can be `nn.Module`: Standard loss (CrossEntropyLoss, MSELoss, etc.). Default is CrossEntropyLoss.
            - Can be `callable`: Function that takes (net, dataloader, use_cuda) and returns dict of metrics
        - use_cuda (:obj:`bool`, optional): Whether to use GPU for computation. Default is False.
        - surf_file (:obj:`str`, optional): Path to save HDF5 surface file. Default is 'loss_surface.h5'.

    Examples::
        >>> # 1D landscape with standard loss
        >>> landscape = LossLandscape(model, dataloader, use_cuda=True)
        >>> result_1d = landscape.compute_1d(xrange=(-1, 1, 51))
        >>> landscape.plot_1d()

        >>> # 2D landscape with custom metrics
        >>> def custom_metrics(net, loader, use_cuda):
        ...     return {'loss': 0.5, 'accuracy': 0.95}
        >>> landscape = LossLandscape(model, dataloader, criterion=custom_metrics)
        >>> result_2d = landscape.compute_2d(xrange=(-1, 1, 21), yrange=(-1, 1, 21))
        >>> landscape.plot_2d_contour(surf_name='auto')
    """

    def __init__(
        self,
        net: nn.Module,
        dataloader: DataLoader,
        criterion: Optional[Union[nn.Module, Callable]] = None,
        use_cuda: bool = False,
        surf_file: Optional[str] = None
    ) -> None:
        """
        Overview:
            Initialize LossLandscape instance with model and data.

        Arguments:
            - net (:obj:`torch.nn.Module`): PyTorch neural network model to analyze
            - dataloader (:obj:`DataLoader`): DataLoader for evaluating loss and metrics
            - criterion (:obj:`nn.Module` or :obj:`Callable`, optional): Loss or metrics function
                - Can be `nn.Module`: Standard loss (CrossEntropyLoss, MSELoss, etc.). Default is CrossEntropyLoss.
                - Can be `callable`: Function that takes (net, dataloader, use_cuda) and returns dict of metrics
            - use_cuda (:obj:`bool`, optional): Whether to use GPU for computation. Default is False.
            - surf_file (:obj:`str`, optional): Path to save HDF5 surface file. Default is 'loss_surface.h5'.
        """
        self.net = net
        self.dataloader = dataloader
        self.use_cuda = use_cuda

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Check if it's a custom metrics function (callable but not nn.Module)
        if callable(criterion) and not isinstance(criterion, nn.Module):
            self.custom_metrics = criterion
            self.criterion = None
            self.use_custom_metrics = True
        else:
            # Standard PyTorch criterion
            self.criterion = criterion
            self.custom_metrics = None
            self.use_custom_metrics = False

        # Store original weights and states
        self.weights = direction.get_weights(net)
        self.states = direction.get_states(net)

        # Default surface file
        if surf_file is None:
            surf_file = 'loss_surface.h5'
        self.surf_file = surf_file

        self.directions = None

    def compute_1d(
        self,
        directions: str = 'random',
        xrange: Tuple[float, float, int] = (-1, 1, 51),
        dir_type: str = 'weights',
        normalize: str = 'filter',
        ignore: str = 'biasbn',
        target_model: Optional[nn.Module] = None,
        save: bool = True
    ) -> Dict:
        """
        Overview:
            Compute 1D loss landscape by evaluating the model along a single direction.
            Sweeps from xmin to xmax with xnum equally spaced points.

        Arguments:
            - directions (:obj:`str`): Type of direction to use
                - 'random': Generate random direction with specified normalization
                - 'target': Use direction towards target_model (requires target_model)
            - xrange (:obj:`Tuple[float, float, int]`): (xmin, xmax, num_points) for x-axis sweep
            - dir_type (:obj:`str`): Type of parameters to perturb
                - 'weights': Perturb only model weights
                - 'states': Perturb weights + batch norm running statistics
            - normalize (:obj:`str`): Normalization method for direction vectors
                - 'filter': Normalize by filter-wise magnitude (default, layer-wise)
                - 'layer': Normalize by entire layer magnitude
                - 'weight': Scale by weight magnitude
                - 'dfilter': Unit norm per filter
                - 'dlayer': Unit norm per layer
            - ignore (:obj:`str`): Ignore certain parameters
                - 'biasbn': Ignore bias and batch norm parameters (recommended)
            - target_model (:obj:`torch.nn.Module`, optional): Target model for 'target' direction
            - save (:obj:`bool`): Whether to save computed surface to HDF5 file. Default is True.

        Returns:
            - result (:obj:`dict`): Dictionary containing:
                - 'losses': np.array or dict of arrays (for custom metrics)
                - 'accuracies': np.array (only for standard loss mode)
                - 'xcoordinates': np.array of x-axis values

        Notes:
            - Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
            - Normalization is crucial for meaningful landscape interpretation
            - Filter-wise normalization recommended for CNNs, layer-wise for MLPs
            - Computation cost is O(num_points) forward passes through the model
            - Use multiple batches (via dataloader) for stable loss estimates

        Examples::
            >>> landscape = LossLandscape(model, dataloader)
            >>> # Random direction landscape
            >>> result = landscape.compute_1d(xrange=(-1, 1, 51))
            >>> print(f"Loss range: {result['losses'].min():.4f} to {result['losses'].max():.4f}")
            >>> # Target direction (towards another model)
            >>> result = landscape.compute_1d(
            ...     directions='target',
            ...     target_model=pretrained_model,
            ...     xrange=(-0.5, 1.5, 51)
            ... )
        """
        xmin, xmax, xnum = xrange

        # Create direction
        if directions == 'random':
            direction_vec = [direction.create_random_direction(
                self.net, dir_type, ignore, normalize)]
        elif directions == 'target':
            assert target_model is not None, "target_model required for 'target' direction"
            direction_vec = [direction.create_target_direction(
                self.net, target_model, dir_type)]
        else:
            raise ValueError(f"Unknown direction type: {directions}")

        self.directions = direction_vec

        # Compute loss surface
        xcoords = np.linspace(xmin, xmax, int(xnum))

        if self.use_custom_metrics:
            # Initialize dict of lists for each metric
            metrics_dict = {}
        else:
            # Standard loss and accuracy
            losses = []
            accuracies = []

        print("Computing 1D loss landscape...")
        for i, x in enumerate(xcoords):
            if dir_type == 'weights':
                perturbation.set_weights(self.net, self.weights, direction_vec, x)
            else:  # states
                perturbation.set_states(self.net, self.states, direction_vec, x)

            if self.use_custom_metrics:
                metrics = self.custom_metrics(self.net, self.dataloader, self.use_cuda)
                if i == 0:
                    # Initialize dict on first iteration
                    metrics_dict = {k: [] for k in metrics.keys()}

                # Append all metrics
                for k, v in metrics.items():
                    metrics_dict[k].append(v)

                # Print first metric for progress display
                first_key = list(metrics.keys())[0]
                print(f"  [{i+1:3d}/{int(xnum)}] {first_key}: {metrics[first_key]:.4f}")
            else:
                # Standard evaluation
                loss, acc = evaluator.eval_loss(self.net, self.criterion,
                                                self.dataloader, self.use_cuda)
                losses.append(loss)
                accuracies.append(acc)
                print(f"  [{i+1:3d}/{int(xnum)}] Loss: {loss:.4f}, Acc: {acc:.1f}%")

        # Restore original weights
        if dir_type == 'weights':
            perturbation.set_weights(self.net, self.weights)
        else:
            perturbation.set_states(self.net, self.states)

        if self.use_custom_metrics:
            # Convert lists to numpy arrays
            losses_result = {k: np.array(v) for k, v in metrics_dict.items()}
            result = {
                'losses': losses_result,  # Dict of arrays
                'xcoordinates': xcoords
            }
        else:
            # Standard format (backward compatible)
            result = {
                'losses': np.array(losses),
                'accuracies': np.array(accuracies),
                'xcoordinates': xcoords
            }

        if save:
            self._save_1d_surface(result)

        return result

    def compute_2d(
        self,
        xrange: Tuple[float, float, int] = (-1, 1, 51),
        yrange: Tuple[float, float, int] = (-1, 1, 51),
        dir_type: str = 'weights',
        normalize: str = 'filter',
        ignore: str = 'biasbn',
        x_target: Optional[nn.Module] = None,
        y_target: Optional[nn.Module] = None,
        save: bool = True
    ) -> Dict:
        """
        Overview:
            Compute 2D loss landscape by evaluating the model along two orthogonal directions.
            Creates a grid of (xnum × ynum) points with loss values at each point.

        Arguments:
            - xrange (:obj:`Tuple[float, float, int]`): (xmin, xmax, num_points_x) for x-axis sweep
            - yrange (:obj:`Tuple[float, float, int]`): (ymin, ymax, num_points_y) for y-axis sweep
            - dir_type (:obj:`str`): Type of parameters to perturb
                - 'weights': Perturb only model weights
                - 'states': Perturb weights + batch norm running statistics
            - normalize (:obj:`str`): Normalization method for direction vectors
                - 'filter': Normalize by filter-wise magnitude (recommended for CNNs)
                - 'layer': Normalize by entire layer magnitude
                - 'weight': Scale by weight magnitude
                - 'dfilter': Unit norm per filter
                - 'dlayer': Unit norm per layer
            - ignore (:obj:`str`): Ignore certain parameters
                - 'biasbn': Ignore bias and batch norm parameters (recommended)
            - x_target (:obj:`torch.nn.Module`, optional): Target model for x-direction
            - y_target (:obj:`torch.nn.Module`, optional): Target model for y-direction
            - save (:obj:`bool`): Whether to save computed surface to HDF5 file. Default is True.

        Returns:
            - result (:obj:`dict`): Dictionary containing:
                - 'losses': 2D np.array (or dict of 2D arrays for custom metrics)
                - 'accuracies': 2D np.array (only for standard loss mode)
                - 'xcoordinates': 1D np.array of x-axis values
                - 'ycoordinates': 1D np.array of y-axis values

        Notes:
            - Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
            - Computation cost is O(num_points_x × num_points_y) forward passes
            - Two directions are typically orthogonal random directions or directions to target models
            - 21×21 grid (441 evaluations) takes ~30 min on H200 GPU for typical models
            - Use 11×11 for quick exploration, 21×21 for detailed analysis
            - Filter-wise normalization preserves layer structure and is most interpretable

        Examples::
            >>> landscape = LossLandscape(model, dataloader)
            >>> # 2D landscape with random directions
            >>> result = landscape.compute_2d(
            ...     xrange=(-1, 1, 21),
            ...     yrange=(-1, 1, 21),
            ...     normalize='filter'
            ... )
            >>> # Access results
            >>> print(f"Landscape shape: {result['losses'].shape}")  # (21, 21)
            >>> # Compare with two different models
            >>> result = landscape.compute_2d(
            ...     xrange=(-1, 1, 15),
            ...     yrange=(-1, 1, 15),
            ...     x_target=model1,
            ...     y_target=model2
            ... )
        """
        xmin, xmax, xnum = xrange
        ymin, ymax, ynum = yrange

        # Create directions
        if x_target is not None:
            x_direction = direction.create_target_direction(
                self.net, x_target, dir_type)
        else:
            x_direction = direction.create_random_direction(
                self.net, dir_type, ignore, normalize)

        if y_target is not None:
            y_direction = direction.create_target_direction(
                self.net, y_target, dir_type)
        else:
            y_direction = direction.create_random_direction(
                self.net, dir_type, ignore, normalize)

        self.directions = [x_direction, y_direction]

        # Compute loss surface
        xcoords = np.linspace(xmin, xmax, int(xnum))
        ycoords = np.linspace(ymin, ymax, int(ynum))

        if self.use_custom_metrics:
            # Initialize placeholder for metrics dict
            metrics_dict = None
        else:
            # Standard loss and accuracy arrays
            losses = np.zeros((len(xcoords), len(ycoords)))
            accuracies = np.zeros((len(xcoords), len(ycoords)))

        print("Computing 2D loss landscape...")
        for i, x in enumerate(xcoords):
            for j, y in enumerate(ycoords):
                if dir_type == 'weights':
                    perturbation.set_weights(self.net, self.weights,
                                            self.directions, [x, y])
                else:  # states
                    perturbation.set_states(self.net, self.states,
                                           self.directions, [x, y])

                if self.use_custom_metrics:
                    metrics = self.custom_metrics(self.net, self.dataloader, self.use_cuda)
                    if metrics_dict is None:
                        # Initialize dict on first iteration
                        metrics_dict = {
                            k: np.zeros((len(xcoords), len(ycoords)))
                            for k in metrics.keys()
                        }
                    # Store all metrics
                    for k, v in metrics.items():
                        metrics_dict[k][i, j] = v
                else:
                    # Standard evaluation
                    loss, acc = evaluator.eval_loss(self.net, self.criterion,
                                                    self.dataloader, self.use_cuda)
                    losses[i, j] = loss
                    accuracies[i, j] = acc

            print(f"  Row {i+1}/{len(xcoords)} complete")

        # Restore original weights
        if dir_type == 'weights':
            perturbation.set_weights(self.net, self.weights)
        else:
            perturbation.set_states(self.net, self.states)

        if self.use_custom_metrics:
            result = {
                'losses': metrics_dict,  # Dict of 2D arrays
                'xcoordinates': xcoords,
                'ycoordinates': ycoords
            }
        else:
            # Standard format (backward compatible)
            result = {
                'losses': losses,
                'accuracies': accuracies,
                'xcoordinates': xcoords,
                'ycoordinates': ycoords
            }

        if save:
            self._save_2d_surface(result)

        return result

    def _save_1d_surface(self, result: Dict) -> None:
        """Save 1D loss surface to HDF5 file.

        Compatible with both single loss and multiple metrics.
        """
        f = h5py.File(self.surf_file, 'w')
        f['xcoordinates'] = result['xcoordinates']

        if isinstance(result['losses'], dict):
            # Multiple metrics (custom)
            for metric_name, metric_values in result['losses'].items():
                f[f'train_loss_{metric_name}'] = metric_values
        else:
            # Single loss (standard - backward compatible)
            f['train_loss'] = result['losses']
            if 'accuracies' in result:
                f['train_acc'] = result['accuracies']

        f.close()
        print(f"\nSaved to: {self.surf_file}")

    def _save_2d_surface(self, result: Dict) -> None:
        """Save 2D loss surface to HDF5 file.

        Compatible with both single loss and multiple metrics.
        """
        f = h5py.File(self.surf_file, 'w')
        f['xcoordinates'] = result['xcoordinates']
        f['ycoordinates'] = result['ycoordinates']

        if isinstance(result['losses'], dict):
            # Multiple metrics (custom)
            for metric_name, metric_values in result['losses'].items():
                f[f'train_loss_{metric_name}'] = metric_values
        else:
            # Single loss (standard - backward compatible)
            f['train_loss'] = result['losses']
            if 'accuracies' in result:
                f['train_acc'] = result['accuracies']

        f.close()
        print(f"\nSaved to: {self.surf_file}")

    def plot_1d(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        loss_max: int = 5,
        log: bool = False,
        show: bool = False
    ) -> None:
        """
        Overview:
            Plot 1D loss curve from previously computed surface file.
            Generates PDF with loss and accuracy curves on dual y-axes.

        Arguments:
            - xmin (:obj:`float`, optional): Minimum x value for plot range. Default uses file min.
            - xmax (:obj:`float`, optional): Maximum x value for plot range. Default uses file max.
            - loss_max (:obj:`float`, optional): Maximum loss value for y-axis. Default is 5.
            - log (:obj:`bool`, optional): Use logarithmic scale for loss axis. Default is False.
            - show (:obj:`bool`, optional): Display plot in interactive window. Default is False.

        Returns:
            None. Saves PDF files to disk.

        Notes:
            - Output files are named: surf_file + '_1d_loss_acc.pdf' or '_1d_loss_acc_log.pdf'
            - Requires compute_1d() to be called first and save=True
            - Dual-axis plot shows loss (left, blue) and accuracy (right, red)
        """
        viz.plot_1d_loss(self.surf_file, xmin or -1, xmax or 1,
                         loss_max, log, show)

    def plot_2d_contour(
        self,
        surf_name: str = 'train_loss',
        vmin: Optional[float] = 0.1,
        vmax: Optional[float] = 10,
        vlevel: Optional[float] = 0.5,
        show: bool = False
    ) -> None:
        """
        Overview:
            Generate 2D contour, filled contour, and heatmap plots from saved surface file.
            Supports automatic detection and visualization of multiple metrics.

        Arguments:
            - surf_name (:obj:`str`, optional): Name of surface to plot. Default is 'train_loss'.
                - Use 'auto' to auto-detect and plot all available metrics
                - For custom metrics, use the metric name (e.g., 'train_loss_accuracy')
            - vmin (:obj:`float`, optional): Minimum value for contour levels. Default is 0.1.
            - vmax (:obj:`float`, optional): Maximum value for contour levels. Default is 10.
            - vlevel (:obj:`float`, optional): Spacing between contour levels. Default is 0.5.
            - show (:obj:`bool`, optional): Display plot in interactive window. Default is False.

        Returns:
            None. Saves three PDF files per metric:
            - *_2dcontour.pdf: Contour lines with labels
            - *_2dcontourf.pdf: Filled contour plot
            - *_2dheat.pdf: Heatmap visualization

        Notes:
            - Requires compute_2d() to be called first and save=True
            - Auto-adjusts contour levels if data has small variance
            - Useful for understanding landscape structure and minima
            - vmin/vmax can be None for auto-scaling
        """
        viz.plot_2d_contour(self.surf_file, surf_name,
                            vmin, vmax, vlevel, show)

    def plot_2d_surface(self, surf_name: str = 'train_loss', show: bool = False) -> None:
        """
        Overview:
            Generate 3D surface plots from saved 2D landscape.
            Supports automatic detection and visualization of multiple metrics.

        Arguments:
            - surf_name (:obj:`str`, optional): Name of surface to plot. Default is 'train_loss'.
                - Use 'auto' to auto-detect and plot all available metrics
                - For custom metrics, use the metric name (e.g., 'train_loss_accuracy')
            - show (:obj:`bool`, optional): Display plot in interactive window. Default is False.

        Returns:
            None. Saves PDF file: surf_file + '_key_3dsurface.pdf' for each metric

        Notes:
            - Requires compute_2d() to be called first and save=True
            - Produces publication-quality figures with 300 dpi
            - 3D visualization helps interpret landscape topology
            - Color gradient represents loss values (coolwarm colormap)
        """
        viz.plot_2d_surface(self.surf_file, surf_name, show)

    def export_paraview(
        self,
        surf_name: str = 'train_loss',
        log: bool = False,
        zmax: float = -1,
        interp: int = -1
    ) -> str:
        """
        Overview:
            Export 2D loss landscape to VTP format for professional 3D rendering in ParaView.
            Enables advanced visualization with lighting, shading, and high-resolution output.

        Arguments:
            - surf_name (:obj:`str`, optional): Name of surface to export. Default is 'train_loss'.
            - log (:obj:`bool`, optional): Apply logarithmic scale to z values. Default is False.
                - Useful when loss values span large dynamic range (e.g., 0.01 to 100)
            - zmax (:obj:`float`, optional): Clip z values to maximum. Default is -1 (no clipping).
                - Use to remove outliers (e.g., zmax=10 to focus on range [0, 10])
            - interp (:obj:`int`, optional): Interpolate to higher resolution. Default is -1 (no interpolation).
                - Use positive integer (e.g., 100) for smooth, high-resolution surface

        Returns:
            - vtp_file (:obj:`str`): Path to generated VTP file

        Notes:
            - Requires compute_2d() to be called first and save=True
            - VTP (VTK XML PolyData) format compatible with ParaView and Meshlab
            - Recommended for publication-quality figures
            - File naming: surf_file + '_{surf_name}_zmax={zmax}_log.vtp'

        Examples::
            >>> landscape = LossLandscape(model, dataloader)
            >>> landscape.compute_2d(xrange=(-1, 1, 51), yrange=(-1, 1, 51))
            >>> # Export with log scale and clipping
            >>> vtp_file = landscape.export_paraview(log=True, zmax=10)
            >>> # Then open vtp_file in ParaView for visualization
        """
        vtp_file = self.surf_file.replace('.h5', '.vtp')
        viz.h5_to_vtp(self.surf_file, surf_name, log, zmax, interp, vtp_file)
        print(f"\nExported to: {vtp_file}")
        return vtp_file
