"""
Overview:
    Vector projection and geometric computation utilities for loss landscape analysis.
    This module provides essential mathematical operations for projecting optimization
    trajectories onto loss landscape directions and computing angular relationships.

This module provides:
    - Tensor list to single vector concatenation for unified parameter space operations
    - NumPy array to tensor conversion with automatic shape handling
    - 1D vector to structured tensor list conversion for model parameter reconstruction
    - Cosine similarity computation between parameter vectors or trajectories
    - 1D and 2D projection operations for visualizing high-dimensional paths on landscapes

Key Functions:
    - tensorlist_to_tensor: Flatten parameter list into single 1D tensor
    - nplist_to_tensor: Convert numpy array list to concatenated tensor
    - npvec_to_tensorlist: Reshape 1D direction vector to match model parameter structure
    - cal_angle: Compute cosine similarity (angular alignment) between two vectors
    - project_1D: Project weight vector onto single direction for 1D landscape coordinates
    - project_2D: Project weight vector onto 2D plane for 2D landscape coordinates

Notes:
    - These utilities bridge between model parameters and geometric landscape coordinates
    - Projections essential for plotting optimization paths on precomputed landscapes
    - Cosine similarity ranges from -1 (opposite) to +1 (aligned), 0 means orthogonal
"""

import torch
import numpy as np
import copy
from typing import List, Dict, Union, Tuple


def tensorlist_to_tensor(weights: List[torch.Tensor]) -> torch.Tensor:
    """
    Overview:
        Concatenate a list of parameter tensors into a single 1D vector.
        Flattens all parameter shapes into unified representation for geometric operations.

    Arguments:
        - weights (:obj:`List[torch.Tensor]`): List of parameter tensors with arbitrary shapes

    Returns:
        - vec (:obj:`torch.Tensor`): Single 1D tensor containing all parameters concatenated

    Notes:
        - Multi-dimensional tensors are flattened using view(w.numel())
        - Scalar or 1D tensors are converted to FloatTensor if needed
        - Preserves parameter order from input list
        - Useful for computing norms, distances, and projections in parameter space

    Examples::
        >>> # Flatten model parameters
        >>> weights = [torch.randn(10, 5), torch.randn(5)]  # Weight matrix and bias
        >>> vec = tensorlist_to_tensor(weights)
        >>> print(vec.shape)  # torch.Size([55])  (10*5 + 5 = 55)
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def nplist_to_tensor(nplist: List[np.ndarray]) -> torch.Tensor:
    """
    Overview:
        Concatenate a list of numpy arrays into a single 1D PyTorch tensor.
        Handles conversion from numpy to torch and flattens all array shapes.

    Arguments:
        - nplist (:obj:`List[numpy.ndarray]`): List of numpy arrays with arbitrary shapes

    Returns:
        - vec (:obj:`torch.Tensor`): Single 1D tensor containing all arrays concatenated

    Notes:
        - Converts numpy arrays to torch tensors with float64 precision
        - Multi-dimensional arrays are flattened, 1D arrays preserved
        - Useful for loading directions stored in HDF5 (numpy format) into torch operations
        - Maintains numerical precision during numpy-to-torch conversion

    Examples::
        >>> # Convert numpy direction to tensor
        >>> np_dirs = [np.random.randn(10, 5), np.random.randn(5)]
        >>> tensor_dir = nplist_to_tensor(np_dirs)
        >>> print(tensor_dir.shape)  # torch.Size([55])
    """
    v = []
    for d in nplist:
        w = torch.tensor(d * np.float64(1.0))
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction: np.ndarray, params: Union[List[torch.Tensor], Dict]) -> List[torch.Tensor]:
    """
    Overview:
        Reshape a 1D numpy direction vector into a list of tensors matching model parameter shapes.
        Inverse operation of tensorlist_to_tensor, reconstructs structured parameters from flat vector.

    Arguments:
        - direction (:obj:`numpy.ndarray`): 1D numpy array containing flattened parameter values
        - params (:obj:`List[torch.Tensor]` or :obj:`dict`): Reference structure for reshaping.
            Can be list of tensors from model.parameters() or state_dict from model.state_dict()

    Returns:
        - shaped_tensors (:obj:`List[torch.Tensor]`): List of tensors with shapes matching params

    Notes:
        - Total elements in direction must exactly match sum of parameter elements
        - Assertion ensures no mismatch between direction length and parameter count
        - Uses deepcopy for list params to avoid modifying original references
        - Supports both parameter list and state_dict formats for flexibility

    Examples::
        >>> # Reconstruct model parameters from direction vector
        >>> direction = np.random.randn(55)  # Flattened parameters
        >>> params = [torch.zeros(10, 5), torch.zeros(5)]  # Reference shapes
        >>> reconstructed = npvec_to_tensorlist(direction, params)
        >>> print([p.shape for p in reconstructed])  # [torch.Size([10, 5]), torch.Size([5])]
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx : idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert idx == len(direction)
        return w2
    else:
        s2 = []
        idx = 0
        for k, w in params.items():
            s2.append(torch.Tensor(direction[idx : idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert idx == len(direction)
        return s2


def cal_angle(vec1: Union[torch.Tensor, np.ndarray], vec2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Overview:
        Calculate cosine similarity between two vectors to measure angular alignment.
        Returns normalized dot product, useful for comparing optimization directions or trajectories.

    Arguments:
        - vec1 (:obj:`torch.Tensor` or :obj:`numpy.ndarray`): First vector (1D)
        - vec2 (:obj:`torch.Tensor` or :obj:`numpy.ndarray`): Second vector (1D, same length as vec1)

    Returns:
        - similarity (:obj:`float`): Cosine similarity value in range [-1, 1]
            - 1.0: Vectors perfectly aligned (same direction)
            - 0.0: Vectors orthogonal (perpendicular)
            - -1.0: Vectors opposite (antiparallel)

    Notes:
        - Automatically detects input type (torch.Tensor or numpy.ndarray) and uses appropriate operations
        - Returns Python float (scalar) for easy comparison and logging
        - Cosine similarity is scale-invariant: only direction matters, not magnitude
        - Useful for analyzing gradient alignment, direction independence, etc.

    Examples::
        >>> # Check if two directions are orthogonal
        >>> dir1 = torch.tensor([1.0, 0.0, 0.0])
        >>> dir2 = torch.tensor([0.0, 1.0, 0.0])
        >>> similarity = cal_angle(dir1, dir2)
        >>> print(f"Similarity: {similarity:.4f}")  # 0.0000 (orthogonal)

        >>> # Check gradient alignment with descent direction
        >>> grad = np.array([0.5, 0.5])
        >>> descent = np.array([0.7, 0.7])
        >>> alignment = cal_angle(grad, descent)
        >>> print(f"Alignment: {alignment:.4f}")  # ~1.0 (aligned)
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
        return (torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm())).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        raise TypeError("Inputs must be either torch.Tensor or numpy.ndarray")


def project_1D(w: torch.Tensor, d: torch.Tensor) -> float:
    """
    Overview:
        Project a weight vector onto a single direction to obtain 1D landscape coordinate.
        Computes scalar position along direction for plotting optimization paths on 1D landscapes.

    Arguments:
        - w (:obj:`torch.Tensor`): 1D tensor containing flattened weights or weight difference
        - d (:obj:`torch.Tensor`): 1D tensor representing the landscape direction (should be normalized)

    Returns:
        - coordinate (:obj:`float`): Scalar projection value (x-coordinate on 1D landscape)

    Notes:
        - Both vectors must have same length (assertion enforces this)
        - Formula: projection = (w · d) / ||d||
        - If d is normalized (||d|| = 1), this simplifies to dot product
        - Used to map optimization trajectory onto precomputed 1D loss landscape

    Examples::
        >>> # Project weight change onto random direction
        >>> weights_diff = torch.randn(100)  # w_current - w_initial
        >>> direction = torch.randn(100)
        >>> x_coord = project_1D(weights_diff, direction)
        >>> print(f"Position on landscape: {x_coord:.4f}")
    """
    assert len(w) == len(d), "Dimension mismatch between w and d"
    scale = torch.dot(w, d) / d.norm()
    return scale.item()


def project_2D(d: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, proj_method: str = 'cos') -> Tuple[float, float]:
    """
    Overview:
        Project a weight vector onto a 2D plane to obtain 2D landscape coordinates.
        Computes (x, y) position on plane spanned by two basis directions for 2D landscape visualization.

    Arguments:
        - d (:obj:`torch.Tensor`): 1D tensor containing flattened weights or weight difference
        - dx (:obj:`torch.Tensor`): 1D tensor for x-axis direction (should be normalized)
        - dy (:obj:`torch.Tensor`): 1D tensor for y-axis direction (should be normalized)
        - proj_method (:obj:`str`, optional): Projection method to use. Default is 'cos'
            - 'cos': Simple cosine projection (fast, assumes dx and dy are orthogonal)
            - 'lstsq': Least squares projection (robust, handles non-orthogonal directions)

    Returns:
        - x (:obj:`float`): X-coordinate on 2D landscape
        - y (:obj:`float`): Y-coordinate on 2D landscape

    Notes:
        - 'cos' method assumes orthogonal basis (dx ⊥ dy), projects independently on each axis
        - 'lstsq' method solves for best linear combination when basis is non-orthogonal
        - Use 'cos' for filter-normalized random directions (typically orthogonal)
        - Use 'lstsq' when plotting on arbitrary direction pairs or when orthogonality unclear
        - All three vectors must have same length

    Examples::
        >>> # Project optimization path onto 2D landscape
        >>> weights_diff = torch.randn(100)
        >>> dir_x = torch.randn(100)
        >>> dir_y = torch.randn(100)
        >>> x, y = project_2D(weights_diff, dir_x, dir_y, proj_method='cos')
        >>> print(f"2D coordinates: ({x:.4f}, {y:.4f})")

        >>> # Use least squares for non-orthogonal basis
        >>> x, y = project_2D(weights_diff, dir_x, dir_y, proj_method='lstsq')
    """
    if proj_method == 'cos':
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy(), rcond=None)[0]
    else:
        raise ValueError(f"Unknown projection method: {proj_method}")

    return x, y
