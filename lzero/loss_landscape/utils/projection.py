"""
Vector projection and angle calculation utilities.
"""

import torch
import numpy as np
import copy


def tensorlist_to_tensor(weights):
    """Concatenate a list of tensors into a single 1D tensor.

    Args:
        weights: List of parameter tensors

    Returns:
        Concatenated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def nplist_to_tensor(nplist):
    """Concatenate a list of numpy arrays into a single 1D tensor.

    Args:
        nplist: List of numpy arrays

    Returns:
        Concatenated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d * np.float64(1.0))
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction, params):
    """Convert a 1D numpy array to a list of tensors with given shapes.

    Args:
        direction: 1D numpy array
        params: List of tensors or state_dict

    Returns:
        List of tensors with same shapes as params
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


def cal_angle(vec1, vec2):
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: Torch tensor or numpy array
        vec2: Torch tensor or numpy array

    Returns:
        Cosine similarity (scalar between -1 and 1)
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
        return (torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm())).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        raise TypeError("Inputs must be either torch.Tensor or numpy.ndarray")


def project_1D(w, d):
    """Project vector w onto direction d.

    Args:
        w: 1D tensor (vectorized weights)
        d: 1D tensor (direction)

    Returns:
        Projection scalar
    """
    assert len(w) == len(d), "Dimension mismatch between w and d"
    scale = torch.dot(w, d) / d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method='cos'):
    """Project vector d onto 2D plane spanned by dx and dy.

    Args:
        d: 1D tensor (vectorized weights/direction)
        dx: 1D tensor (first basis direction)
        dy: 1D tensor (second basis direction)
        proj_method: Projection method
                     - 'cos': simple cosine projection (assumes orthogonal)
                     - 'lstsq': least squares projection

    Returns:
        Tuple of (x, y) projection coordinates
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
