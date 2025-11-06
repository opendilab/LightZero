"""
Simplified DeepspeedStrategy for evaluation only.
This is a minimal version that only provides the interface needed for data processing.
"""

from abc import ABC


class DeepspeedStrategy(ABC):
    """
    Minimal strategy class for evaluation.
    The full version is used for training, but evaluation only needs this empty interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize strategy (no-op for evaluation)"""
        pass
