# priorzero_evaluator.py
"""
[PRIORZERO] PriorZero Evaluator

Simple evaluator that inherits from MuZeroEvaluator.
Since the policy already integrates LLM priors in its _forward_collect method,
the evaluator can use the parent implementation directly.

Author: PriorZero Team
Date: 2025-01-20
"""

from typing import Optional

from ding.worker.collector.base_serial_evaluator import SERIAL_EVALUATOR_REGISTRY
from lzero.worker.muzero_evaluator import MuZeroEvaluator as OriginalEvaluator
from vllm import AsyncLLMEngine


@SERIAL_EVALUATOR_REGISTRY.register('priorzero', force_overwrite=True)
class PriorZeroEvaluator(OriginalEvaluator):
    """
    [PRIORZERO-MODIFIED]
    Evaluator for PriorZero.

    Since the PriorZero policy already integrates LLM priors in its
    _forward_collect method, this evaluator simply inherits all
    functionality from MuZeroEvaluator.

    The vLLM engine is passed for potential future enhancements
    (e.g., comparative evaluation with/without LLM priors).
    """

    def __init__(
        self,
        vllm_engine: Optional[AsyncLLMEngine] = None,
        **kwargs
    ):
        """
        Initialize PriorZeroEvaluator.

        Args:
            vllm_engine: vLLM async engine (optional, for future use)
            **kwargs: Arguments for parent MuZeroEvaluator
        """
        super().__init__(**kwargs)
        self.vllm_engine = vllm_engine

        if vllm_engine is not None:
            self._logger.info("✓ PriorZeroEvaluator initialized with vLLM engine")
        else:
            self._logger.info("✓ PriorZeroEvaluator initialized (no vLLM engine)")

    # All other methods are inherited from MuZeroEvaluator
    # The policy's _forward_collect already handles LLM prior integration
