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
        **kwargs
    ):
        """
        Initialize PriorZeroEvaluator.

        Args:
            vllm_engine: vLLM async engine (optional, for future use)
            **kwargs: Arguments for parent MuZeroEvaluator
        """
        super().__init__(**kwargs)

    # All other methods are inherited from MuZeroEvaluator
    # The policy's _forward_collect already handles LLM prior integration
