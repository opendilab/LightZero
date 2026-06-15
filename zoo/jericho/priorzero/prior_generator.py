"""
Unified Prior Generator Interface

This module provides a unified interface for generating action priors
from different types of observations (text or image).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import numpy as np
import torch
from PIL import Image


class PriorGenerator(ABC):
    """
    Abstract base class for prior generators.

    Subclasses should implement generate_prior() to generate action prior
    distributions from observations.
    """

    def __init__(self, model_name: str, obs_type: str):
        """
        Args:
            model_name: Name/path of the model
            obs_type: Type of observation ('text' or 'image')
        """
        self.model_name = model_name
        self.obs_type = obs_type

    @abstractmethod
    def generate_prior(
        self,
        observation: Any,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate action prior distribution from observation.

        Args:
            observation: Observation (text string or image array/PIL Image)
            action_candidates: List of valid action strings
            history: Optional history of previous (obs, action, reward) tuples
            temperature: Temperature for sampling
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing:
                - 'action_probs': np.ndarray of shape (num_actions,) with probabilities
                - 'action_logits': np.ndarray of shape (num_actions,) with logits
                - 'raw_output': Raw model output (for logging/debugging)
        """
        pass

    @abstractmethod
    def batch_generate_prior(
        self,
        observations: List[Any],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch version of generate_prior for efficiency.

        Args:
            observations: List of observations
            action_candidates_list: List of action candidate lists
            histories: Optional list of histories
            temperature: Temperature for sampling
            **kwargs: Additional arguments

        Returns:
            List of prior dictionaries (same format as generate_prior)
        """
        pass


class LLMPriorGenerator(PriorGenerator):
    """
    Prior generator using Language Models for text observations.

    This is a wrapper around the existing vLLM engine and DataProcessor.
    """

    def __init__(
        self,
        vllm_engine,
        data_processor,
        model_name: str,
        use_cot: bool = True,
        **kwargs
    ):
        """
        Args:
            vllm_engine: vLLM engine instance
            data_processor: DataProcessor instance
            model_name: LLM model name
            use_cot: Whether to use Chain-of-Thought
        """
        super().__init__(model_name, obs_type='text')
        self.vllm_engine = vllm_engine
        self.data_processor = data_processor
        self.use_cot = use_cot

    def generate_prior(
        self,
        observation: str,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prior from text observation using LLM.

        Args:
            observation: Text observation string
            action_candidates: List of valid action strings
            history: Optional history buffer
            temperature: Sampling temperature

        Returns:
            Prior dictionary with action_probs, action_logits, raw_output
        """
        # Use existing DataProcessor logic
        # This delegates to the existing implementation
        result = self.data_processor.get_action_prior_single(
            text_obs=observation,
            action_candidates=action_candidates,
            history=history,
            temperature=temperature,
            use_cot=self.use_cot,
        )

        return result

    def batch_generate_prior(
        self,
        observations: List[str],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch generate priors from text observations.
        """
        # Use existing DataProcessor batch logic
        results = self.data_processor.get_action_prior_batch(
            text_obs_list=observations,
            action_candidates_list=action_candidates_list,
            histories=histories,
            temperature=temperature,
            use_cot=self.use_cot,
        )

        return results


class VLPriorGenerator(PriorGenerator):
    """
    Prior generator using Vision-Language (VL) models for image observations.

    Supports models like Qwen-VL, LLaVA, InternVL, etc.

    Includes training sample construction for PPO optimization with advantages.
    """

    def __init__(
        self,
        vl_engine,
        model_name: str,
        use_cot: bool = True,
        tokenizer=None,
        game_description: str = "",
        vlm_image_mode: str = "current_only",
        prompt_style: str = "concise",
        logprob_extraction_mode: str = "exact",
        **kwargs
    ):
        """
        Args:
            vl_engine: VL engine instance (to be implemented)
            model_name: VL model name
            use_cot: Whether to use Chain-of-Thought reasoning
            tokenizer: Tokenizer for building training samples
            game_description: Game-specific description for prompts
            vlm_image_mode: Image mode - "current_only", "first_and_current", or "all_history"
            prompt_style: "concise" (shorter, better for small VLMs) or "legacy" (verbose, original)
            logprob_extraction_mode: "exact" (LLM-aligned, default) or "approximate" (fallback with pseudo logprobs)
        """
        super().__init__(model_name, obs_type='image')
        self.vl_engine = vl_engine
        self.use_cot = use_cot
        self.tokenizer = tokenizer
        self.game_description = game_description
        self.vlm_image_mode = vlm_image_mode
        self.prompt_style = prompt_style
        self.logprob_extraction_mode = logprob_extraction_mode

        # For logging VL outputs
        self.episode_output = []

        # Log control: only log every N calls
        self.log_interval = 100  # Log every 100 calls
        self.call_count = 0
        self.batch_call_count = 0

    def _convert_obs_to_pil_image(self, obs: np.ndarray) -> Image.Image:
        """
        Robustly convert observation array to PIL Image.

        Handles various input formats:
        - CHW format (C, H, W): channels first, e.g., (3, 64, 64)
        - HWC format (H, W, C): channels last, e.g., (64, 64, 3)
        - Grayscale (H, W): single channel, e.g., (64, 64)
        - Stacked frames (N, H, W): takes the last frame

        Args:
            obs: Observation array

        Returns:
            PIL Image in RGB format

        Raises:
            ValueError: If observation shape is invalid
        """
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(obs)}")

        # Ensure uint8 dtype
        if obs.dtype != np.uint8:
            # Normalize to [0, 255] if needed
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)

        # Handle different shapes
        if obs.ndim == 2:
            # Grayscale (H, W) -> convert to RGB
            return Image.fromarray(obs, mode='L').convert('RGB')

        elif obs.ndim == 3:
            # Determine if CHW or HWC format
            c, h, w = obs.shape

            # If first dimension is small (1-4), likely CHW format
            if c <= 4 and h > c and w > c:
                # CHW format -> transpose to HWC
                if c == 1:
                    # Single channel (1, H, W) -> (H, W)
                    obs = obs[0]
                    return Image.fromarray(obs, mode='L').convert('RGB')
                elif c == 3:
                    # RGB (3, H, W) -> (H, W, 3)
                    obs = np.transpose(obs, (1, 2, 0))
                    return Image.fromarray(obs)
                elif c == 4:
                    # RGBA or stacked frames
                    # Take last 3 channels as RGB
                    obs = np.transpose(obs[-3:], (1, 2, 0))
                    return Image.fromarray(obs)
                else:
                    # Stacked grayscale frames (N, H, W) -> take last frame
                    obs = obs[-1]
                    return Image.fromarray(obs, mode='L').convert('RGB')

            # Otherwise, assume HWC format
            elif w <= 4 and h > w and c > w:
                # HWC format
                if w == 1:
                    # Single channel (H, W, 1) -> (H, W)
                    obs = obs[:, :, 0]
                    return Image.fromarray(obs, mode='L').convert('RGB')
                elif w == 3:
                    # RGB (H, W, 3)
                    return Image.fromarray(obs)
                elif w == 4:
                    # RGBA (H, W, 4) -> take first 3 channels
                    obs = obs[:, :, :3]
                    return Image.fromarray(obs)

            # Ambiguous shape - provide detailed error
            raise ValueError(
                f"Cannot determine image format from shape {obs.shape}. "
                f"Expected CHW (C, H, W) with C<=4 or HWC (H, W, C) with C<=4. "
                f"Please ensure observation is in correct format."
            )

        elif obs.ndim == 4:
            # Batch dimension (B, C, H, W) or (B, H, W, C) -> take first image
            raise ValueError(
                f"Observation has batch dimension {obs.shape}. "
                f"Please pass individual observations, not batches."
            )

        else:
            raise ValueError(
                f"Invalid observation shape {obs.shape}. "
                f"Expected 2D (H, W) or 3D (C, H, W) or (H, W, C)."
            )

    def _assemble_images(
        self,
        current_obs: Union[np.ndarray, Image.Image],
        history: Optional[List] = None,
    ) -> List[Image.Image]:
        """
        Assemble image list based on vlm_image_mode.

        Args:
            current_obs: Current frame observation
            history: History entries, each is (raw_obs, action, reward, timestep)

        Returns:
            List of PIL Images to send to the VL model
        """
        if self.vlm_image_mode == "current_only":
            current_image = self._convert_obs_to_pil_image(current_obs) if isinstance(current_obs, np.ndarray) else current_obs
            return [current_image]

        # Extract history images
        history_images = []
        if history:
            for entry in history:
                obs = entry[0]  # (raw_obs, action, reward, timestep)
                if isinstance(obs, np.ndarray):
                    history_images.append(self._convert_obs_to_pil_image(obs))
                elif isinstance(obs, Image.Image):
                    history_images.append(obs)
                # Skip non-image observations (e.g. text strings)

        current_image = self._convert_obs_to_pil_image(current_obs) if isinstance(current_obs, np.ndarray) else current_obs

        if self.vlm_image_mode == "first_and_current":
            if history_images:
                return [history_images[0], current_image]
            return [current_image]

        elif self.vlm_image_mode == "all_history":
            return history_images + [current_image]

        # Fallback (should not reach here due to validation)
        return [current_image]

    def get_system_prompt(self) -> str:
        """System prompt — dispatches to concise or legacy style."""
        if self.prompt_style == "concise":
            return self._get_system_prompt_concise()
        return self._get_system_prompt_legacy()

    def _get_system_prompt_concise(self) -> str:
        """Short system prompt optimized for small VLMs (2B-7B)."""
        if self.use_cot:
            return (
                "You play an image-based game. Pick the best action.\n"
                "Reply EXACTLY:\nReasoning: <1 sentence>\nAction: <action>"
            )
        return "You play an image-based game. Pick the best action.\nReply EXACTLY:\nAction: <action>"

    def _get_system_prompt_legacy(self) -> str:
        """
        System prompt for VL — mirrors LLM's get_system_prompt(),
        only replacing "text-based adventure game" with image-based context.
        """
        parts = [
            "You are an expert player in an image-based game. Your goal is to maximize the score by choosing the optimal next action.",
            "Analyze the game screen and history to decide the single best next action.",
            "IMPORTANT: You MUST choose EXACTLY ONE action from the provided valid actions list. Output the action name EXACTLY as given.",
        ]

        if self.use_cot:
            parts.append(
                "OUTPUT FORMAT (you MUST follow this EXACTLY):\n"
                "Reasoning: <brief analysis in 1-3 sentences>\n"
                "Action: <exact_action_name>\n\n"
                "RULES:\n"
                "- Keep reasoning SHORT (1-3 sentences max).\n"
                "- The Action line MUST contain exactly one action name from the valid actions list.\n"
                "- Do NOT add any text after the action name."
            )
        else:
            parts.append(
                "OUTPUT FORMAT:\n"
                "Action: <exact_action_name>\n\n"
                "Output ONLY this single line. No other text."
            )
        return "\n".join(parts)

    def get_user_prompt(
        self,
        action_candidates: List[str],
        history: Optional[List] = None,
        num_images: int = 1,
    ) -> str:
        """User prompt — dispatches to concise or legacy style."""
        if self.prompt_style == "concise":
            return self._get_user_prompt_concise(action_candidates, history, num_images)
        return self._get_user_prompt_legacy(action_candidates, history, num_images)

    def _get_user_prompt_concise(
        self,
        action_candidates: List[str],
        history: Optional[List] = None,
        num_images: int = 1,
    ) -> str:
        """
        Concise user prompt: minimal tokens, maximum signal.
        Designed for small VLMs (2B-7B) where instruction-following degrades with long prompts.
        """
        parts = []

        # Game description — one line only
        if self.game_description:
            # Take only the first sentence of game_description
            first_sentence = self.game_description.split('\n')[0].strip()
            parts.append(first_sentence)

        # Multi-image labelling
        if self.vlm_image_mode != "current_only" and num_images > 1:
            img_idx = 1
            if history and len(history) > 0:
                for entry in history:
                    action = entry[1]
                    reward = entry[2]
                    has_image = isinstance(entry[0], (np.ndarray, Image.Image))
                    if self.vlm_image_mode == "all_history" and has_image and img_idx < num_images:
                        parts.append(f"[Image {img_idx}] Action: {action}, Reward: {reward}")
                        img_idx += 1
                    elif self.vlm_image_mode == "first_and_current" and has_image and img_idx == 1:
                        parts.append(f"[Image {img_idx} - initial] Action: {action}, Reward: {reward}")
                        img_idx += 1
                    else:
                        parts.append(f"Action: {action}, R: {reward}")
            parts.append(f"[Image {num_images}] Current screen.")
        else:
            # Single image — text-only history
            if history and len(history) > 0:
                hist_strs = []
                for entry in history:
                    action, reward = entry[1], entry[2]
                    hist_strs.append(f"{action}(R:{reward})")
                parts.append("History: " + " → ".join(hist_strs))
            parts.append("Current screen shown above.")

        # Valid actions — compact
        actions_str = ", ".join(action_candidates)
        parts.append(f"Actions: [{actions_str}]")

        # LunarLander-specific compact hints
        if set(action_candidates) == {"NOOP", "LEFT_ENGINE", "MAIN_ENGINE", "RIGHT_ENGINE"}:
            parts.append(
                "NOOP=do nothing | LEFT_ENGINE=push right,rotate CW(-0.03) | "
                "MAIN_ENGINE=slow descent(-0.3) | RIGHT_ENGINE=push left,rotate CCW(-0.03)\n"
                "Goal: land on pad horizontally. Crash=-100, land=+100."
            )

        # Instruction
        if self.use_cot:
            parts.append("Reasoning: <1 sentence>\nAction: <one action>")
        else:
            parts.append("Action: <one action>")

        return "\n".join(parts)

    def _get_user_prompt_legacy(
        self,
        action_candidates: List[str],
        history: Optional[List] = None,
        num_images: int = 1,
    ) -> str:
        """
        User prompt for VL — mirrors LLM's get_user_prompt() structure,
        replacing text observation with image vision tokens.

        Args:
            action_candidates: List of valid action names
            history: Optional history entries
            num_images: Number of images being sent (for multi-image labelling)
        """
        prompt_parts = []

        # Multi-image mode: label each image in the prompt
        if self.vlm_image_mode != "current_only" and num_images > 1:
            img_idx = 1  # 1-based image index for the prompt

            if history and len(history) > 0:
                prompt_parts.append("=== GAME HISTORY ===")
                for entry in history:
                    if len(entry) >= 4:
                        obs, action, reward, timestep = entry[0], entry[1], entry[2], entry[3]
                    else:
                        obs, action, reward = entry[0], entry[1], entry[2]
                        timestep = None

                    # Check if this history entry has a corresponding image
                    has_image = isinstance(obs, (np.ndarray, Image.Image))

                    if self.vlm_image_mode == "all_history" and has_image and img_idx < num_images:
                        step_label = f"Step {timestep}" if timestep is not None else "Step"
                        prompt_parts.append(f"=== HISTORICAL OBSERVATION ({step_label}) ===")
                        prompt_parts.append(f"[See image {img_idx} above]")
                        if timestep is not None:
                            prompt_parts.append(f"Action: {action}, Reward: {reward}")
                        else:
                            prompt_parts.append(f"Action: {action}, Reward: {reward}")
                        img_idx += 1
                    elif self.vlm_image_mode == "first_and_current" and has_image and img_idx == 1:
                        step_label = f"Step {timestep}" if timestep is not None else "First Step"
                        prompt_parts.append(f"=== INITIAL OBSERVATION ({step_label}) ===")
                        prompt_parts.append(f"[See image {img_idx} above]")
                        prompt_parts.append(f"Action: {action}, Reward: {reward}")
                        img_idx += 1
                    else:
                        # Text-only history entry
                        if timestep is not None:
                            prompt_parts.append(f"Step {timestep}: Action: {action}, Reward: {reward}")
                        else:
                            prompt_parts.append(f"Action: {action}, Reward: {reward}")

                prompt_parts.append("")  # empty line separator

            prompt_parts.append("=== CURRENT OBSERVATION ===")
            prompt_parts.append(f"[See image {num_images} above]")
            prompt_parts.append("\nLook at the image carefully and analyze:")
            prompt_parts.append("- The lander's tilt angle (horizontal, tilted left, or tilted right?)")
            prompt_parts.append("- The lander's horizontal position relative to the landing pad")
            prompt_parts.append("- Visual indicators of descent speed")

        else:
            # Original single-image prompt (current_only mode or only 1 image)
            if history and len(history) > 0:
                prompt_parts.append("=== GAME HISTORY ===")
                for entry in history:
                    if len(entry) >= 4:
                        obs, action, reward, timestep = entry[0], entry[1], entry[2], entry[3]
                        prompt_parts.append(f"Step {timestep}: Action: {action}, Reward: {reward}")
                    else:
                        obs, action, reward = entry[0], entry[1], entry[2]
                        prompt_parts.append(f"Action: {action}, Reward: {reward}")
                prompt_parts.append("")  # empty line separator

            prompt_parts.append("=== CURRENT OBSERVATION ===")
            prompt_parts.append("[See the game screen image above]")
            prompt_parts.append("\nLook at the image carefully and analyze:")
            prompt_parts.append("- The lander's tilt angle (horizontal, tilted left, or tilted right?)")
            prompt_parts.append("- The lander's horizontal position relative to the landing pad")
            prompt_parts.append("- Visual indicators of descent speed")
        if self.game_description:
            prompt_parts.append(self.game_description)

        if action_candidates and len(action_candidates) > 0:
            actions_str = ", ".join(action_candidates)
            prompt_parts.append(f"\nValid actions: [{actions_str}]")

            # Add per-action descriptions for LunarLander
            # (For other games, the game_description already covers action semantics)
            if set(action_candidates) == {"NOOP", "LEFT_ENGINE", "MAIN_ENGINE", "RIGHT_ENGINE"}:
                prompt_parts.append(
                    "- NOOP: Do nothing (0 cost).\n"
                    "- LEFT_ENGINE: Fires the left thruster. Pushes the lander RIGHT and rotates it clockwise. (-0.03 cost)\n"
                    "- MAIN_ENGINE: Fires the bottom thruster. Slows descent. (-0.3 cost)\n"
                    "- RIGHT_ENGINE: Fires the right thruster. Pushes the lander LEFT and rotates it counter-clockwise. (-0.03 cost)\n"
                    "\n"
                    "=== STRATEGY GUIDE ===\n"
                    "1. Keep Horizontal: The game penalizes tilt. Correct tilt immediately. If tilted left, fire LEFT_ENGINE to rotate clockwise. If tilted right, fire RIGHT_ENGINE.\n"
                    "2. Conserve Main Fuel: MAIN_ENGINE is very expensive (-0.3). Use it ONLY if falling too fast.\n"
                    "3. Steer to Center: Use side engines to adjust horizontal position toward the flags.\n"
                    "4. Coasting: If the lander is horizontal, aligned with the pad, and descending slowly, use NOOP to save points."
                )

        prompt_parts.append("\n=== INSTRUCTION ===")
        if self.use_cot:
            prompt_parts.append(
                "Choose the best action. You MUST respond in EXACTLY this format:\n"
                "Reasoning: <Sentence 1: Describe the lander's current tilt, vertical/horizontal speed, and position. "
                "Sentence 2: Explain why the action is optimal based on the reward structure and physics.>\n"
                "Action: <EXACTLY ONE word from: NOOP, LEFT_ENGINE, MAIN_ENGINE, RIGHT_ENGINE>\n"
                "\n"
                "CRITICAL RULES:\n"
                "- Write ONLY the action name after 'Action:', nothing else.\n"
                "- Do NOT add punctuation, arrows (->), or explanations after the action.\n"
                "- Do NOT write 'NOPE' or 'NO_OP', only 'NOOP'.\n"
                "\n"
                "Example 1:\n"
                "Reasoning: The lander is tilted left and drifting left of the pad; firing LEFT_ENGINE will rotate it clockwise back to horizontal and push it right toward the center at a low cost.\n"
                "Action: LEFT_ENGINE\n"
                "\n"
                "Example 2:\n"
                "Reasoning: The lander is horizontal and centered, but falling too rapidly; despite the high cost, MAIN_ENGINE is strictly necessary to slow the descent and prevent a -100 crash penalty.\n"
                "Action: MAIN_ENGINE\n"
                "\n"
                "Example 3:\n"
                "Reasoning: The lander is perfectly horizontal, aligned above the pad, and descending at a safe, slow speed; no thrust is needed, so doing nothing avoids point deductions.\n"
                "Action: NOOP"
            )
        else:
            example_action = action_candidates[1] if len(action_candidates) >= 2 else (action_candidates[0] if action_candidates else "NOOP")
            prompt_parts.append(
                f"Choose the best action. Output ONLY:\n"
                f"Action: <one action from the valid actions list>\n\n"
                f"Example:\nAction: {example_action}"
            )
        return "\n".join(prompt_parts)

    def _parse_vl_output_with_cot(
        self,
        raw_output: str,
        action_candidates: List[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Parse VL output to extract action and optional CoT reasoning.

        Args:
            raw_output: Raw VL output string
            action_candidates: List of valid action names

        Returns:
            Tuple of (chosen_action, cot_prefix)
            - chosen_action: The selected action name
            - cot_prefix: The reasoning part (if use_cot=True), else None
        """
        import re

        cot_prefix = None
        chosen_action = None

        # Extract reasoning part (if present)
        if self.use_cot:
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Action:|$)', raw_output, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                cot_prefix = reasoning_match.group(1).strip()

        # Strategy 1: Extract text after "Action:" and match against candidates
        # Use .+ instead of \S+ to capture multi-word or underscore-separated actions
        action_match = re.search(r'Action:\s*(.+)', raw_output, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip().strip("'\"`.,:;")
            # Exact match (case-insensitive)
            for candidate in action_candidates:
                if candidate.upper() == action_str.upper():
                    chosen_action = candidate
                    break
            # If no exact match, try if candidate is contained in the extracted text
            if chosen_action is None:
                for candidate in action_candidates:
                    if candidate.upper() in action_str.upper():
                        chosen_action = candidate
                        break

        # Strategy 2: If no "Action:" line found, scan entire output for action names
        if chosen_action is None:
            # Search for exact action name mentions in the output (prefer later mentions)
            last_found = None
            for candidate in action_candidates:
                # Use word boundary to avoid partial matches
                pattern = re.escape(candidate)
                matches = list(re.finditer(pattern, raw_output, re.IGNORECASE))
                if matches:
                    pos = matches[-1].start()
                    if last_found is None or pos > last_found[1]:
                        last_found = (candidate, pos)
            if last_found is not None:
                chosen_action = last_found[0]

        # Fallback: if no valid action found, use first candidate
        if chosen_action is None:
            chosen_action = action_candidates[0] if action_candidates else "NOOP"

        return chosen_action, cot_prefix

    def _extract_action_logprobs_batch(
        self,
        image_list: List[Image.Image],
        prompt: str,
        action_candidates: List[str],
        cot_prefix: Optional[str],
        temperature: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Extract action log probabilities with configurable mode.
        """
        if self.logprob_extraction_mode == "exact":
            return self._extract_logprobs_exact_mode(
                image_list, prompt, action_candidates, cot_prefix, temperature
            )
        else:  # approximate mode (default)
            return self._extract_logprobs_approximate_mode(
                image_list, prompt, action_candidates, cot_prefix, temperature
            )

    def _extract_logprobs_approximate_mode(
        self,
        image_list: List[Image.Image],
        prompt: str,
        action_candidates: List[str],
        cot_prefix: Optional[str],
        temperature: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Approximate mode: Use fallback with pseudo token data.
        Fast but less accurate.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            rollout_action_logprob_dict = {}
            full_ids_dict = {}
            label_ids_dict = {}

            for action in action_candidates:
                if self.use_cot and cot_prefix:
                    label_text = cot_prefix + " " + action
                else:
                    label_text = "Action: " + action

                label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
                full_prompt = prompt + "\n" + label_text
                full_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

                pseudo_logprobs = [0.0] * len(label_ids)

                rollout_action_logprob_dict[action] = pseudo_logprobs
                full_ids_dict[action] = full_ids
                label_ids_dict[action] = label_ids

            return None, rollout_action_logprob_dict, full_ids_dict, label_ids_dict

        except Exception as e:
            logger.error(f"⚠️ Approximate mode failed: {e}", exc_info=True)

        return None, {}, {}, {}

    def _extract_logprobs_exact_mode(
        self,
        image_list: List[Image.Image],
        prompt: str,
        action_candidates: List[str],
        cot_prefix: Optional[str],
        temperature: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Exact mode: Use token IDs like LLM (bypassing chat template).
        """
        import logging
        import math
        logger = logging.getLogger(__name__)

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            if self.use_cot and cot_prefix:
                label_texts = [cot_prefix + " " + action for action in action_candidates]
                label_texts_no_cots = [" " + action for action in action_candidates]
            else:
                label_texts = ["Action: " + action for action in action_candidates]
                label_texts_no_cots = label_texts

            label_ids_list = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in label_texts]
            label_ids_no_cots_list = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in label_texts_no_cots]
            full_ids_list = [prompt_ids + label_ids for label_ids in label_ids_list]

            results = self.vl_engine.batch_generate_with_token_ids(
                images=[image_list] * len(action_candidates),
                prompt_token_ids=full_ids_list,
                temperature=temperature,
                max_new_tokens=1,
                return_logprobs=True,
            )

            action_scores = []
            rollout_action_logprob_dict = {}
            full_ids_dict = {}
            label_ids_dict = {}

            for action, label_ids, label_ids_no_cot, full_ids, result in zip(
                action_candidates, label_ids_list, label_ids_no_cots_list, full_ids_list, results
            ):
                prompt_logprobs = result.get('prompt_logprobs') if isinstance(result, dict) else None

                if not prompt_logprobs or len(prompt_logprobs) == 0:
                    action_scores.append(float("-inf"))
                    rollout_action_logprob_dict[action] = []
                    full_ids_dict[action] = []
                    label_ids_dict[action] = []
                    continue

                token_lps = []
                for j in range(1, len(full_ids)):
                    tok_id = full_ids[j]
                    lp_dict = prompt_logprobs[j]

                    if lp_dict is None or tok_id not in lp_dict:
                        break

                    logprob_obj = lp_dict[tok_id]
                    logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else float(logprob_obj)

                    if math.isnan(logprob):
                        break

                    token_lps.append(logprob)

                if len(token_lps) > 0:
                    l_len = len(label_ids)
                    l_no_cots_len = len(label_ids_no_cot)
                    label_lps = token_lps[-l_len:]

                    if self.use_cot:
                        target_lps = label_lps
                    else:
                        target_lps = label_lps[-l_no_cots_len:]

                    score = sum(target_lps) / len(target_lps)
                    action_scores.append(score)
                    rollout_action_logprob_dict[action] = label_lps
                    full_ids_dict[action] = full_ids
                    label_ids_dict[action] = label_ids
                else:
                    action_scores.append(float("-inf"))
                    rollout_action_logprob_dict[action] = []
                    full_ids_dict[action] = []
                    label_ids_dict[action] = []

            valid_count = sum(1 for s in action_scores if s > float("-inf"))
            if valid_count == len(action_candidates):
                return np.array(action_scores, dtype=np.float32), rollout_action_logprob_dict, full_ids_dict, label_ids_dict

            logger.warning(f"⚠️ Exact mode: {valid_count}/{len(action_candidates)} valid")

        except Exception as e:
            logger.error(f"⚠️ Exact mode failed: {e}")

        return None, {}, {}, {}

    def _action_to_logprob(
        self,
        chosen_action: str,
        action_candidates: List[str],
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Convert chosen action to log probability distribution.

        For training, we need to store the "old" log probabilities that were used
        to select the action. This creates a peaked distribution around the chosen action.

        Args:
            chosen_action: The action selected by VL
            action_candidates: List of all valid actions
            temperature: Temperature for softening the distribution

        Returns:
            Log probability array of shape (num_actions,)
        """
        num_actions = len(action_candidates)

        # Create peaked but NOT one-hot distribution to preserve MCTS exploration.
        # Use moderate logit gap (2.0 vs 0.0) instead of extreme (10.0 vs -10.0),
        # so the prior is informative but not deterministic.
        logits = np.zeros(num_actions, dtype=np.float32)

        try:
            chosen_idx = action_candidates.index(chosen_action)
            logits[chosen_idx] = 2.0  # Moderate logit for chosen action
        except ValueError:
            # If chosen action not in candidates, uniform distribution
            logits = np.zeros(num_actions)

        # Apply temperature and convert to log probabilities (numerically stable)
        logits = logits / temperature
        max_logit = np.max(logits)
        log_probs = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit)) + 1e-10)

        return log_probs

    def _parse_vl_output(
        self,
        raw_output: str,
        action_candidates: List[str]
    ) -> np.ndarray:
        """
        Parse VL output to extract action probabilities.

        Args:
            raw_output: Raw text output from VL
            action_candidates: List of valid action names (e.g., ['NOOP', 'FIRE', 'RIGHT'])

        Returns:
            Action probabilities as numpy array
        """
        import json
        import re

        # Try to extract JSON from output
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{[^}]+\}', raw_output)
            if json_match:
                action_probs_dict = json.loads(json_match.group())

                # Convert to array aligned with action_candidates
                probs = []
                for action in action_candidates:
                    # Try exact match and case-insensitive match
                    prob = action_probs_dict.get(action,
                           action_probs_dict.get(action.upper(),
                           action_probs_dict.get(action.lower(), 0.0)))
                    probs.append(prob)

                probs = np.array(probs, dtype=np.float32)

                # Normalize
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # Fallback to uniform
                    probs = np.ones(len(action_candidates), dtype=np.float32) / len(action_candidates)

                return probs
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse VL output: {e}. Using uniform prior.")
            logger.debug(f"Raw output: {raw_output}")

        # Fallback: uniform distribution
        return np.ones(len(action_candidates), dtype=np.float32) / len(action_candidates)

    def generate_prior(
        self,
        observation: Union[np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prior with LLM-aligned token-level data.
        """
        self.call_count += 1

        # Assemble images
        image_list = self._assemble_images(observation, history)
        prompt = self.get_user_prompt(action_candidates, history, num_images=len(image_list))

        # Step 1: Generate to get chosen action and CoT prefix
        result = self.vl_engine.generate(
            image=image_list,
            prompt=prompt,
            temperature=temperature,
            system_prompt=self.get_system_prompt(),
            return_logprobs=False,
            **kwargs
        )
        raw_output = result.get('text', '') if isinstance(result, dict) else result
        chosen_action, cot_prefix = self._parse_vl_output_with_cot(raw_output, action_candidates)

        # Step 2: Extract logprobs with token-level data (same as LLM)
        action_log_probs, rollout_logprob_dict, full_ids_dict, label_ids_dict = self._extract_action_logprobs_batch(
            image_list, prompt, action_candidates, cot_prefix, temperature
        )

        # Fallback if batch extraction failed
        if action_log_probs is None:
            action_log_probs = self._action_to_logprob(chosen_action, action_candidates, temperature)
            rollout_logprob_dict = {}
            full_ids_dict = {}
            label_ids_dict = {}

        action_probs = np.exp(action_log_probs)

        return {
            'action_probs': action_probs,
            'action_logits': action_log_probs,
            'raw_output': raw_output,
            'cot_prefix': cot_prefix,
            'chosen_action': chosen_action,
            'rollout_action_logprob': rollout_logprob_dict,
            'full_ids': full_ids_dict,
            'label_ids': label_ids_dict,
        }

    def batch_generate_prior(
        self,
        observations: List[Union[np.ndarray, Image.Image]],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch generate priors with LLM-aligned token-level data.
        """
        if histories is None:
            histories = [None] * len(observations)

        # Assemble images and prompts
        image_lists = []
        prompts = []
        for obs, history, action_candidates in zip(observations, histories, action_candidates_list):
            image_list = self._assemble_images(obs, history)
            prompt = self.get_user_prompt(action_candidates, history, num_images=len(image_list))
            image_lists.append(image_list)
            prompts.append(prompt)

        # Step 1: Generate to get chosen actions and CoT prefixes
        raw_outputs = self.vl_engine.batch_generate(
            images=image_lists,
            prompts=prompts,
            temperature=temperature,
            system_prompt=self.get_system_prompt(),
            return_logprobs=False,
            **kwargs
        )

        # Parse outputs
        chosen_actions = []
        cot_prefixes = []
        for result, action_candidates in zip(raw_outputs, action_candidates_list):
            raw_output = result.get('text', '') if isinstance(result, dict) else result
            chosen_action, cot_prefix = self._parse_vl_output_with_cot(raw_output, action_candidates)
            chosen_actions.append(chosen_action)
            cot_prefixes.append(cot_prefix)

        # Step 2: Extract logprobs with token-level data for each observation
        results = []
        for idx, (image_list, prompt, action_candidates, raw_output, chosen_action, cot_prefix) in enumerate(
            zip(image_lists, prompts, action_candidates_list,
                [r.get('text', '') if isinstance(r, dict) else r for r in raw_outputs],
                chosen_actions, cot_prefixes)
        ):
            action_log_probs, rollout_logprob_dict, full_ids_dict, label_ids_dict = self._extract_action_logprobs_batch(
                image_list, prompt, action_candidates, cot_prefix, temperature
            )

            if action_log_probs is None:
                action_log_probs = self._action_to_logprob(chosen_action, action_candidates, temperature)
                rollout_logprob_dict = {}
                full_ids_dict = {}
                label_ids_dict = {}

            action_probs = np.exp(action_log_probs)

            results.append({
                'action_probs': action_probs,
                'action_logits': action_log_probs,
                'raw_output': raw_output,
                'cot_prefix': cot_prefix,
                'chosen_action': chosen_action,
                'rollout_action_logprob': rollout_logprob_dict,
                'full_ids': full_ids_dict,
                'label_ids': label_ids_dict,
            })

        return results

    def build_vl_train_samples(
        self,
        raw_obs_list: List[List[np.ndarray]],
        history_obs_list: List[List[List]],
        vl_prior_per_tok_list: List[List[Dict]],
        pred_values: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        cot_prefix_list: Optional[List[List[str]]] = None,
        vl_action_list: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build training samples for VL - ALIGNED with LLM's build_llm_samples.

        Args:
            raw_obs_list: [B, T] Raw image observations
            history_obs_list: [B, T] History observations
            vl_prior_per_tok_list: [B, T] VL prior per token (with rollout_logprob, full_ids, label_ids)
            pred_values: [B, T-1] Predicted values
            target_values: [B, T-1] Target values
            cot_prefix_list: [B, T] CoT prefixes
            vl_action_list: [B, T] Action names

        Returns:
            List of training samples
        """
        import logging
        logger = logging.getLogger(__name__)

        samples = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]

                # Convert obs to PIL Image
                if isinstance(current_obs, np.ndarray):
                    image = self._convert_obs_to_pil_image(current_obs)
                else:
                    image = current_obs

                # Build prompt (same structure as LLM)
                image_list = self._assemble_images(current_obs, current_hist)
                instruction = self.get_user_prompt(
                    action_candidates=None,  # Will be filled from vl_prior_per_tok_list
                    history=current_hist,
                    num_images=len(image_list)
                )

                # Get action and logprobs (same as LLM)
                true_action = vl_action_list[b][t+1]
                rollout_logprob = vl_prior_per_tok_list[b][t+1]['rollout_action_logprob'][true_action]
                full_ids = vl_prior_per_tok_list[b][t+1]['full_ids'][true_action]
                label_ids = vl_prior_per_tok_list[b][t+1]['label_ids'][true_action]

                if len(label_ids) == 0:
                    continue

                # Get values (same as LLM)
                target_value = None
                if target_values is not None:
                    target_value = float(target_values[b][t].item())

                pred_value = None
                if pred_values is not None:
                    pred_value = float(pred_values[b][t].item())

                # Get CoT prefix (same as LLM)
                prefix_cot = None
                if self.use_cot and cot_prefix_list is not None:
                    prefix_cot = cot_prefix_list[b][t+1]

                samples.append({
                    "image": image,
                    "image_list": image_list,
                    "instruction": instruction,
                    "target": true_action,
                    "pred_value": pred_value,
                    "target_value": target_value,
                    "rollout_logprob": rollout_logprob,
                    "prefix_cot": prefix_cot,
                    "full_ids": full_ids,
                    "label_ids": label_ids,
                })

        return samples

    def compute_action_log_prob(
        self,
        vl_output: str,
        target_action: str,
        valid_actions: List[str],
        temperature: float = 1.0
    ) -> float:
        """
        Compute log probability of target action from VL output.

        This is used during training to compute the new log probability
        for PPO ratio calculation.

        Args:
            vl_output: Raw VL output string
            target_action: The action that was actually taken
            valid_actions: List of valid action names
            temperature: Temperature for scaling

        Returns:
            Log probability of target action
        """
        if self.use_cot:
            # Parse CoT output to get chosen action
            chosen_action, _ = self._parse_vl_output_with_cot(vl_output, valid_actions)

            # Get log prob distribution
            log_probs = self._action_to_logprob(chosen_action, valid_actions, temperature)

            # Return log prob of target action
            try:
                target_idx = valid_actions.index(target_action)
                return float(log_probs[target_idx])
            except ValueError:
                # Target action not in valid actions
                return -10.0  # Very low log prob
        else:
            # Parse probability distribution
            probs = self._parse_vl_output(vl_output, valid_actions)
            log_probs = np.log(probs + 1e-10)

            try:
                target_idx = valid_actions.index(target_action)
                return float(log_probs[target_idx])
            except ValueError:
                return -10.0


    def get_vl_output_log(
        self,
        wm_train_iter: int,
        vl_train_iter: int,
    ) -> None:
        """
        Log VL output statistics (similar to LLM's get_llm_output_log).

        Args:
            wm_train_iter: World model training iteration
            vl_train_iter: VL training iteration
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(self.episode_output) == 0:
            return

        logger.info(
            f"\n{'='*80}\n"
            f"[VL Output Log] WM Iter: {wm_train_iter} | VL Iter: {vl_train_iter}\n"
            f"{'='*80}"
        )

        for i, tmp_dict in enumerate(self.episode_output[:15]):
            instruction = tmp_dict["Instruction"]
            response = tmp_dict["Response"]
            vl_prior = tmp_dict["vl_prior_per_seq"]
            chosen_action = tmp_dict.get("chosen_action", "N/A")
            cot_prefix = tmp_dict.get("cot_prefix", "")

            logger.info(
                f"\n{'-'*80}\n"
                f"[Step {i}]\n"
                f"{'-'*80}\n"
                f"Instruction:\n{instruction}\n\n"
                f"Response:\n{response}\n\n"
                f"Chosen Action: {chosen_action}\n"
            )

            if cot_prefix:
                logger.info(f"CoT Reasoning:\n{cot_prefix}\n")

            logger.info("Action Probabilities:")

            # Sort actions by probability (descending)
            sorted_actions = sorted(vl_prior.items(), key=lambda x: x[1], reverse=True)

            for action, prob in sorted_actions:
                logger.info(f"  {action:30s} | prob={prob:.6f}")

        self.episode_output = []


def create_prior_generator(
    obs_type: str,
    model_config: Dict[str, Any],
    **kwargs
) -> PriorGenerator:
    """
    Factory function to create appropriate prior generator.

    Args:
        obs_type: 'text' or 'image'
        model_config: Model configuration dictionary
        **kwargs: Additional arguments

    Returns:
        PriorGenerator instance (LLMPriorGenerator or VLPriorGenerator)
    """
    if obs_type == 'text':
        # Create LLM prior generator
        from vllm_utils.vllm_engine import create_vllm_engine

        vllm_engine = create_vllm_engine(
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            pretrain=model_config['model_path'],
            enable_prefix_caching=model_config.get('enable_prefix_caching', True),
            max_model_len=model_config.get('max_model_len', 8192),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.3),
        )

        # Note: data_processor needs to be passed separately
        # This is a placeholder - actual implementation needs data_processor
        raise NotImplementedError(
            "LLMPriorGenerator requires data_processor. "
            "Use the existing implementation or pass data_processor explicitly."
        )

    elif obs_type == 'image':
        # Create VL prior generator
        from vl_engine import create_vl_engine

        vl_engine = create_vl_engine(
            model_name=model_config['model_name'],
            model_path=model_config['model_path'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.3),
        )

        return VLPriorGenerator(
            vl_engine=vl_engine,
            model_name=model_config['model_name'],
        )

    else:
        raise ValueError(f"Unknown obs_type: {obs_type}. Must be 'text' or 'image'.")


if __name__ == "__main__":
    # Example usage
    print("Prior Generator Interface")
    print("=" * 80)
    print("\nThis module provides unified interface for generating action priors.")
    print("\nSupported generators:")
    print("  - LLMPriorGenerator: For text observations (Jericho games)")
    print("  - VLPriorGenerator: For image observations (Atari games)")
    print("\nUsage:")
    print("  generator = create_prior_generator(obs_type='image', model_config={...})")
    print("  prior = generator.generate_prior(observation, action_candidates)")
