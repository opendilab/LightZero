import torch
from typing import List, Dict, Any, Tuple, Union, Optional

def build_llm_prompt(
    current_obs: str,
    history: Optional[List[Tuple[str, str, float]]] = None,
    action_descriptions: Optional[Dict[str, str]] = None,
    use_cot: bool = True
) -> str:
    prompt_parts = []

    prompt_parts.append(
        "You are an expert player in a text-based adventure game. "
        "Your goal is to maximize the score by choosing the best possible next action. "
        "You must choose exactly ONE best next action."
    )
    if history is not None and len(history) > 0:
        history = list(history)
        prompt_parts.append("\n=== Recent History ===")

        for i, (obs, action, reward) in enumerate(history, start=1):  
            obs_str = obs
            prompt_parts.append(f"Step {i}:")
            prompt_parts.append(f"  Observation: {obs_str}")
            prompt_parts.append(f"  Action: {action}")
            prompt_parts.append(f"  Reward: {reward}")

    # Current observation
    prompt_parts.append("\n=== Current Situation ===")
    prompt_parts.append(current_obs)

    # Available actions (if provided)
    if action_descriptions:
        prompt_parts.append("\n=== Available Actions ===")
        prompt_parts.append(
            "You MUST choose the best action from the list below. "
            "Do not invent actions that are not in this list."
        )
        for action_text, desc in action_descriptions.items():
            # action_text: should match exactly the string we want inside <action>...</action>
            prompt_parts.append(f"- {action_text}: {desc}")

    # Task + output format
    if use_cot:
        prompt_parts.append(
            "\n=== Task ===\n"
            "You must produce TWO parts in order: (1) Reasoning, then (2) Action.\n\n"
            "1) Reasoning:\n"
            "- Perform a detailed reasoning process based ONLY on the current state and the recent interaction history.\n"
            "- Analyze what environment or situation you are currently in.\n"
            "- Identify what actions are available or valid at this step, and the relevant constraints.\n"
            "- You may discuss observations, uncertainties, and implications of different possibilities.\n"
            "- IMPORTANT: Do NOT state, imply, or reveal which action will be chosen, and the reasoning section MUST output exactly in the following format: Reasoning:<REASONING>.\n\n"
            "2) Action:\n"
            "- After finishing the reasoning, output exactly ONE line in the following format:\nAction: <ACTION>\n"
            "Your output MUST strictly follow this format: \nReasoning: <your reasoning content>\nAction: <the chosen action>"
        )
    else:
        prompt_parts.append(
            "\n=== Task ===\n"
            "Analyze the recent history and the current situation, and decide on the SINGLE best next action."
            "Please keep the output concise, avoiding any other content.\n\n"
        )
    return "\n".join(prompt_parts)


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio