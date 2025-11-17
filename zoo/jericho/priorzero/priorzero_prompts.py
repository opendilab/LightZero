"""
PriorZero LLM Prompts Module

This module provides optimized prompt templates for PriorZero's LLM policy,
based on the successful prompt structure from Open-Reasoner-Zero.

Key Features:
- Structured reasoning with <think> and <answer> tags
- Clear role definitions (User/Assistant paradigm)
- Explicit format examples to guide the LLM
- Game-specific context integration

Author: PriorZero Team
Date: 2025-10-21
"""

from jinja2 import Template
from typing import List, Dict, Any, Optional


class PriorZeroPromptTemplates:
    """
    Centralized prompt templates for PriorZero LLM policy.

    Prompt Structure:
    1. System instruction (role definition)
    2. Format specification (<think> and <answer> tags)
    3. Example format to prime the model
    4. User query with game state
    5. Start reasoning with "<think>" tag
    """

    # ==============================================================================
    # MCTS Policy Guidance Prompts
    # ==============================================================================

    MCTS_POLICY_TEMPLATE = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game \
and needs to decide the next action. The Assistant carefully analyzes the current game state, \
considers the available actions, and recommends the best action to take. \
The reasoning process is enclosed within <think> </think> tags, and the recommended action \
is enclosed within <answer> </answer> tags. For example: \
<think> The player is in a dark room and needs light. The lamp is available. </think> \
<answer> take lamp </answer>. \

User: Current game state:
{{game_state}}

Available actions:
{{valid_actions}}

Recent history:
{{history}}

What is the best action to take?
Assistant: <think>\
"""

    # ==============================================================================
    # Supervised Fine-Tuning (SFT) Prompts - Learning from MCTS Policy
    # ==============================================================================

    SFT_FROM_MCTS_TEMPLATE = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game. \
The Assistant provides step-by-step reasoning and selects the best action based on MCTS search results. \
The reasoning is in <think> </think> tags and the action is in <answer> </answer> tags. \

User: Game state: {{game_state}}
Available actions: {{valid_actions}}
MCTS recommended action: {{mcts_action}}
MCTS value estimate: {{mcts_value}}

Please explain why this is the best action and then select it.
Assistant: <think>\
"""

    # ==============================================================================
    # Reward Fine-Tuning (RFT) Prompts - Learning from Environment Rewards
    # ==============================================================================

    RFT_TEMPLATE = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game \
and wants to maximize the total reward. The Assistant analyzes the game state, considers past rewards, \
and selects actions that lead to higher rewards. \
The reasoning is in <think> </think> tags and the action is in <answer> </answer> tags. \

User: Current game state:
{{game_state}}

Available actions:
{{valid_actions}}

Recent trajectory:
{{trajectory_with_rewards}}

Cumulative reward so far: {{cumulative_reward}}

What action should I take to maximize future rewards?
Assistant: <think>\
"""

    # ==============================================================================
    # Evaluation Prompts - For Testing LLM Policy
    # ==============================================================================

    EVAL_TEMPLATE = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game. \
The Assistant thinks carefully about the situation and provides the best action. \
Format: <think> reasoning </think> <answer> action </answer>. \

User: {{game_state}}
Available actions: {{valid_actions}}
Assistant: <think>\
"""

    # ==============================================================================
    # Few-Shot Learning Prompts - With Example Demonstrations
    # ==============================================================================

    FEW_SHOT_TEMPLATE = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game. \
The Assistant learns from examples and applies similar reasoning to new situations. \

Example 1:
User: You are in a dark room. You can't see anything.
Available actions: [go north, take lamp, light lamp]
Assistant: <think> I need light to see. I should take the lamp first, then light it. </think> <answer> take lamp </answer>

Example 2:
User: You are holding a lamp. It is dark.
Available actions: [go north, light lamp, drop lamp]
Assistant: <think> I have the lamp but it's not lit. I should light it to see. </think> <answer> light lamp </answer>

Now your turn:
User: {{game_state}}
Available actions: {{valid_actions}}
Assistant: <think>\
"""


class PriorZeroPromptBuilder:
    """
    Builder class for constructing prompts with specific game context.
    """

    def __init__(self, tokenizer):
        """
        Initialize the prompt builder.

        Args:
            tokenizer: HuggingFace tokenizer with bos_token
        """
        self.tokenizer = tokenizer
        self.templates = PriorZeroPromptTemplates()

    def _get_bos_token(self) -> str:
        """Get the beginning-of-sequence token."""
        if self.tokenizer.bos_token_id is None:
            return ""
        return self.tokenizer.decode([self.tokenizer.bos_token_id])

    def build_mcts_policy_prompt(
        self,
        game_state: str,
        valid_actions: List[str],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a prompt for MCTS policy guidance.

        Args:
            game_state: Current observation text from the game
            valid_actions: List of valid action strings
            history: Recent trajectory [(obs, action, reward), ...]

        Returns:
            Formatted prompt string
        """
        # Format valid actions as a numbered list
        actions_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(valid_actions)])

        # Format history
        if history is None or len(history) == 0:
            history_str = "This is the beginning of the game."
        else:
            history_lines = []
            for i, step in enumerate(history[-5:]):  # Last 5 steps
                obs = step.get('observation', 'N/A')
                action = step.get('action', 'N/A')
                reward = step.get('reward', 0)
                history_lines.append(f"Step {i+1}: Observation: {obs[:100]}... | Action: {action} | Reward: {reward}")
            history_str = "\n".join(history_lines)

        # Render template
        template = Template(self.templates.MCTS_POLICY_TEMPLATE)
        return template.render(
            bos_token=self._get_bos_token(),
            game_state=game_state,
            valid_actions=actions_str,
            history=history_str,
        )

    def build_sft_prompt(
        self,
        game_state: str,
        valid_actions: List[str],
        mcts_action: str,
        mcts_value: float,
    ) -> str:
        """
        Build a prompt for supervised fine-tuning from MCTS policy.

        Args:
            game_state: Current observation text
            valid_actions: List of valid action strings
            mcts_action: Action recommended by MCTS
            mcts_value: Value estimate from MCTS

        Returns:
            Formatted prompt string
        """
        actions_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(valid_actions)])

        template = Template(self.templates.SFT_FROM_MCTS_TEMPLATE)
        return template.render(
            bos_token=self._get_bos_token(),
            game_state=game_state,
            valid_actions=actions_str,
            mcts_action=mcts_action,
            mcts_value=f"{mcts_value:.3f}",
        )

    def build_rft_prompt(
        self,
        game_state: str,
        valid_actions: List[str],
        trajectory: List[Dict[str, Any]],
        cumulative_reward: float,
    ) -> str:
        """
        Build a prompt for reward fine-tuning.

        Args:
            game_state: Current observation text
            valid_actions: List of valid action strings
            trajectory: Recent trajectory with rewards
            cumulative_reward: Total reward accumulated

        Returns:
            Formatted prompt string
        """
        actions_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(valid_actions)])

        # Format trajectory with rewards
        traj_lines = []
        for i, step in enumerate(trajectory[-5:]):
            action = step.get('action', 'N/A')
            reward = step.get('reward', 0)
            traj_lines.append(f"  Step {i+1}: Action: {action} → Reward: {reward:+.2f}")
        trajectory_str = "\n".join(traj_lines)

        template = Template(self.templates.RFT_TEMPLATE)
        return template.render(
            bos_token=self._get_bos_token(),
            game_state=game_state,
            valid_actions=actions_str,
            trajectory_with_rewards=trajectory_str,
            cumulative_reward=f"{cumulative_reward:+.2f}",
        )

    def build_eval_prompt(
        self,
        game_state: str,
        valid_actions: List[str],
    ) -> str:
        """
        Build a simple prompt for evaluation.

        Args:
            game_state: Current observation text
            valid_actions: List of valid action strings

        Returns:
            Formatted prompt string
        """
        actions_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(valid_actions)])

        template = Template(self.templates.EVAL_TEMPLATE)
        return template.render(
            bos_token=self._get_bos_token(),
            game_state=game_state,
            valid_actions=actions_str,
        )


# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_action_from_llm_output(llm_output: str, valid_actions: List[str]) -> Optional[str]:
    """
    Extract the action from LLM output with <answer> tags.

    Args:
        llm_output: Full LLM response including <think> and <answer> tags
        valid_actions: List of valid action strings to match against

    Returns:
        Extracted action string, or None if extraction fails

    Example:
        >>> output = "<think>I need light</think> <answer>take lamp</answer>"
        >>> extract_action_from_llm_output(output, ["take lamp", "go north"])
        "take lamp"
    """
    import re

    # Pattern to extract content between <answer> and </answer>
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

    if not match:
        return None

    extracted = match.group(1).strip()

    # Try exact match first
    if extracted in valid_actions:
        return extracted

    # Try case-insensitive match
    extracted_lower = extracted.lower()
    for action in valid_actions:
        if action.lower() == extracted_lower:
            return action

    # Try fuzzy match (substring)
    for action in valid_actions:
        if extracted_lower in action.lower() or action.lower() in extracted_lower:
            return action

    return None


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PriorZero Prompt Templates - Example Usage")
    print("="*80)

    # Mock tokenizer
    class MockTokenizer:
        bos_token_id = 1
        def decode(self, ids):
            return "<BOS>"

    tokenizer = MockTokenizer()
    builder = PriorZeroPromptBuilder(tokenizer)

    # Example game state
    game_state = "You are standing in an open field west of a white house."
    valid_actions = ["go north", "go south", "go east", "open mailbox", "take mailbox"]
    history = [
        {"observation": "West of House", "action": "look", "reward": 0},
        {"observation": "You see a mailbox", "action": "examine mailbox", "reward": 0},
    ]

    print("\n1. MCTS Policy Prompt:")
    print("-"*80)
    prompt = builder.build_mcts_policy_prompt(game_state, valid_actions, history)
    print(prompt)

    print("\n2. SFT Prompt:")
    print("-"*80)
    sft_prompt = builder.build_sft_prompt(game_state, valid_actions, "open mailbox", 0.75)
    print(sft_prompt)

    print("\n3. RFT Prompt:")
    print("-"*80)
    trajectory = [
        {"action": "go east", "reward": 0},
        {"action": "open mailbox", "reward": 5},
    ]
    rft_prompt = builder.build_rft_prompt(game_state, valid_actions, trajectory, 5.0)
    print(rft_prompt)

    print("\n4. Action Extraction:")
    print("-"*80)
    llm_output = "<think>The mailbox might contain something useful.</think> <answer>open mailbox</answer>"
    extracted = extract_action_from_llm_output(llm_output, valid_actions)
    print(f"LLM Output: {llm_output}")
    print(f"Extracted Action: {extracted}")

    print("\n" + "="*80)
    print("✓ All prompt templates demonstrated successfully!")
    print("="*80)
