import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional

_jericho_df_path = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "jericho" / "priorzero" / "src" / "priorzero_datafactory.py"
)
_spec = importlib.util.spec_from_file_location("jericho_datafactory", _jericho_df_path)
_jericho_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_jericho_mod)
JerichoDataProcessor = _jericho_mod.DataProcessor


class DataProcessor(JerichoDataProcessor):
    """BabyAI-specific DataProcessor with grid-world appropriate prompts."""

    def get_system_prompt(self):
        parts = [
            "You are an expert agent navigating a BabyAI grid-world environment. "
            "You are placed in rooms and must accomplish goals by choosing optimal actions.",
            "",
            "Available action types:",
            "- turn left / turn right / move forward: basic movement",
            "- go to <obj> <id>: navigate to a specific object",
            "- pick up <obj> <id>: pick up an object",
            "- go through <door> <id>: go through an open door",
            "- toggle and go through <door> <id>: open and go through a closed/locked door (locked doors require a matching color key)",
            "- toggle: open/close a door directly in front of you",
            "",
            "Your goal is to complete the given task efficiently to maximize your score.",
            "",
            "OUTPUT FORMAT:",
        ]
        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts in the following order:\n"
                "1. Reasoning: Analyze the current observation, your position, nearby objects, "
                "and which action best progresses toward the goal.\n"
                "2. Action: The final chosen action (must be one of the valid actions).\n"
                "Strict Format Example:\n"
                "Reasoning: <detailed_analysis>\n"
                "Action: <single_action>"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:'.\n"
                "Example:\n"
                "Action: <your_action_here>"
            )
        return "\n".join(parts)

    def get_user_prompt(self, history=None, current_obs=None, valid_actions=None):
        prompt_parts = []
        user_prompt_dict = self.args.user_prompt_dict

        if history and len(history) > 0:
            prompt_parts.append("=== ACTION HISTORY ===")
            for i, (obs, action, reward) in enumerate(history, start=1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"Observation: {obs.strip()}")
                prompt_parts.append(f"Action: {action.strip()}")
                if user_prompt_dict.history_with_reward:
                    prompt_parts.append(f"Reward: {reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        prompt_parts.append(current_obs.strip())

        if user_prompt_dict.observation_with_valid_actions:
            if valid_actions and len(valid_actions) > 0:
                actions_str = ", ".join([f"'{act}'" for act in valid_actions])
                prompt_parts.append(f"\n[Valid Actions]\nChoose from: {actions_str}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        if self.use_cot:
            prompt_parts.append(
                "Analyze the observation and provide your response:\n"
                "Reasoning: <detailed_analysis>\n"
                "Action: <single_action>"
            )
        else:
            prompt_parts.append(
                "Choose the best action:\n"
                "Action: <your_action_here>"
            )
        return "\n".join(prompt_parts)
