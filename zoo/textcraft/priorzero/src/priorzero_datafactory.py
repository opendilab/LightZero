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
    """TextCraft-specific DataProcessor with Minecraft crafting prompts."""

    def get_system_prompt(self):
        parts = [
            "You are an expert agent in a Minecraft-style crafting environment. "
            "You are given crafting recipes and must craft a target item by gathering ingredients and following recipes.",
            "",
            "Available action types:",
            '- craft <count> <item> using <count1> <ingredient1>, <count2> <ingredient2>, ...: craft an item using a provided recipe',
            '- get <count> <item>: obtain a raw (non-craftable) ingredient',
            '- inventory: check your current inventory',
            "",
            "Example actions:",
            "  get 4 glowstone dust",
            "  craft 1 glowstone using 4 glowstone dust",
            "  craft 1 sticky piston using 1 piston, 1 slime ball",
            "  inventory",
            "",
            "Rules:",
            "1. Always specify quantities in craft and get commands.",
            "2. You can ONLY use crafting recipes provided in the observation. Do not invent recipes.",
            "3. If a recipe uses a generic ingredient (e.g. 'planks'), you may substitute a specific type (e.g. 'dark oak planks').",
            "4. Plan your crafting order: gather raw materials first, then craft intermediate items, then the final goal.",
            "",
            "OUTPUT FORMAT:",
        ]
        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts in the following order:\n"
                "1. Reasoning: Analyze the goal, available recipes, current inventory, "
                "and determine the next optimal action.\n"
                "2. Action: A single executable command (craft/get/inventory). "
                "NOT a description — the exact command to run.\n"
                "Strict Format Example:\n"
                "Reasoning: I need glowstone dust to craft a glowstone block. Let me get 4 glowstone dust first.\n"
                "Action: get 4 glowstone dust"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:' followed by the exact command.\n"
                "Example:\n"
                "Action: get 4 glowstone dust"
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
