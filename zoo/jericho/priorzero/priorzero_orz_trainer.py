from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
from loguru import logger

from jinja2 import Template

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp
from orz.ppo import RayPPOTrainer, PromptDataset
from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig


class TempExp(BasePPOExp):
    def __init__(self, orz_cfg, orz_tokenizer, orz_strategy):
        self.cfg = orz_cfg
        self.tokenizer = orz_tokenizer
        self.strategy = orz_strategy
        
class JerichoPromptDataset(PromptDataset):
    """
    Custom dataset for Jericho text adventure games in ORZ format.
    Adapts PriorZero game_segments to ORZ PPO training format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        """
        Process a single dialogue (observation + action pair) into ORZ format.

        Args:
            dialogue: Dict with 'prompt', 'final_answer', 'file_name'

        Returns:
            prompt: Formatted prompt string
            extra: Dict with answer and metadata
        """
        # Template for Jericho text adventure prompts
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game \
and needs to decide the next action. The Assistant carefully analyzes the current game state, \
considers the available actions, and recommends the best action to take. \
The reasoning process is enclosed within <think> </think> tags, and the recommended action \
is enclosed within <answer> </answer> tags. For example: \
<think> The player is in a dark room and needs light. The lamp is available. </think> \
<answer> take lamp </answer>. User: {{prompt}}
Assistant: <think>\
"""

        prompt_instruction_template_jinja = """\
Current game state:
{{prompt}}

What is the best action to take? Put your answer inside <answer> </answer> tags.
"""

        # Validate dialogue format
        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"

        # Build prompt
        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(
            prompt=dialogue["prompt"][0]["value"]
        )

        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        prompt = prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction
        )

        extra = {
            "answer": dialogue["final_answer"],
            "file_name": dialogue.get("file_name", "unknown")
        }

        return prompt, extra

class GameSegmentToORZAdapter:
    """
    Convert PriorZero game_segments to ORZ-compatible format.
    """

    @staticmethod
    def convert_segments_to_prompts(game_segments: List[Any], tokenizer) -> List[Dict]:
        """
        Convert game_segments to ORZ prompt format.

        Args:
            game_segments: List of GameSegment from PriorZero
            tokenizer: HuggingFace tokenizer

        Returns:
            List of ORZ-compatible prompt dictionaries
        """
        prompts = []
        for segment in game_segments:
            if hasattr(segment, 'raw_obs_segment') and segment.raw_obs_segment:
                for i, (obs, action) in enumerate(zip(
                    segment.raw_obs_segment,
                    segment.action_segment
                )):
                    prompt_dict = {
                        "prompt": [{"value": obs}],
                        "final_answer": action,
                        "file_name": f"segment_{id(segment)}_step_{i}"
                    }
                    prompts.append(prompt_dict)

        return prompts

    @staticmethod
    def extract_training_data(game_segments: List[Any]) -> Dict[str, List]:
        """
        Extract training data from game_segments for ORZ.

        Returns:
            Dictionary containing:
            - states: List of state descriptions
            - actions: List of actions taken
            - rewards: List of rewards received
            - mcts_policies: List of MCTS visit distributions
        """
        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'mcts_policies': []
        }

        for segment in game_segments:
            # Extract raw observations (states)
            if hasattr(segment, 'raw_obs_segment'):
                training_data['states'].extend(segment.raw_obs_segment)

            # Extract actions
            if hasattr(segment, 'action_segment'):
                training_data['actions'].extend(segment.action_segment)

            # Extract rewards
            if hasattr(segment, 'reward_segment'):
                training_data['rewards'].extend(segment.reward_segment)

            # Extract MCTS policies
            if hasattr(segment, 'mcts_policy_segment'):
                training_data['mcts_policies'].extend(segment.mcts_policy_segment)

        return training_data


class JerichoRewardTrainer(RayPPOTrainer):
    """Custom reward trainer for Jericho text adventures"""

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn,
    ):
        """
        Compute rewards for Jericho actions.
        Reward is 1.0 if action matches ground truth, else 0.0
        """
        import torch
        scores = []
        responses = []

        for output, extra in zip(outputs, extras):
            response = output["response"]
            responses.append(response)

            # Extract action from response
            # Look for <answer>...</answer> tags
            import re
            pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            matches = re.findall(pattern, response)
            predicted_action = matches[-1].strip() if matches else ""

            # Ground truth action
            true_action = extra["answer"]

            # Simple exact match for now
            # TODO: Could use fuzzy matching or LLM-based similarity
            score = 1.0 if predicted_action.lower() == true_action.lower() else 0.0
            scores.append(score)

        # Log statistics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"    ORZ reward - avg: {avg_score:.3f}, samples: {len(scores)}")

        # Create score tensors (reward only on last token)
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # Remove empty responses
        res_prompts, res_responses, res_score_tensors = [], [], []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors