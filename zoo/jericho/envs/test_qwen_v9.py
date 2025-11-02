# -*- coding: utf-8 -*-
"""
使用大型语言模型（LLM）作为智能体来玩 Jericho 文本冒险游戏的脚本。

该脚本实现了以下核心功能:
1.  一个基于 LLM 的策略智能体，能够理解游戏状态并做出决策。
2.  一个记忆管理器（MemoryManager），用于存储和管理游戏过程中的“好记忆”（成功策略）和“坏记忆”（失败教训），并使用语义相似度来避免冗余。
3.  一个自主反思（Autonomous Reflection）机制，在每轮游戏结束后，让 LLM 分析游戏过程，总结失败原因并提炼出可操作的经验。
4.  支持多种 LLM 后端，例如本地部署的 Qwen 模型。
5.  通过命令行参数高度可配置，方便进行实验和调整。

如何运行:
- 使用 torchrun 进行分布式训练（即使是单节点单GPU）。
- 示例命令:
  export CUDA_VISIBLE_DEVICES=1
  cd /fs-computility/niuyazhe/puyuan/code/LightZero
  torchrun --nproc_per_node=1 --master-port 20092 /fs-computility/niuyazhe/puyuan/code/LightZero/zoo/jericho/envs/test_qwen_v9.py \
    --model-path /fs-computility/niuyazhe/shared/xiongjyu/model/Qwen2.5-7B-Instruct \
    --env-type detective \
    --max-steps 100 \
    --num-episodes 20 \
    --good-memory-maxlen 20 \
    --temperature 0.01 \
    --log-dir ./priorzero_log_optimized
"""

import logging
import copy
import os
import json
import re
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from easydict import EasyDict
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from jericho_env import JerichoEnv


# =====================================================================================
# 1. 记忆管理模块 (MemoryManager)
# =====================================================================================
class MemoryManager:
    """
    管理智能体的长期记忆，包括成功经验（好记忆）和失败教训（坏记忆）。

    该类使用 SentenceTransformer 来计算记忆文本的嵌入向量，并实现了一套
    基于余弦相似度的驱逐策略，以防止记忆库中出现过多语义重复的内容。
    """
    def __init__(self, maxlen: int, similarity_threshold: float = 0.85, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 MemoryManager.
        
        Args:
            maxlen (int): 记忆库的最大容量。
            similarity_threshold (float): 用于判断记忆是否冗余的相似度阈值。
            device (str): 用于计算嵌入向量的设备 ('cuda' 或 'cpu')。
        """
        self.maxlen = maxlen
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.memories: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        logging.info(f"MemoryManager initialized with maxlen={maxlen}, threshold={similarity_threshold} on device='{self.device}'")

    def get_memories(self) -> List[str]:
        """返回当前所有的记忆文本。"""
        return self.memories

    def __len__(self) -> int:
        """返回当前记忆的数量。"""
        return len(self.memories)

    def add_memory(self, new_memory_text: str) -> str:
        """
        添加一条新记忆，并根据冗余检查和容量限制进行管理。

        该方法首先检查新记忆是否与现有记忆高度相似，如果相似度超过阈值，则拒绝添加。
        如果记忆库已满，则采用先进先出（FIFO）策略移除最旧的记忆。

        Args:
            new_memory_text (str): 要添加的新记忆文本。

        Returns:
            str: 描述本次操作的日志信息。
        """
        if not new_memory_text or not isinstance(new_memory_text, str):
            return "Skipped adding empty or invalid memory."

        with torch.no_grad():
            new_embedding = self.model.encode(new_memory_text, convert_to_tensor=True, device=self.device)

        # 检查新记忆是否与任何现有记忆过于相似
        if self.embeddings is not None and len(self.memories) > 0:
            similarities = F.cosine_similarity(new_embedding, self.embeddings)
            max_similarity = torch.max(similarities).item()
            if max_similarity > self.similarity_threshold:
                return f"Rejected redundant memory (sim: {max_similarity:.2f} > {self.similarity_threshold}). New: '{new_memory_text}'"

        # 如果记忆库未满，直接添加
        if len(self.memories) < self.maxlen:
            self.memories.append(new_memory_text)
            if self.embeddings is None:
                self.embeddings = new_embedding.unsqueeze(0)
            else:
                self.embeddings = torch.cat([self.embeddings, new_embedding.unsqueeze(0)], dim=0)
            return f"Added new memory: '{new_memory_text}'"

        # 如果记忆库已满，执行FIFO替换
        else:
            removed_memory = self.memories.pop(0)
            self.memories.append(new_memory_text)
            
            # 更新嵌入张量
            self.embeddings = torch.cat([self.embeddings[1:], new_embedding.unsqueeze(0)], dim=0)
            return (f"FIFO eviction. "
                    f"Removed: '{removed_memory}', Added: '{new_memory_text}'")

# =====================================================================================
# 2. 分布式环境初始化
# =====================================================================================
def init_distributed():
    """初始化 PyTorch 分布式环境。"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

# =====================================================================================
# 3. LLM 策略抽象基类
# =====================================================================================
class BaseLLMPolicy(ABC):
    """
    所有 LLM 策略的抽象基类，定义了通用接口。
    """
    def __init__(self, reflection_history_len=9999):
        self.reflection_history_len = reflection_history_len
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def format_prompt_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None) -> str:
        """
        构建用于动作选择的 Prompt。

        该 Prompt 结构清晰，包含系统指令、战略记忆、近期历史、当前观察和决策指令，
        旨在引导 LLM 做出更具策略性的决策。

        Args:
            history: 近期的动作和观察历史。
            full_scene_description: 当前的完整场景描述。
            inventory: 当前的物品栏。
            location: 当前的地点。
            valid_actions: 当前可用的动作列表。
            good_memory: 成功策略列表。
            bad_memory: 需要避免的错误列表。

        Returns:
            str: 格式化后的完整 Prompt 字符串。
        """
        system_prompt = """You are an intelligent agent playing a text-based adventure game. Your primary goal is to solve puzzles, uncover the story, and maximize your score.
Think step-by-step. First, analyze the Current Obeservation and Recent Game History. Second, review your Strategic Memory. Third, decide on the single best action to make progress.
"""

        system_prompt += "\n--- Strategic Memory ---\n"
        if good_memory:
            system_prompt += "✅ Successful Strategies (Good Memories):\n" + "".join(f"- {mem}\n" for mem in good_memory)
        else:
            system_prompt += "✅ No successful strategies recorded yet.\n"

        if bad_memory:
            system_prompt += "❌ Mistakes to Avoid (Bad Memories):\n" + "".join(f"- {mem}\n" for mem in bad_memory)
        else:
            system_prompt += "❌ No mistakes recorded yet.\n"

        system_prompt += "\n--- Recent Game History ---\n"

        if not history:
            system_prompt += "The game has just begun.\n"
        else:
            history_lines = []
            for i, h in enumerate(history):
                loc_info = ""
                if i < len(history)-1:
                    history_lines.append(f"{loc_info}YOU DID ACTION: > {h['action']}\nTHEN YOU OBSERVED: {h['obs']}")
                else:
                    history_lines.append(f"{loc_info}YOU DID ACTION: > {h['action']}\nTHEN YOU OBSERVED:")   

            system_prompt += "\n".join(history_lines) + "\n"

        system_prompt += f"\n--- Current Obeservation ---\n{full_scene_description}\n"

        system_prompt += (
            "\n--- Your Decision ---\n"
            "Based on the Current Obeservation, Recent Game History and Strategic Memory, what is the single most promising next action to progress? "
            "If a past strategy is listed in 'Mistakes to Avoid', you MUST NOT repeat it. If a strategy is in 'Successful Strategies', consider applying a similar logic.\n"
            f"Choose ONLY ONE from the following valid actions: {', '.join(valid_actions)}\n"
            "Respond with the action phrase only.\n"
            "> "
        )
        return system_prompt

    @abstractmethod
    def sample_action_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None) -> tuple[str, str]:
        """
        根据当前状态采样一个动作。这是一个抽象方法，需要由子类实现。

        Returns:
            tuple[str, str]: (选择的动作, 用于生成的 Prompt)
        """
        pass


    @abstractmethod
    def generate_autonomous_reflection(self, history: List[Dict[str, str]], final_score: float) -> Dict[str, str]:
        """
        生成自主反思。这是一个抽象方法，需要由子类实现。

        Returns:
            Dict[str, str]: 包含 'critical_mistake_analysis' 和 'alternative_action_suggestion' 的字典。
        """
        pass

# =====================================================================================
# 4. 本地 Qwen 模型策略实现
# =====================================================================================
class QwenLocalPolicy(BaseLLMPolicy):
    """
    使用本地部署的 Qwen 系列模型的策略实现。
    """
    def __init__(self, model_path: str, reflection_history_len: int = 9999, local_rank: int = 0):
        super().__init__(reflection_history_len)
        self.device = torch.device(f"cuda:{local_rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(self.device)
        print(f"QwenLocalPolicy initialized on device {self.device}")

    def _generate_text(self, prompt: str, gen_config: GenerationConfig) -> str:
        """内部方法，用于调用模型生成文本。"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**model_inputs, generation_config=gen_config)
        output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content

    def sample_action_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None) -> tuple[str, str]:
        """
        实现动作采样逻辑。

        首先生成 Prompt，然后调用 LLM 生成动作文本。之后，将生成的文本与
        有效的动作列表进行匹配，以找出最合适的动作。如果匹配失败，则选择
        一个安全动作（如 'look'）或列表中的第一个动作作为后备。
        """
        prompt = self.format_prompt_v2(history, full_scene_description, inventory, location, valid_actions, good_memory, bad_memory)
        
        gen_config = GenerationConfig(
            temperature=TEMPERATURE, top_p=0.9, do_sample=True, max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        content = self._generate_text(prompt, gen_config).lower()
        
        # 优先匹配更长、更具体的动作
        sorted_actions = sorted(valid_actions, key=len, reverse=True)
        for va in sorted_actions:
            cleaned_va = va.lower().strip()
            cleaned_content = content.replace(">", "").strip()
            if cleaned_va in cleaned_content or cleaned_content in cleaned_va:
                return va, prompt

        # 后备安全动作
        safe_actions = ['look', 'inventory', 'l', 'i']
        for sa in safe_actions:
            if sa in valid_actions:
                return sa, prompt
        
        return (valid_actions[0] if valid_actions else 'look'), prompt

    def generate_autonomous_reflection(self, history: List[Dict[str, str]], final_score: float) -> Dict[str, str]:
        """
        实现自主反思逻辑。

        构建一个专门的 Prompt，要求 LLM 分析游戏轨迹，特别是最后几步，
        找出导致游戏结束或停滞的关键错误，并以 JSON 格式返回分析和改进建议。
        """
        recent_history = history[-self.reflection_history_len:]
        # 确保历史记录包含最后一步和最终结果
        # TODO: test inpact, 加入每步的reward
        trajectory_str = "\n".join([f"Step {i+1}: YOU DID ACTION: > {h['action']}', THEN YOU OBSERVED: \n{h['obs']}\n" for i, h in enumerate(recent_history)])

        # trajectory_str = "\n".join([f"STEP {i+1}: ACTION: > {h['action']}', OBSERVATION: \n{h['obs']}\n" for i, h in enumerate(recent_history)])
        # trajectory_str = "\n".join([f"Step {i+1}: I did '> {h['action']}' and the outcome was:\n{h['obs']}\n" for i, h in enumerate(recent_history)])

        # TODO: test inpact of different prompts
        # 20eps reward_mean:110.5   Total Scores: [array([60.]), array([140.]), array([90.]), array([90.]), array([140.]), array([160.]), array([90.]), array([90.]), array([90.]), array([140.]), array([160.]), array([90.]), array([90.]), array([140.]), array([90.]), array([140.]), array([140.]), array([90.]), array([90.]), array([90.])]
        prompt = (
            "You are an expert strategy analyst for a text-based adventure game. Your task is to analyze a game trajectory, identify the critical mistake that led to a low score or game over, and produce actionable advice.\n\n"
            f"--- Final Score ---\n{final_score}\n\n"
            f"--- Full Gameplay Log of the Final Attempt ---\n{trajectory_str}\n"
            "--- Analysis Task ---\n"
            "Analyze the final steps of the gameplay log. Pinpoint the single critical mistake that ended the game or caused it to get stuck. Provide your analysis as a single, clean JSON object with two keys: `critical_mistake_analysis` and `alternative_action_suggestion`.\n"
            # "- `critical_mistake_analysis`: A concise sentence explaining what the final wrong move was and why it was wrong, based on the final outcome. Example: 'The final action 'go north' led to a dead end, wasting a turn when exploring 'east' towards the Mayor's home was a more promising option based on the description.'\n"
            # "- `alternative_action_suggestion`: A concrete, actionable suggestion for what to do instead of the mistaken action, phrased as a general rule for the future. Example: 'When at the 'Outside' location after moving west, the next logical step is to explore 'east' to investigate the Mayor's home mentioned in the description.'\n\n"
            "- `critical_mistake_analysis`: A concise sentence explaining what the final wrong move was and why it was wrong, based on the final outcome. Example: 'The final action 'north' led to a dead end, while exploring 'east' towards the Mayor's home was a more promising option.'\n"
            "- `alternative_action_suggestion`: A concrete, actionable suggestion for what to do instead of the mistaken action, phrased as a general rule for the future. Example: 'When at the 'Outside' location after moving west, the next promising step is to explore 'east' to investigate the Mayor's home.'\n\n"
            "Respond ONLY with the JSON object, without any surrounding text or explanations.\n"
        )
        # TODO(pu): 为什么提示词差一点，性能就差很多？

        # prompt = (
        #     "You are an expert strategy analyst for a text-based adventure game. Your task is to analyze a game trajectory, identify the critical mistake that led to a low score or game over, and produce actionable advice.\n\n"
        #     f"--- Final Score ---\n{final_score}\n\n"
        #     f"--- Full Gameplay Log of the Final Attempt ---\n{trajectory_str}\n"
        #     "--- Analysis Task ---\n"
        #     "Analyze the final steps of the gameplay log. Pinpoint the single critical mistake that ended the game or caused it to get stuck. Provide your analysis as a single, clean JSON object with two keys: `critical_mistake_analysis` and `alternative_action_suggestion`.\n"
        #     "- `critical_mistake_analysis`: A concise sentence explaining what the final wrong move was and why it was wrong, based on the final outcome. Example: 'The ACTION <take paper> in the Location <Dead End> repeatedly led to the player getting stuck, as there was no useful purpose for the paper in that location, and it prevented further exploration. \n"
        #     "- `alternative_action_suggestion`: A concrete, actionable suggestion for what to do instead of the mistaken action, phrased as a general rule for the future. Example: 'When at the Location <Dead End>, avoid taking or interacting with items unless there is a clear purpose or hint that it will lead to progress. Instead, explore other directions or examine the environment for clues.'\n\n"
        #     "Respond ONLY with the JSON object, without any surrounding text or explanations.\n"
        # )

        # prompt = f"""
        # # Role and Mission
        # You are an expert strategy analyst for a text-based adventure game.
        # Your mission is to analyze the gameplay log provided below, identify the **single critical mistake** that resulted in a low score or a game-over state, and provide specific, actionable advice for improvement.
        # ---
        # # Game Data
        # ## Final Score
        # {final_score}
        # ## Full Gameplay Log of the Final Attempt
        # {trajectory_str}
        # ---
        # # Analysis Task
        # Analyze the final steps of the gameplay log to pinpoint the **one critical mistake** that halted progress or ended the game.
        # Your analysis must be delivered as a **single, clean JSON object**, containing only the following two keys: `critical_mistake_analysis` and `alternative_action_suggestion`.
        # ## JSON Structure Definition
        # 1.  `critical_mistake_analysis`: (Type: String)
        #     -   **Content Requirement**: A concise sentence explaining **what the final wrong move was** and **why it was wrong**, based on the game's outcome.
        #     -   **Good Example**: "After finding the key, the player's final command 'go north' was a redundant and ineffective move because the logical next step was to use the key with the command 'unlock door with key'."
        # 2.  `alternative_action_suggestion`: (Type: String)
        #     -   **Content Requirement**: A concrete, actionable suggestion for what to do instead. This advice should be framed as a **general rule for future gameplay**.
        #     -   **Good Example**: "When a new item (like a key) is acquired, immediately consider its purpose and try related interaction commands (e.g., 'unlock door') rather than repeating previously failed actions."
        # ---
        # # Output Format Requirement
        # **Strictly adhere to this rule**: Your response must **ONLY** be the JSON object itself. It must **NOT** be enclosed in markdown code blocks (like ```json ... ```) or contain any surrounding text, explanations, or comments.
        # """


        gen_config = GenerationConfig(
            temperature=TEMPERATURE, top_p=0.9, do_sample=True, max_new_tokens=250,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        content = self._generate_text(prompt, gen_config)
        
        reflections = {'critical_mistake_analysis': '', 'alternative_action_suggestion': ''}
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).replace('\n', ' ').replace('\r', '').strip()
                parsed_json = json.loads(json_str)
                reflections['critical_mistake_analysis'] = parsed_json.get('critical_mistake_analysis', '').strip()
                reflections['alternative_action_suggestion'] = parsed_json.get('alternative_action_suggestion', '').strip()
            else:
                logging.warning(f"Could not find a JSON object in the LLM's reflection output. Content: {content}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from reflection output: {e}. Content: {content}")
        return reflections

# =====================================================================================
# 5. 参数解析
# =====================================================================================
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run Jericho text-based game with an LLM agent.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- LLM 和模型配置 ---
    parser.add_argument('--llm-provider', type=str, default='Qwen', choices=['Qwen', 'PoeAPI'], help='The LLM provider to use.')
    parser.add_argument('--model-path', type=str, default='/fs-computility/niuyazhe/shared/xiongjyu/model/Qwen2.5-7B-Instruct', help='Path to the local LLM model (e.g., for Qwen).')
    parser.add_argument('--model-name', type=str, default='Qwen2.5-7B-Instruct', help='Identifier for the model, used for logging.')
    
    # --- 实验和游戏配置 ---
    parser.add_argument('--env-type', type=str, default='detective', help='The Jericho game to play (e.g., zork1, detective).')
    parser.add_argument('--num-episodes', type=int, default=20, help='Number of episodes to run.')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps per episode.')
    parser.add_argument('--log-dir', type=str, default='./priorzero_log_optimized', help='Base directory for saving logs.')
    
    # --- 智能体行为配置 ---
    parser.add_argument('--use-autonomous-reflection', action=argparse.BooleanOptionalAction, default=True, help='Enable or disable autonomous reflection after each episode.')
    parser.add_argument('--use-structured-info', action=argparse.BooleanOptionalAction, default=True, help='Provide structured info (location, inventory) to the LLM.')
    parser.add_argument('--max-history-len', type=int, default=10, help='Maximum number of recent history steps to include in the prompt.')
    
    # --- 记忆系统配置 ---
    parser.add_argument('--good-memory-maxlen', type=int, default=20, help='Maximum capacity for good memories.')
    parser.add_argument('--bad-memory-maxlen', type=int, default=20, help='Maximum capacity for bad memories.')
    parser.add_argument('--similarity-threshold', type=float, default=0.9, help='Similarity threshold for memory eviction.')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature in llm.')


    return parser.parse_args()

# =====================================================================================
# 6. 主执行逻辑
# =====================================================================================
def main(args):
    """主执行函数，包含游戏循环和智能体交互。"""
    # --- 初始化分布式环境和日志 ---
    rank, world_size = init_distributed()
    print(f"[RANK {rank}] Initialized. World size: {world_size}")
    
    log_dir = os.path.join(args.log_dir, args.model_name, f"{args.env_type}_structured_{args.use_structured_info}_auto_reflect_{args.use_autonomous_reflection}_v2")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.txt")
    f = open(log_file, "w", encoding="utf-8")
    
    global TEMPERATURE;TEMPERATURE=args.temperature

    # --- 实例化策略对象 ---
    llm_policy: BaseLLMPolicy
    print(f"Using LLM Provider: {args.llm_provider}")
    if args.llm_provider == "Qwen":
        llm_policy = QwenLocalPolicy(
            model_path=args.model_path,
            local_rank=rank
        )
    # elif args.llm_provider == "PoeAPI":
    #     # PoeAPIPolicy 的实现可以放在这里
    #     raise NotImplementedError("PoeAPIPolicy is not fully implemented in this example.")
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {args.llm_provider}")
    
    # --- 配置并初始化游戏环境 ---
    env_cfg = EasyDict(
        dict(
            max_steps=args.max_steps,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{args.env_type}.z5",
            add_location_and_inventory=args.use_structured_info,
            env_type=args.env_type,
            # 以下为保持原有配置的参数
            max_action_num=55,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            remove_stuck_actions=False,
            for_unizero=False,
            collector_env_num=1,
            evaluator_env_num=1,
            save_replay=False,
            save_replay_path=None,
            collect_policy_mode='expert'
        )
    )
    env = JerichoEnv(env_cfg)
    
    # --- 初始化记忆管理器 ---
    good_trial_memory = MemoryManager(maxlen=args.good_memory_maxlen, similarity_threshold=args.similarity_threshold)
    bad_trial_memory = MemoryManager(maxlen=args.bad_memory_maxlen, similarity_threshold=args.similarity_threshold)

    total_scores = []

    # --- 游戏主循环 ---
    for episode_id in range(args.num_episodes):
        f.write(f"{'='*80}\n")
        f.write(f"STARTING EPISODE: {episode_id} | Good Memories: {len(good_trial_memory)}, Bad Memories: {len(bad_trial_memory)}\n")
        f.write(f"Current Good Memories: {good_trial_memory.get_memories()}\n")
        f.write(f"Current Bad Memories: {bad_trial_memory.get_memories()}\n")
        f.write(f"CONFIG: USE_STRUCTURED_INFO = {args.use_structured_info}, USE_AUTONOMOUS_REFLECTION = {args.use_autonomous_reflection}\n")
        f.write(f"{'='*80}\n")
        f.flush()
        
        obs_or_dict = env.reset(return_str=True)
        done = False
        step_count = 0
        episode_reward = 0
        episode_history = []
        last_actions = deque(maxlen=4)

        while not done:
            # --- 解析观察值 ---
            if isinstance(obs_or_dict, dict):
                full_scene_description = obs_or_dict.get('observation', '')
                inventory_str = obs_or_dict.get('inventory', 'not tracked')
                location_str = obs_or_dict.get('location', 'not tracked')
            else:
                full_scene_description = obs_or_dict
                inventory_str, location_str = "not tracked", "not tracked"
                try:
                    inventory_str = env._env.get_inventory()
                    location_str = env._env.get_player_location()
                except Exception:
                    pass

            valid_actions = env._env.get_valid_actions()

            # --- 动作选择：处理循环并调用 LLM ---
            # TODO(pu): 
            # if len(last_actions) == 4 and last_actions[0] == last_actions[2] and last_actions[1] == last_actions[3]:
            #     # 20eps return_mean=102.5
            #     f.write("[SYSTEM] Stuck in a 2-step loop. Forcing 'look' to re-evaluate.\n")
            #     action, prompt = ('look', "[SYSTEM] Stuck in a loop. Forcing 'look'.") if 'look' in valid_actions else (np.random.choice([a for a in valid_actions if a not in last_actions] or valid_actions), "[SYSTEM] Stuck in a loop. Forcing random action.")
            # else: # 20eps return_mean=154.0
            action, prompt = llm_policy.sample_action_v2(
                history=list(episode_history)[-args.max_history_len:],
                full_scene_description=full_scene_description,
                inventory=inventory_str, 
                location=location_str,
                valid_actions=valid_actions, 
                good_memory=good_trial_memory.get_memories(),
                bad_memory=bad_trial_memory.get_memories()
            )
            
            last_actions.append(action)
            
            # --- 与环境交互 ---
            next_obs_or_dict, reward, done, info = env.step(action, return_str=True)
            episode_reward += reward

            next_obs_str = next_obs_or_dict.get('observation', '') if isinstance(next_obs_or_dict, dict) else next_obs_or_dict
            episode_history.append({'action': action, 'obs': next_obs_str})

            # --- 更新“好记忆” ---
            if reward > 0:
                good_memory_suggestion = f"At the location '{location_str}', performing the action '{action}' was successful and yielded a positive reward."
                log_msg = good_trial_memory.add_memory(good_memory_suggestion)
                f.write(f"[GOOD MEMORY UPDATE]: {log_msg}\n")
                print(f"[GOOD MEMORY UPDATE]: {log_msg}")

            # --- 记录日志 ---
            f.write(f"--- Step {step_count} ---\n")
            f.write(f"[Prompt Sent to LLM]:\n==============================================\n{prompt}\n")
            f.write(f"==============================================\n")
            f.write(f"[LLM Chose Action]: {action}\n")
            if done:
                f.write(f"[Final Observation]:\n{next_obs_str}\n")
            else:
                f.write(f"[Next Observation]:\n{next_obs_str}\n")
            f.write(f"---------------------------------\n")
            f.write(f"[Reward]: {reward}, [Done]: {done}, [Total Score]: {episode_reward}\n")
            f.write(f"---------------------------------\n\n")
            f.flush()

            obs_or_dict = next_obs_or_dict
            step_count += 1
            if step_count >= env_cfg.max_steps:
                done = True

        final_score = info.get('eval_episode_return', episode_reward)
        total_scores.append(final_score)

        # --- 回合结束后的自主反思 ---
        if args.use_autonomous_reflection:
            f.write("[Reflection Mode]: Autonomous\n")
            print("[Reflection Mode]: Autonomous")
            
            reflections = llm_policy.generate_autonomous_reflection(episode_history, final_score=final_score)
            f.write(f"Reflections JSON from LLM: {reflections}\n")
            print(f"Reflections JSON from LLM: {reflections}")

            bad_reflection_suggestion = reflections.get('alternative_action_suggestion')
            if bad_reflection_suggestion:
                log_msg = bad_trial_memory.add_memory(bad_reflection_suggestion)
                f.write(f"[BAD MEMORY UPDATE]: {log_msg}\n")
                print(f"[BAD MEMORY UPDATE]: {log_msg}")
            else:
                 f.write(f"[BAD MEMORY UPDATE]: LLM did not provide a valid suggestion.\n")
                 print(f"[BAD MEMORY UPDATE]: LLM did not provide a valid suggestion.")
        
        f.write(f"Episode {episode_id} finished. Final Score: {final_score}\n")
        print(f"Episode {episode_id} finished. Final Score: {final_score}")

    # --- 实验结束，清理和总结 ---
    f.write(f"\n\n{'='*80}\n")
    f.write(f"All episodes finished. Total Scores: {total_scores}\n")
    f.write(f"Average score: {np.mean(total_scores)}\n")
    f.close()
    print(f"[RANK {rank}] Finished. Log written to {log_file}")
    del env

    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"[RANK {rank}] Process group destroyed.")

if __name__ == '__main__':
    args = parse_args()
    main(args)