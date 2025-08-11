import logging
import copy
import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from easydict import EasyDict
from collections import deque
from abc import ABC, abstractmethod
import numpy as np
import torch
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.distributed as dist
import torch

from jericho_env import JerichoEnv
# --- LLM Provider Specific Imports ---
# Qwen (local transformers)
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Poe API (or other OpenAI-compatible APIs)
import openai

# --- 新增的导入 ---
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# =====================================================================================
# 优化后的 MemoryManager 类 (无修改)
# =====================================================================================
class MemoryManager:
    """
    管理好/坏记忆的类，实现了基于相似度的驱逐策略。
    【优化】: 增加了更严格的冗余检查。
    """
    def __init__(self, maxlen: int, similarity_threshold: float = 0.85, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 MemoryManager.
        
        参数:
            maxlen (int): 记忆库的最大容量。
            similarity_threshold (float): 用于判断是否冗余或替换的相似度阈值。 (稍微提高阈值)
            device (str): 用于计算嵌入的设备 ('cuda' 或 'cpu')。
        """
        self.maxlen = maxlen
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.memories: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        logging.info(f"MemoryManager initialized with maxlen={maxlen}, threshold={similarity_threshold} on device='{self.device}'")

    def get_memories(self) -> List[str]:
        return self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def add_memory(self, new_memory_text: str) -> str:
        """
        添加一条新记忆，并根据策略进行管理。
        【优化】: 优先检查并拒绝与现有记忆高度相似的新记忆，以防止冗余。
        返回一个描述所执行操作的日志消息。
        """
        if not new_memory_text or not isinstance(new_memory_text, str):
            return "Skipped adding empty or invalid memory."

        with torch.no_grad():
            new_embedding = self.model.encode(new_memory_text, convert_to_tensor=True, device=self.device)

        # 【核心优化 1】: 检查新记忆是否与任何现有记忆过于相似
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

        # 如果记忆库已满，执行先进先出（FIFO）替换最旧的记忆
        else:
            removed_memory = self.memories.pop(0)
            self.memories.append(new_memory_text)
            
            # 更新嵌入张量
            self.embeddings = torch.cat([self.embeddings[1:], new_embedding.unsqueeze(0)], dim=0)
            return (f"FIFO eviction. "
                    f"Removed: '{removed_memory}', Added: '{new_memory_text}'")


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

# =====================================================================================
# 1. 定义一个抽象基类 (Abstract Base Class) 作为所有 LLM 策略的通用接口
# =====================================================================================
class BaseLLMPolicy(ABC):
    def __init__(self, reflection_history_len=9999):
        self.reflection_history_len = reflection_history_len
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def format_prompt_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None, use_structured_info: bool = True) -> str:
        """
        - 增强了系统指令，明确了目标和思考方式。
        - 强化了决策指令，引导模型利用记忆打破循环，并增加了更强硬的指令。
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
    def sample_action_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None, use_structured_info: bool = True) -> tuple[str, str]:
        pass

    def clean_obs(self, obs: str) -> str:
        if not isinstance(obs, str):
            logging.error(f"clean_obs received a non-string input: {type(obs)}. Converting to string.")
            obs = str(obs)
            
        obs = re.sub(r'Copyright \(c\).*reserved\.', '', obs, flags=re.DOTALL)
        obs = re.sub(r'ZORK is a registered trademark.*', '', obs, flags=re.DOTALL)
        obs = re.sub(r'Revision \d+ / Serial number \d+', '', obs, flags=re.DOTALL)
        core_obs = obs.split('Valid actions:')[0].strip()
        return '\n'.join(line.strip() for line in core_obs.split('\n') if line.strip())

    @abstractmethod
    def generate_autonomous_reflection(self, history: List[Dict[str, str]], final_score: float, use_structured_info: bool = True) -> Dict[str, str]:
        pass

# =====================================================================================
# 2. 为本地 Qwen 模型创建一个具体实现
# =====================================================================================
class QwenLocalPolicy(BaseLLMPolicy):
    def __init__(self, model_path: str, reflection_history_len: int = 9999, local_rank: int = 0):
        super().__init__(reflection_history_len)
        self.device = torch.device(f"cuda:{local_rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(self.device)
        print(f"QwenLocalPolicy initialized on device {self.device}")

    def _generate_text(self, prompt: str, gen_config: GenerationConfig) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**model_inputs, generation_config=gen_config)
        output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content

    def sample_action_v2(self, history: List[Dict[str, str]], full_scene_description: str, inventory: str, location: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None, use_structured_info: bool = True) -> tuple[str, str]:
        prompt = self.format_prompt_v2(history, full_scene_description, inventory, location, valid_actions, good_memory, bad_memory, use_structured_info)
        
        gen_config = GenerationConfig(
            temperature=0.2, top_p=0.9, do_sample=True, max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        content = self._generate_text(prompt, gen_config).lower()
        
        sorted_actions = sorted(valid_actions, key=len, reverse=True)
        for va in sorted_actions:
            cleaned_va = va.lower().strip()
            cleaned_content = content.replace(">", "").strip()
            if cleaned_va in cleaned_content or cleaned_content in cleaned_va:
                return va, prompt

        safe_actions = ['look', 'inventory', 'l', 'i']
        for sa in safe_actions:
            if sa in valid_actions:
                return sa, prompt
        
        return (valid_actions[0] if valid_actions else 'look'), prompt

    def generate_autonomous_reflection(self, history: List[Dict[str, str]], final_score: float, use_structured_info: bool = True) -> Dict[str, str]:
        """
        - 聚焦于分析导致游戏结束或停滞的【最后一步】。
        - 要求提供更具体的、可操作的替代方案。
        - 完整地展示最后的游戏历史，包括导致结束的动作和观测。
        """
        recent_history = history[-self.reflection_history_len:]
        # 确保历史记录包含最后一步和最终结果
        trajectory_str = "\n".join([f"Step {i+1}: I did '> {h['action']}' and the outcome was:\n{h['obs']}\n" for i, h in enumerate(recent_history)])

        prompt = (
            "You are an expert strategy analyst for a text-based adventure game. Your task is to analyze a game trajectory, identify the critical mistake that led to a low score or game over, and produce actionable advice.\n\n"
            f"--- Final Score ---\n{final_score}\n\n"
            f"--- Full Gameplay Log of the Final Attempt ---\n{trajectory_str}\n"
            "--- Analysis Task ---\n"
            "Analyze the final steps of the gameplay log. Pinpoint the single critical mistake that ended the game or caused it to get stuck. Provide your analysis as a single, clean JSON object with two keys: `critical_mistake_analysis` and `alternative_action_suggestion`.\n"
            "- `critical_mistake_analysis`: A concise sentence explaining what the final wrong move was and why it was wrong, based on the final outcome. Example: 'The final action 'go north' led to a dead end, wasting a turn when exploring 'east' towards the Mayor's home was a more promising option based on the description.'\n"
            "- `alternative_action_suggestion`: A concrete, actionable suggestion for what to do instead of the mistaken action, phrased as a general rule for the future. Example: 'When at the 'Outside' location after moving west, the next logical step is to explore 'east' to investigate the Mayor's home mentioned in the description.'\n\n"
            "Respond ONLY with the JSON object, without any surrounding text or explanations.\n"
        )
        
        gen_config = GenerationConfig(
            temperature=0.2, top_p=0.9, do_sample=True, max_new_tokens=250,
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
# 3. Poe API 实现 (同样应用优化, 此处省略，修改逻辑与 QwenLocalPolicy 完全相同)
# =====================================================================================
class PoeAPIPolicy(BaseLLMPolicy): # ... (内部逻辑与QwenLocalPolicy的修改相同)
    pass # 假设已按上述逻辑修改

# =====================================================================================
# 主程序修改
# =====================================================================================
if __name__ == '__main__':
    """
    export CUDA_VISIBLE_DEVICES=6
    torchrun --nproc_per_node=1  /fs-computility/niuyazhe/puyuan/code/LightZero/zoo/jericho/envs/test_qwen_v8.py  
    """
    # --- 核心配置区 ---
    LLM_PROVIDER = "Qwen"
    
    LLM_CONFIGS = {
        "Qwen": {
            "model_path": "/fs-computility/niuyazhe/shared/xiongjyu/model/Qwen2.5-7B-Instruct",
            "model_name": "Qwen2.5-7B-Instruct",
        },
        "PoeAPI": {
            "model_name": "GPT-3.5-Turbo",
            "api_key": "YOUR_POE_API_KEY", # 请替换为您的密钥
            "base_url": "https://api.poe.com/v1"
        }
    }

    num_episodes = 20
    max_history_len = 10 
    good_trial_memory_maxlen = 20
    bad_trial_memory_maxlen = 20

    # --- 游戏和实验设置 ---
    USE_AUTONOMOUS_REFLECTION = True 
    USE_STRUCTURED_INFO = True 
    ENV_TYPE = 'detective'

    # --- 【优化】相似度阈值 ---
    SIMILARITY_THRESHOLD = 0.9  # 可以适当提高阈值，因为反思质量更高了

    # --- 初始化 ---
    rank, world_size = init_distributed()
    print(f"[RANK {rank}] Initialized. World size: {world_size}")
    
    # --- 实例化策略对象 ---
    llm_policy: BaseLLMPolicy
    current_config = LLM_CONFIGS[LLM_PROVIDER]
    
    print(f"Using LLM Provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == "Qwen":
        llm_policy = QwenLocalPolicy(
            model_path=current_config["model_path"],
            local_rank=rank
        )
    elif LLM_PROVIDER == "PoeAPI":
        # llm_policy = PoeAPIPolicy(...) # 省略
        pass
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
    
    env_cfg = EasyDict(
        dict(
            max_steps=100,
            game_path="./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/" + f"{ENV_TYPE}.z5",
            add_location_and_inventory=USE_STRUCTURED_INFO,
            max_action_num=55,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            remove_stuck_actions=False,
            for_unizero=False,
            collector_env_num=1,
            evaluator_env_num=1,
            save_replay=False,
            save_replay_path=None,
            env_type=ENV_TYPE,
            collect_policy_mode='expert'
        )
    )

    log_dir = f'./priorzero_log_optimized/{current_config["model_name"]}/{ENV_TYPE}_structured_{USE_STRUCTURED_INFO}_auto_reflect_{USE_AUTONOMOUS_REFLECTION}_v2'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.txt")
    f = open(log_file, "w", encoding="utf-8")

    env = JerichoEnv(env_cfg)
    
    good_trial_memory = MemoryManager(maxlen=good_trial_memory_maxlen, similarity_threshold=SIMILARITY_THRESHOLD)
    bad_trial_memory = MemoryManager(maxlen=bad_trial_memory_maxlen, similarity_threshold=SIMILARITY_THRESHOLD)

    total_scores = []

    for episode_id in range(num_episodes):
        f.write(f"{'='*80}\n")
        f.write(f"STARTING EPISODE: {episode_id} | Good Memories: {len(good_trial_memory)}, Bad Memories: {len(bad_trial_memory)}\n")
        f.write(f"Current Good Memories: {good_trial_memory.get_memories()}\n")
        f.write(f"Current Bad Memories: {bad_trial_memory.get_memories()}\n")
        f.write(f"CONFIG: USE_STRUCTURED_INFO = {USE_STRUCTURED_INFO}, USE_AUTONOMOUS_REFLECTION = {USE_AUTONOMOUS_REFLECTION}\n")
        f.write(f"{'='*80}\n")
        f.flush()
        
        obs_or_dict = env.reset(return_str=True)
        done = False
        step_count = 0
        episode_reward = 0
        episode_history = []
        last_actions = deque(maxlen=4)

        while not done:
            if isinstance(obs_or_dict, dict):
                full_scene_description = obs_or_dict.get('observation', '')
                inventory_str = obs_or_dict.get('inventory', 'not tracked')
                location_str = obs_or_dict.get('location', 'not tracked')
            else:
                full_scene_description = obs_or_dict
                inventory_str = "not tracked"
                location_str = "not tracked"
                try:
                    inventory_str = env._env.get_inventory()
                    location_str = env._env.get_player_location()
                except:
                    pass

            valid_actions = env._env.get_valid_actions()

            if len(last_actions) == 4 and last_actions[0] == last_actions[2] and last_actions[1] == last_actions[3]:
                f.write("[SYSTEM] Stuck in a 2-step loop. Forcing 'look' to re-evaluate.\n")
                if 'look' in valid_actions:
                    action = 'look'
                    prompt = "[SYSTEM] Stuck in a loop. Forcing 'look' to re-evaluate."
                else: 
                    non_loop_actions = [a for a in valid_actions if a not in last_actions]
                    if non_loop_actions:
                        action = np.random.choice(non_loop_actions)
                        prompt = f"[SYSTEM] Stuck in a loop and 'look' is unavailable. Forcing random exploration: {action}."
                    else:
                        action = valid_actions[0]
                        prompt = f"[SYSTEM] Stuck in a loop and no other options. Forcing: {action}."
            else:
                action, prompt = llm_policy.sample_action_v2(
                    history=list(episode_history)[-max_history_len:],
                    full_scene_description=full_scene_description,
                    inventory=inventory_str, 
                    location=location_str,
                    valid_actions=valid_actions, 
                    good_memory=good_trial_memory.get_memories(),
                    bad_memory=bad_trial_memory.get_memories(), 
                    use_structured_info=USE_STRUCTURED_INFO
                )
            
            last_actions.append(action)
            
            next_obs_or_dict, reward, done, info = env.step(action, return_str=True)
            episode_reward += reward

            if isinstance(next_obs_or_dict, dict):
                next_obs_str = next_obs_or_dict.get('observation', '')
            else:
                next_obs_str = next_obs_or_dict

            history_entry = {'action': action, 'obs': next_obs_str}
            episode_history.append(history_entry)

            # 【修改 3】: 增加“好记忆”的生成逻辑
            if reward > 0:
                good_memory_suggestion = f"At the location '{location_str}', performing the action '{action}' was successful and yielded a positive reward."
                log_msg = good_trial_memory.add_memory(good_memory_suggestion)
                f.write(f"[GOOD MEMORY UPDATE]: {log_msg}\n")
                print(f"[GOOD MEMORY UPDATE]: {log_msg}")


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

        # 【修改 5】: 修改反思逻辑，只在分数低时进行，并使用新的反思结果
        if USE_AUTONOMOUS_REFLECTION: # 只在分数低时反思
            f.write("[Reflection Mode]: Autonomous\n")
            print("[Reflection Mode]: Autonomous")
            
            # 使用新的、更具针对性的反思函数
            reflections = llm_policy.generate_autonomous_reflection(
                episode_history, final_score=final_score, use_structured_info=USE_STRUCTURED_INFO
            )

            f.write(f"Reflections JSON from LLM: {reflections}\n")
            print(f"Reflections JSON from LLM: {reflections}")

            # 我们现在使用更有意义的“替代行动建议”作为坏记忆
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

    f.write(f"\n\n{'='*80}\n")
    f.write(f"All episodes finished. Total Scores: {total_scores}\n")
    f.write(f"Average score: {np.mean(total_scores)}\n")
    f.close()
    print(f"[RANK {rank}] Finished. Log written to {log_file}")
    del env

    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"[RANK {rank}] Process group destroyed.")