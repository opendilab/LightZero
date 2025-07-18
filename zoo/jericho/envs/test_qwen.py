import logging
import copy
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from easydict import EasyDict
from collections import deque

import numpy as np
import torch
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.distributed as dist
import torch

from jericho_env import JerichoEnv  


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

class Qwen3Policy:
    def __init__(self, model_path=None, local_rank=0):
        self.device = torch.device(f"cuda:{local_rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(self.device)
        self.max_history_len = 5

    def format_prompt(self, history: List[Dict[str, str]], current_obs: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None) -> str:
        system_prompt = (
            "You are an expert at playing text-based games.\n"
            "You will be given a history of game states and actions taken, as well as your memory from previous gameplay.\n"
            "Use this information to select the best next action.\n\n"
        )
        if good_memory:
            system_prompt += "Good memory from past games:\n"
            for mem in good_memory:
                system_prompt += f"- {mem}\n"
        else:
            system_prompt += "Good memory from past games: None\n"

        if bad_memory:
            system_prompt += "Bad memory from past failures:\n"
            for mem in bad_memory:
                system_prompt += f"- {mem}\n"
        else:
            system_prompt += "Bad memory from past failures: None\n"

        
        system_prompt += "\nHistory:\n"
        if history is None or len(history) == 0:
            system_prompt += "None\n"
        for h in history:
            system_prompt += f"[State]: {h['obs']}\n[Action]: {h['action']}\n"
        
        state = current_obs.split('Valid actions:')[0]
       
        system_prompt += (
            f"\nCurrent:\n[State]: {state}\n"
            f"[Valid Actions]: {', '.join(valid_actions)}\n"
            "Please choose the best next action from the valid actions above, and answer with only one word or phrase, without any explanation.\n"
            "[Action]:"
        )
        return system_prompt
    
    def generate_reflection(self, history: List[Dict[str, str]], positive: bool) -> str:
        trajectory_str = "\n".join([f"[State]: {h['obs']}\n[Action]: {h['action']}" for h in history])
        if positive:
            prompt = (
                "You will receive a log of successful gameplay from a text-based adventure game.\n"
                "Summarize a good strategy or useful lesson learned from the following playthrough in one sentence.\n"
                f"{trajectory_str}\n"
            )
        else:
            prompt = (
                "You will receive a log of unsuccessful gameplay from a text-based adventure game.\n"
                "Please identify the reasons for failure and provide a short suggestion for improving the strategy next time.\n"
                "Respond with one sentence only.\n"
                f"{trajectory_str}\n"
            )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = self.model.generate(
            **model_inputs,
            generation_config=gen_config
        )
        output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content

    def sample_action(self, history: List[Dict[str, str]], current_obs: str, valid_actions: List[str], good_memory: List[str] = None, bad_memory: List[str] = None) -> str:
        prompt = self.format_prompt(history, current_obs, valid_actions, good_memory, bad_memory)
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = self.model.generate(
            **model_inputs,
            generation_config=gen_config
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        res_action = None

        for va in valid_actions:
            if va.lower() in content:
                res_action = va
        if res_action is None:
            if valid_actions:
                res_action = valid_actions[0]
            else:
                res_action = 'go'
        
        return res_action, prompt
   
if __name__ == '__main__':
    rank, world_size = init_distributed()
    print(f"[RANK {rank}] Initialized. World size: {world_size}")
    
    
    # env_type='detective' # zork1, acorncourt, detective, omniquest
    env_type='zork1'
    model_name = "Qwen2.5-7B-Instruct"  # Path to the Qwen model
    # Configuration dictionary for the environment.
    env_cfg = EasyDict(
        dict(
            max_steps=100,
            game_path="./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/" + f"{env_type}.z5",
            max_action_num=12,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            remove_stuck_actions=False,
            # add_location_and_inventory=True, # TODO 尝试打开或者不打开该参数
            add_location_and_inventory=False, # TODO 尝试打开或者不打开该参数
            for_unizero=False,
            collector_env_num=1,
            evaluator_env_num=1,
            save_replay=False,
            save_replay_path=None,
            env_type=env_type,
            collect_policy_mode='expert'    # random, human, expert
        )
    )

    if env_cfg.add_location_and_inventory:
        log_dir = f'/fs-computility/niuyazhe/shared/xiongjyu/jericho/LightZero/log/{model_name}/{env_type}_add_locAndinv'
    else:
        log_dir = f'/fs-computility/niuyazhe/shared/xiongjyu/jericho/LightZero/log/{model_name}/{env_type}'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"rank_{rank}.txt")
    f = open(log_file, "w", encoding="utf-8")

    env = JerichoEnv(env_cfg)
    qwen_policy = Qwen3Policy(model_path=f"/fs-computility/niuyazhe/shared/xiongjyu/model/{model_name}", local_rank=rank)
    history = deque(maxlen=qwen_policy.max_history_len)

    num_episodes = 20  # 可设置为任意 N
    good_trial_memory = deque(maxlen=5)  
    bad_trial_memory = deque(maxlen=5)  

    for episode_id in range(num_episodes):
        f.write(f"{'='*60}\n")
        f.write(f'current episode: {episode_id}\n')
        f.write(f"{'='*60}\n")
        f.flush()
        obs = env.reset(return_str=True)
        done = False
        step_count = 0
        history.clear()

        while not done:
            obs_str = obs['observation']
            action, prompt = qwen_policy.sample_action(history=list(history),current_obs=obs_str, valid_actions=env._action_list,
                                                        good_memory=list(good_trial_memory), bad_memory=list(bad_trial_memory))
            obs, reward, done, info = env.step(action, return_str=True)
            history.append({'obs': obs_str, 'action': action})

            # 每步写入日志
            f.write(f"Step {step_count}\n")
            f.write(f"[Prompt]:\n{prompt}\n")
            f.write(f"[Qwen Action]: {action}\n")
            f.write(f"[Env Feedback] Reward: {reward}, Done: {done}\n")
            f.write(f"{'-'*60}\n")
            f.flush()

            step_count += 1

            if "*** you have died ***" in obs_str.lower():
                reflection = qwen_policy.generate_reflection(list(history), positive=False)
                bad_trial_memory.append(reflection)
                f.write(f"[BAD Reflection]: {reflection}\n")
                print(f'[BAD Reflection]: {reflection}')
            elif "your score has just gone up by" in obs_str.lower():
                reflection = qwen_policy.generate_reflection(list(history), positive=True)
                good_trial_memory.append(reflection)
                f.write(f"[GOOD Reflection]: {reflection}\n")
                print(f'[GOOD Reflection]: {reflection}')

        
        f.write(f"Episode finished. Final return: {info.get('eval_episode_return', 0.0)}\n")

    f.close()
    print(f"[RANK {rank}] Finished. Log written to {log_file}")
    del env  