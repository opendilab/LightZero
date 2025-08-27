# test_world_model_cache.py
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import csv
import os

# 确保lzero和toy_env在Python路径中
from lzero.model.unizero_world_models.world_model import WorldModel
from toy_env import ToyEnv
from lzero.model.unizero_world_models.utils import hash_state

# ==============================================================================
# Helper classes and functions for the test
# ==============================================================================

class DummyTokenizer:
    """一个用于向量观测的简化分词器。"""
    def __init__(self, obs_shape, embed_dim, device):
        self.encoder = nn.Linear(obs_shape[0], embed_dim).to(device)
        self.device = device

    def encode_to_obs_embeddings(self, obs):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if len(obs_tensor.shape) == 2:
            return self.encoder(obs_tensor).unsqueeze(1)
        elif len(obs_tensor.shape) == 3:
            return self.encoder(obs_tensor).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported observation tensor shape: {obs_tensor.shape}")

def print_cache_summary(name: str, kv_cache, context_length: int):
    """打印 KeysValues 缓存对象的摘要，并高亮显示截断行为。"""
    if kv_cache is None:
        print(f"  {name}: None")
        return 0, "None"
    
    size = kv_cache.size
    shape = kv_cache._keys_values[0]._k_cache._cache.shape
    status_msg = ""
    # 模型在截断时会为未来的(act, obs)等留出空间，所以我们检查是否接近限制
    if size >= context_length - 3:
        status_msg = f" (!! Approaching/Exceeded Context Limit of {context_length}. Truncation will occur.)"
        
    print(f"  {name}: Size = {size}, Shape = {shape}{status_msg}")
    return size, f"Size={size}"

# ==============================================================================
# Main Test Function
# ==============================================================================
def test_cache_logic():
    # 1. 设置环境和模型配置
    env_cfg = ToyEnv.default_config()
    env = ToyEnv(env_cfg)
    
    world_model_cfg = EasyDict(
        dict(
            continuous_action_space=False, num_layers=2, num_heads=4, embed_dim=64,
            context_length=8, max_tokens=100, tokens_per_block=2,
            action_space_size=env.cfg.action_space_size, env_num=1, obs_type='vector',
            device='cuda' if torch.cuda.is_available() else 'cpu', rotary_emb=False,
            policy_entropy_weight=0, predict_latent_loss_type='mse', group_size=8,
            gamma=0.99, dormant_threshold=0.0, analysis_dormant_ratio=False,
            latent_recon_loss_weight=0, perceptual_loss_weight=0, support_size=11,
            max_cache_size=1000, final_norm_option_in_obs_head='SimNorm', norm_type='LN',
            embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, max_blocks=10, gru_gating=False,
        )
    )
    
    # 2. 实例化世界模型
    tokenizer = DummyTokenizer(env.cfg.observation_shape, world_model_cfg.embed_dim, world_model_cfg.device)
    world_model = WorldModel(world_model_cfg, tokenizer).to(world_model_cfg.device)
    world_model.eval()

    # 3. 设置日志文件
    log_filename = "cache_log.csv"
    log_filepath = os.path.join(os.getcwd(), log_filename)
    print(f"\nLogging statistics to: {log_filepath}")
    
    with open(log_filepath, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        header = [
            'Timestep', 'Action_Taken', 'Current_State',
            'Root_Cache_Hit', 'Root_Cache_Size',
            'Recurrent_Cache_Hit', 'Recurrent_Cache_Size',
            'Comment'
        ]
        csv_writer.writerow(header)

        # 4. 运行一个 episode 并检查缓存
        obs_dict = env.reset()
        last_action = -1
        last_obs_for_infer = np.zeros_like(obs_dict['observation'])

        for t in range(env.cfg.collect_max_episode_steps):
            print(f"\n{'='*25} Timestep {t} {'='*25}")
            print(f"Environment State: Obs = {obs_dict['observation']}, Timestep from Env = {obs_dict['timestep']}")
            
            log_row = {'Timestep': t, 'Current_State': str(obs_dict['observation'])}

            # --- 模拟 MCTS 搜索开始 ---
            obs_act_dict = {
                'obs': last_obs_for_infer, 'action': np.array([last_action]),
                'current_obs': obs_dict['observation']
            }
            print("\n[1. Initial Inference] -> Simulating root node creation for MCTS.")
            print(f"  Inputs: last_obs={obs_act_dict['obs']}, last_action={obs_act_dict['action']}, current_obs={obs_act_dict['current_obs']}")
            
            with torch.no_grad():
                # 注意：start_pos 应该是一个列表或数组，以适应模型的批处理逻辑
                _, latent_state, _, _, _ = world_model.forward_initial_inference(
                    obs_act_dict, start_pos=[obs_dict['timestep']]
                )

            # --- 检查根节点缓存 ---
            print("\n[2. Inspecting Root Node Cache]")
            cache_key = hash_state(latent_state.cpu().numpy().flatten())
            cache_index = world_model.past_kv_cache_init_infer_envs[0].get(cache_key)
            
            if cache_index is not None:
                root_kv_cache = world_model.shared_pool_init_infer[0][cache_index]
                log_row['Root_Cache_Hit'] = 'Stored'
                size, _ = print_cache_summary("Stored Root KV Cache", root_kv_cache, world_model_cfg.context_length)
                log_row['Root_Cache_Size'] = size
            else:
                log_row['Root_Cache_Hit'] = 'Not_Found'
                log_row['Root_Cache_Size'] = 0
                print("  Status: Cache Not Found! (This is unexpected after the first step).")

            # --- 模拟一步 MCTS 循环推断 ---
            action_to_take = env.action_space.sample()
            log_row['Action_Taken'] = action_to_take
            print(f"\n[3. Recurrent Inference] -> Simulating one search step from the root.")
            print(f"  Action to explore: {action_to_take}")
            
            state_action_history = [(latent_state.cpu().numpy(), np.array([action_to_take]))]
            
            print("  Checking if root cache is available for recurrent step...")
            root_cache_key_for_recur = hash_state(state_action_history[0][0].flatten())
            root_cache_index = world_model.past_kv_cache_init_infer_envs[0].get(root_cache_key_for_recur)
            if root_cache_index is not None:
                 log_row['Comment'] = 'Recurrent step found root cache.'
                 print("  -> Cache Hit! The recurrent step will build upon the existing root cache.")
            else:
                 log_row['Comment'] = 'Recurrent step MISSES root cache!'
                 print("  -> Cache Miss! The recurrent step will have to regenerate context. (This indicates a problem)")

            with torch.no_grad():
                # 注意：start_pos 应该是一个列表或数组
                _, next_latent_state, _, _, _ = world_model.forward_recurrent_inference(
                    state_action_history,
                    start_pos=[obs_dict['timestep']]
                )
            
            # --- 检查循环推断节点的缓存 ---
            print("\n[4. Inspecting Recurrent Node Cache]")
            cache_key_recur = hash_state(next_latent_state.cpu().numpy().flatten())
            cache_index_recur = world_model.past_kv_cache_recurrent_infer.get(cache_key_recur)
            if cache_index_recur is not None:
                recurrent_kv_cache = world_model.shared_pool_recur_infer[cache_index_recur]
                log_row['Recurrent_Cache_Hit'] = 'Stored'
                size, _ = print_cache_summary("Stored Recurrent KV Cache", recurrent_kv_cache, world_model_cfg.context_length)
                log_row['Recurrent_Cache_Size'] = size
            else:
                log_row['Recurrent_Cache_Hit'] = 'Not_Found'
                log_row['Recurrent_Cache_Size'] = 0
                print("  Status: Recurrent Cache Not Found! (This is unexpected).")

            # --- 环境步进 ---
            print("\n[5. Stepping Environment]")
            timestep_obj = env.step(action_to_take)
            
            last_action = action_to_take
            last_obs_for_infer = obs_dict['observation']
            obs_dict = timestep_obj.obs
            
            # 写入日志行
            csv_writer.writerow([log_row.get(h, '') for h in header])
            
            if timestep_obj.done:
                print("\n" + "="*20 + " Episode Finished " + "="*20)
                break
                
    world_model.clear_caches()
    print(f"\nTest finished. Log saved to {log_filepath}")

if __name__ == "__main__":
    test_cache_logic()