from zoo.babyai.priorzero.envs.babyai_env import BabyAIEnv
cfg = dict(env_addr='http://127.0.0.1:8000', data_idx=0, max_steps=64,
    max_action_num=20, tokenizer_path='/mnt/shared-storage-user/puyuan/xiongjyu/models/bge-base-en-v1.5',
    max_seq_len=512, for_unizero=True, use_high_level_actions=True,
    collector_env_num=1, evaluator_env_num=1)
env = BabyAIEnv(cfg)
obs = env.reset(return_str=True)
print('=== RESET ===')
print('mission:', obs.get('raw_obs_text', '')[:200])
print('valid_actions:', obs['valid_actions'])
print('action_mask:', obs['action_mask'][:10])
print('num_actions:', sum(obs['action_mask']))
for i in range(5):
    action = obs['valid_actions'][0] if obs['valid_actions'] else 'check available actions'
    ts = env.step(action, return_str=True)
    print(f'step {i}: action={action}, reward={ts.reward:.4f}, done={ts.done}')
    if ts.done: break
    obs = ts.obs
env.close()
print('=== DONE ===')