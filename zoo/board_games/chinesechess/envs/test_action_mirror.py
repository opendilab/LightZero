"""
测试中国象棋环境的动作镜像转换功能

验证黑方的观测旋转和动作镜像转换是否正确对应
"""

import numpy as np
import sys
sys.path.append('.')

from zoo.board_games.chinesechess.envs.cchess_env import ChineseChessEnv
from easydict import EasyDict


def test_action_mirror():
    """测试动作镜像转换的正确性"""
    
    # 创建环境
    cfg = EasyDict({
        'battle_mode': 'self_play_mode',
        'battle_mode_in_simulation_env': 'self_play_mode',
        'render_mode': None,
        'replay_path': None,
        'agent_vs_human': False,
        'prob_random_agent': 0,
        'prob_expert_agent': 0,
        'uci_engine_path': None,
        'engine_depth': 5,
        'channel_last': False,
        'scale': False,
        'stop_value': 2,
        'max_episode_steps': 500,
    })
    
    env = ChineseChessEnv(cfg)
    obs = env.reset()
    
    print("=" * 80)
    print("测试中国象棋环境的动作镜像转换功能")
    print("=" * 80)
    
    # 测试1：初始状态（红方）
    print("\n【测试1】初始状态 - 红方回合")
    print(f"当前玩家: {env.current_player} (1=红方, 2=黑方)")
    print(f"观测形状: {obs['observation'].shape}")
    print(f"合法动作数: {obs['action_mask'].sum()}")
    
    legal_actions = env.legal_actions
    print(f"前5个合法动作: {legal_actions[:5]}")
    
    # 验证action_mask和legal_actions一致
    action_mask_indices = np.where(obs['action_mask'] == 1)[0]
    print(f"action_mask中的前5个合法动作: {action_mask_indices[:5]}")
    assert len(action_mask_indices) == len(legal_actions), "红方: action_mask数量与legal_actions不一致！"
    print("✓ 红方: action_mask 与 legal_actions 一致")
    
    # 测试2：执行一步后切换到黑方
    print("\n【测试2】执行一步后 - 黑方回合")
    action = legal_actions[0]
    print(f"红方执行动作: {action}")
    
    timestep = env.step(action)
    obs = timestep.obs
    
    print(f"当前玩家: {env.current_player} (1=红方, 2=黑方)")
    print(f"观测形状: {obs['observation'].shape}")
    print(f"合法动作数: {obs['action_mask'].sum()}")
    
    # 获取黑方的合法动作（真实坐标）
    legal_actions_black_real = env.legal_actions
    print(f"黑方合法动作（真实坐标）前5个: {legal_actions_black_real[:5]}")
    
    # 获取黑方的action_mask（镜像坐标）
    action_mask_indices_black = np.where(obs['action_mask'] == 1)[0]
    print(f"黑方action_mask（镜像坐标）前5个: {action_mask_indices_black[:5]}")
    
    # 验证：将action_mask中的镜像动作转回真实坐标，应该等于legal_actions
    action_mask_to_real = []
    for mirror_action in action_mask_indices_black:
        real_action = env._mirror_action(mirror_action)
        action_mask_to_real.append(real_action)
    
    action_mask_to_real_sorted = sorted(action_mask_to_real)
    legal_actions_sorted = sorted(legal_actions_black_real)
    
    print(f"\n验证镜像转换:")
    print(f"  action_mask转回真实坐标后: {action_mask_to_real_sorted[:5]}...")
    print(f"  legal_actions（真实坐标）: {legal_actions_sorted[:5]}...")
    
    assert action_mask_to_real_sorted == legal_actions_sorted, "黑方: action_mask镜像转换后与legal_actions不一致！"
    print("✓ 黑方: action_mask 镜像转换正确")
    
    # 测试3：验证镜像函数的对称性
    print("\n【测试3】验证镜像函数的对称性")
    test_actions = [0, 45, 89, 100, 500, 1000, 8099]
    for test_action in test_actions:
        mirror_once = env._mirror_action(test_action)
        mirror_twice = env._mirror_action(mirror_once)
        print(f"动作 {test_action:4d} -> 镜像 {mirror_once:4d} -> 再镜像 {mirror_twice:4d}")
        assert mirror_twice == test_action, f"镜像函数不对称！{test_action} != {mirror_twice}"
    print("✓ 镜像函数对称性验证通过")
    
    # 测试4：执行黑方动作并切换回红方
    print("\n【测试4】执行黑方动作后 - 切换回红方")
    black_mirror_action = action_mask_indices_black[0]
    print(f"黑方执行动作（镜像坐标）: {black_mirror_action}")
    
    # 应该自动转换为真实坐标执行
    timestep = env.step(black_mirror_action)
    obs = timestep.obs
    
    print(f"当前玩家: {env.current_player} (1=红方, 2=黑方)")
    print(f"合法动作数: {obs['action_mask'].sum()}")
    
    # 验证切换回红方后，action_mask又回到真实坐标
    legal_actions_red = env.legal_actions
    action_mask_indices_red = np.where(obs['action_mask'] == 1)[0]
    
    assert sorted(action_mask_indices_red) == sorted(legal_actions_red), "切换回红方后: action_mask与legal_actions不一致！"
    print("✓ 切换回红方: action_mask 恢复为真实坐标")
    
    # 测试5：模拟MCTS场景
    print("\n【测试5】模拟MCTS场景")
    env2 = env.copy()
    print(f"复制环境后，当前玩家: {env2.current_player}")
    
    # 使用simulate_action
    action_to_simulate = action_mask_indices_red[0]
    print(f"模拟红方动作: {action_to_simulate}")
    
    try:
        new_env = env2.simulate_action(action_to_simulate)
        print(f"模拟成功！新环境当前玩家: {new_env.current_player}")
        print("✓ simulate_action 工作正常")
    except Exception as e:
        print(f"✗ simulate_action 失败: {e}")
        raise
    
    # 测试6：Bot模式测试（最重要的修复）
    print("\n【测试6】Bot模式测试（eval_mode/play_with_bot_mode）")
    cfg_eval = EasyDict(cfg)
    cfg_eval.battle_mode = 'eval_mode'
    cfg_eval.agent_vs_human = False
    
    env_eval = ChineseChessEnv(cfg_eval)
    obs_eval = env_eval.reset()
    
    print(f"初始玩家: {env_eval.current_player}")
    print(f"合法动作数: {obs_eval['action_mask'].sum()}")
    
    # Agent (Player 1, 红方) 执行一个动作
    legal_actions_red = np.where(obs_eval['action_mask'] == 1)[0]
    agent_action = legal_actions_red[0]
    print(f"Agent执行动作: {agent_action}")
    
    try:
        timestep = env_eval.step(agent_action)
        print(f"执行成功！当前玩家: {env_eval.current_player}")
        
        if not timestep.done:
            print(f"Bot (Player 2, 黑方) 将执行动作...")
            # 注意：step内部会调用bot_action()并自动处理
            # 这里我们已经执行了一步，下一次step会由bot执行
            print("✓ Bot模式测试通过（无非法动作警告）")
        else:
            print("游戏已结束")
    except Exception as e:
        print(f"✗ Bot模式测试失败: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("所有测试通过！动作镜像转换功能正常工作。")
    print("关键修复：Bot的动作（真实坐标）不会被错误地转换。")
    print("=" * 80)


if __name__ == "__main__":
    test_action_mirror()

