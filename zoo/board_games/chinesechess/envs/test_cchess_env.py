import pytest
import os
from datetime import datetime
from easydict import EasyDict
from zoo.board_games.chinesechess.envs.cchess_env import ChineseChessEnv


@pytest.mark.envtest
class TestChineseChessEnv:

    def test_self_play_mode(self):
        """测试自对弈模式"""
        cfg = EasyDict(
            battle_mode='self_play_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        env.reset()
        print('=' * 50)
        print('自对弈模式测试')
        print('=' * 50)
        env.render(mode='human')

        step_count = 0
        while True:
            # player 1 (红方)
            action = env.random_action()
            print(f'第 {step_count + 1} 步 - 红方走棋')
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            step_count += 1

            if done:
                if reward > 0:
                    print('红方获胜!')
                elif reward < 0:
                    print('黑方获胜!')
                else:
                    print('和棋!')
                break

            # player 2 (黑方)
            action = env.random_action()
            print(f'第 {step_count + 1} 步 - 黑方走棋')
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            step_count += 1

            if done:
                if reward > 0:
                    print('黑方获胜!')
                elif reward < 0:
                    print('红方获胜!')
                else:
                    print('和棋!')
                break

        print(f'游戏结束，共 {step_count} 步')
        env.close()

    def test_play_with_bot_mode(self):
        """测试人机对战模式 (Agent vs Bot)"""
        cfg = EasyDict(
            battle_mode='play_with_bot_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        env.reset()
        print('=' * 50)
        print('人机对战模式测试 (Agent vs Random Bot)')
        print('=' * 50)
        env.render(mode='human')

        step_count = 0
        while True:
            # Agent (红方) 走棋
            action = env.random_action()
            print(f'第 {step_count + 1} 步 - Agent (红方) 走棋')
            obs, reward, done, info = env.step(action)
            # 在 play_with_bot_mode 下，step 会自动执行 bot 的回合
            env.render(mode='human')
            step_count += 2  # Agent + Bot 各走一步

            if done:
                eval_return = info.get('eval_episode_return', reward)
                if eval_return > 0:
                    print('Agent (红方) 获胜!')
                elif eval_return < 0:
                    print('Bot (黑方) 获胜!')
                else:
                    print('和棋!')
                break

        print(f'游戏结束，共约 {step_count} 步')
        env.close()

    def test_eval_mode(self):
        """测试评估模式"""
        cfg = EasyDict(
            battle_mode='eval_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        env.reset()
        print('=' * 50)
        print('评估模式测试')
        print('=' * 50)
        env.render(mode='human')

        step_count = 0
        while True:
            # Agent (红方) 走棋
            action = env.random_action()
            print(f'第 {step_count + 1} 步 - Agent (红方) 走棋')
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            step_count += 2

            if done:
                eval_return = info.get('eval_episode_return', reward)
                if eval_return > 0:
                    print('Agent (红方) 获胜!')
                elif eval_return < 0:
                    print('Bot (黑方) 获胜!')
                else:
                    print('和棋!')
                break

        print(f'游戏结束，共约 {step_count} 步')
        env.close()

    def test_observation_space(self):
        """测试观测空间"""
        cfg = EasyDict(
            battle_mode='self_play_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        obs = env.reset()

        print('=' * 50)
        print('观测空间测试')
        print('=' * 50)
        print(f"observation shape: {obs['observation'].shape}")
        print(f"action_mask shape: {obs['action_mask'].shape}")
        print(f"action_mask sum (合法动作数): {obs['action_mask'].sum()}")
        print(f"board shape: {obs['board'].shape}")
        print(f"to_play: {obs['to_play']}")
        print(f"current_player_index: {obs['current_player_index']}")

        assert obs['observation'].shape == (57, 10, 9), f"Expected (57, 10, 9), got {obs['observation'].shape}"
        assert obs['action_mask'].shape == (8100,), f"Expected (8100,), got {obs['action_mask'].shape}"
        assert obs['board'].shape == (10, 9), f"Expected (10, 9), got {obs['board'].shape}"

        print('观测空间测试通过!')
        env.close()

    def test_legal_actions(self):
        """测试合法动作"""
        cfg = EasyDict(
            battle_mode='self_play_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        env.reset()

        print('=' * 50)
        print('合法动作测试')
        print('=' * 50)

        legal_actions = env.legal_actions
        print(f'初始局面合法动作数: {len(legal_actions)}')
        print(f'前10个合法动作索引: {legal_actions[:10]}')

        # 中国象棋初始局面，红方有44个合法走法
        assert len(legal_actions) > 0, "合法动作数不应为0"
        print('合法动作测试通过!')
        env.close()

    def test_simulate_action(self):
        """测试 MCTS 模拟动作"""
        cfg = EasyDict(
            battle_mode='self_play_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            render_mode=None,
            replay_path=None,
            uci_engine_path=None,
            engine_depth=5,
            max_episode_steps=200,
        )
        env = ChineseChessEnv(cfg)
        env.reset()

        print('=' * 50)
        print('MCTS 模拟动作测试')
        print('=' * 50)

        # 获取原始状态
        original_step = env.current_step
        original_player = env.current_player

        # 执行模拟
        action = env.random_action()
        simulated_env = env.simulate_action(action)

        # 验证原环境未被修改
        assert env.current_step == original_step, "原环境步数被修改"
        assert env.current_player == original_player, "原环境玩家被修改"

        # 验证模拟环境已更新
        assert simulated_env.current_step == original_step + 1, "模拟环境步数未更新"
        assert simulated_env.current_player != original_player, "模拟环境玩家未切换"

        print(f'原环境步数: {env.current_step}, 模拟环境步数: {simulated_env.current_step}')
        print(f'原环境玩家: {env.current_player}, 模拟环境玩家: {simulated_env.current_player}')
        print('MCTS 模拟动作测试通过!')
        env.close()


def play_human_vs_bot(engine_path: str = None):
    """
    人类 vs Bot 对战
    人类执红先手，Bot 执黑后手

    Args:
        engine_path: UCI 引擎路径，如 pikafish。为 None 则 Bot 使用随机策略。
    """
    # 生成replay目录路径（SVG会保存在此目录下）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bot_name = "uci" if engine_path else "random"
    replay_dir = os.path.join(os.path.dirname(__file__), 'replay_log')
    replay_path = os.path.join(replay_dir, f'human_vs_{bot_name}_{timestamp}')

    cfg = EasyDict(
        battle_mode='eval_mode',
        channel_last=False,
        scale=False,
        agent_vs_human=False,  # False: Bot 是黑方; True: 人类是黑方
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode=None,
        replay_path=replay_path,
        uci_engine_path=engine_path,  # 设置为 pikafish 路径以使用更强的 Bot
        engine_depth=10,
        max_episode_steps=500,
    )
    env = ChineseChessEnv(cfg)
    env.reset()

    bot_name = "UCI引擎" if engine_path else "随机Bot"
    print('=' * 60)
    print(f'人类 vs {bot_name} 对战')
    print('你执红方 (先手)，Bot 执黑方 (后手)')
    print('走法格式: UCI 格式，如 h2e2 (炮二平五)')
    print('棋盘坐标: 列 a-i (左到右), 行 0-9 (下到上)')
    print('=' * 60)
    env.render(mode='human')

    step_count = 0
    while True:
        # 人类输入红方走法
        action = env.human_to_action()
        print(f'\n第 {step_count + 1} 步 - 你 (红方) 走棋')
        # step() 内部会自动调用 bot_action() 让 Bot (黑方) 走棋
        obs, reward, done, info = env.step(action)
        step_count += 2
        print(f'第 {step_count} 步 - Bot (黑方) 走棋')
        env.render(mode='human')

        if done:
            eval_return = info.get('eval_episode_return', reward)
            if eval_return > 0:
                print('\n恭喜！你 (红方) 获胜!')
            elif eval_return < 0:
                print(f'\n{bot_name} (黑方) 获胜!')
            else:
                print('\n和棋!')
            break

    print(f'\n游戏结束，共 {step_count} 步')
    print(f'对局已保存至: {replay_path}')
    env.close()


def play_bot_vs_bot():
    """
    Bot vs Bot 对战 (观战模式)
    """
    # 生成replay目录路径（SVG会保存在此目录下）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    replay_dir = os.path.join(os.path.dirname(__file__), 'replay_log')
    replay_path = os.path.join(replay_dir, f'bot_vs_bot_{timestamp}')

    cfg = EasyDict(
        battle_mode='self_play_mode',
        channel_last=False,
        scale=False,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode=None,
        replay_path=replay_path,
        uci_engine_path=None,
        engine_depth=10,
        max_episode_steps=500,
    )
    env = ChineseChessEnv(cfg)
    env.reset()

    print('=' * 60)
    print('Bot vs Bot 对战 (观战模式)')
    print('=' * 60)
    env.render(mode='human')

    step_count = 0
    while True:
        # 红方 Bot
        action = env.random_action()
        obs, reward, done, info = env.step(action)
        step_count += 1
        print(f'\n第 {step_count} 步 - 红方')
        env.render(mode='human')

        if done:
            if reward > 0:
                print('\n红方获胜!')
            elif reward < 0:
                print('\n黑方获胜!')
            else:
                print('\n和棋!')
            break

        # 黑方 Bot
        action = env.random_action()
        obs, reward, done, info = env.step(action)
        step_count += 1
        print(f'\n第 {step_count} 步 - 黑方')
        env.render(mode='human')

        if done:
            if reward > 0:
                print('\n黑方获胜!')
            elif reward < 0:
                print('\n红方获胜!')
            else:
                print('\n和棋!')
            break

    print(f'\n游戏结束，共 {step_count} 步')
    print(f'对局已保存至: {replay_path}')
    env.close()


def play_uci_vs_random(engine_path: str = "/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish",
                       depth: int = 5):
    """
    UCI引擎 vs 随机Bot对战（红方引擎 vs 黑方随机）
    采用和 play_human_vs_bot 相同的设计模式

    Args:
        engine_path: UCI引擎路径，默认为 pikafish
        depth: 引擎搜索深度（1-20，默认5）
    """
    # 生成replay目录路径（SVG会保存在此目录下）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    replay_dir = os.path.join(os.path.dirname(__file__), 'replay_log')
    replay_path = os.path.join(replay_dir, f'uci_vs_random_depth{depth}_{timestamp}')

    cfg = EasyDict(
        battle_mode='play_with_bot_mode',
        channel_last=False,
        scale=False,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode=None,
        replay_path=replay_path,
        uci_engine_path=engine_path,  # 传给环境处理
        engine_depth=depth,
        max_episode_steps=1000,
    )
    env = ChineseChessEnv(cfg)
    env.reset()

    engine_name = "UCI引擎" if engine_path else "随机Bot"
    print('=' * 60)
    print(f'UCI vs Random 对战')
    print(f'红方: {engine_name} (深度{depth})')
    print('黑方: 随机Bot')
    print('=' * 60)
    env.render(mode='human')

    step_count = 0
    while True:
        # Agent (红方) 走棋，step()内部自动调用bot_action()
        action = env.random_action()
        print(f'\n第 {step_count + 1} 步 - 红方 ({engine_name}深度{depth}) 走棋')
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        step_count += 2

        if done:
            eval_return = info.get('eval_episode_return', reward)
            if eval_return > 0:
                print(f'\n红方 ({engine_name}) 获胜!')
            elif eval_return < 0:
                print('\n黑方 (随机Bot) 获胜!')
            else:
                print('\n和棋!')
            break

    print(f'\n游戏结束，共 {step_count} 步')
    print(f'对局已保存至: {replay_path}')
    env.close()


def play_uci_vs_uci(engine_path: str = "/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish",
                    depth_red: int = 5,
                    depth_black: int = 5):
    """
    两个UCI引擎对战（红 vs 黑）
    红方和黑方分别使用独立的UCI引擎实例，支持不同搜索深度

    Args:
        engine_path: UCI引擎路径，默认为 pikafish
        depth_red: 红方搜索深度（1-20，默认5）
        depth_black: 黑方搜索深度（1-20，默认5）
    """
    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module
    from zoo.board_games.chinesechess.envs.cchess_env import move_to_action

    # 生成replay目录路径（SVG会保存在此目录下）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    replay_dir = os.path.join(os.path.dirname(__file__), 'replay_log')
    replay_path = os.path.join(replay_dir, f'uci_vs_uci_red{depth_red}_black{depth_black}_{timestamp}')

    cfg = EasyDict(
        battle_mode='play_with_bot_mode',
        channel_last=False,
        scale=False,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode=None,
        replay_path=replay_path,
        uci_engine_path=engine_path,  # 黑方Bot使用这个引擎
        engine_depth=depth_black,      # 黑方Bot使用这个深度
        max_episode_steps=1000,
    )
    env = ChineseChessEnv(cfg)
    env.reset()

    # 为红方创建独立的UCI引擎实例
    red_engine = None
    try:
        red_engine = engine_module.SimpleEngine.popen_uci(engine_path)
        print(f"红方UCI引擎加载成功: {engine_path}")
    except Exception as e:
        print(f"红方UCI引擎加载失败: {e}，将使用随机策略")
        red_engine = None

    engine_name = "UCI引擎" if engine_path else "随机Bot"
    print('=' * 60)
    print(f'UCI vs UCI 对战')
    print(f'红方: {engine_name} (深度{depth_red})')
    print(f'黑方: {engine_name} (深度{depth_black})')
    print('=' * 60)
    env.render(mode='human')

    step_count = 0
    try:
        while True:
            # 红方使用UCI引擎计算最佳走法
            if red_engine is not None:
                limit = engine_module.Limit(depth=depth_red)
                result = red_engine.play(env.board, limit)
                action = move_to_action(result.move)
            else:
                action = env.random_action()

            print(f'\n第 {step_count + 1} 步 - 红方 ({engine_name}深度{depth_red}) 走棋')
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            step_count += 2

            if done:
                eval_return = info.get('eval_episode_return', reward)
                if eval_return > 0:
                    print(f'\n红方 ({engine_name}深度{depth_red}) 获胜!')
                elif eval_return < 0:
                    print(f'\n黑方 ({engine_name}深度{depth_black}) 获胜!')
                else:
                    print('\n和棋!')
                break
    finally:
        # 确保红方引擎被正确关闭
        if red_engine is not None:
            try:
                red_engine.quit()
                print("红方UCI引擎已关闭")
            except:
                pass

    print(f'\n游戏结束，共 {step_count} 步')
    print(f'对局已保存至: {replay_path}')
    env.close()


if __name__ == '__main__':
    import sys

    print('\n' + '=' * 60)
    print('中国象棋环境测试')
    print('=' * 60)
    print('1. 运行自动化测试')
    print('2. 人类 vs 随机Bot 对战')
    print('3. 人类 vs UCI引擎 对战 (需要输入引擎路径)')
    print('4. Bot vs Bot 观战')
    print('5. UCI vs Random 对战 (UCI引擎 vs 随机Bot)')
    print('6. UCI vs UCI 对战 (pikafish vs pikafish)')
    print('=' * 60)

    choice = input('请选择 (1/2/3/4/5/6): ').strip()

    if choice == '1':
        test = TestChineseChessEnv()
        print('\n开始自动化测试...\n')
        test.test_observation_space()
        print()
        test.test_legal_actions()
        print()
        test.test_simulate_action()
        print()
        test.test_self_play_mode()
        print()
        test.test_play_with_bot_mode()
        print()
        test.test_eval_mode()
        print('\n所有测试完成!')

    elif choice == '2':
        play_human_vs_bot(engine_path=None)

    elif choice == '3':
        engine_path = input('请输入 UCI 引擎路径 (如 pikafish 或完整路径): ').strip()
        if not engine_path:
            print('引擎路径不能为空，退出')
            sys.exit(1)
        play_human_vs_bot(engine_path=engine_path)

    elif choice == '4':
        play_bot_vs_bot()

    elif choice == '5':
        # UCI vs Random 对战
        engine_path = input('请输入 UCI 引擎路径 (默认: /mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish): ').strip()
        if not engine_path:
            engine_path = "/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish"

        depth_str = input('请输入引擎搜索深度 (1-20, 默认: 5): ').strip()
        depth = int(depth_str) if depth_str.isdigit() else 5
        depth = max(1, min(20, depth))  # 限制在1-20之间

        play_uci_vs_random(engine_path=engine_path, depth=depth)

    elif choice == '6':
        # UCI vs UCI 对战
        engine_path = input('请输入 UCI 引擎路径 (默认: /mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish): ').strip()
        if not engine_path:
            engine_path = "/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish"

        depth_red_str = input('请输入红方搜索深度 (1-20, 默认: 5): ').strip()
        depth_red = int(depth_red_str) if depth_red_str.isdigit() else 5
        depth_red = max(1, min(20, depth_red))  # 限制在1-20之间

        depth_black_str = input('请输入黑方搜索深度 (1-20, 默认: 5): ').strip()
        depth_black = int(depth_black_str) if depth_black_str.isdigit() else 5
        depth_black = max(1, min(20, depth_black))  # 限制在1-20之间

        play_uci_vs_uci(engine_path=engine_path, depth_red=depth_red, depth_black=depth_black)

    else:
        print('无效选择，退出')
        sys.exit(1)
    
    # count=0
    # try:
    #     play_uci_vs_random(engine_path="/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish")
    #     # for i in range(1):

    #     #     count+=1
    # except:
    #     pass
    # finally:
    #     print("count")
    #     print(count)


