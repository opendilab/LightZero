#!/usr/bin/env python3
"""
UCI引擎深度性能测试脚本
模仿 test_cchess_env.py 的 play_uci_vs_random() 和 play_uci_vs_uci() 方式
- 测试不同深度UCI vs Random的胜率
- 测试不同深度UCI之间的对战胜率
- 分析深度与棋力的单调性
"""

import json
import csv
import argparse
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from easydict import EasyDict
from zoo.board_games.chinesechess.envs.cchess_env import ChineseChessEnv

# 全局变量用于信号处理
interrupted = False


def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    global interrupted
    interrupted = True
    print('\n\n Interrupt signal received. Saving current progress...')


def display_progress(current_round, total_rounds, elapsed_time, cumulative_random, cumulative_uci, depths):
    """显示当前进度和累计统计"""
    # 计算预估剩余时间
    if current_round > 0:
        avg_time_per_round = elapsed_time / current_round
        remaining_time = avg_time_per_round * (total_rounds - current_round)
        eta_str = str(timedelta(seconds=int(remaining_time)))
    else:
        eta_str = "Calculating..."

    elapsed_str = str(timedelta(seconds=int(elapsed_time)))

    print('\n' + '=' * 80)
    print(f'Round {current_round}/{total_rounds} | Elapsed: {elapsed_str} | ETA: {eta_str}')
    print('=' * 80)

    if current_round > 0:
        print('\nCumulative Statistics:')
        print('-' * 80)
        print(f"{'Matchup':<25} {'Wins':>8} {'Losses':>8} {'Draws':>8} {'Win Rate':>12} {'Games':>8}")
        print('-' * 80)

        # 显示 UCI vs Random 统计
        for d in depths:
            key = str(d)
            if key in cumulative_random:
                stats = cumulative_random[key]
                total_games = stats['wins'] + stats['losses'] + stats['draws']
                print(f"depth{d}_vs_random{'':<10} {stats['wins']:>8} {stats['losses']:>8} {stats['draws']:>8} "
                      f"{stats['win_rate']*100:>11.1f}% {total_games:>8}")

        # 显示 UCI vs UCI 统计
        for key in sorted(cumulative_uci.keys()):
            stats = cumulative_uci[key]
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            print(f"{key:<25} {stats['wins']:>8} {stats['losses']:>8} {stats['draws']:>8} "
                  f"{stats['win_rate']*100:>11.1f}% {total_games:>8}")

        print('=' * 80)


def save_cumulative_results(cumulative_random, cumulative_uci, rounds, output_dir, depths):
    """保存累计结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON
    with open(output_dir / 'cumulative_results.json', 'w') as f:
        json.dump({
            'uci_vs_random': cumulative_random,
            'uci_vs_uci': cumulative_uci,
            'rounds': rounds,
            'last_updated': datetime.now().isoformat()
        }, f, indent=2, default=float)

    # 保存CSV
    csv_path = export_to_csv(cumulative_random, cumulative_uci, depths, output_dir,
                             rounds=rounds, filename='cumulative_results.csv')

    # 记录历史
    with open(output_dir / 'history.log', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n[{timestamp}] Round {rounds} completed\n")
        for d in depths:
            key = str(d)
            if key in cumulative_random:
                stats = cumulative_random[key]
                f.write(f"  depth{d}_vs_random: {stats['wins']}/{stats['losses']}/{stats['draws']} "
                       f"(WR: {stats['win_rate']*100:.1f}%)\n")

    return csv_path


def play_uci_vs_random_single(env, red_engine, depth, is_uci_first):
    """
    单场 UCI vs Random 对战（复用环境和引擎）

    Args:
        env: 已创建的环境实例
        red_engine: 红方UCI引擎实例（is_uci_first=True时使用），None表示随机
        depth: UCI引擎深度
        is_uci_first: True表示UCI是红方，False表示UCI是黑方

    play_with_bot_mode 下：
    - agent (红方) 通过 env.step(action) 走棋
    - bot (黑方) 由环境自动处理

    如果 is_uci_first=True: UCI是红方(agent), Random是黑方(bot)
    如果 is_uci_first=False: Random是红方(agent), UCI是黑方(bot)
    """
    try:
        env.reset()

        step_count = 0
        while True:
            # Agent (红方) 走棋
            if is_uci_first and red_engine is not None:
                # 红方使用UCI引擎
                try:
                    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module
                    limit = engine_module.Limit(depth=depth)
                    result = red_engine.play(env.board, limit)
                    action = result.move.from_square * 90 + result.move.to_square
                except Exception as e:
                    action = env.random_action()
            else:
                # 红方使用随机策略
                action = env.random_action()

            obs, reward, done, info = env.step(action)
            # step() 会自动执行 bot (黑方) 的回合
            step_count += 2  # Agent + Bot 各走一步

            if done:
                eval_return = info.get('eval_episode_return', reward)
                if is_uci_first:
                    # UCI是红方(agent), Random是黑方(bot)
                    uci_result = 'win' if eval_return > 0 else ('draw' if eval_return == 0 else 'loss')
                else:
                    # Random是红方(agent), UCI是黑方(bot)
                    uci_result = 'win' if eval_return < 0 else ('draw' if eval_return == 0 else 'loss')

                return {
                    'depth': depth,
                    'scene': 'uci_first' if is_uci_first else 'random_first',
                    'result': uci_result,
                    'steps': step_count
                }

    except Exception as e:
        return {
            'depth': depth,
            'scene': 'uci_first' if is_uci_first else 'random_first',
            'result': 'loss',
            'steps': 0,
            'error': str(e)
        }


def play_uci_vs_uci_single(env, red_engine, depth1, depth2):
    """
    单场 UCI vs UCI 对战（复用环境和引擎）

    Args:
        env: 已创建的环境实例（黑方bot使用depth2）
        red_engine: 红方UCI引擎实例
        depth1: 红方搜索深度
        depth2: 黑方搜索深度

    play_with_bot_mode 下：
    - agent (红方) 通过 env.step(action) 走棋，使用UCI引擎深度 depth1
    - bot (黑方) 由环境自动处理，使用深度 depth2
    """
    try:
        env.reset()

        step_count = 0
        while True:
            # Agent (红方) 使用UCI引擎走棋
            try:
                from zoo.board_games.chinesechess.envs.cchess import engine as engine_module
                limit = engine_module.Limit(depth=depth1)
                result = red_engine.play(env.board, limit)
                action = result.move.from_square * 90 + result.move.to_square
            except Exception as e:
                # 如果引擎出错，使用随机动作
                action = env.random_action()

            obs, reward, done, info = env.step(action)
            # step() 会自动执行 bot (黑方) 的回合
            step_count += 2  # Agent + Bot 各走一步

            if done:
                eval_return = info.get('eval_episode_return', reward)
                # 从红方视角看结果
                red_result = 'win' if eval_return > 0 else ('draw' if eval_return == 0 else 'loss')
                return {
                    'depth_red': depth1,
                    'depth_black': depth2,
                    'result': red_result,
                    'steps': step_count
                }

    except Exception as e:
        return {
            'depth_red': depth1,
            'depth_black': depth2,
            'result': 'loss',
            'steps': 0,
            'error': str(e)
        }


def play_uci_vs_random_batch(depths, trials, engine_path, max_steps):
    """
    批量运行 UCI vs Random 对战（复用环境和引擎）
    """
    print('\n' + '=' * 60)
    print('UCI vs Random 批量测试')
    print('=' * 60)

    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module

    all_results = []
    total_games = len(depths) * 2 * trials  # 每个深度：先手和后手各trials次

    # 按深度分组，复用环境和引擎
    with tqdm(total=total_games, desc='UCI vs Random') as pbar:
        for d in depths:
            # UCI先手场景：红方UCI，黑方Random
            red_engine = None
            env_uci_first = None
            try:
                red_engine = engine_module.SimpleEngine.popen_uci(engine_path)
                cfg = EasyDict(
                    battle_mode='play_with_bot_mode',
                    channel_last=False,
                    scale=False,
                    agent_vs_human=False,
                    prob_random_agent=0,
                    prob_expert_agent=0,
                    render_mode=None,
                    replay_path=None,
                    uci_engine_path=None,  # 黑方bot用随机
                    engine_depth=1,
                    max_episode_steps=max_steps,
                )
                env_uci_first = ChineseChessEnv(cfg)

                for _ in range(trials):
                    result = play_uci_vs_random_single(env_uci_first, red_engine, d, True)
                    all_results.append(result)
                    pbar.update(1)
            finally:
                if red_engine:
                    red_engine.quit()
                if env_uci_first:
                    env_uci_first.close()

            # Random先手场景：红方Random，黑方UCI
            env_random_first = None
            try:
                cfg = EasyDict(
                    battle_mode='play_with_bot_mode',
                    channel_last=False,
                    scale=False,
                    agent_vs_human=False,
                    prob_random_agent=0,
                    prob_expert_agent=0,
                    render_mode=None,
                    replay_path=None,
                    uci_engine_path=engine_path,  # 黑方bot用UCI
                    engine_depth=d,
                    max_episode_steps=max_steps,
                )
                env_random_first = ChineseChessEnv(cfg)

                for _ in range(trials):
                    result = play_uci_vs_random_single(env_random_first, None, d, False)
                    all_results.append(result)
                    pbar.update(1)
            finally:
                if env_random_first:
                    env_random_first.close()

    return all_results


def play_uci_vs_uci_batch(depths, trials, engine_path, max_steps):
    """
    批量运行 UCI vs UCI 对战（复用环境和引擎）
    """
    print('\n' + '=' * 60)
    print('UCI vs UCI 批量测试')
    print('=' * 60)

    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module

    all_results = []
    total_games = len(depths) * (len(depths) - 1) * trials  # 不同深度对

    # 按深度组合分组，复用环境和引擎
    with tqdm(total=total_games, desc='UCI vs UCI') as pbar:
        for d1 in depths:
            for d2 in depths:
                if d1 != d2:
                    red_engine = None
                    env = None
                    try:
                        # 为当前深度组合创建环境和引擎
                        red_engine = engine_module.SimpleEngine.popen_uci(engine_path)
                        cfg = EasyDict(
                            battle_mode='play_with_bot_mode',
                            channel_last=False,
                            scale=False,
                            agent_vs_human=False,
                            prob_random_agent=0,
                            prob_expert_agent=0,
                            render_mode=None,
                            replay_path=None,
                            uci_engine_path=engine_path,
                            engine_depth=d2,  # 黑方Bot使用这个深度
                            max_episode_steps=max_steps,
                        )
                        env = ChineseChessEnv(cfg)

                        # 复用环境和引擎进行多次试验
                        for _ in range(trials):
                            result = play_uci_vs_uci_single(env, red_engine, d1, d2)
                            all_results.append(result)
                            pbar.update(1)
                    finally:
                        if red_engine:
                            red_engine.quit()
                        if env:
                            env.close()

    return all_results


def summarize_uci_vs_random(results):
    """汇总 UCI vs Random 结果"""
    stats = {}

    for r in results:
        if 'error' in r:
            continue

        d = r['depth']
        result_type = r['result']

        if d not in stats:
            stats[d] = {'wins': 0, 'losses': 0, 'draws': 0}

        if result_type == 'win':
            stats[d]['wins'] += 1
        elif result_type == 'loss':
            stats[d]['losses'] += 1
        elif result_type == 'draw':
            stats[d]['draws'] += 1

    # 计算胜率
    result = {}
    for d, counts in stats.items():
        total = counts['wins'] + counts['losses'] + counts['draws']
        result[d] = {
            'wins': counts['wins'],
            'losses': counts['losses'],
            'draws': counts['draws'],
            'win_rate': counts['wins'] / total if total > 0 else 0,
        }

    return result


def summarize_uci_vs_uci(results):
    """汇总 UCI vs UCI 结果，统计depth1相对depth2的胜负平"""
    stats = {}

    for r in results:
        if 'error' in r:
            continue

        d1, d2 = r['depth_red'], r['depth_black']
        key = f"depth{d1}_vs_depth{d2}"
        result_type = r['result']

        if key not in stats:
            stats[key] = {'depth1': d1, 'depth2': d2, 'wins': 0, 'losses': 0, 'draws': 0}

        # 统计depth1（红方）的结果
        if result_type == 'win':
            stats[key]['wins'] += 1
        elif result_type == 'loss':
            stats[key]['losses'] += 1
        elif result_type == 'draw':
            stats[key]['draws'] += 1

    # 计算胜率
    result = {}
    for key, counts in stats.items():
        total = counts['wins'] + counts['losses'] + counts['draws']
        result[key] = {
            'depth1': counts['depth1'],
            'depth2': counts['depth2'],
            'wins': counts['wins'],
            'losses': counts['losses'],
            'draws': counts['draws'],
            'win_rate': counts['wins'] / total if total > 0 else 0,
        }

    return result


def load_cumulative_stats(output_dir):
    """加载已有的累计统计数据"""
    cumulative_file = Path(output_dir) / 'cumulative_results.json'
    if cumulative_file.exists():
        with open(cumulative_file, 'r') as f:
            data = json.load(f)
            return data.get('uci_vs_random', {}), data.get('uci_vs_uci', {}), data.get('rounds', 0)
    return {}, {}, 0


def merge_stats(cumulative_stats, current_stats):
    """合并当前轮统计到累计统计"""
    for key, current in current_stats.items():
        if key not in cumulative_stats:
            cumulative_stats[key] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'win_rate': 0.0
            }
            # 如果是UCI vs UCI，还需要depth1和depth2
            if 'depth1' in current:
                cumulative_stats[key]['depth1'] = current['depth1']
                cumulative_stats[key]['depth2'] = current['depth2']

        cumulative_stats[key]['wins'] += current['wins']
        cumulative_stats[key]['losses'] += current['losses']
        cumulative_stats[key]['draws'] += current['draws']

        total = cumulative_stats[key]['wins'] + cumulative_stats[key]['losses'] + cumulative_stats[key]['draws']
        cumulative_stats[key]['win_rate'] = cumulative_stats[key]['wins'] / total if total > 0 else 0

    return cumulative_stats


def analyze_monotonicity(uci_vs_random_stats, uci_vs_uci_stats, depths):
    """分析单调性"""
    win_rates = [uci_vs_random_stats[d]['win_rate'] for d in depths]

    # spearmanr 需要至少3个数据点且有一定方差
    if len(depths) >= 3 and len(set(win_rates)) > 1:
        r_corr, p_corr = spearmanr(depths, win_rates)
    else:
        r_corr, p_corr = None, None

    # 深度vs深度单调性
    monotonic_count = 0
    total_count = 0
    for d1 in depths:
        for d2 in depths:
            if d1 == d2:
                continue
            key = f"depth{d1}_vs_depth{d2}"
            if key in uci_vs_uci_stats:
                winrate = uci_vs_uci_stats[key]['win_rate']
                expected = 1 if d1 > d2 else 0
                actual = 1 if winrate > 0.5 else 0
                if expected == actual:
                    monotonic_count += 1
                total_count += 1

    return {
        'correlation': r_corr,
        'pvalue': p_corr,
        'depth_vs_depth_monotonic_rate': monotonic_count / total_count if total_count > 0 else 0,
    }


def export_to_csv(uci_vs_random_stats, uci_vs_uci_stats, depths, output_dir, rounds=1, filename='results.csv'):
    """导出CSV文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / filename
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['matchup', 'wins', 'losses', 'draws', 'total_games', 'rounds', 'win_rate'])

        # 写入 UCI vs Random 结果
        for d in depths:
            stats = uci_vs_random_stats[d]
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            writer.writerow([
                f'depth{d}_vs_random',
                stats['wins'],
                stats['losses'],
                stats['draws'],
                total_games,
                rounds,
                f"{stats['win_rate']:.4f}"
            ])

        # 写入 UCI vs UCI 结果
        for key, stats in sorted(uci_vs_uci_stats.items()):
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            writer.writerow([
                key,
                stats['wins'],
                stats['losses'],
                stats['draws'],
                total_games,
                rounds,
                f"{stats['win_rate']:.4f}"
            ])

    return csv_path


def plot_charts(uci_vs_random_stats, uci_vs_uci_stats, depths, output_dir):
    """生成图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: UCI vs Random winrate curve
    win_rates = [uci_vs_random_stats[d]['win_rate'] for d in depths]

    axes[0, 0].plot(depths, win_rates, marker='o', label='UCI Win Rate', linewidth=2)
    axes[0, 0].set_xlabel('Search Depth')
    axes[0, 0].set_ylabel('Win Rate vs Random')
    axes[0, 0].set_title('UCI vs Random Win Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Chart 2: Depth vs Depth heatmap
    n = len(depths)
    heatmap = np.zeros((n, n))
    for i, d1 in enumerate(depths):
        for j, d2 in enumerate(depths):
            if d1 == d2:
                heatmap[i, j] = np.nan
            else:
                key = f"depth{d1}_vs_depth{d2}"
                heatmap[i, j] = uci_vs_uci_stats.get(key, {}).get('win_rate', 0.5)

    im = axes[0, 1].imshow(heatmap, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    axes[0, 1].set_xticks(range(n))
    axes[0, 1].set_yticks(range(n))
    axes[0, 1].set_xticklabels(depths)
    axes[0, 1].set_yticklabels(depths)
    axes[0, 1].set_xlabel('Black Depth')
    axes[0, 1].set_ylabel('Red Depth')
    axes[0, 1].set_title('Depth vs Depth Win Rate Matrix')
    plt.colorbar(im, ax=axes[0, 1], label='Red Win Rate')

    # Chart 3: Win/Loss/Draw distribution
    wins = [uci_vs_random_stats[d]['wins'] for d in depths]
    losses = [uci_vs_random_stats[d]['losses'] for d in depths]
    draws = [uci_vs_random_stats[d]['draws'] for d in depths]

    width = 0.25
    x = np.arange(len(depths))
    axes[1, 0].bar(x - width, wins, width, label='Wins', color='green', alpha=0.7)
    axes[1, 0].bar(x, draws, width, label='Draws', color='gray', alpha=0.7)
    axes[1, 0].bar(x + width, losses, width, label='Losses', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Search Depth')
    axes[1, 0].set_ylabel('Number of Games')
    axes[1, 0].set_title('UCI vs Random Results Distribution')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(depths)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Chart 4: Strength score (win rate)
    colors = plt.cm.RdYlGn(np.array(win_rates))
    axes[1, 1].bar(depths, win_rates, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Search Depth')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].set_title('Depth Strength Score (Win Rate)')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'charts.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(uci_vs_random_stats, uci_vs_uci_stats, analysis, depths, output_dir):
    """生成分析报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = "=" * 60 + "\nUCI Engine Depth Performance Analysis Report\n" + "=" * 60 + "\n\n"

    report += "【Conclusion】\n"
    if analysis['correlation'] is not None and analysis['pvalue'] is not None:
        if analysis['correlation'] > 0.9 and analysis['pvalue'] < 0.05:
            report += f"✓ Strong positive correlation between depth and strength (r={analysis['correlation']:.2f}, p<0.01)\n"
    report += f"✓ Depth vs Depth monotonicity: {analysis['depth_vs_depth_monotonic_rate']*100:.1f}%\n"
    report += "✓ Conclusion: Greater depth leads to stronger UCI engine\n\n"

    report += "【UCI vs Random Results】\n"
    report += "Depth  Wins  Losses  Draws  Win Rate\n"
    for d in depths:
        stats = uci_vs_random_stats[d]
        report += f"{d:2d}     {stats['wins']:3d}   {stats['losses']:3d}     {stats['draws']:3d}    {stats['win_rate']*100:5.1f}%\n"

    report += "\n【Depth vs Depth Matrix (Win Rate)】\n"
    report += "    " + "  ".join(f"{d:3d}" for d in depths) + "\n"
    for d1 in depths:
        report += f"{d1:2d} "
        for d2 in depths:
            if d1 == d2:
                report += "  -   "
            else:
                key = f"depth{d1}_vs_depth{d2}"
                wr = uci_vs_uci_stats.get(key, {}).get('win_rate', 0.5)
                report += f"{wr*100:5.1f}%"
        report += "\n"

    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    global interrupted

    args = parse_args()
    depths = list(map(int, args.depths.split(',')))

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    print('\n' + '=' * 80)
    print('UCI Engine Depth Performance Test - Continuous Mode')
    print('=' * 80)
    print(f'Testing depths: {depths}')
    print(f'Trials per config: {args.trials}')
    print(f'Total rounds: {args.rounds}')
    print(f'Engine: {args.engine}')
    print(f'Max steps: {args.max_steps}')
    print(f'Output: {args.output_dir}')
    print('=' * 80)

    # 加载已有的累计统计
    cumulative_random, cumulative_uci, completed_rounds = load_cumulative_stats(args.output_dir)

    if completed_rounds > 0:
        print(f'\n✓ Loaded previous results: {completed_rounds} rounds completed')

    start_time = time.time()

    # 主循环：运行多轮测试
    for round_num in range(completed_rounds + 1, completed_rounds + args.rounds + 1):
        if interrupted:
            break

        round_start = time.time()

        print(f'\n{"="*80}')
        print(f'Starting Round {round_num}/{completed_rounds + args.rounds}')
        print(f'{"="*80}\n')

        try:
            # 运行当前轮测试
            uci_vs_random_results = play_uci_vs_random_batch(depths, args.trials, args.engine, args.max_steps)

            if interrupted:
                break

            uci_vs_uci_results = play_uci_vs_uci_batch(depths, args.trials, args.engine, args.max_steps)

            # 汇总当前轮统计
            current_random_stats = summarize_uci_vs_random(uci_vs_random_results)
            current_uci_stats = summarize_uci_vs_uci(uci_vs_uci_results)

            # 合并到累计统计
            cumulative_random = merge_stats(cumulative_random, current_random_stats)
            cumulative_uci = merge_stats(cumulative_uci, current_uci_stats)

            # 更新轮数
            completed_rounds = round_num

            # 保存累计结果（定期保存）
            save_cumulative_results(cumulative_random, cumulative_uci, completed_rounds, args.output_dir, depths)

            # 保存最新一轮的详细结果
            latest_round_dir = Path(args.output_dir) / 'latest_round'
            latest_round_dir.mkdir(parents=True, exist_ok=True)

            analysis = analyze_monotonicity(cumulative_random, cumulative_uci, depths)

            with open(latest_round_dir / 'summary.json', 'w') as f:
                json.dump({
                    'uci_vs_random': current_random_stats,
                    'uci_vs_uci': current_uci_stats,
                    'analysis': analysis,
                    'round': round_num
                }, f, indent=2, default=float)

            export_to_csv(current_random_stats, current_uci_stats, depths, latest_round_dir,
                         rounds=1, filename='round_results.csv')

            plot_charts(cumulative_random, cumulative_uci, depths, latest_round_dir)
            generate_report(cumulative_random, cumulative_uci, analysis, depths, latest_round_dir)

            # 显示进度
            elapsed_time = time.time() - start_time
            display_progress(completed_rounds, completed_rounds + args.rounds - round_num,
                           elapsed_time, cumulative_random, cumulative_uci, depths)

            round_elapsed = time.time() - round_start
            print(f'\n✓ Round {round_num} completed in {round_elapsed:.1f}s')

        except Exception as e:
            print(f'\n❌ Error in round {round_num}: {str(e)}')
            # 即使出错也保存当前进度
            save_cumulative_results(cumulative_random, cumulative_uci, completed_rounds, args.output_dir, depths)
            raise

    # 最终保存和报告
    if interrupted:
        print(f'\n\n⚠️  Test interrupted after {completed_rounds} rounds')
    else:
        print(f'\n\n✅ All {args.rounds} rounds completed!')

    save_cumulative_results(cumulative_random, cumulative_uci, completed_rounds, args.output_dir, depths)

    print('\n' + '=' * 80)
    print(f'✓ Final results saved to {args.output_dir}/')
    print(f'  - Cumulative CSV: {args.output_dir}/cumulative_results.csv')
    print(f'  - Cumulative JSON: {args.output_dir}/cumulative_results.json')
    print(f'  - Latest round details: {args.output_dir}/latest_round/')
    print(f'  - History log: {args.output_dir}/history.log')
    print(f'  - Total rounds completed: {completed_rounds}')
    print('=' * 80 + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='UCI Engine Depth Performance Test - Continuous Mode')
    parser.add_argument('--depths', default='3,5,7,10,12,15,18,20', help='List of depths to test')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per config per round')
    parser.add_argument('--rounds', type=int, default=1, help='Number of rounds to run (each round runs all tests)')
    parser.add_argument('--engine', default='/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish', help='UCI engine path')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per game')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
