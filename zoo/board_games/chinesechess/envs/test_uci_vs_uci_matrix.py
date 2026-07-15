#!/usr/bin/env python3
"""
UCI vs UCI 多深度对战矩阵测试脚本
- 基于 test_uci_vs_uci_detailed.py 的直接棋盘操作方式
- 测试多个深度之间的对战胜率
- 生成胜率矩阵和可视化图表
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def play_single_game(engine_red, engine_black, depth_red, depth_black, engine_module, max_steps=500):
    """
    单局UCI vs UCI对战（直接棋盘操作，不使用环境包装器）

    Args:
        engine_red: 红方UCI引擎实例
        engine_black: 黑方UCI引擎实例
        depth_red: 红方搜索深度
        depth_black: 黑方搜索深度
        engine_module: cchess.engine模块
        max_steps: 最大步数

    Returns:
        dict: 对局结果
    """
    from zoo.board_games.chinesechess.envs import cchess

    board = cchess.Board()
    step_count = 0

    while not board.is_game_over() and step_count < max_steps:
        if board.turn == cchess.RED:
            # 红方走棋
            limit = engine_module.Limit(depth=depth_red)
            result = engine_red.play(board, limit)
            move = result.move
        else:
            # 黑方走棋
            limit = engine_module.Limit(depth=depth_black)
            result = engine_black.play(board, limit)
            move = result.move

        board.push(move)
        step_count += 1

    # 判断结果
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        winner = "draw"
    elif outcome.winner == cchess.RED:
        winner = "red"
    else:
        winner = "black"

    return {
        'depth_red': depth_red,
        'depth_black': depth_black,
        'winner': winner,
        'total_steps': step_count,
        'termination': str(outcome.termination) if outcome else 'max_steps'
    }


def run_depth_matrix_test(
    depths=[3, 5, 8, 10, 12],
    games_per_pair=10,
    engine_path="/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish",
    output_dir=None,
    max_steps=500
):
    """
    运行多深度对战矩阵测试

    Args:
        depths: 要测试的深度列表
        games_per_pair: 每对深度的对战局数（会交换先后手，所以实际是 games_per_pair * 2）
        engine_path: UCI引擎路径
        output_dir: 输出目录
        max_steps: 每局最大步数
    """
    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module

    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'test_results' / f'uci_matrix_{timestamp}'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算总对战数
    num_pairs = len(depths) * (len(depths) - 1)  # 不同深度的组合数
    total_games = num_pairs * games_per_pair

    print('=' * 70)
    print('UCI vs UCI 多深度对战矩阵测试')
    print('=' * 70)
    print(f'测试深度: {depths}')
    print(f'每对深度对战局数: {games_per_pair} (交换先后手)')
    print(f'深度组合数: {num_pairs}')
    print(f'总对战局数: {total_games}')
    print(f'引擎路径: {engine_path}')
    print(f'输出目录: {output_dir}')
    print('=' * 70)

    # 初始化统计矩阵
    # stats[d1][d2] = {'wins': x, 'losses': y, 'draws': z}
    # 表示 d1 作为红方 vs d2 作为黑方时，d1 的胜负平
    stats = {d1: {d2: {'wins': 0, 'losses': 0, 'draws': 0} for d2 in depths} for d1 in depths}

    # 详细结果列表
    all_results = []

    # 创建两个引擎实例
    engine1 = None
    engine2 = None

    start_time = time.time()

    try:
        print('\n正在加载UCI引擎...')
        engine1 = engine_module.SimpleEngine.popen_uci(engine_path)
        engine2 = engine_module.SimpleEngine.popen_uci(engine_path)
        print('UCI引擎加载成功!\n')

        game_id = 0

        # 遍历所有深度组合
        with tqdm(total=total_games, desc='总进度') as pbar:
            for d1 in depths:
                for d2 in depths:
                    if d1 == d2:
                        continue  # 跳过相同深度

                    # d1 作为红方，d2 作为黑方
                    for _ in range(games_per_pair):
                        game_id += 1
                        result = play_single_game(
                            engine_red=engine1,
                            engine_black=engine2,
                            depth_red=d1,
                            depth_black=d2,
                            engine_module=engine_module,
                            max_steps=max_steps
                        )
                        result['game_id'] = game_id
                        all_results.append(result)

                        # 更新统计（d1的视角）
                        if result['winner'] == 'red':
                            stats[d1][d2]['wins'] += 1
                        elif result['winner'] == 'black':
                            stats[d1][d2]['losses'] += 1
                        else:
                            stats[d1][d2]['draws'] += 1

                        pbar.update(1)

                        # 显示当前对战进度
                        pbar.set_postfix({
                            'matchup': f'd{d1}vs d{d2}',
                            'game': game_id
                        })

    finally:
        # 关闭引擎
        if engine1:
            try:
                engine1.quit()
            except:
                pass
        if engine2:
            try:
                engine2.quit()
            except:
                pass

    elapsed_time = time.time() - start_time

    # 计算综合胜率（考虑先后手）
    combined_stats = calculate_combined_stats(stats, depths)

    # 保存结果
    save_matrix_csv(stats, depths, output_dir / 'winrate_matrix.csv')
    save_combined_csv(combined_stats, depths, output_dir / 'combined_stats.csv')
    save_detailed_csv(all_results, output_dir / 'game_details.csv')
    save_json_results(stats, combined_stats, depths, output_dir / 'results.json')

    # 生成图表
    plot_winrate_matrix(stats, depths, output_dir / 'winrate_matrix.png')
    plot_combined_winrate(combined_stats, depths, output_dir / 'combined_winrate.png')

    # 打印统计结果
    print_statistics(stats, combined_stats, depths, elapsed_time)

    print(f'\n结果已保存至: {output_dir}')
    print(f'  - 胜率矩阵: winrate_matrix.csv')
    print(f'  - 综合统计: combined_stats.csv')
    print(f'  - 对局详情: game_details.csv')
    print(f'  - JSON结果: results.json')
    print(f'  - 胜率热力图: winrate_matrix.png')
    print(f'  - 综合胜率图: combined_winrate.png')

    return stats, combined_stats, all_results


def calculate_combined_stats(stats, depths):
    """
    计算综合统计（合并先后手）

    对于 d1 vs d2，合并：
    - d1 红方 vs d2 黑方 的结果
    - d2 红方 vs d1 黑方 的结果（从d1视角看是输赢互换）
    """
    combined = {}

    for i, d1 in enumerate(depths):
        for j, d2 in enumerate(depths):
            if d1 >= d2:
                continue  # 只计算 d1 < d2 的组合，避免重复

            key = f'd{d1}_vs_d{d2}'

            # d1 红方时的结果
            d1_as_red = stats[d1][d2]
            # d1 黑方时的结果（从d2红方的统计反推）
            d2_as_red = stats[d2][d1]

            # d1 的综合胜负（先手+后手）
            d1_total_wins = d1_as_red['wins'] + d2_as_red['losses']  # d1赢 = d1红方赢 + d2红方输
            d1_total_losses = d1_as_red['losses'] + d2_as_red['wins']  # d1输 = d1红方输 + d2红方赢
            d1_total_draws = d1_as_red['draws'] + d2_as_red['draws']

            total_games = d1_total_wins + d1_total_losses + d1_total_draws

            combined[key] = {
                'depth1': d1,
                'depth2': d2,
                'd1_wins': d1_total_wins,
                'd1_losses': d1_total_losses,
                'draws': d1_total_draws,
                'total_games': total_games,
                'd1_winrate': d1_total_wins / total_games if total_games > 0 else 0,
                'd2_winrate': d1_total_losses / total_games if total_games > 0 else 0,
            }

    return combined


def save_matrix_csv(stats, depths, csv_path):
    """保存胜率矩阵到CSV"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 表头：红方深度 vs 黑方深度
        header = ['红方深度\\黑方深度'] + [f'd{d}' for d in depths]
        writer.writerow(header)

        for d1 in depths:
            row = [f'd{d1}']
            for d2 in depths:
                if d1 == d2:
                    row.append('-')
                else:
                    s = stats[d1][d2]
                    total = s['wins'] + s['losses'] + s['draws']
                    winrate = s['wins'] / total if total > 0 else 0
                    row.append(f"{winrate*100:.1f}%")
            writer.writerow(row)


def save_combined_csv(combined_stats, depths, csv_path):
    """保存综合统计到CSV"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['对战组合', '深度1胜', '深度2胜', '和棋', '总局数', '深度1胜率', '深度2胜率', '结论'])

        for key, s in sorted(combined_stats.items()):
            conclusion = ''
            if s['d1_winrate'] > 0.6:
                conclusion = f"深度{s['depth1']}明显较弱"
            elif s['d2_winrate'] > 0.6:
                conclusion = f"深度{s['depth2']}明显较弱"
            elif s['d1_winrate'] > 0.5:
                conclusion = f"深度{s['depth2']}略强"
            else:
                conclusion = f"深度{s['depth1']}略强"

            writer.writerow([
                f"d{s['depth1']} vs d{s['depth2']}",
                s['d1_wins'],
                s['d1_losses'],
                s['draws'],
                s['total_games'],
                f"{s['d1_winrate']*100:.1f}%",
                f"{s['d2_winrate']*100:.1f}%",
                conclusion
            ])


def save_detailed_csv(results, csv_path):
    """保存对局详情到CSV"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['game_id', 'depth_red', 'depth_black', 'winner', 'total_steps', 'termination'])

        for r in results:
            writer.writerow([
                r['game_id'],
                r['depth_red'],
                r['depth_black'],
                r['winner'],
                r['total_steps'],
                r['termination']
            ])


def save_json_results(stats, combined_stats, depths, json_path):
    """保存JSON格式结果"""
    # 将stats转换为可序列化格式
    stats_serializable = {
        str(d1): {str(d2): stats[d1][d2] for d2 in depths}
        for d1 in depths
    }

    data = {
        'depths': depths,
        'matrix_stats': stats_serializable,
        'combined_stats': combined_stats,
        'timestamp': datetime.now().isoformat()
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_winrate_matrix(stats, depths, save_path):
    """绘制胜率矩阵热力图"""
    n = len(depths)
    matrix = np.zeros((n, n))

    for i, d1 in enumerate(depths):
        for j, d2 in enumerate(depths):
            if d1 == d2:
                matrix[i, j] = np.nan
            else:
                s = stats[d1][d2]
                total = s['wins'] + s['losses'] + s['draws']
                matrix[i, j] = s['wins'] / total if total > 0 else 0.5

    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热力图
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=[f'd{d}' for d in depths],
        yticklabels=[f'd{d}' for d in depths],
        mask=np.isnan(matrix),
        cbar_kws={'label': 'Win Rate'}
    )

    plt.xlabel('Black Depth (Opponent)', fontsize=12)
    plt.ylabel('Red Depth (Self)', fontsize=12)
    plt.title('UCI Engine Win Rate Matrix\n(Red vs Black)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_winrate(combined_stats, depths, save_path):
    """绘制综合胜率对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: 各深度对战胜率柱状图
    matchups = []
    d1_winrates = []
    d2_winrates = []
    draw_rates = []

    for key, s in sorted(combined_stats.items()):
        matchups.append(f"d{s['depth1']} vs d{s['depth2']}")
        d1_winrates.append(s['d1_winrate'])
        d2_winrates.append(s['d2_winrate'])
        draw_rate = s['draws'] / s['total_games'] if s['total_games'] > 0 else 0
        draw_rates.append(draw_rate)

    x = np.arange(len(matchups))
    width = 0.25

    axes[0].bar(x - width, d1_winrates, width, label='Smaller Depth Wins', color='#ff6b6b', alpha=0.8)
    axes[0].bar(x, draw_rates, width, label='Draws', color='#95a5a6', alpha=0.8)
    axes[0].bar(x + width, d2_winrates, width, label='Larger Depth Wins', color='#4ecdc4', alpha=0.8)

    axes[0].set_xlabel('Matchup')
    axes[0].set_ylabel('Rate')
    axes[0].set_title('Win/Draw/Loss Rate by Matchup')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(matchups, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')

    # 图2: 深度强度排名（基于综合胜率）
    depth_scores = {d: [] for d in depths}

    for key, s in combined_stats.items():
        d1, d2 = s['depth1'], s['depth2']
        # 较小深度的得分 = 1 - 较大深度的胜率
        depth_scores[d1].append(s['d1_winrate'])
        depth_scores[d2].append(s['d2_winrate'])

    avg_scores = {d: np.mean(scores) if scores else 0 for d, scores in depth_scores.items()}
    sorted_depths = sorted(avg_scores.keys())
    scores = [avg_scores[d] for d in sorted_depths]

    colors = plt.cm.RdYlGn(np.array(scores))
    axes[1].bar([f'd{d}' for d in sorted_depths], scores, color=colors, alpha=0.8)
    axes[1].set_xlabel('Search Depth')
    axes[1].set_ylabel('Average Win Rate')
    axes[1].set_title('Depth Strength Ranking\n(Higher = Stronger)')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (d, score) in enumerate(zip(sorted_depths, scores)):
        axes[1].text(i, score + 0.02, f'{score:.1%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_statistics(stats, combined_stats, depths, elapsed_time):
    """打印统计结果"""
    print('\n' + '=' * 70)
    print('统计结果')
    print('=' * 70)

    print(f'\n耗时: {timedelta(seconds=int(elapsed_time))}')

    print('\n【胜率矩阵】（红方深度 vs 黑方深度）')
    print('-' * 70)

    # 打印表头
    header = '红方\\黑方 |'
    for d in depths:
        header += f'  d{d:2d}  |'
    print(header)
    print('-' * 70)

    # 打印每行
    for d1 in depths:
        row = f'  d{d1:2d}    |'
        for d2 in depths:
            if d1 == d2:
                row += '   -   |'
            else:
                s = stats[d1][d2]
                total = s['wins'] + s['losses'] + s['draws']
                winrate = s['wins'] / total if total > 0 else 0
                row += f' {winrate*100:4.1f}% |'
        print(row)

    print('-' * 70)

    print('\n【综合统计】（合并先后手）')
    print('-' * 70)
    print(f'{"对战组合":<15} {"小深度胜":>8} {"大深度胜":>8} {"和棋":>6} {"小深度胜率":>10} {"结论":<20}')
    print('-' * 70)

    for key, s in sorted(combined_stats.items()):
        conclusion = ''
        if s['d2_winrate'] > 0.7:
            conclusion = f'深度{s["depth2"]}明显更强 ✓'
        elif s['d2_winrate'] > 0.55:
            conclusion = f'深度{s["depth2"]}略强'
        elif s['d1_winrate'] > 0.55:
            conclusion = f'深度{s["depth1"]}略强（异常!）'
        else:
            conclusion = '势均力敌'

        print(f'd{s["depth1"]:2d} vs d{s["depth2"]:2d}     '
              f'{s["d1_wins"]:>8} {s["d1_losses"]:>8} {s["draws"]:>6} '
              f'{s["d1_winrate"]*100:>9.1f}% {conclusion:<20}')

    print('=' * 70)

    # 检验单调性
    print('\n【单调性检验】')
    monotonic_violations = 0
    total_pairs = 0

    for key, s in combined_stats.items():
        total_pairs += 1
        if s['d1_winrate'] > 0.5:  # 小深度胜率 > 50%，违反单调性
            monotonic_violations += 1
            print(f'  ⚠ {key}: 深度{s["depth1"]}胜率({s["d1_winrate"]*100:.1f}%) > 深度{s["depth2"]}')

    if monotonic_violations == 0:
        print('  ✓ 所有对战符合单调性（深度越大越强）')
    else:
        print(f'\n  单调性符合率: {(total_pairs - monotonic_violations) / total_pairs * 100:.1f}%')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UCI vs UCI 多深度对战矩阵测试')
    parser.add_argument('--depths', default='3,5,8,10,12',
                       help='要测试的深度列表，逗号分隔 (默认: 3,5,8,10,12)')
    parser.add_argument('--games', type=int, default=10,
                       help='每对深度的对战局数 (默认: 10)')
    parser.add_argument('--engine', default='/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish',
                       help='UCI引擎路径')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='每局最大步数 (默认: 500)')
    parser.add_argument('--output', default=None,
                       help='输出目录 (默认: 自动生成)')

    args = parser.parse_args()

    depths = list(map(int, args.depths.split(',')))

    run_depth_matrix_test(
        depths=depths,
        games_per_pair=args.games,
        engine_path=args.engine,
        output_dir=args.output,
        max_steps=args.max_steps
    )
