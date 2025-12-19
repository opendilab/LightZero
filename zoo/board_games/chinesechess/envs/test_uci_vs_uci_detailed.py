#!/usr/bin/env python3
"""
UCI vs UCI 详细对战测试脚本
- 深度12 vs 深度8，共100局
- 交换先后手（各50局）
- CSV记录每局详细信息：每一步走法、胜负、步数等
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict


def play_single_game(engine_red, engine_black, depth_red, depth_black, engine_module, max_steps=500):
    """
    单局UCI vs UCI对战

    Args:
        engine_red: 红方UCI引擎实例
        engine_black: 黑方UCI引擎实例
        depth_red: 红方搜索深度
        depth_black: 黑方搜索深度
        engine_module: cchess.engine模块
        max_steps: 最大步数

    Returns:
        dict: 对局详细信息
    """
    from zoo.board_games.chinesechess.envs import cchess

    board = cchess.Board()
    moves_red = []  # 红方走法列表
    moves_black = []  # 黑方走法列表
    all_moves = []  # 所有走法（按顺序）
    step_count = 0

    while not board.is_game_over() and step_count < max_steps:
        if board.turn == cchess.RED:
            # 红方走棋
            limit = engine_module.Limit(depth=depth_red)
            result = engine_red.play(board, limit)
            move = result.move
            moves_red.append(move.uci())
        else:
            # 黑方走棋
            limit = engine_module.Limit(depth=depth_black)
            result = engine_black.play(board, limit)
            move = result.move
            moves_black.append(move.uci())

        all_moves.append(move.uci())
        board.push(move)
        step_count += 1

    # 判断结果
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        winner = "draw"
        winner_depth = None
    elif outcome.winner == cchess.RED:
        winner = "red"
        winner_depth = depth_red
    else:
        winner = "black"
        winner_depth = depth_black

    return {
        'depth_red': depth_red,
        'depth_black': depth_black,
        'winner': winner,
        'winner_depth': winner_depth,
        'total_steps': step_count,
        'moves_red': moves_red,
        'moves_black': moves_black,
        'all_moves': all_moves,
        'termination': str(outcome.termination) if outcome else 'max_steps'
    }


def run_uci_vs_uci_test(
    depth1=12,
    depth2=8,
    games_per_side=50,
    engine_path="/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish",
    output_dir=None,
    max_steps=500
):
    """
    运行UCI vs UCI详细测试

    Args:
        depth1: 第一个深度
        depth2: 第二个深度
        games_per_side: 每方先手的局数
        engine_path: UCI引擎路径
        output_dir: 输出目录
        max_steps: 每局最大步数
    """
    from zoo.board_games.chinesechess.envs.cchess import engine as engine_module

    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'test_results' / f'uci_d{depth1}_vs_d{depth2}_{timestamp}'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_games = games_per_side * 2

    print('=' * 70)
    print(f'UCI vs UCI 详细对战测试')
    print('=' * 70)
    print(f'深度1: {depth1}')
    print(f'深度2: {depth2}')
    print(f'每方先手局数: {games_per_side}')
    print(f'总局数: {total_games}')
    print(f'引擎路径: {engine_path}')
    print(f'输出目录: {output_dir}')
    print('=' * 70)

    # 统计变量
    results = []
    stats = {
        f'd{depth1}_red_wins': 0,
        f'd{depth1}_red_losses': 0,
        f'd{depth1}_red_draws': 0,
        f'd{depth2}_red_wins': 0,
        f'd{depth2}_red_losses': 0,
        f'd{depth2}_red_draws': 0,
        f'd{depth1}_total_wins': 0,
        f'd{depth2}_total_wins': 0,
        'total_draws': 0,
    }

    # 创建两个引擎实例
    engine1 = None
    engine2 = None

    try:
        print('\n正在加载UCI引擎...')
        engine1 = engine_module.SimpleEngine.popen_uci(engine_path)
        engine2 = engine_module.SimpleEngine.popen_uci(engine_path)
        print('UCI引擎加载成功!\n')

        game_id = 0

        # 第一轮: depth1 先手 (红方)
        print(f'\n--- 第一轮: 深度{depth1}先手 (红方) vs 深度{depth2}后手 (黑方) ---\n')
        for i in tqdm(range(games_per_side), desc=f'd{depth1}先手'):
            game_id += 1
            result = play_single_game(
                engine_red=engine1,
                engine_black=engine2,
                depth_red=depth1,
                depth_black=depth2,
                engine_module=engine_module,
                max_steps=max_steps
            )
            result['game_id'] = game_id
            result['red_is_depth1'] = True
            results.append(result)

            # 更新统计
            if result['winner'] == 'red':
                stats[f'd{depth1}_red_wins'] += 1
                stats[f'd{depth1}_total_wins'] += 1
            elif result['winner'] == 'black':
                stats[f'd{depth1}_red_losses'] += 1
                stats[f'd{depth2}_total_wins'] += 1
            else:
                stats[f'd{depth1}_red_draws'] += 1
                stats['total_draws'] += 1

        # 第二轮: depth2 先手 (红方)
        print(f'\n--- 第二轮: 深度{depth2}先手 (红方) vs 深度{depth1}后手 (黑方) ---\n')
        for i in tqdm(range(games_per_side), desc=f'd{depth2}先手'):
            game_id += 1
            result = play_single_game(
                engine_red=engine2,
                engine_black=engine1,
                depth_red=depth2,
                depth_black=depth1,
                engine_module=engine_module,
                max_steps=max_steps
            )
            result['game_id'] = game_id
            result['red_is_depth1'] = False
            results.append(result)

            # 更新统计
            if result['winner'] == 'red':
                stats[f'd{depth2}_red_wins'] += 1
                stats[f'd{depth2}_total_wins'] += 1
            elif result['winner'] == 'black':
                stats[f'd{depth2}_red_losses'] += 1
                stats[f'd{depth1}_total_wins'] += 1
            else:
                stats[f'd{depth2}_red_draws'] += 1
                stats['total_draws'] += 1

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

    # 保存详细CSV
    csv_path = output_dir / 'game_details.csv'
    save_detailed_csv(results, csv_path, depth1, depth2)

    # 保存统计CSV
    stats_path = output_dir / 'statistics.csv'
    save_statistics_csv(stats, stats_path, depth1, depth2, total_games)

    # 打印统计结果
    print_statistics(stats, depth1, depth2, total_games)

    print(f'\n结果已保存至: {output_dir}')
    print(f'  - 详细记录: {csv_path}')
    print(f'  - 统计汇总: {stats_path}')

    return results, stats


def save_detailed_csv(results, csv_path, depth1, depth2):
    """保存详细对局记录到CSV"""

    # 找出最大步数，用于确定列数
    max_moves = max(len(r['all_moves']) for r in results)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 表头
        header = [
            'game_id',
            'red_depth',
            'black_depth',
            'red_is_stronger',  # 红方是否是更深的那个
            'winner',
            'winner_depth',
            'total_steps',
            'termination',
        ]
        # 添加每一步的列
        for i in range(max_moves):
            step_num = i + 1
            if i % 2 == 0:
                header.append(f'move_{step_num}_red')
            else:
                header.append(f'move_{step_num}_black')

        writer.writerow(header)

        # 写入每局数据
        for r in results:
            row = [
                r['game_id'],
                r['depth_red'],
                r['depth_black'],
                'yes' if r['depth_red'] > r['depth_black'] else 'no',
                r['winner'],
                r['winner_depth'] if r['winner_depth'] else 'N/A',
                r['total_steps'],
                r['termination'],
            ]
            # 添加每一步走法
            for i in range(max_moves):
                if i < len(r['all_moves']):
                    row.append(r['all_moves'][i])
                else:
                    row.append('')

            writer.writerow(row)


def save_statistics_csv(stats, csv_path, depth1, depth2, total_games):
    """保存统计汇总到CSV"""

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(['统计项', '数值', '百分比'])
        writer.writerow([])

        # 总体统计
        writer.writerow(['=== 总体统计 ===', '', ''])
        writer.writerow(['总局数', total_games, '100%'])
        writer.writerow([f'深度{depth1}总胜场', stats[f'd{depth1}_total_wins'],
                        f"{stats[f'd{depth1}_total_wins']/total_games*100:.1f}%"])
        writer.writerow([f'深度{depth2}总胜场', stats[f'd{depth2}_total_wins'],
                        f"{stats[f'd{depth2}_total_wins']/total_games*100:.1f}%"])
        writer.writerow(['和棋总数', stats['total_draws'],
                        f"{stats['total_draws']/total_games*100:.1f}%"])
        writer.writerow([])

        # 深度1先手统计
        games_d1_red = stats[f'd{depth1}_red_wins'] + stats[f'd{depth1}_red_losses'] + stats[f'd{depth1}_red_draws']
        writer.writerow([f'=== 深度{depth1}先手(红方)统计 ===', '', ''])
        writer.writerow(['局数', games_d1_red, ''])
        writer.writerow(['胜', stats[f'd{depth1}_red_wins'],
                        f"{stats[f'd{depth1}_red_wins']/games_d1_red*100:.1f}%" if games_d1_red > 0 else 'N/A'])
        writer.writerow(['负', stats[f'd{depth1}_red_losses'],
                        f"{stats[f'd{depth1}_red_losses']/games_d1_red*100:.1f}%" if games_d1_red > 0 else 'N/A'])
        writer.writerow(['和', stats[f'd{depth1}_red_draws'],
                        f"{stats[f'd{depth1}_red_draws']/games_d1_red*100:.1f}%" if games_d1_red > 0 else 'N/A'])
        writer.writerow([])

        # 深度2先手统计
        games_d2_red = stats[f'd{depth2}_red_wins'] + stats[f'd{depth2}_red_losses'] + stats[f'd{depth2}_red_draws']
        writer.writerow([f'=== 深度{depth2}先手(红方)统计 ===', '', ''])
        writer.writerow(['局数', games_d2_red, ''])
        writer.writerow(['胜', stats[f'd{depth2}_red_wins'],
                        f"{stats[f'd{depth2}_red_wins']/games_d2_red*100:.1f}%" if games_d2_red > 0 else 'N/A'])
        writer.writerow(['负', stats[f'd{depth2}_red_losses'],
                        f"{stats[f'd{depth2}_red_losses']/games_d2_red*100:.1f}%" if games_d2_red > 0 else 'N/A'])
        writer.writerow(['和', stats[f'd{depth2}_red_draws'],
                        f"{stats[f'd{depth2}_red_draws']/games_d2_red*100:.1f}%" if games_d2_red > 0 else 'N/A'])


def print_statistics(stats, depth1, depth2, total_games):
    """打印统计结果"""

    print('\n' + '=' * 70)
    print('统计结果')
    print('=' * 70)

    print(f'\n总局数: {total_games}')
    print(f'\n【总体胜率】')
    print(f'  深度{depth1}总胜场: {stats[f"d{depth1}_total_wins"]} ({stats[f"d{depth1}_total_wins"]/total_games*100:.1f}%)')
    print(f'  深度{depth2}总胜场: {stats[f"d{depth2}_total_wins"]} ({stats[f"d{depth2}_total_wins"]/total_games*100:.1f}%)')
    print(f'  和棋: {stats["total_draws"]} ({stats["total_draws"]/total_games*100:.1f}%)')

    games_d1_red = stats[f'd{depth1}_red_wins'] + stats[f'd{depth1}_red_losses'] + stats[f'd{depth1}_red_draws']
    games_d2_red = stats[f'd{depth2}_red_wins'] + stats[f'd{depth2}_red_losses'] + stats[f'd{depth2}_red_draws']

    print(f'\n【深度{depth1}先手(红方)】 共{games_d1_red}局')
    if games_d1_red > 0:
        print(f'  胜: {stats[f"d{depth1}_red_wins"]} ({stats[f"d{depth1}_red_wins"]/games_d1_red*100:.1f}%)')
        print(f'  负: {stats[f"d{depth1}_red_losses"]} ({stats[f"d{depth1}_red_losses"]/games_d1_red*100:.1f}%)')
        print(f'  和: {stats[f"d{depth1}_red_draws"]} ({stats[f"d{depth1}_red_draws"]/games_d1_red*100:.1f}%)')

    print(f'\n【深度{depth2}先手(红方)】 共{games_d2_red}局')
    if games_d2_red > 0:
        print(f'  胜: {stats[f"d{depth2}_red_wins"]} ({stats[f"d{depth2}_red_wins"]/games_d2_red*100:.1f}%)')
        print(f'  负: {stats[f"d{depth2}_red_losses"]} ({stats[f"d{depth2}_red_losses"]/games_d2_red*100:.1f}%)')
        print(f'  和: {stats[f"d{depth2}_red_draws"]} ({stats[f"d{depth2}_red_draws"]/games_d2_red*100:.1f}%)')

    print('=' * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UCI vs UCI 详细对战测试')
    parser.add_argument('--depth1', type=int, default=12, help='第一个深度 (默认: 12)')
    parser.add_argument('--depth2', type=int, default=8, help='第二个深度 (默认: 8)')
    parser.add_argument('--games', type=int, default=50, help='每方先手的局数 (默认: 50，总共100局)')
    parser.add_argument('--engine', default='/mnt/shared-storage-user/tangjia/chess/Pikafish/src/pikafish',
                       help='UCI引擎路径')
    parser.add_argument('--max-steps', type=int, default=500, help='每局最大步数 (默认: 500)')
    parser.add_argument('--output', default=None, help='输出目录 (默认: 自动生成)')

    args = parser.parse_args()

    run_uci_vs_uci_test(
        depth1=args.depth1,
        depth2=args.depth2,
        games_per_side=args.games,
        engine_path=args.engine,
        output_dir=args.output,
        max_steps=args.max_steps
    )
