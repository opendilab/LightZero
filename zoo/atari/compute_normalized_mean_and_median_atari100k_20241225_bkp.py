import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

def compute_normalized_mean_and_median(
    random_scores: List[float],
    human_scores: List[float],
    algo_scores: List[float]
) -> Tuple[float, float]:
    """
    计算基于随机、人类和算法分数的归一化均值和中位数。

    参数:
        random_scores (List[float]): 每个游戏的随机分数列表。
        human_scores (List[float]): 每个游戏的人类分数列表。
        algo_scores (List[float]): 每个游戏的算法分数列表。

    返回:
        Tuple[float, float]: 归一化均值和中位数。

    异常:
        ValueError: 如果任一输入列表为空或长度不一致。
    """
    if not random_scores or not human_scores or not algo_scores:
        raise ValueError("输入的分数列表不能为空。")
    if len(random_scores) != len(human_scores) or len(human_scores) != len(algo_scores):
        raise ValueError("输入的分数列表长度必须一致。")

    # 计算归一化分数
    normalized_scores = [
        (algo_score - random_score) / (human_score - random_score)
        if human_score != random_score else 0
        for random_score, human_score, algo_score in zip(random_scores, human_scores, algo_scores)
    ]

    # 计算均值和中位数
    normalized_mean = np.mean(normalized_scores)
    normalized_median = np.median(normalized_scores)

    return normalized_mean, normalized_median


def plot_normalized_scores(
    algorithms: List[str],
    means: List[float],
    medians: List[float],
    filename: str = "normalized_scores.png"
) -> None:
    """
    绘制不同算法的归一化均值和中位数的柱状图。

    参数:
        algorithms (List[str]): 算法名称列表。
        means (List[float]): 归一化均值列表。
        medians (List[float]): 归一化中位数列表。
        filename (str, optional): 保存图表的文件名（默认是 'normalized_scores.png'）。

    返回:
        None

    异常:
        ValueError: 如果算法、均值或中位数列表长度不一致。
    """
    if not (len(algorithms) == len(means) == len(medians)):
        raise ValueError("算法、均值和中位数列表的长度必须一致。")

    # 设置适合学术论文的风格
    sns.set(style="whitegrid", font_scale=1.2)

    x = np.arange(len(algorithms))  # 算法位置
    width = 0.35  # 柱状图的宽度

    # 设置图表尺寸
    plt.figure(figsize=(10, 6))

    # 定义颜色：均值为蓝色，中位数为红色
    mean_color = '#1f77b4'   # Matplotlib 默认的蓝色
    median_color = '#d62728' # Matplotlib 默认的红色

    # 绘制均值和中位数的柱状图
    # bars_mean = plt.bar(x - width/2, means, width, label='归一化均值', color=mean_color)
    # bars_median = plt.bar(x + width/2, medians, width, label='归一化中位数', color=median_color)

    bars_mean = plt.bar(x - width/2, means, width, label='Normalized Mean', color=mean_color)
    bars_median = plt.bar(x + width/2, medians, width, label='Normalized Median', color=median_color)


    # 添加标签和标题
    # plt.ylabel('分数', fontsize=14)
    # plt.title('Atari 100k 人类归一化分数', fontsize=16, pad=20)
    plt.ylabel('Score', fontsize=14)
    plt.title('Human Normalized Score (Atari 100k)', fontsize=16, pad=20)

    plt.xticks(x, algorithms, fontsize=12)
    plt.legend(fontsize=12)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在每个柱子上添加数值标签
    def attach_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 偏移量
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11)

    attach_labels(bars_mean)
    attach_labels(bars_median)

    # 调整布局避免标签被截断
    plt.tight_layout()

    # 保存图表
    plt.savefig(filename, dpi=300)
    print(f"图表已保存为 {filename}")
    plt.close()


def main():
    # 游戏随机、人类和算法分数
    scores_data: Dict[str, Dict[str, List[float]]] = {
        '随机': {
            'scores': [
                227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
                152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
                -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
            ]
        },
        '人类': {
            'scores': [
                7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
                1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
                14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
            ]
        },
        'EZ': {
            'scores': [
                808.5, 149, 1263, 25558, 351, 13871, 53, 414, 1117, 83940,
                13004, 22, 296, 3260, 9315, 517, 724, 5663, 30945, 1281,
                20, 97, 13782, 17751, 1100, 17264
            ]
        },
        'MZ': {
            'scores': [
                530.0, 39, 500, 1734, 193, 7688, 15, 48, 1350, 56937,
                3527, 22, 255, 1256, 3095, 88, 63, 4891, 18813, 1266,
                -7, 56, 3952, 2500, 208, 2897
            ]
        },
        'MZ with SSL': {
            'scores': [
                700, 90, 600, 1400, 33, 7587, 20, 4, 2050, 26060,
                4601, 12, 260, 646, 9315, 300, 600, 2700, 25100, 1410,
                -15, 100, 4700, 3400, 566, 5213
            ]
        },
        'UniZero': {
            'scores': [
                1000, 96, 609, 1016, 50, 11410, 7, 12, 3205, 13666,
                1001, 7, 310, 1153, 8005, 305, 1285, 3484, 15600, 1927,
                18, 1048, 3056, 11000, 620, 4523
            ]
        }
    }

    # 提取随机和人类分数
    random_scores = scores_data['随机']['scores']
    human_scores = scores_data['人类']['scores']

    # 需要计算归一化分数的算法
    # algo_names = ['EZ', 'MZ', 'MZ with SSL', 'UniZero']
    algo_names = ['MZ', 'MZ with SSL', 'UniZero']

    normalized_results = {}

    for algo in algo_names:
        algo_scores = scores_data[algo]['scores']
        mean, median = compute_normalized_mean_and_median(random_scores, human_scores, algo_scores)
        normalized_results[algo] = {'mean': mean, 'median': median}
        print(f"{algo} - 归一化均值: {mean:.4f}, 归一化中位数: {median:.4f}")

    # 准备绘图数据
    algorithms = list(normalized_results.keys())
    means = [normalized_results[algo]['mean'] for algo in algorithms]
    medians = [normalized_results[algo]['median'] for algo in algorithms]

    algorithms_x_stick = ['MuZero', 'MuZero (Reproduced)', 'UniZero']
    # 绘制图表
    plot_normalized_scores(
        algorithms=algorithms_x_stick,
        means=means,
        medians=medians,
        filename="atari100k_normalized_scores_3algo_20241225.png"
    )

if __name__ == "__main__":
    main()