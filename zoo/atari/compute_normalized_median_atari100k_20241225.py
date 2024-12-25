import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


def compute_normalized_median(
    random_scores: List[float],
    human_scores: List[float],
    algo_scores: List[float],
) -> float:
    """
    计算基于随机、人类和算法分数的归一化中位数。

    参数:
        random_scores (List[float]): 每个游戏的随机分数列表。
        human_scores (List[float]): 每个游戏的人类分数列表。
        algo_scores (List[float]): 每个游戏的算法分数列表。

    返回:
        float: 归一化中位数。

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

    # 计算中位数
    normalized_median = np.median(normalized_scores)
    return normalized_median


def plot_normalized_medians(
    algorithms: List[str],
    medians: List[float],
    filename: str = "normalized_medians.png",
) -> None:
    """
    绘制不同算法的归一化中位数的柱状图。

    参数:
        algorithms (List[str]): 算法名称列表。
        medians (List[float]): 归一化中位数列表。
        filename (str, optional): 保存图表的文件名（默认是 'normalized_medians.png'）。

    返回:
        None

    异常:
        ValueError: 如果算法或中位数列表长度不一致。
    """
    if not (len(algorithms) == len(medians)):
        raise ValueError("算法和中位数列表的长度必须一致。")

    # 设置学术风格和字体比例
    sns.set(style="whitegrid", font_scale=1.5)

    # 定义颜色：所有算法有独立颜色，UniZero 为蓝色
    colors = ['#2ca02c', '#d62728', '#1f77b4']  # 绿色, 红色, 蓝色
    color_mapping = {algo: colors[i] for i, algo in enumerate(algorithms)}

    x = np.arange(len(algorithms))  # 算法位置
    width = 0.6  # 柱宽

    # 设置图表尺寸
    plt.figure(figsize=(14, 8))

    # 绘制柱状图
    bars = plt.bar(
        x, medians, width, color=[color_mapping[algo] for algo in algorithms], edgecolor="black"
    )

    # 添加标签和标题
    # plt.ylabel('Normalized Median Score', fontsize=30)
    plt.ylabel('Score', fontsize=30)

    plt.title('Human Normalized Median Score (Atari 100k)', fontsize=30, pad=20)

    plt.xticks(x, algorithms, fontsize=30)
    plt.yticks(fontsize=20)
    plt.ylim(0, max(medians) * 1.2)  # 设置 y 轴范围，留空间给标签
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加图例，放在图表顶部，避免与柱状图重叠
    for algo, color in color_mapping.items():
        plt.bar(0, 0, color=color, label=algo, edgecolor="black")  # 虚拟柱子用于图例
    # plt.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3)

    # 在每个柱子上添加数值标签
    def attach_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 偏移量
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=30,
            )

    attach_labels(bars)

    # 调整布局避免标签和内容被截断
    plt.tight_layout()

    # 保存图表
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
        'MuZero': {
            'scores': [
                530.0, 39, 500, 1734, 193, 2688, 15, 48, 1350, 56937,
                3527, 22, 255, 1256, 3095, 88, 63, 4891, 18813, 1266,
                -7, 56, 3952, 2500, 208, 2897
            ]
        },
        'MuZero (Reproduced)': {
            'scores': [
                300, 90, 609, 1400, 223, 7587, 20, 3, 1050, 22060,
                4601, 12, 260, 346, 3315, 90, 200, 5191, 6100, 1010,
                -15, 100, 1700, 4400, 466, 1213
            ]
        },
        'UniZero (Ours)': {
            'scores': [
                600, 96, 608, 1216, 400, 11410, 7, 8, 2205, 13666,
                991, 10, 310, 853, 2005, 405, 1885, 4484, 11400, 900,
                -10, 500, 1056, 1100, 620, 2823
            ]
        }
    }

    # 提取随机和人类分数
    random_scores = scores_data['随机']['scores']
    human_scores = scores_data['人类']['scores']

    # 需要计算归一化分数的算法
    algo_names = ['MuZero', 'MuZero (Reproduced)', 'UniZero (Ours)']

    normalized_results = {}

    for algo in algo_names:
        algo_scores = scores_data[algo]['scores']
        median = compute_normalized_median(random_scores, human_scores, algo_scores)
        normalized_results[algo] = {'median': median}
        print(f"{algo} - 归一化中位数: {median:.4f}")

    # 准备绘图数据
    algorithms = list(normalized_results.keys())
    medians = [normalized_results[algo]['median'] for algo in algorithms]

    plot_normalized_medians(
        algorithms=algorithms,
        medians=medians,
        filename="atari100k_normalized_medians_20241225.png"
    )


if __name__ == "__main__":
    main()