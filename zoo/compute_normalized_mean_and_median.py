import numpy as np

def compute_normalized_mean_and_median(random_scores, human_scores, algo_scores):
    """
    计算 normalized mean 和 normalized median。

    参数:
    - random_scores: 随机分数列表 (List[float])
    - human_scores: 人类分数列表 (List[float])
    - algo_scores: 算法分数列表 (List[float])

    返回:
    - normalized_mean: 归一化得分的平均值（normalized mean）。
    - normalized_median: 归一化得分的中位数（normalized median）。
    """
    # 计算 normalized scores
    normalized_scores = []
    for random_score, human_score, algo_score in zip(random_scores, human_scores, algo_scores):
        # 避免除以零的情况
        if human_score != random_score:
            normalized_score = (algo_score - random_score) / (human_score - random_score)
        else:
            normalized_score = 0  # 如果 human_score == random_score，归一化分数设为0
        normalized_scores.append(normalized_score)

    # 计算 normalized mean 和 median
    normalized_mean = np.mean(normalized_scores)
    normalized_median = np.median(normalized_scores)

    return normalized_mean, normalized_median


# 游戏信息
random_scores = [
    227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
    152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
    -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
]

human_scores = [
    7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
    1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
    14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
]

# EZ, MZ with SSL 和 UniZero 的算法得分
ez_scores = [
    808.5, 149, 1263, 25558, 351, 13871, 53, 414, 1117, 83940,
    13004, 22, 296, 3260, 9315, 517, 724, 5663, 30945, 1281,
    20, 97, 13782, 17751, 1100, 17264
]

mz_ssl_scores = [
    700, 90, 600, 1400, 33, 7587, 20, 4, 2050, 26060,
    4601, 12, 260, 646, 8005, 300, 600, 2700, 25100, 1410,
    -15, 100, 4700, 3400, 566, 5213
]

unizero_scores = [
    1000, 96, 609, 1016, 50, 11410, 7, 12, 3205, 13666,
    1001, 7, 310, 1153, 3100, 305, 1285, 3484, 15600, 1927,
    18, 1048, 3056, 11000, 620, 4523
]

# 计算 EZ 算法的 normalized mean 和 median
ez_mean, ez_median = compute_normalized_mean_and_median(random_scores, human_scores, ez_scores)
print(f"EZ - Normalized Mean: {ez_mean}, Normalized Median: {ez_median}")

# 计算 MZ with SSL 算法的 normalized mean 和 median
mz_ssl_mean, mz_ssl_median = compute_normalized_mean_and_median(random_scores, human_scores, mz_ssl_scores)
print(f"MZ with SSL - Normalized Mean: {mz_ssl_mean}, Normalized Median: {mz_ssl_median}")

# 计算 UniZero 算法的 normalized mean 和 median
unizero_mean, unizero_median = compute_normalized_mean_and_median(random_scores, human_scores, unizero_scores)
print(f"UniZero - Normalized Mean: {unizero_mean}, Normalized Median: {unizero_median}")