import numpy as np

# 数据
tasks = [
    "acrobot-swingup", "cartpole-balance", "cartpole-balance_sparse", "cartpole-swingup",
    "cartpole-swingup_sparse", "cheetah-run", "ball_in_cup-catch", "finger-spin",
    "finger-turn_easy", "finger-turn_hard", "hopper-hop", "hopper-stand",
    "pendulum-swingup", "reacher-easy", "reacher-hard", "walker-run", "walker-stand", "walker-walk"
]

unizero_scores = [
    400, 952, 1000, 801, 752, 517, 961, 810, 1000, 884, 120, 602, 865, 993, 988, 587, 976, 954
]

dreamerv3_scores = [
    154.5, 990.5, 996.8, 850.0, 468.1, 585.9, 958.2, 937.2, 745.4, 841.0, 111.0, 573.2, 766.0,
    947.1, 936.2, 632.7, 956.9, 935.7
]

# 计算 mean 和 median
unizero_mean = np.mean(unizero_scores)
unizero_median = np.median(unizero_scores)

dreamerv3_mean = np.mean(dreamerv3_scores)
dreamerv3_median = np.median(dreamerv3_scores)

# 打印详细统计结果
print("UniZero Statistics:")
print(f"Mean: {unizero_mean:.2f}")
print(f"Median: {unizero_median:.2f}")
print("\nDreamerV3 Statistics:")
print(f"Mean: {dreamerv3_mean:.2f}")
print(f"Median: {dreamerv3_median:.2f}")