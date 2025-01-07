


import numpy as np
import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 归一化，减少目标值的幅度差异。
    symlog(x) = sign(x) * log(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 的逆操作，用于恢复原始值。
    inv_symlog(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def compute_task_weights(
    task_rewards: dict,
    epsilon: float = 1e-6,
    min_weight: float = 0.1,
    max_weight: float = 0.5,
    temperature: float = 1.0,
    use_symlog: bool = True,
) -> dict:
    """
    改进后的任务权重计算函数，加入 symlog 处理和鲁棒性设计。

    Args:
        task_rewards (dict): 每个任务的字典，键为 task_id，值为评估奖励。
        epsilon (float): 避免分母为零的小值。
        min_weight (float): 权重的最小值，用于裁剪。
        max_weight (float): 权重的最大值，用于裁剪。
        temperature (float): 控制权重分布的温度系数。
        use_symlog (bool): 是否使用 symlog 对 task_rewards 进行矫正。

    Returns:
        dict: 每个任务的权重，键为 task_id，值为归一化并裁剪后的权重。
    """
    # Step 1: 矫正奖励值（可选，使用 symlog）
    if use_symlog:
        rewards_tensor = torch.tensor(list(task_rewards.values()), dtype=torch.float32)
        corrected_rewards = symlog(rewards_tensor).numpy()  # 使用 symlog 矫正
        task_rewards = dict(zip(task_rewards.keys(), corrected_rewards))

    # Step 2: 计算初始权重（反比例关系）
    raw_weights = {task_id: 1 / (reward + epsilon) for task_id, reward in task_rewards.items()}

    # Step 3: 温度缩放
    scaled_weights = {task_id: weight ** (1 / temperature) for task_id, weight in raw_weights.items()}

    # Step 4: 归一化权重
    total_weight = sum(scaled_weights.values())
    normalized_weights = {task_id: weight / total_weight for task_id, weight in scaled_weights.items()}

    # Step 5: 裁剪权重，确保在 [min_weight, max_weight] 范围内
    clipped_weights = {task_id: np.clip(weight, min_weight, max_weight) for task_id, weight in normalized_weights.items()}

    final_weights = clipped_weights
    return final_weights

task_rewards_list = [
    {"task1": 10, "task2": 100, "task3": 1000, "task4": 500, "task5": 300},
    {"task1": 1, "task2": 10, "task3": 100, "task4": 1000, "task5": 10000},
    {"task1": 0.1, "task2": 0.5, "task3": 0.9, "task4": 5, "task5": 10},
]

for i, task_rewards in enumerate(task_rewards_list, start=1):
    print(f"Case {i}: Original Rewards: {task_rewards}")
    print("Original Weights:")
    print(compute_task_weights(task_rewards, use_symlog=False))
    print("Improved Weights with Symlog:")
    print(compute_task_weights(task_rewards, use_symlog=True))
    print()