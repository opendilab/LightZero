import torch

# 参数设置
seq_batch_size = 3
num_unroll_steps = 5
history_length = 3

# 模拟输入：(seq_batch_size*(num_unroll_steps+history_length), 3, 64, 64) = (21, 3, 64, 64)
obs = torch.randn(seq_batch_size * (num_unroll_steps + history_length), 3, 64, 64)

# Step 1: 重构 shape 为 [seq_batch_size, (num_unroll_steps+history_length), 3, 64, 64]
obs = obs.view(seq_batch_size, num_unroll_steps + history_length, 3, 64, 64)
# 此时 obs.shape = [3, 7, 3, 64, 64]

# Step 2: 对时间维度应用 sliding window 操作（unfold）；
# unfolding 参数: 在 dim=1 上，窗口大小为 history_length，步长为 1.
# unfolding 后形状：[seq_batch_size, (7 - history_length + 1), history_length, 3, 64, 64]
windows = obs.unfold(dimension=1, size=history_length, step=1)  # 形状：[3, 6, 3, 64, 64, 2]
print("Step 2 windows.shape:", windows.shape)

# Step 3: 根据要求，仅使用时间步从 history_length 开始的部分
# 即丢弃第一个窗口，保留从索引 1 开始的 5 个窗口：最终形状为 [3, 5, 2, 3, 64, 64]
# windows = windows[:, 1:]

# Step 4: 将窗口中的观测在通道维度上进行拼接
# 原本每个窗口形状为 [2, 3, 64, 64]，将 2 (history_length) 个通道拼接后变为 [6, 64, 64]
# 整体结果 shape 最终为 [seq_batch_size, num_unroll_steps, history_length*3, 64, 64] = [3, 5, 6, 64, 64]
windows = windows.reshape(seq_batch_size, num_unroll_steps+1, history_length * 3, 64, 64)

print("最终结果 shape:", windows.shape)
# 输出应为: torch.Size([3, 5, 6, 64, 64])