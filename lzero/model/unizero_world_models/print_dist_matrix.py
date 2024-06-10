import torch

# 假设 z 是一个形状为 [8, 512, 4, 4] 的张量
z = torch.randn(8, 512, 4, 4)  # 这里是用随机数生成的数据作为例子

# z=self.embedding.weight

# 将每个样本展平成一维向量
z_flat = z.view(z.size(0), -1)  # 结果的形状将是 [8, 512*4*4]

# 初始化一个矩阵来存储距离
dist_matrix = torch.zeros((z.size(0), z.size(0)))

# 计算两两样本之间的欧几里得距离
for i in range(z.size(0)):
    for j in range(i+1, z.size(0)):
        dist_matrix[i, j] = torch.norm(z_flat[i] - z_flat[j])
        dist_matrix[j, i] = dist_matrix[i, j]  # 距离矩阵是对称的

print(dist_matrix)