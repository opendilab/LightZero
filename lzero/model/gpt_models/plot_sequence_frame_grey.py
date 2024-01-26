import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
batch_observations = reconstructed_images.detach().view(32, 5, 4, 64, 64)[:,:,0:1,:,:]
# batch_observations = batch['observations'][:,:,0:1,:,:]
B, N, C, H, W = batch_observations.shape  # 自动检测维度

# 分隔条的宽度（可以根据需要调整）
separator_width = 2

# 遍历每个样本
for i in range(B):
    # 提取当前样本中的所有帧
    frames = batch_observations[i]

    # 计算拼接图像的总宽度（包括分隔条）
    total_width = N * W + (N - 1) * separator_width

    # 创建一个新的图像，其中包含分隔条，模式为'L'代表灰度图
    concat_image = Image.new('L', (total_width, H), color='black')

    # 拼接每一帧及分隔条
    for j in range(N):
        # 如果输入是灰度图像，那么C应该为1，我们可以直接使用numpy的squeeze方法去掉长度为1的维度
        frame = np.squeeze(frames[j].cpu().numpy(), axis=0)  # 转换为(H, W)
        frame_image = Image.fromarray((frame * 255).astype('uint8'), 'L')

        # 计算当前帧在拼接图像中的位置
        x_position = j * (W + separator_width)
        concat_image.paste(frame_image, (x_position, 0))

    # 显示图像
    plt.imshow(concat_image, cmap='gray')
    plt.title(f'Sample {i+1}')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

    # 保存图像到文件
    concat_image.save(f'sample_{i+1}_recs_0125.png')




import torch
# 假设x是一个具有形状[160, 64, 8, 8]的张量
x = torch.randn(160, 64, 8, 8)
# 首先将特征图展平，以便每个样本都变成一个长向量
x_flat = x.view(x.size(0), -1)

# 初始化一个矩阵来保存所有样本对的相似度比率
similarity_matrix = torch.zeros((x.size(0), x.size(0)))

# 计算所有样本对的相似度
for i in range(x.size(0)):
    for j in range(i+1, x.size(0)):  # 只计算上三角矩阵
        # 计算两个样本之间完全相同的特征比率
        similarity = (x_flat[i] == x_flat[j]).float().mean()
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # 矩阵是对称的

# 打印出相似度矩阵
print(similarity_matrix)

# 获取平均相似度
average_similarity = similarity_matrix.sum() / (x.size(0) * (x.size(0) - 1))
print(f"Average similarity ratio between all pairs of samples: {average_similarity.item()}")