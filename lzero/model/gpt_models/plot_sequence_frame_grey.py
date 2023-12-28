import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
    concat_image.save(f'sample_{i+1}_0105.png')