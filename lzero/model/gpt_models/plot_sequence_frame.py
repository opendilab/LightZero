import torch
from PIL import Image
import matplotlib.pyplot as plt

# 假设batch是一个字典，其中包含了observations键，
# 并且它的形状是torch.Size([B, N, C, H, W])
# batch_observations = batch_for_gpt['observations']
# batch_observations = batch['observations']
batch_observations = obs.unsqueeze(0)
# batch_observations = rec_img.unsqueeze(0)

# batch_observations = observations.unsqueeze(0)
# batch_observations = x.unsqueeze(0)
# batch_observations = reconstructions.unsqueeze(0)



B, N, C, H, W = batch_observations.shape  # 自动检测维度

# 分隔条的宽度（可以根据需要调整）
separator_width = 2

# 遍历每个样本
for i in range(B):
    # 提取当前样本中的所有帧
    frames = batch_observations[i]

    # 计算拼接图像的总宽度（包括分隔条）
    total_width = N * W + (N - 1) * separator_width

    # 创建一个新的图像，其中包含分隔条
    concat_image = Image.new('RGB', (total_width, H), color='black')

    # 拼接每一帧及分隔条
    for j in range(N):
        frame = frames[j].permute(1, 2, 0).cpu().numpy()  # 转换为(H, W, C)
        frame_image = Image.fromarray((frame * 255).astype('uint8'), 'RGB')

        # 计算当前帧在拼接图像中的位置
        x_position = j * (W + separator_width)
        concat_image.paste(frame_image, (x_position, 0))

    # 显示图像
    plt.imshow(concat_image)
    plt.title(f'Sample {i+1}')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

    # 保存图像到文件
    concat_image.save(f'sample_{i+1}.png')