import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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






x = torch.randn(192, 64, 8, 8).to('cuda:0')

def check_layer_output(model, x):
    for name, layer in model.named_children():
        x = layer(x)
        if torch.any(x == 0):
            print(f"After {name}, there are zeros in the output.")
        else:
            print(f"After {name}, there are no zeros in the output.")
        print(f"Layer: {name} | Output mean: {x.mean():.4f} | Output std: {x.std():.4f} |")
    return x

output = check_layer_output(model, x)