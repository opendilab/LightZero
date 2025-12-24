# Loss Landscape 核心库

一个简洁、模块化的 PyTorch 库，用于神经网络 loss landscape 的可视化。

本库基于 [loss-landscape](https://github.com/tomgoldstein/loss-landscape) 仓库进行抽象和重构，提供了通用的神经网络 loss landscape 可视化工具，适用于任何 PyTorch 模型。下面将展示通用示例（CIFAR-10）和专门针对 UniZero 模型的应用示例。

## 安装

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## 使用

### 利用 Loss Landscape 模块绘制 Loss 曲面

```python
from lzero.loss_landscape import LossLandscape

# 创建 LossLandscape 对象
# net: 你的 PyTorch 模型
# dataloader: 数据加载器
# criterion: 损失函数（如 nn.CrossEntropyLoss()）
# use_cuda: 是否使用 GPU
landscape = LossLandscape(net, dataloader, criterion, use_cuda=True)

# 计算 2D landscape
result = landscape.compute_2d(
    xrange=(-1, 1, 51),
    yrange=(-1, 1, 51)
)

# 多种格式的可视化
landscape.plot_2d_contour()      # 轮廓线
landscape.plot_2d_surface()       # 3D 曲面
landscape.export_paraview()       # 高质量渲染
```

**简单示例的可视化结果：**

| 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- |
| <img src="./images/simple/simple_example_2dcontour.png" width="200" /> | <img src="./images/simple/simple_example_2dcontourf.png" width="200" /> | <img src="./images/simple/simple_example_2dheat.png" width="200" /> | <img src="./images/simple/simple_example_3dsurface.png" width="200" /> |

### 更多的简单示例（CIFAR-10）

详见 `examples/` 目录中的示例：
- **example_1d.py** - 1D Loss 曲线
- **example_2d.py** - 2D Loss 曲面与 ParaView

```bash
cd lzero/loss_landscape
python examples/example_1d.py
python examples/example_2d.py
```

### 利用 Loss Landscape 绘制 UniZero 的 Loss Landscape

对于 UniZero 模型，可以使用批处理脚本对多个训练迭代的 checkpoint 进行 loss landscape 评估：

```bash
bash lzero/loss_landscape/run_loss_landscape_batch.sh
```

该脚本会自动：
1. 遍历指定目录下的所有 checkpoint 文件（iteration_10000.pth.tar 到 iteration_100000.pth.tar）
2. 为每个 checkpoint 加载模型权重
3. 从游戏环境收集评估数据
4. 计算 21×21 网格的 loss landscape（441 个评估点）
5. 生成多种可视化图像（轮廓图、填充轮廓、热力图、3D 曲面）

**使用前需要修改脚本中的路径配置：**
- `CKPT_BASE_DIR`: checkpoint 文件所在目录
- `CONFIG_SCRIPT`: loss landscape 配置脚本路径
- `BASE_LOG_DIR`: 输出结果保存目录
- `ENV_ID`: Atari 游戏环境 ID（如 "PongNoFrameskip-v4"）

### UniZero 的 Loss Landscape 可视化结果

本库已应用于 UniZero 模型在 Atari 游戏中的 loss landscape 可视化。结果通过以下方式生成：

**批处理评估流程：**
运行 `run_loss_landscape_batch.sh` 脚本，对多个训练迭代的 checkpoint (10K-100K) 进行批量 loss landscape 评估。脚本会自动：
1. 加载每个迭代的 checkpoint
2. 从游戏环境收集一批数据
3. 计算 21×21 网格的 loss landscape（441 个评估点）
4. 生成多种可视化图像（轮廓图、填充轮廓、热力图、3D 曲面）

**结果展示：**

不同训练迭代中 Pong 和 MsPacman 环境的 Total Loss 的 loss landscape 可视化。每行显示不同的迭代检查点，列显示不同的可视化样式。

#### Pong 环境 - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

#### MsPacman 环境 - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

## 可视化文件

loss landscape 计算完成后，系统会生成以下类型的输出文件：

- **HDF5 数据文件** (`.h5`) - 原始数据存储，包含坐标和 loss 值
- **PDF 可视化文件** - 用于报告和演示的图表文件，包括以下 4 种视图：
  - `*_2dcontour.pdf` - 轮廓线图，显示 loss 等值线
  - `*_2dcontourf.pdf` - 填充轮廓图，用颜色填充不同 loss 区域
  - `*_2dheat.pdf` - 热力图，用颜色强度表示 loss 值
  - `*_3dsurface.pdf` - 3D 曲面图，立体呈现 loss landscape
- **ParaView 文件** (`.vtp`) - 高质量 3D 渲染专业格式

## loss landscape 模块结构

```
loss_landscape/
├── core/                       # 核心功能
│   ├── direction.py            # 方向生成和归一化
│   ├── evaluator.py            # Loss 和准确率评估
│   └── perturbation.py         # 权重扰动
├── utils/                      # 实用程序
│   ├── storage.py              # HDF5 文件 I/O
│   ├── plot_1d.py              # 1D 绘图
│   ├── plot_2d.py              # 2D 绘图（支持多指标）
│   ├── projection.py           # 方向投影
│   └── paraview.py             # ParaView 导出
├── loss_landscape_api.py       # 高级 API
├── __init__.py                 # 包初始化
├── examples/                   # 示例脚本
│   ├── example_1d.py
│   └── example_2d.py
├── images/                     # 可视化结果
│   ├── simple/                 # 简单示例
│   ├── Pong/                   # Pong 环境结果
│   └── MsPacman/               # MsPacman 环境结果
└── run_loss_landscape_batch.sh # 批处理脚本
```

## 性能测试

### 计算时间

- **1D landscape**：O(num_points) - 与采样点数线性相关
- **2D landscape**：O(num_points_x × num_points_y) - 二次方
- 每个点需要对所有数据的一次前向传播

### 估计运行时间（现代 GPU）

**对于单个 checkpoint 在 21×21 分辨率下的评估：** 评估流程包括：(1) 加载 checkpoint，(2) 从环境中收集一批数据，(3) 通过扰动模型权重在参数空间的 441 个点（21×21 网格）计算 loss 值，(4) 生成多种可视化图像（轮廓图、热力图、3D 曲面图）。在 H200 GPU 上约需 30 分钟。

### 如何加快计算

1. **减少网格分辨率**：测试时使用 11×11 或 15×15 而不是 21×21
2. **使用更少的批**：减少 num_batches 参数（例如 20-50 而不是 100）
3. **使用 GPU 加速**：启用 `use_cuda=True` 以获得 ~10-100 倍加速
4. **减少 batch size**：较小的 batch 适合 GPU 内存
5. **并行评估**：使用多个 GPU 和数据并行

## 更好结果的提示

1. **使用归一化数据**：确保数据加载器使用归一化数据
2. **充分采样**：每个维度至少使用 51 个点
3. **适当的 loss 范围**：在轮廓图中调整 `vmin`/`vmax`
4. **对数刻度**：在 ParaView 导出中对大动态范围使用 `log=True`
5. **分辨率**：导出到 ParaView 以获得发表质量的图表

## 参考资料

- 原始论文：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- 原始代码：https://github.com/tomgoldstein/loss-landscape
- PyTorch 文档：https://pytorch.org/docs/
- ParaView：https://www.paraview.org/

## 引用

如果在您的研究中使用此库，请引用原始工作和我们的工作：[UniZero](https://openreview.net/pdf?id=Gl6dF9soQo) 和 [LightZero](https://proceedings.neurips.cc/paper_files/paper/2023/file/765043fe026f7d704c96cec027f13843-Paper-Datasets_and_Benchmarks.pdf)

```bibtex
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}

@article{niu2024lightzero,
  title={LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios},
  author={Niu, Yazhe and Pu, Yuan and Yang, Zhenjie and Li, Xueyan and Zhou, Tong and Ren, Jiyuan and Hu, Shuai and Li, Hongsheng and Liu, Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{puunizero,
  title={UniZero: Generalized and Efficient Planning with Scalable Latent World Models},
  author={Pu, Yuan and Niu, Yazhe and Yang, Zhenjie and Ren, Jiyuan and Li, Hongsheng and Liu, Yu},
  journal={Transactions on Machine Learning Research}
}
```

## 许可证

MIT 许可证 - 参见原始 [loss-landscape](https://github.com/tomgoldstein/loss-landscape) 仓库
