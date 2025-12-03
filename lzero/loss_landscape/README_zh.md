# Loss Landscape 核心库

**本工具专门用于为 UniZero 模型检查点（checkpoint）绘制 loss landscape 可视化。**

一个简洁、模块化的 PyTorch 库，用于神经网络 loss landscape 的可视化。

## 安装

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## 使用

### 2D Loss 曲面

```python
from lzero.loss_landscape import LossLandscape

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

### 运行示例

详见 `examples/` 目录中的示例：
- **example_1d.py** - 1D Loss 曲线
- **example_2d.py** - 2D Loss 曲面与 ParaView
- **example_custom_metrics.py** - 自定义指标函数

```bash
python examples/example_1d.py
python examples/example_2d.py
```

## 模块结构

```
loss_landscape/
├── core/                  # 核心功能
│   ├── direction.py       # 方向生成和归一化
│   ├── evaluator.py       # Loss 和准确率评估
│   └── perturbation.py    # 权重扰动
├── utils/                 # 实用程序
│   ├── storage.py         # HDF5 文件 I/O
│   ├── plot_1d.py         # 1D 绘图
│   ├── plot_2d.py         # 2D 绘图（支持多指标）
│   └── paraview.py        # ParaView 导出
├── api.py                 # 高级 API
├── __init__.py            # 包初始化
└── examples/              # 示例脚本
    ├── example_1d.py
    ├── example_2d.py
    └── example_custom_metrics.py
```

## API 参考

### LossLandscape 类

用于 loss landscape 计算和可视化的主类。

**构造函数：**
```python
LossLandscape(net, dataloader, criterion=None, use_cuda=False, surf_file=None)
```

**参数：**
- `net`: PyTorch 模型
- `dataloader`: 用于评估的数据加载器
- `criterion`: Loss 函数或指标函数
  - 可以是 `nn.Module`（CrossEntropyLoss、MSELoss 等）← 标准模式
  - 可以是返回指标 `dict` 的 `callable` ← 自定义模式
  - 默认：`nn.CrossEntropyLoss()`
- `use_cuda`: 是否使用 GPU（默认：False）
- `surf_file`: HDF5 结果保存路径（默认：'loss_surface.h5'）

### 方法

#### `compute_1d()`

```python
result = landscape.compute_1d(
    directions='random',      # 或 'target'（需要 target_model）
    xrange=(-1, 1, 51),       # (min, max, num_points)
    dir_type='weights',       # 或 'states'
    normalize='filter',       # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',          # 忽略偏置和批归一化参数
    target_model=None,        # 当 directions='target' 时必需
    save=True                 # 保存到 HDF5 文件
)
```

**返回：**
- 标准模式：`{'losses': np.array([...]), 'accuracies': np.array([...]), 'xcoordinates': np.array([...])}`
- 自定义模式：`{'losses': {metric_name: np.array([...]), ...}, 'xcoordinates': np.array([...])}`

#### `compute_2d()`

```python
result = landscape.compute_2d(
    xrange=(-1, 1, 51),     # (min, max, num_points)
    yrange=(-1, 1, 51),     # (min, max, num_points)
    dir_type='weights',     # 或 'states'
    normalize='filter',     # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',        # 忽略偏置和批归一化参数
    x_target=None,          # X 方向的可选目标模型
    y_target=None,          # Y 方向的可选目标模型
    save=True               # 保存到 HDF5 文件
)
```

#### `plot_1d()`

```python
landscape.plot_1d(xmin=-1, xmax=1, loss_max=5, log=False, show=False)
```

#### `plot_2d_contour()`

```python
landscape.plot_2d_contour(
    surf_name='train_loss',  # 要绘制的指标名称，或 'auto' 绘制所有
    vmin=0.1,                # 轮廓级别的最小值
    vmax=10,                 # 轮廓级别的最大值
    vlevel=0.5,              # 轮廓级别之间的间距
    show=False               # 是否显示绘图
)
```

**输出文件：**
- `*_2dcontour.pdf`: 轮廓线
- `*_2dcontourf.pdf`: 填充轮廓
- `*_2dheat.pdf`: 热力图

#### `plot_2d_surface()`

```python
landscape.plot_2d_surface(
    surf_name='train_loss',  # 要绘制的指标名称，或 'auto' 绘制所有
    show=False               # 是否显示绘图
)
```

**输出文件：**
- `*_3dsurface.pdf`: 3D 曲面图

#### `export_paraview()`

```python
vtp_file = landscape.export_paraview(
    surf_name='train_loss',  # 要导出的曲面
    log=False,               # 使用对数刻度
    zmax=-1,                 # 剪切最大 z 值（-1：不剪切）
    interp=-1                # 插值分辨率（-1：不插值）
)
```

## 性能考虑

### 计算时间

- **1D landscape**：O(num_points) - 与采样点数线性相关
- **2D landscape**：O(num_points_x × num_points_y) - 二次方
- 每个点需要对所有数据的一次前向传播

### GPU 要求

- 支持任何具有 CUDA 支持的 NVIDIA GPU
- 如果 CUDA 不可用，自动回退到 CPU

### 估计运行时间（现代 GPU）

**对于 21×21 loss landscape 评估：**
- **H200 GPU**：约 30 分钟（441 次评估）
- **A100 GPU**：45-60 分钟
- **H100 GPU**：20-25 分钟

**影响运行时的因素：**
- 模型大小：更大的模型需要更多计算
- 网格分辨率：更高的分辨率二次增加评估计数
- 批数：更多批改进 loss 估计但增加计算
- 数据大小：更大的数据集意味着更长的 loss 评估时间

### 加快计算的提示

1. **减少网格分辨率**：测试时使用 11×11 或 15×15 而不是 21×21
2. **使用更少的批**：减少 num_batches 参数（例如 20-50 而不是 100）
3. **使用 GPU 加速**：启用 `use_cuda=True` 以获得 ~10-100 倍加速
4. **减少 batch size**：较小的 batch 适合 GPU 内存
5. **并行评估**：使用多个 GPU 和数据并行

## 输出文件

### HDF5 数据文件 (`.h5`)

**标准模式：**
```
键：
  - xcoordinates、ycoordinates：坐标
  - train_loss：训练 loss 值
  - train_acc：训练准确率
```

**自定义模式：**
```
键：
  - xcoordinates、ycoordinates：坐标
  - train_loss_metric1：指标 1 值
  - train_loss_metric2：指标 2 值
  - （每个自定义指标的更多内容）
```

### PDF 可视化文件

对于每个指标，生成 4 个文件：
```
*_2dcontour.pdf    # 轮廓线
*_2dcontourf.pdf   # 填充轮廓
*_2dheat.pdf       # 热力图
*_3dsurface.pdf    # 3D 曲面
```

### ParaView 文件 (`.vtp`)

与 ParaView 兼容的 VTK 格式用于专业渲染。

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

如果在您的研究中使用此库，请引用原始工作：

```bibtex
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```

## 许可证

MIT 许可证 - 参见原始 [loss-landscape](https://github.com/tomgoldstein/loss-landscape) 仓库

---

## 可视化结果

不同训练迭代中 Pong 和 MsPacman 环境的 Total Loss 的 loss landscape 可视化。
每行显示不同的迭代检查点，列显示不同的可视化样式。

### Pong 环境 - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

### MsPacman 环境 - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
