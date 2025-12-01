# Loss Landscape 核心库

一个简洁、模块化的 PyTorch 库，用于神经网络 loss landscape 的可视化。

从原始 [loss-landscape](https://github.com/tomgoldstein/loss-landscape) 项目提取并重构，包含以下改进：
- 针对单 GPU 使用的简化 API
- 移除 MPI 依赖
- 模块化架构，易于集成
- **自定义指标函数支持** - 一次计算多个指标
- **自动检测和多指标可视化**
- 完整支持 1D 曲线、2D 轮廓、3D 曲面和 ParaView 导出
- 与标准 PyTorch loss 完全向后兼容

**目录**
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细示例](#详细示例)
- [自定义指标指南](#自定义指标指南)
- [API 参考](#api-参考)
- [关键概念](#关键概念)
- [高级用法](#高级用法)
- [更好结果的提示](#更好结果的提示)
- [性能和运行时考虑](#性能和运行时考虑)

## 特性

✨ **主要特性**:

- ✅ **简化的 API** - 易于在单 GPU 机器上使用
- ✅ **1D/2D Landscape** - 计算 loss 曲线和 2D 曲面
- ✅ **多种可视化类型**
  - 轮廓图（线条和填充）
  - 热力图和颜色条
  - 3D 曲面图
  - ParaView 兼容的 VTP 导出
- ✅ **自定义指标函数**
  - 定义您自己的指标函数
  - 同时计算多个 loss 值和指标
  - 返回指标字典
  - 自动存储和可视化
- ✅ **自动指标检测**
  - 自动检测所有计算的指标
  - 为每个指标生成单独的可视化
  - 使用 `surf_name='auto'` 绘制所有内容
- ✅ **灵活的 Criterion 类型**
  - 标准 PyTorch loss 模块（CrossEntropyLoss、MSELoss 等）
  - 返回指标字典的自定义可调用函数
- ✅ **类型安全的自动检测**
  - 自动区分 PyTorch loss 和自定义函数
  - 无需额外标志或配置
- ✅ **完全向后兼容**
  - 所有现有代码继续工作
  - 现有的 HDF5 文件仍然可读
  - 绘图函数在两种模式下都工作

## 安装

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## 快速开始

### 2D Loss 曲面

```python
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

**示例 1：1D Loss 曲线**
```bash
cd examples
python example_1d.py
```

**示例 2：2D Loss 曲面与 ParaView**
```bash
python example_2d.py
```

**示例 3：批量处理多个检查点**

使用提供的批量处理脚本：
```bash
# 基础用法（处理 10K 到 100K 的检查点）
bash run_loss_landscape_batch.sh --ckpt-dir ./path/to/checkpoints

# 自定义环境和迭代版本
bash run_loss_landscape_batch.sh \
    --ckpt-dir ./checkpoints \
    --env PongNoFrameskip-v4 \
    --iterations 10000,50000,100000 \
    --log-dir ./results
```

详见 [API 参考](#api-参考) 部分获取参数的详细说明。

## 详细示例

### 1D Loss 曲线（标准 Loss）

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lzero.loss_landscape import LossLandscape

# 设置模型和数据
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

# 创建带有标准 PyTorch loss 的 landscape
landscape = LossLandscape(model, dataloader,
                         criterion=nn.CrossEntropyLoss(),
                         use_cuda=True)

# 计算 1D landscape
result = landscape.compute_1d(
    directions='random',
    xrange=(-1, 1, 51),
    normalize='filter',
    ignore='biasbn'
)

# 可视化
landscape.plot_1d(loss_max=5, show=True)
```

### 2D Loss 曲面

```python
# 计算 2D landscape
result = landscape.compute_2d(
    xrange=(-1, 1, 51),
    yrange=(-1, 1, 51),
    normalize='filter'
)

# 绘制轮廓和 3D 曲面
landscape.plot_2d_contour(vmin=0.1, vmax=10, vlevel=0.5)
landscape.plot_2d_surface(show=True)
```

### 自定义指标

同时计算多个自定义指标：

```python
import torch
import torch.nn as nn
from lzero.loss_landscape import LossLandscape

def compute_custom_metrics(net, dataloader, use_cuda):
    """
    计算多个指标的自定义指标函数。

    参数：
        net: PyTorch 模型
        dataloader: 数据加载器
        use_cuda: 是否使用 GPU

    返回：
        包含指标值的字典
    """
    net.eval()
    device = 'cuda' if use_cuda else 'cpu'

    total_ce = 0.0
    total_smooth_l1 = 0.0
    total_correct = 0
    total_samples = 0

    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # 计算不同的 loss 函数
            ce_loss = criterion_ce(outputs, targets)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.size(1)).float()
            smooth_l1_loss = criterion_smooth_l1(outputs, targets_onehot)

            # 累积指标
            total_ce += ce_loss.item() * inputs.size(0)
            total_smooth_l1 += smooth_l1_loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

    # 以字典形式返回多个指标
    return {
        'ce_loss': total_ce / total_samples,
        'smooth_l1_loss': total_smooth_l1 / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
    }

# 与 LossLandscape 一起使用
landscape = LossLandscape(model, dataloader,
                         criterion=compute_custom_metrics,  # 传递函数
                         use_cuda=True)

# 使用自定义指标计算 2D landscape
result = landscape.compute_2d(
    xrange=(-1, 1, 11),
    yrange=(-1, 1, 11)
)

# result['losses'] 现在是一个字典：
# {
#     'ce_loss': shape (11, 11),
#     'smooth_l1_loss': shape (11, 11),
#     'accuracy': shape (11, 11)
# }

# 自动绘制所有指标
landscape.plot_2d_contour(surf_name='auto', vmin=0.1, vmax=10)
landscape.plot_2d_surface(surf_name='auto')

# 或绘制特定指标
landscape.plot_2d_contour(surf_name='train_loss_ce_loss', vmin=0.1, vmax=5)
```

### 导出到 ParaView（高质量渲染）

```python
# 导出 2D 曲面用于 ParaView 渲染
vtp_file = landscape.export_paraview(surf_name='train_loss_ce_loss',
                                     log=False, zmax=10, interp=-1)
```

然后用 [ParaView](https://www.paraview.org/) 打开 `.vtp` 文件进行专业可视化。

## 自定义指标指南

### 什么是自定义指标？

自定义指标函数允许您：
1. **计算多个 loss 值** - 同时比较不同的 loss 函数
2. **跟踪各种指标** - 准确率、F1-score、精度、召回率等
3. **联合可视化** - 查看不同指标在权重空间中的变化
4. **分析权衡** - 理解不同目标之间的关系

### 工作原理

**类型检测机制：**

库自动检测 criterion 类型：

```python
# 标准 PyTorch loss (nn.Module) → 标准模式
criterion = nn.CrossEntropyLoss()

# 自定义指标函数（可调用，非 nn.Module）→ 自定义模式
def my_metrics(net, dataloader, use_cuda):
    return {'metric1': value1, 'metric2': value2}
```

**存储格式：**

- **标准模式**：保存为 `'train_loss'`、`'train_acc'`（向后兼容）
- **自定义模式**：保存为 `'train_loss_metric_name'`、`'train_loss_metric2'` 等

**自动绘制：**

```python
# 绘制所有检测到的指标
landscape.plot_2d_contour(surf_name='auto')  # 检测所有 train_loss_* 键

# 绘制特定指标
landscape.plot_2d_contour(surf_name='train_loss_f1_score')

# 默认：绘制标准 loss（向后兼容）
landscape.plot_2d_contour()  # 绘制 'train_loss'
```

### 函数签名

自定义指标函数必须遵循以下签名：

```python
def my_metrics(net: nn.Module,
               dataloader: DataLoader,
               use_cuda: bool) -> dict:
    """
    为给定的数据加载器上的模型计算指标。

    参数：
        net: PyTorch 模型（eval 模式）
        dataloader: 用于评估的数据加载器
        use_cuda: 模型是否在 GPU 上

    返回：
        将指标名称映射到值的字典：
        {'metric_name': float_value, ...}
    """
    pass
```

### 完整示例

查看 `example_custom_metrics.py` 获取完整的工作示例：
- 标准 loss 模式（向后兼容）
- 自定义指标的 1D landscape
- 自定义指标的 2D landscape
- 所有指标的自动绘制

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

**方法：**

#### `compute_1d()`
计算 1D loss landscape。

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

**标准模式**（nn.Module criterion）：
```python
{
    'losses': np.array([...]),      # 1D loss 值数组
    'accuracies': np.array([...]),  # 1D 准确率数组
    'xcoordinates': np.array([...]) # X 坐标
}
```

**自定义模式**（callable criterion）：
```python
{
    'losses': {
        'metric1': np.array([...]),
        'metric2': np.array([...]),
        ...
    },
    'xcoordinates': np.array([...])
}
```

#### `compute_2d()`
计算 2D loss landscape。

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

**返回：**

**标准模式：**
```python
{
    'losses': np.array([...]).shape(nx, ny),      # 2D 数组
    'accuracies': np.array([...]).shape(nx, ny),  # 2D 数组
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

**自定义模式：**
```python
{
    'losses': {
        'metric1': np.array([...]).shape(nx, ny),
        'metric2': np.array([...]).shape(nx, ny),
        ...
    },
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

#### `plot_1d()`
可视化 1D landscape。

```python
landscape.plot_1d(xmin=-1, xmax=1, loss_max=5, log=False, show=False)
```

#### `plot_2d_contour()`
绘制 2D 轮廓可视化。

```python
landscape.plot_2d_contour(
    surf_name='train_loss',  # 要绘制的指标名称
                             # 使用 'auto' 绘制所有检测到的指标
    vmin=0.1,                # 轮廓级别的最小值
    vmax=10,                 # 轮廓级别的最大值
    vlevel=0.5,              # 轮廓级别之间的间距
    show=False               # 是否显示绘图
)
```

**参数：**
- `surf_name`:
  - `'train_loss'`（默认）：绘制标准 loss（向后兼容）
  - `'auto'`：自动检测并绘制所有指标（新增！）
  - `'train_loss_metric_name'`：绘制特定的自定义指标

**输出文件：**
- `*_2dcontour.pdf`: 轮廓线
- `*_2dcontourf.pdf`: 填充轮廓
- `*_2dheat.pdf`: 热力图

#### `plot_2d_surface()`
绘制 3D 曲面可视化。

```python
landscape.plot_2d_surface(
    surf_name='train_loss',  # 要绘制的指标名称
                             # 使用 'auto' 绘制所有检测到的指标
    show=False               # 是否显示绘图
)
```

**输出文件：**
- `*_3dsurface.pdf`: 3D 曲面图

#### `export_paraview()`
导出到 ParaView VTP 格式。

```python
vtp_file = landscape.export_paraview(
    surf_name='train_loss',  # 要导出的曲面
    log=False,               # 使用对数刻度
    zmax=-1,                 # 剪切最大 z 值（-1：不剪切）
    interp=-1                # 插值分辨率（-1：不插值）
)
```

## 关键概念

### 标准模式 vs 自定义模式

| 方面 | 标准模式 | 自定义模式 |
|------|---------|---------|
| Criterion | `nn.Module`（loss 函数） | `callable` 函数 |
| 返回值 | 单个 loss + 准确率 | 指标字典 |
| 示例 | `nn.CrossEntropyLoss()` | `def my_metrics(...)` |
| 输出键 | `'train_loss'`、`'train_acc'` | `'train_loss_metric1'`、`'train_loss_metric2'` |
| 绘图 | 固定为 `'train_loss'` | 自动检测所有指标 |
| 向后兼容 | ✅ 是（原始行为） | ✅ 是（新增） |

### 方向类型

- **`weights`**: 权重空间中的方向（包括所有参数）
- **`states`**: state_dict 空间中的方向（包括 BN 运行统计）

对于一般用途分析使用 `weights`，分析 BatchNorm 层时使用 `states`。

### 归一化方法

- **`filter`**: 在 filter 级别归一化（推荐）
  - 每个 filter 的范数与原始权重中相同
  - 适用于卷积网络
- **`layer`**: 在层级别归一化
- **`weight`**: 按权重幅度缩放
- **`dfilter`**: 每个 filter 的单位范数
- **`dlayer`**: 每层的单位范数

### 忽略选项

- **`biasbn`**: 忽略偏置和批归一化参数
  - 将其方向分量设置为零
  - 推荐用于大多数分析

## 模块结构

```
loss_landscape_core/
├── core/                  # 核心功能
│   ├── direction.py       # 方向生成和归一化
│   ├── evaluator.py       # Loss 和准确率评估
│   └── perturbation.py    # 权重扰动
├── utils/                 # 实用程序
│   ├── storage.py         # HDF5 文件 I/O
│   └── projection.py      # 投影和角度计算
├── viz/                   # 可视化
│   ├── plot_1d.py         # 1D 绘图
│   ├── plot_2d.py         # 2D 绘图（支持多指标）
│   └── paraview.py        # ParaView 导出
├── api.py                 # 高级 API（支持自定义指标）
├── __init__.py            # 包初始化
└── README.md              # 本文件
```

## 高级用法

### 使用目标方向（模型之间的插值）

在两个训练模型之间可视化 loss landscape：

```python
# 使用不同超参数训练两个模型
model1 = train_model(lr=0.1, batch_size=128)
model2 = train_model(lr=0.01, batch_size=256)

# 创建 landscape
landscape = LossLandscape(model1, dataloader)

# 计算从 model1 到 model2 的 landscape
result = landscape.compute_1d(
    directions='target',
    target_model=model2,
    xrange=(0, 1, 51)  # 0 = model1, 1 = model2
)
```

有关更全面的多指标示例，请参阅 [详细示例](#详细示例) 部分。

### 保存和加载曲面

计算的 loss 曲面自动保存为 HDF5 文件：

```python
# 加载之前计算的曲面
import h5py

f = h5py.File('loss_surface.h5', 'r')
print(f.keys())

# 标准模式键：['xcoordinates', 'ycoordinates', 'train_loss', 'train_acc']
# 自定义模式键：['xcoordinates', 'ycoordinates', 'train_loss_metric1', 'train_loss_metric2', ...]

losses = f['train_loss'][:]
f.close()
```

### 直接使用低级函数

为了获得更多控制权，直接使用核心模块：

```python
from lzero.loss_landscape.core import direction, evaluator, perturbation
from lzero.loss_landscape import utils

# 创建方向
d = direction.create_random_direction(model, dir_type='weights')

# 扰动权重
original_weights = direction.get_weights(model)
perturbation.set_weights(model, original_weights, [d], step=0.5)

# 评估
loss, acc = evaluator.eval_loss(model, criterion, dataloader, use_cuda=True)

# 恢复
perturbation.set_weights(model, original_weights)
```

## 输出文件

库生成以下输出文件：

### HDF5 数据文件 (`.h5`)

**标准模式**（nn.Module criterion）：
```
键：
  - xcoordinates、ycoordinates：坐标
  - train_loss：训练 loss 值
  - train_acc：训练准确率
```

**自定义模式**（callable criterion）：
```
键：
  - xcoordinates、ycoordinates：坐标
  - train_loss_ce_loss：CE loss 值
  - train_loss_smooth_l1：平滑 L1 loss 值
  - train_loss_accuracy：准确率值
  - （每个自定义指标的更多内容）
```

### PDF 可视化文件

对于每个指标，生成 4 个文件：

```
*_train_loss_metricname_2dcontour.pdf    # 轮廓线
*_train_loss_metricname_2dcontourf.pdf   # 填充轮廓
*_train_loss_metricname_2dheat.pdf       # 热力图
*_train_loss_metricname_3dsurface.pdf    # 3D 曲面
```

**3 个指标的示例：**
```
共 8 个文件：
  - 指标 1 的 4 个文件
  - 指标 2 的 4 个文件
  - 指标 3 的 4 个文件
```

### ParaView 文件 (`.vtp`)

与 ParaView 兼容的 VTK 格式用于专业渲染：
```
loss_surface.h5_train_loss_ce_loss.vtp
loss_surface.h5_train_loss_accuracy.vtp
```

## 示例脚本

提供了两个示例脚本：

### 1. `example_custom_metrics.py`

演示：
- 标准 loss 模式（向后兼容）
- 自定义指标的 1D landscape
- 自定义指标的 2D landscape
- 所有指标的自动绘制

运行：
```bash
python example_custom_metrics.py
```

### 2. `test_2d_landscape_fast_multi_metrics.py`

快速 2D landscape 演示，包含优化：
- ResNet56 on CIFAR-10（数据子集 1/10）
- 11×11 网格（vs 标准 21×21）
- 3 个指标：CE Loss、平滑 L1 Loss、准确率
- ~2-3 分钟运行时间
- 自动生成 12 个可视化

运行：
```bash
python test_2d_landscape_fast_multi_metrics.py
```

## 更好结果的提示

1. **使用归一化数据**：确保您的数据加载器使用归一化数据（对获得有意义的 loss 值很重要）

2. **充分采样**：每个维度至少使用 51 个点（2D 为 51×51）以捕获曲面特征

3. **适当的 loss 范围**：在轮廓图中调整 `vmin`/`vmax` 以突出有趣的特征

4. **对数刻度**：在 ParaView 导出中对动态范围大的 landscape 使用 `log=True`

5. **分辨率**：对于出版质量的图表，导出到 ParaView 并以更高分辨率渲染

6. **自定义指标**：为获得最佳效果，请使用自定义指标：
   - 确保所有数据点的指标计算一致
   - 谨慎使用批归一化（考虑 `dir_type='states'`）
   - 计算指标前归一化输出

## 性能和运行时考虑

### 计算时间

- **1D landscape**：O(num_points) - 与采样点数线性相关
- **2D landscape**：O(num_points_x × num_points_y) - 二次方
- 每个点需要对所有数据的一次前向传播

### 内存使用

- 存储 2D landscape：~4 MB 每个指标（21×21 float32 数组）
- 多个指标线性增加
- GPU 内存需求：~2× 模型 + 1× batch size

### GPU 要求

- 支持任何具有 CUDA 支持的 NVIDIA GPU
- 如果 CUDA 不可用，自动回退到 CPU

### UniZero 模型的计算需求

对于大型模型，loss landscape 计算计算密集。以下是现代 GPU 上的估计运行时间：

**对于使用 UniZero 模型评估 checkpoint：**
- **H200 GPU**：21×21 loss landscape 约 30 分钟（441 次评估）
- **A100 GPU**：45-60 分钟
- **H100 GPU**：20-25 分钟
- **多 GPU**：计算是按 GPU；为分布式评估设置数据并行

**影响运行时的因素：**
- 模型大小：更大的模型需要更多计算
- 网格分辨率：更高的分辨率（如 51×51）二次增加评估计数
- 批数：更多批改进 loss 估计但增加计算
- 数据大小：更大的数据集意味着更长的 loss 评估时间

### 加快计算的提示

1. **减少网格分辨率**：测试时使用 11×11 或 15×15 而不是 21×21
2. **使用更少的批**：减少 `num_batches` 参数（例如 20-50 而不是 100）
3. **使用 GPU 加速**：启用 `use_cuda=True` 以获得 ~10-100 倍加速
4. **减少 batch size**：较小的 batch 适合 GPU 内存但可能需要更长的计算时间
5. **并行评估**：使用多个 GPU 和数据并行

## 故障排除

### CUDA 内存不足

- 减少 DataLoader 中的 batch size
- 使用更小的数据集子集
- 使用 `use_cuda=False` 用于 CPU 模式

### 动态范围大

在 ParaView 导出中使用 `log=True` 以对数刻度可视化

### 缺少指标

确保自定义指标函数始终返回相同的键：

```python
def my_metrics(net, dataloader, use_cuda):
    # 必须每次都返回具有相同键的 dict
    return {
        'loss': loss_val,
        'accuracy': acc_val
    }  # 每次都相同的键
```

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

MIT 许可证 - 参见原始 [loss-landscape](https://github.com/zingyi-li/Loss-Surfaces) 仓库

## 参考资料

- 原始论文：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- 原始代码：https://github.com/zingyi-li/Loss-Surfaces
- PyTorch 文档：https://pytorch.org/docs/
- ParaView：https://www.paraview.org/

## 最新更新

### 最近的添加

- **自定义指标支持**：使用自定义函数定义和计算多个指标
- **自动检测**：自动检测并绘制所有计算的指标
- **类型安全的 API**：智能 criterion 类型检测（无需标志）
- **增强的绘图**：`plot_2d_contour()` 和 `plot_2d_surface()` 现在支持 `surf_name` 参数
- **更好的文档**：全面的示例和 API 参考

### 向后兼容性

所有更改都是完全向后兼容的：
- 使用 `nn.CrossEntropyLoss()` 的现有代码继续工作
- 默认行为未改变
- 新功能是可选的

### UniZero Loss Landscape 示例命令

```bash
# 典型 H200 设置用于完整 landscape 计算
python train_unizero_with_loss_landscape.py \
    --env PongNoFrameskip-v4 \
    --seed 0 \
    --ckpt /path/to/checkpoint.pth.tar
# 预期运行时：~30 分钟
```

## 支持

如有问题、疑问或建议：
1. 查看示例脚本
2. 查看 API 参考
3. 参考原始论文和代码
