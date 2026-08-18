## One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning — MoE 梯度冲突实验复现

本文档为论文 **《One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning》** 中 **MoE 梯度冲突相关实验** 的复现说明。  
论文链接：`https://arxiv.org/pdf/2509.07945`

本目录下两个主要脚本用于复现 **Figure 16–20** 相关实验：

| 配置文件 | 模型类型 | 在论文中的角色 |
| --- | --- | --- |
| `atari_unizero_nomoe_multitask_segment_ddp_config.py` | 多任务 UniZero，**无 MoE** | 作为 **“MLP / Dense Transformer” 基线** |
| `atari_unizero_moe_multitask_segment_ddp_config.py` | 使用 **MoE backbone** 的 ScaleZero 风格模型 | 作为 **“MoE Transformer” 版本** |

两份脚本都会在 Atari 多任务基准上训练，并记录复现以下图所需的统计量：  
**Figure 16, 17, 18, 19, 20**（梯度冲突、专家选择熵、专家利用热力图、任务间 Wasserstein 距离等）。

> **Figure 18 提醒**：图中纵轴为 **梯度冲突的 log 值**，画图时请先对原始冲突数值取对数（如 log10）。

### 运行示例

在 LightZero 根目录下执行：

```bash
# 非 MoE 基线
torchrun --nproc_per_node=4 zoo/atari/config/atari_unizero_nomoe_multitask_segment_ddp_config.py

# MoE 版本
torchrun --nproc_per_node=4 zoo/atari/config/atari_unizero_moe_multitask_segment_ddp_config.py
```

跑完这两个脚本后，结合日志与论文附录（尤其 E.1–E.2）的说明，即可复现 Figure 16–20。

### 参考图片

**moe_expert_selection_wasserstein_distance**

<img src="../../../assets/moe_expert_selection_wasserstein_distance.png" alt="moe_expert_selection_wasserstein_distance" width="800"/>

**moe_expert_selection_js_divergence**

<img src="../../../assets/moe_expert_selection_js_divergence.png" alt="moe_expert_selection_js_divergence" width="800"/>

**gradient_conflict_comparison_moe_vs_nomoe**

<img src="../../../assets/gradient_conflict_comparison_moe_vs_nomoe.png" alt="gradient_conflict_comparison_moe_vs_nomoe" width="800"/>

**expert_selection_heatmaps**

<img src="../../../assets/expert_selection_heatmaps.jpg" alt="expert_selection_heatmaps" width="800"/>
