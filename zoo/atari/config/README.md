## One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning — MoE Gradient Conflict Experiment Reproduction

This README describes how to reproduce the **MoE gradient conflict experiments** from the paper **"One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning"**.  
Paper link: `https://arxiv.org/pdf/2509.07945`

The two main configs in this folder are used to reproduce **Figures 16–20**:

| Config file | Model type | Role in the paper |
| --- | --- | --- |
| `atari_unizero_nomoe_multitask_segment_ddp_config.py` | Multi-task UniZero, **no MoE** | **"MLP / dense Transformer"** baseline |
| `atari_unizero_moe_multitask_segment_ddp_config.py` | ScaleZero-style model with **MoE backbone** | **"MoE Transformer"** variant |

Both scripts train on a multi-task Atari benchmark and log the statistics needed to rebuild  
**Figure 16, 17, 18, 19, 20** (gradient conflicts, expert-selection entropy, expert usage heatmaps, task-wise Wasserstein distances, etc.).

> **Figure 18 note**: the y-axis uses **log-scaled gradient conflict values**.  
> When plotting, first apply a log transform (e.g., log10) to the raw conflict metrics.

### How to run

From the LightZero project root:

```bash
# Non-MoE baseline
torchrun --nproc_per_node=4 zoo/atari/config/atari_unizero_nomoe_multitask_segment_ddp_config.py

# MoE version
torchrun --nproc_per_node=4 zoo/atari/config/atari_unizero_moe_multitask_segment_ddp_config.py
```

After running both scripts, you can follow the descriptions in the paper appendix (especially Sections E.1–E.2) plus the logged statistics to reproduce Figures 16–20.

### Reference Figures

**moe_expert_selection_wasserstein_distance**

<img src="../../../assets/moe_expert_selection_wasserstein_distance.png" alt="moe_expert_selection_wasserstein_distance" width="800"/>

**moe_expert_selection_js_divergence**

<img src="../../../assets/moe_expert_selection_js_divergence.png" alt="moe_expert_selection_js_divergence" width="800"/>

**gradient_conflict_comparison_moe_vs_nomoe**

<img src="../../../assets/gradient_conflict_comparison_moe_vs_nomoe.png" alt="gradient_conflict_comparison_moe_vs_nomoe" width="800"/>

**expert_selection_heatmaps**

<img src="../../../assets/expert_selection_heatmaps.jpg" alt="expert_selection_heatmaps" width="800"/>
