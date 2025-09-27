<font style="color:rgb(27, 28, 29);">The core of this version update revolves around the Mixture of Experts (MoE) architecture in multi-task reinforcement learning, introducing a powerful suite of tools for analysis and validation. Based on recent experimental research (see "MoE Experimental Analysis Summary"), we have developed features to monitor gradient conflicts and expert specialization in real-time, aiming to provide a deeper understanding of MoE's mechanisms and support its optimization.</font>

### <font style="color:rgb(27, 28, 29);">1. New Core Feature: Gradient Conflict Analysis System</font>
+ <font style="color:rgb(27, 28, 29);">Feature Introduction:</font>

<font style="color:rgb(27, 28, 29);">An advanced, distributed-training-compatible gradient conflict analysis system has been introduced. This system can compute and visualize gradient conflicts between different model components in real-time, including the encoder, MoE layers, and shared experts.</font>

+ <font style="color:rgb(27, 28, 29);">Experimental Relevance (Experiments 1 & 3):</font>

<font style="color:rgb(27, 28, 29);">This feature directly stems from the experimental findings that MoE architectures effectively mitigate gradient conflicts, with most conflicts concentrated in the shared expert. This tool allows developers to quantify this effect, monitor training stability, and provide a data-driven basis for future routing and load-balancing strategies.</font>

+ **<font style="color:rgb(27, 28, 29);">Technical Implementation:</font>**
    - **<font style="color:rgb(27, 28, 29);">Conflict Calculation Logic:</font>**<font style="color:rgb(27, 28, 29);"> Multi-level gradient conflict calculation and logging are integrated into the policy module at </font>`<font style="color:rgb(87, 91, 95);">lzero/policy/unizero_multitask.py</font>`<font style="color:rgb(27, 28, 29);">.</font>
    - **<font style="color:rgb(27, 28, 29);">Distributed Calculation & Visualization:</font>**<font style="color:rgb(27, 28, 29);"> High-efficiency functions for distributed gradient computation and heatmap generation are implemented in the utility library at </font>`<font style="color:rgb(87, 91, 95);">lzero/policy/utils.py</font>`<font style="color:rgb(27, 28, 29);">.</font>

### <font style="color:rgb(27, 28, 29);">2. New Core Feature: Expert Selection and Specialization Tracking</font>
+ <font style="color:rgb(27, 28, 29);">Feature Introduction:</font>

<font style="color:rgb(27, 28, 29);">A new module for in-depth tracking of MoE expert selection behavior has been added. This module uses multi-granularity sliding windows (from an immediate 100 steps to a long-term 100,000 steps) to track the usage frequency of experts for each task, thereby quantifying the expert specialization process.</font>

+ <font style="color:rgb(27, 28, 29);">Experimental Relevance (Experiment 2):</font>

<font style="color:rgb(27, 28, 29);">This feature is designed to validate the conclusion from Experiment 2: as training progresses, experts gradually "specialize" for specific tasks (evidenced by a decrease in expert selection entropy). It provides key insights into how tasks are automatically partitioned among different experts.</font>

+ **<font style="color:rgb(27, 28, 29);">Technical Implementation:</font>**
    - **<font style="color:rgb(27, 28, 29);">Core Statistics Module:</font>**<font style="color:rgb(27, 28, 29);"> Task-aware routing, a multi-window statistics collector, and the </font>`<font style="color:rgb(87, 91, 95);">get_expert_selection_stats</font>`<font style="color:rgb(27, 28, 29);"> data retrieval interface are implemented in </font>`<font style="color:rgb(87, 91, 95);">lzero/model/unizero_world_models/moe.py</font>`<font style="color:rgb(27, 28, 29);">.</font>

### <font style="color:rgb(27, 28, 29);">3. Architecture Refactoring and Experimental Support</font>
+ **<font style="color:rgb(27, 28, 29);">Core Architecture Enhancements:</font>**
    - **<font style="color:rgb(27, 28, 29);">Task ID Propagation:</font>**<font style="color:rgb(27, 28, 29);"> The </font>`<font style="color:rgb(87, 91, 95);">lzero/model/unizero_world_models/transformer.py</font>`<font style="color:rgb(27, 28, 29);"> and </font>`<font style="color:rgb(87, 91, 95);">world_model_multitask.py</font>`<font style="color:rgb(27, 28, 29);"> have been refactored to support the propagation of the </font>`<font style="color:rgb(87, 91, 95);">task_id</font>`<font style="color:rgb(27, 28, 29);"> throughout the entire forward pass.</font>
    - **<font style="color:rgb(27, 28, 29);">Gradient Hooks:</font>**<font style="color:rgb(27, 28, 29);"> Flexible gradient extraction hooks have been added in </font>`<font style="color:rgb(87, 91, 95);">world_model_multitask.py</font>`<font style="color:rgb(27, 28, 29);"> to provide the underlying data for the analysis systems mentioned above.</font>
+ **<font style="color:rgb(27, 28, 29);">Comprehensive Experimental Configurations:</font>**
    - **<font style="color:rgb(27, 28, 29);">Dedicated Configurations:</font>**<font style="color:rgb(27, 28, 29);"> A new set of MoE-specific configuration files, such as </font>`<font style="color:rgb(87, 91, 95);">atari_unizero_multitask_segment_ddp_config_moe.py</font>`<font style="color:rgb(27, 28, 29);">, has been added to the </font>`<font style="color:rgb(87, 91, 95);">zoo/atari/config/</font>`<font style="color:rgb(27, 28, 29);"> directory to facilitate comparative experiments.</font>
+ **<font style="color:rgb(27, 28, 29);">Performance and Debugging:</font>**
    - **<font style="color:rgb(27, 28, 29);">Performance Profiling:</font>**<font style="color:rgb(27, 28, 29);"> The </font>`<font style="color:rgb(87, 91, 95);">LineProfiler</font>`<font style="color:rgb(27, 28, 29);"> tool has been integrated into </font>`<font style="color:rgb(87, 91, 95);">lzero/policy/unizero_multitask.py</font>`<font style="color:rgb(27, 28, 29);">.</font>
    - **<font style="color:rgb(27, 28, 29);">Entry Points & Utilities:</font>**<font style="color:rgb(27, 28, 29);"> Corresponding modifications have been made in </font>`<font style="color:rgb(87, 91, 95);">lzero/entry/train_unizero_multitask_segment_ddp.py</font>`<font style="color:rgb(27, 28, 29);"> and </font>`<font style="color:rgb(87, 91, 95);">lzero/entry/utils.py</font>`<font style="color:rgb(27, 28, 29);"> to support the new features and configurations.</font>

# <font style="color:rgb(27, 28, 29);">SExperimental Analysis for Mixture-of-Experts (MoE)</font>
<font style="color:rgb(27, 28, 29);">This document summarizes the experimental setup and key findings from the analysis of Mixture-of-Experts (MoE) architectures in multitask reinforcement learning. The goal is to understand the mechanisms behind MoE's strong performance.</font>

### <font style="color:rgb(27, 28, 29);">Experiment 1: Analyzing Gradient Conflicts in MoE-based Transformers</font>
**<font style="color:rgb(27, 28, 29);">Experimental Setup:</font>**

+ **<font style="color:rgb(27, 28, 29);">Task Domain:</font>**<font style="color:rgb(27, 28, 29);"> Atari-8.</font>
+ **<font style="color:rgb(27, 28, 29);">Architectures Compared:</font>**
    1. **<font style="color:rgb(27, 28, 29);">Naive Transformer:</font>**<font style="color:rgb(27, 28, 29);"> A backbone with four standard Transformer blocks.</font>
    2. **<font style="color:rgb(27, 28, 29);">MoE-based Transformer:</font>**<font style="color:rgb(27, 28, 29);"> A backbone of four Transformer blocks where each MLP layer is replaced by an MoE layer (consisting of one shared expert and eight non-shared experts).</font>
+ **<font style="color:rgb(27, 28, 29);">Measurement:</font>**<font style="color:rgb(27, 28, 29);"> Gradient conflict between tasks is quantified using the maximum negative cosine similarity.</font>

![](https://cdn.nlark.com/yuque/0/2025/png/22947362/1758900605706-2f47ce39-1eb5-471c-b2aa-9fe98cd6769c.png)

+ **<font style="color:rgb(27, 28, 29);">Analysis Points:</font>**<font style="color:rgb(27, 28, 29);"> Gradient conflicts were measured at three key locations:</font>
    1. <font style="color:rgb(27, 28, 29);">The input right before the MoE layer.</font>
    2. <font style="color:rgb(27, 28, 29);">The output of the encoder.</font>
    3. <font style="color:rgb(27, 28, 29);">The parameters within the MoE layer itself (shared expert, non-shared experts, and the entire layer).</font>

**<font style="color:rgb(27, 28, 29);">Main Conclusion (Observation 1):</font>**

<font style="color:rgb(27, 28, 29);">The primary finding is that the MoE-based Transformer demonstrates significantly fewer gradient conflicts at the MoE layer and its input compared to the standard Transformer with MLP layers. This suggests that the MoE architecture helps mitigate gradient conflicts not just within its own layer but also in other connected components. Conflict levels at the encoder output were comparable for both models, likely because the encoder learns general representations that inherently have fewer conflicts.</font>

![](https://cdn.nlark.com/yuque/0/2025/png/22947362/1758900622719-5b0f776e-8aff-4425-8087-19696ac514a3.png)

### <font style="color:rgb(27, 28, 29);">Experiment 2: Investigating MoE Gating Mechanisms</font>
**<font style="color:rgb(27, 28, 29);">Experimental Setup:</font>**

+ **<font style="color:rgb(27, 28, 29);">Objective:</font>**<font style="color:rgb(27, 28, 29);"> To determine if MoE experts effectively differentiate and specialize when dealing with non-stationary data from agent-environment interactions in RL.</font>
+ **<font style="color:rgb(27, 28, 29);">Metrics:</font>**
    1. **<font style="color:rgb(27, 28, 29);">Expert Selection Entropy:</font>**<font style="color:rgb(27, 28, 29);"> Measures the uncertainty in expert choice for a given task. Lower entropy indicates higher specialization.</font>

![](https://cdn.nlark.com/yuque/0/2025/png/22947362/1758900647827-9bdf07f5-bfea-4ae2-b728-6a053ae3c7da.png)

    2. **<font style="color:rgb(27, 28, 29);">Wasserstein Distance:</font>**<font style="color:rgb(27, 28, 29);"> Measures the similarity between the expert selection distributions of different tasks.</font>
+ **<font style="color:rgb(27, 28, 29);">Procedure:</font>**<font style="color:rgb(27, 28, 29);"> Data on expert choices was collected over time windows of different sizes (</font>_<font style="color:rgb(27, 28, 29);">immediate</font>_<font style="color:rgb(27, 28, 29);"> = 100 steps, </font>_<font style="color:rgb(27, 28, 29);">short</font>_<font style="color:rgb(27, 28, 29);"> = 1,000 steps) to form probability distributions for analysis.</font>

**<font style="color:rgb(27, 28, 29);">Main Conclusion (Observation 2):</font>**

<font style="color:rgb(27, 28, 29);">The key observation from this experiment is that as training progresses, the entropy of the expert selection distribution for tasks gradually decreases. This indicates that the selection of experts becomes more certain and concentrated on a smaller subset over time, demonstrating a clear pattern of expert specialization and differentiation in the multitask setting.</font>

![](https://cdn.nlark.com/yuque/0/2025/png/22947362/1758900661959-e19e904f-f1e3-4832-aa06-2ecf60d6e2b5.png)

### <font style="color:rgb(27, 28, 29);">Experiment 3: Analyzing Gradient Conflicts Between Shared and Non-Shared Experts</font>
**<font style="color:rgb(27, 28, 29);">Experimental Setup:</font>**

+ **<font style="color:rgb(27, 28, 29);">Objective:</font>**<font style="color:rgb(27, 28, 29);"> To further analyze the source of gradient dynamics within the MoE architecture by comparing conflicts between shared and non-shared experts.</font>
+ **<font style="color:rgb(27, 28, 29);">Method:</font>**<font style="color:rgb(27, 28, 29);"> The MoE-based Transformer was used to measure and compare the gradient conflicts experienced by the shared expert versus the eight individual non-shared experts.</font>

**<font style="color:rgb(27, 28, 29);">Main Conclusion (Observation 3):</font>**

<font style="color:rgb(27, 28, 29);">The results show that the shared expert bears a significantly higher level of gradient conflict compared to any of the non-shared, task-specific experts. In fact, most of the gradient conflicts within the entire MoE layer are concentrated on this shared component, while individual experts experience almost no conflict. This is attributed to the gating mechanism, which routes different tasks to different non-shared experts, leading to consistent gradient updates for each. In contrast, the shared expert must handle all tasks simultaneously, causing conflicting updates. Therefore, the introduction of non-shared experts is a key factor in reducing the overall gradient conflict of the MoE layer.</font>

![](https://cdn.nlark.com/yuque/0/2025/png/22947362/1758900675792-d0ee1bd7-5fba-4ee5-ad6d-c0d719c51823.png)

