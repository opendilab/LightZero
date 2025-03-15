# LightZero 中如何设置配置文件？

在 LightZero 框架中，针对特定环境运行特定算法需要设置相应的配置文件。
配置文件主要包含两个部分: `main_config`和`create_config`。
其中，`main_config`定义了算法运行的主要参数，如环境设置、策略设置等；而`create_config`则指定了要使用的具体环境类和策略类及其引用路径。
针对自定义环境运行特定算法，您可以在 `zoo/<env>/config/<env>_<algo>_config` 路径下找到已有环境`<env>`的不同算法`<algo>`对应的默认 `config` 文件，然后在此基础上主要修改 `env` 对应的部分，然后进行调试优化。
下面我们以 [atari_muzero_config.py](https://github.com/opendilab/LightZero/blob/main/zoo/atari/config/atari_muzero_config.py) 为例，来详细说明配置文件的设置。

## 1. `main_config`

`main_config`字典包含了算法运行的主要参数设置，主要分为两部分: `env`和`policy`。

### 1.1 `env`部分的主要参数

- `env_id`: 指定要使用的环境。
- `observation_shape`: 环境观测的维度。
- `collector_env_num`: 经验回放采集器(collector)中并行用于收集数据的环境数目。
- `evaluator_env_num`: 评估器(evaluator)中并行用于评估策略性能的环境数目。 
- `n_evaluator_episode`: 评估器中所有环境运行的总的episode数目。
- `collect_max_episode_steps`: 收集数据时单个episode 允许的最大步数。
- `eval_max_episode_steps`: 评估时单个episode允许的最大步数。
- `frame_stack_num`: 叠帧数。
- `gray_scale`: 是否使用灰度图像。
- `scale`: 是否缩放输入数据。
- `clip_rewards`:  是否裁剪奖励值。
- `episode_life`: 是否在游戏失败时重置生命值。
- `env_type`: 环境类型。
- `frame_skip`: 动作重复的帧数。
- `stop_value`: 训练停止的目标分数。
- `replay_path`: 经验回放的存储路径。
- `save_replay`: 是否存储回放视频。
- `channel_last`: 是否将channel维度放在输入数据的最后一维。
- `warp_frame`: 是否裁剪每一帧的图片。
- `manager`: 指定环境管理器的类型，主要用于控制环境的并行化方式。

### 1.2 `policy`部分的主要参数
- `model`: 指定策略所使用的神经网络模型。
    - `model_type`: 选择使用的模型类型。
    - `observation_shape`: 观测空间的维度。
    - `action_space_size`: 动作空间大小。
    - `continuous_action_space`: 动作空间是否是连续的。
    - `num_res_blocks`: 残差块的数量。
    - `downsample`: 是否进行降采样。
    - `norm_type`: 归一化使用的方法。
    - `num_channels`: 卷积层提取的特征个数。
    - `support_scale`: 价值支持集的范围 (-support_scale, support_scale)。
    - `bias`: 是否使用偏置。
    - `discrete_action_encoding_type`: 离散化动作空间使用的编码类型。
    - `self_supervised_learning_loss`: 是否使用自监督学习损失（efficient muzero）。
    - `image_channel`: 输入图像通道数。
    - `frame_stack_num`: 堆叠帧数。
    - `gray_scale`: 是否使用灰度图像。
    - `use_sim_norm`: 是否使用 SimNorm。
    - `use_sim_norm_kl_loss`: 是否使用 SimNorm 的 KL 散度损失。
    - `res_connection_in_dynamics`: 动力学模型中是否使用残差连接。
    - `world_model_cfg`: unizero中使用的世界模型配置。
      - `continuous_action_space`: 动作空间是否连续。
      - `tokens_per_block`: 每个block包含的token数量（默认为2，观察状态和动作）。
      - `max_blocks`: 最大块数量，等同于num_unroll_steps。
      - `max_tokens`: 最大token数量，结果是max_blocks与tokens_per_block的乘积。
      - `context_length`: Transformer 的上下文长度 (处理多少个时间步的信息)。
      - `gru_gating`: 是否使用 GRU 门控机制。
      - `attention`: 注意力机制类型。
      - `num_layers`: Transformer 层数。
      - `num_heads`: Transformer 注意力头数。
      - `embed_dim`: 嵌入维度。
      - `support_size`: 价值函数支持集大小。
      - `latent_recon_loss_weight`: 潜在状态重建损失权重。
      - `perceptual_loss_weight`: 感知损失权重。
      - `policy_entropy_weight`: 策略熵权重。
      - `obs_type`: 观测数据类型 (图像)。
      - `dormant_threshold`: 休眠神经元阈值。
      - `latent_recon_loss_weight`: 潜在状态重建损失权重。
      - `max_cache_size`: 最大缓存大小。
- `learn`: 学习过程配置
    - `learner`: 学习器配置（字典类型），包括训练迭代次数，检查点保存策略等信息。
    - `resume_training`: 是否恢复训练。
- `collect`: 收集过程配置
    - `collector`: 收集器配置（字典类型），包括类型和输出频率等信息。
- `eval`: 收集过程配置
    - `evaluator`: 评估器配置（字典类型），包括评估频率、评估的episode数量和图片保存路径等。
- `other`: 其它配置
    - `replay_buffer`: 经验回放器配置（字典类型），包括存储大小，经验的最大使用次数和最大陈旧度以及吞吐量控制和监控配置相关的参数。
- `cuda`: 指定是否将模型迁移到GPU上进行训练。
- `on_policy`: 是否为on-policy 算法 。
- `multi_gpu`: 是否开启多GPU训练。
- `bp_update_sync`: 是否开启bp同步更新。
- `use_wandb`: 是否使用 wandb 。
- `mcts_ctree`: 是否使用蒙特卡洛树搜索。
- `collector_env_num`: 收集环境的数量。
- `evaluator_env_num`: 评估环境的数量。
- `env_type`: 环境类型（棋盘游戏或非棋盘游戏）。
- `action_type`: 动作类型 (固定动作空间或其他)。
- `game_segment_length`: 用于自我博弈的序列(game segment)长度。
- `cal_dormant_ratio`: 是否计算休眠神经元比率。
- `use_augmentation`: 是否使用数据增强。
- `augmentation`:  数据增强方法。
- `update_per_collect`: 每次数据收集完以后模型更新的次数。
- `batch_size`: 更新时采样的批量大小。
- `optim_type`: 优化器类型。
- `reanalyze_ratio`: 重分析系数，控制进行重分析的概率。
- `reanalyze_noise`: 是否在MCTS重分析时引入噪声，可以增加探索。
- `reanalyze_batch_size`: 重分析批量大小。
- `reanalyze_partition`: 重分析的比例。
- `random_collect_episode_num`: 随机采集的episode数量，为探索提供初始数据。 
- `eps`: 探索控制参数，包括是否使用epsilon-greedy方法进行控制，控制参数的更新方式、起始值、终止值、衰减速度等。
- `piecewise_decay_lr_scheduler`: 是否使用分段常数学习率衰减。
- `learning_rate`: 初始学习率。
- `num_simulations`: MCTS算法中使用的模拟次数。
- `reward_loss_weight`: 奖励损失函数的权重。
- `policy_loss_weight`: 策略损失函数的权重。
- `value_loss_weight`: 价值损失函数的权重。
- `ssl_loss_weight`: 自监督学习损失函数的权重。
- `n_episode`: 并行采集器中所有环境运行的总episode数量。
- `eval_freq`: 策略评估频率(按照训练步数计)。
- `replay_buffer_size`: 经验回放器的容量。
- `target_update_freq`: 目标网络更新频率。
- `grad_clip_value`: 梯度裁剪值。
- `discount_factor`: 折扣因子。
- `td_steps`: TD 步数。
- `num_unroll_steps`: muzero展开的步数。



这里还特别提到了两个易变参数设定区域，通过注释

```python 
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# 这里是需要根据实际情况经常调整的参数
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

```

标注出来，提醒用户这些参数是经常需要调整的，如`collector_env_num`，`num_simulations`，`update_per_collect`，`batch_size`，`max_env_step`等。调整这些参数可以优化算法性能，加快训练速度。

## 2. `create_config`

`create_config`字典指定了要使用的具体环境类和策略类及其引用路径，主要包含两个部分: `env`和`policy`。

### 2.1 `env`部分的设置

```python
env=dict(
    type='atari_lightzero',
    import_names=['zoo.atari.envs.atari_lightzero_env'],
)
```

其中`type`指定了要使用的环境名，`env_name`则指定了该环境类所在的引用路径。这里使用的是预定义的`atari_lightzero_env`。如果要使用自定义的环境类，则需要将`type`改为自定义环境类名，并相应修改`import_names`参数。

### 2.2 `policy`部分的设置

```python
policy=dict(
    type='muzero',
    import_names=['lzero.policy.muzero'],
)
```

其中`type`指定了要使用的策略名，`import_names`则指定了该策略类所在的引用路径。这里使用的是LightZero中预定义的MuZero算法。如果要使用自定义的策略类，则需要将`type`改为自定义策略类，并修改`import_names`参数为自定义策略所在的引用路径。

## 3. 运行算法

配置完成后，在`main`函数中调用: 

```python  
if __name__ == "__main__": 
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
```

即可运行MuZero算法在配置的环境上进行训练。其中`[main_config， create_config]`指定了训练使用的配置，`seed`指定了随机数种子，`max_env_step`指定了最大的环境交互步数。

## 4. 注意事项

以上为您简要介绍了在 LightZero 框架下针对自定义环境配置算法的方法，希望对您有所帮助。在配置过程中，请注意以下几点：

- 当使用自定义环境时，请务必按照 LightZero 框架定义的环境接口标准编写环境类，否则可能引发错误。
- 不同的算法和环境需要不同的配置参数。在配置之前，您需要详细了解算法的原理及环境的特点，可以参考相关的学术论文来合理设置参数。
- 如果您希望在一个自定义环境上运行 LightZero 支持的算法，可以首先使用该算法的默认`policy`配置，随后根据训练的实际情况进行优化和调整。
- 在配置并行环境的数目时，应根据您的计算资源情况来合理设定，以避免因并行环境过多而导致显存不足的问题。
- 您可以利用 tensorboard 等工具来监控训练情况，及时发现并解决问题。具体可参考[日志系统文档](https://github.com/opendilab/LightZero/tree/main/docs/source/tutorials/logs/logs_zh.md)。

祝您使用 LightZero 框架顺利！

  