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
- `obs_shape`: 环境观测的维度。
- `collector_env_num`: 经验回放采集器(collector)中并行用于收集数据的环境数目。
- `evaluator_env_num`: 评估器(evaluator)中并行用于评估策略性能的环境数目。 
- `n_evaluator_episode`: 评估器中每个环境运行的episode数目。
- `manager`: 指定环境管理器的类型，主要用于控制环境的并行化方式。

### 1.2 `policy`部分的主要参数

- `model`: 指定策略所使用的神经网络模型，包含模型的输入维度、叠帧数、模型输出的动作空间维度、模型是否需要使用降采样、是否使用自监督学习辅助损失、动作编码类型、网络中使用的Normalization模式等。
- `cuda`: 指定是否将模型迁移到GPU上进行训练。
- `reanalyze_noise`: 是否在MCTS重分析时引入噪声，可以增加探索。
- `env_type`: 标记MuZero算法所面对的环境类型，根据不同的环境类型，MuZero算法会在细节处理上有所不同。
- `game_segment_length`: 用于自我博弈的序列(game segment)长度。
- `random_collect_episode_num`: 随机采集的episode数量，为探索提供初始数据。 
- `eps`: 探索控制参数，包括是否使用epsilon-greedy方法进行控制，控制参数的更新方式、起始值、终止值、衰减速度等。
- `use_augmentation`: 是否使用数据增强。
- `update_per_collect`: 每次数据收集后更新的次数。
- `batch_size`: 更新时采样的批量大小。
- `optim_type`: 优化器类型。
- `piecewise_decay_lr_scheduler`: 是否使用分段常数学习率衰减。
- `learning_rate`: 初始学习率。
- `num_simulations`: MCTS算法中使用的模拟次数。
- `reanalyze_ratio`: 重分析系数，控制进行重分析的概率。
- `ssl_loss_weight`: 自监督学习损失函数的权重。
- `n_episode`: 并行采集器中每个环境运行的episode数量。
- `eval_freq`: 策略评估频率(按照训练步数计)。
- `replay_buffer_size`: 经验回放器的容量。

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
),
```

其中`type`指定了要使用的环境名，`env_name`则指定了该环境类所在的引用路径。这里使用的是预定义的`atari_lightzero_env`。如果要使用自定义的环境类，则需要将`type`改为自定义环境类名，并相应修改`import_names`参数。

### 2.2 `policy`部分的设置

```python
policy=dict(
    type='muzero',
    import_names=['lzero.policy.muzero'],
),
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

  