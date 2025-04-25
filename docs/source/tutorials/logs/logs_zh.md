# LightZero的日志监控体系

LightZero是一个功能强大的MCTS+强化学习框架，在模型训练过程中会生成详尽的日志文件和模型检查点。本文将深入剖析LightZero的日志监控体系，重点介绍框架运行后的文件目录结构，以及各个日志文件的内容组成。

## 1. 文件目录结构

当我们利用LightZero开展一项实验时，例如在CartPole环境中训练MuZero智能体，框架会按照如下方式组织输出文件:

```markdown
cartpole_muzero
├── ckpt
│   ├── ckpt_best.pth.tar
│   ├── iteration_0.pth.tar
│   └── iteration_10000.pth.tar
├── log  
│   ├── buffer
│   │   └── buffer_logger.txt
│   ├── collector
│   │   └── collector_logger.txt
│   ├── evaluator
│   │   └── evaluator_logger.txt
│   ├── learner
│   │   └── learner_logger.txt
│   └── serial
│       └── events.out.tfevents.1626453528.CN0014009700M.local
├── formatted_total_config.py
└── total_config.py
```

可以看到，输出文件的主体是 `log` 和 `ckpt` 两个文件夹，分别储存了详细的日志信息和模型检查点。而 `total_config.py` 和 `formatted_total_config.py` 两个文件则记录了本次实验的配置信息，关于它们的具体含义可以参考[配置系统文档](https://di-engine-docs.readthedocs.io/en/latest/03_system/config.html)。

## 2. 日志文件解析

### 采集器日志

`log/collector/collector_logger.txt` 文件记录了采集器在【本次采集阶段】与环境交互的各项指标，主要包括:

- `episode_count`: 本阶段采集的episode数量 
- `envstep_count`: 本阶段采集的环境交互步数
- `train_sample_count`: 本阶段采集的训练样本数
- `avg_envstep_per_episode`: 平均每个episode包含的环境交互步数
- `avg_sample_per_episode`: 平均每个episode包含的样本数
- `avg_envstep_per_sec`: 平均每秒钟采集的环境交互步数
- `avg_train_sample_per_sec`: 平均每秒钟采集的训练样本数  
- `avg_episode_per_sec`: 平均每秒钟采集的episode数
- `collect_time`: 本阶段数据采集总耗时
- `reward_mean`: 本阶段采集过程中获得的平均奖励
- `reward_std`: 本阶段采集奖励的标准差
- `each_reward`: 采集器每个episode与环境交互的奖励值
- `reward_max`: 本阶段采集的最大单个奖励
- `reward_min`: 本阶段采集的最小单个奖励
- `total_envstep_count`: 采集器累计采集的环境交互总步数 
- `total_train_sample_count`: 采集器累计采集的样本总数
- `total_episode_count`: 采集器累计采集的episode总数
- `total_duration`: 采集器运行的总时长

### 评估器日志

`log/evaluator/evaluator_logger.txt` 文件记录了评估器在【本次评估阶段】与环境交互的各项指标，主要包括:

- `[INFO]`: 评估器每完成一个episode的提示日志，包含最终奖励和当前的episode计数
- `train_iter`: 模型完成的训练迭代次数  
- `ckpt_name`: 本次评估所使用的模型检查点路径
- `episode_count`: 本次评估的episode数量
- `envstep_count`: 本次评估与环境交互的总步数
- `evaluate_time`: 本次评估的总耗时
- `avg_envstep_per_episode`: 平均每个评估episode包含的环境交互步数
- `avg_envstep_per_sec`: 本次评估平均每秒钟与环境交互的步数  
- `avg_time_per_episode`: 本次评估每个episode的平均耗时
- `reward_mean`: 本次评估获得的平均奖励
- `reward_std`: 本次评估奖励的标准差 
- `each_reward`: 评估器每个episode与环境交互的奖励值
- `reward_max`: 本次评估获得的最大奖励
- `reward_min`: 本次评估获得的最小奖励

### 学习器日志

`log/learner/learner_logger.txt` 文件记录了模型训练过程中学习器的各项信息，主要包括:

- 神经网络结构:描述了MuZero模型的整体架构，包括表示网络、动力学网络、预测网络等
- 学习器状态:以表格形式展示了当前的学习率、各项损失函数值、优化器监控指标等

### Tensorboard日志文件

为了便于实验管理，LightZero会将`log/serial`文件夹下的所有分散日志文件统一保存为一个tensorboard日志文件，命名格式为`events.out.tfevents.<时间戳>.<主机名>`。通过Tensorboard，用户可以实时监控训练过程的各项指标变化趋势。

## 3. 检查点文件

`ckpt`文件夹下保存了模型参数的检查点文件:

- `ckpt_best.pth.tar`:在评估中取得最佳表现的模型参数 
- `iteration_<迭代次数>.pth.tar`:在训练过程中定期保存的模型参数

如果需要加载保存的模型，可以使用`torch.load('ckpt_best.pth.tar')`等方法进行读取。

## 4. 总结

LightZero为用户提供了一套全面的日志监控体系，帮助研究者和开发者深入洞察强化学习智能体的训练全过程。
我们可以通过分析采集器、评估器、学习器的各项指标，实时掌握算法的进展和效果，并据此优化训练策略。
同时，规范的检查点文件组织方式保证了实验的可复现性。LightZero完善的日志监控体系必将成为用户进行算法研究和落地应用的得力助手。