# LightZero's Logging and Monitoring System

LightZero is a powerful MCTS and reinforcement learning framework that generates comprehensive log files and model checkpoints during the training process. In this article, we will take an in-depth look at LightZero's logging and monitoring system, focusing on the file directory structure after running the framework and the contents of each log file.

## File Directory Structure

When we conduct an experiment using LightZero, such as training a MuZero agent in the CartPole environment, the framework organizes the output files as follows:

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

As we can see, the main body of the output files consists of two folders: `log` and `ckpt`, which store detailed log information and model checkpoints, respectively. The `total_config.py` and `formatted_total_config.py` files record the configuration information for this experiment. For more details on their specific meanings, please refer to the [Configuration System Documentation](https://di-engine-docs.readthedocs.io/en/latest/03_system/config.html).

## Log File Analysis

### Collector Logs

The `log/collector/collector_logger.txt` file records various metrics of the collector's interaction with the environment during the current collection stage, including:

- `episode_count`: The number of episodes collected in this stage
- `envstep_count`: The number of environment interaction steps collected in this stage
- `avg_envstep_per_episode`: The average number of environment interaction steps per episode
- `avg_envstep_per_sec`: The average number of environment interaction steps collected per second
- `avg_episode_per_sec`: The average number of episodes collected per second
- `collect_time`: The total time spent on data collection in this stage
- `reward_mean`: The average reward obtained during the collection process in this stage
- `reward_std`: The standard deviation of rewards collected in this stage
- `reward_max`: The maximum single reward collected in this stage
- `reward_min`: The minimum single reward collected in this stage
- `total_envstep_count`: The cumulative total number of environment interaction steps collected by the collector
- `total_episode_count`: The cumulative total number of episodes collected by the collector
- `total_duration`: The total running time of the collector
- `visit_entropy`: The entropy of the visit distribution at the root node in MCTS, measuring the uniformity of node visits

### Evaluator Logs

The `log/evaluator/evaluator_logger.txt` file records various metrics of the evaluator's interaction with the environment during the current evaluation stage, including:

- `[INFO]`: Log prompts for each completed episode by the evaluator, including the final reward and current episode count
- `train_iter`: The number of completed training iterations of the model
- `ckpt_name`: The path of the model checkpoint used in this evaluation
- `episode_count`: The number of episodes in this evaluation
- `envstep_count`: The total number of environment interaction steps in this evaluation
- `evaluate_time`: The total time spent on this evaluation
- `avg_envstep_per_episode`: The average number of environment interaction steps per evaluation episode
- `avg_envstep_per_sec`: The average number of environment interaction steps per second in this evaluation
- `avg_time_per_episode`: The average time per episode in this evaluation
- `reward_mean`: The average reward obtained in this evaluation
- `reward_std`: The standard deviation of rewards in this evaluation
- `eval_episode_return`: The reward value of each episode's interaction with the environment by the evaluator
- `reward_max`: The maximum reward obtained in this evaluation
- `reward_min`: The minimum reward obtained in this evaluation
- `eval_episode_return_mean`: The average reward obtained in this evaluation

### Learner Logs

The `log/learner/learner_logger.txt` file records various information about the learner during the model training process, including:

- Neural network structure: Describes the overall architecture of the MuZero model, including the representation network, dynamics network, prediction network, etc.
- Learner status: Displays the current learning rate, loss function values, optimizer monitoring metrics, etc., in a tabular format:
    - `analysis/dormant_ratio_encoder_avg`: The average dormant ratio in the encoder, indicating inactive neurons
    - `analysis/dormant_ratio_dynamics_avg`: The average dormant ratio in the dynamics model
    - `analysis/latent_state_l2_norms_avg`: The average L2 norm of the latent state
    - `collect_mcts_temperature_avg`: The average temperature parameter of MCTS during data collection, affecting exploration
    - `cur_lr_avg`: The current learning rate
    - `weighted_total_loss_avg`: The weighted average total loss
    - `total_loss_avg`: The average total loss
    - `policy_loss_avg`: The average policy loss
    - `policy_entropy_avg`: The average policy entropy
    - `target_policy_entropy_avg`: The average target policy entropy
    - `reward_loss_avg`: The average reward loss
    - `value_loss_avg`: The average value loss
    - `consistency_loss_avg`: The average consistency loss
    - `value_priority_avg`: The average priority based on value in experience replay
    - `target_reward_avg`: The average target reward
    - `target_value_avg`: The average target value
    - `predicted_rewards_avg`: The average predicted rewards
    - `predicted_values_avg`: The average predicted values
    - `transformed_target_reward_avg`: The transformed average target reward
    - `transformed_target_value_avg`: The transformed average target value
    - `total_grad_norm_before_clip_avg`: The total gradient norm before clipping

### Tensorboard Log Files

To facilitate experiment management, LightZero saves all scattered log files in the `log/serial` folder as a single Tensorboard log file, named in the format `events.out.tfevents.<timestamp>.<hostname>`. Through Tensorboard, users can monitor the trends of various metrics during the training process in real-time.

## Checkpoint Files

The `ckpt` folder stores the checkpoint files of the model parameters:

- `ckpt_best.pth.tar`: The model parameters that achieved the best performance during evaluation
- `iteration_<iteration_number>.pth.tar`: The model parameters periodically saved during the training process

If you need to load the saved model, you can use methods like `torch.load('ckpt_best.pth.tar')` to read them.

## Conclusion

LightZero provides users with a comprehensive logging and monitoring system, helping researchers and developers gain deep insights into the entire training process of reinforcement learning agents. By analyzing the metrics of the collector, evaluator, and learner, we can grasp the progress and effectiveness of the algorithm in real-time and optimize the training strategy accordingly. At the same time, the standardized organization of checkpoint files ensures the reproducibility of experiments. LightZero's well-developed logging and monitoring system will undoubtedly become a powerful assistant for users in algorithm research and practical applications.