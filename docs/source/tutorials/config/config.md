# How to Set Configuration Files in LightZero

In the LightZero framework, to run a specific algorithm in a specific environment, you need to set the corresponding configuration files. The configuration files mainly consist of two parts: `main_config` and `create_config`. Among them, `main_config` defines the main parameters for running the algorithm, such as environment settings and policy settings, while `create_config` specifies the specific environment class and policy class to be used and their reference paths.

To run a specific algorithm in a custom environment, you can find the default `config` file corresponding to different algorithms `<algo>` for the existing environment `<env>` under the path `zoo/<env>/config/<env>_<algo>_config`. Then, based on this, you can mainly modify the part corresponding to `env` and then perform debugging and optimization.

Below, we use [atari_muzero_config.py](https://github.com/opendilab/LightZero/blob/main/zoo/atari/config/atari_muzero_config.py) as an example to explain the configuration file settings in detail.

## 1. `main_config`

The `main_config` dictionary contains the main parameter settings for running the algorithm, which are mainly divided into two parts: `env` and `policy`.

### 1.1 Main Parameters in the `env` Part

- `env_id`: Specifies the environment to be used.
- `observation_shape`: The dimension of the environment's observations.
- `collector_env_num`: The number of parallel environments used to collect data in the experience replay collector.
- `evaluator_env_num`: The number of parallel environments used to evaluate policy performance in the evaluator.
- `n_evaluator_episode`: The total number of episodes run across all environments in the evaluator.
- `collect_max_episode_steps`: The maximum number of steps allowed per episode during data collection.
- `eval_max_episode_steps`: The maximum number of steps allowed per episode during evaluation.
- `frame_stack_num`: The number of consecutive frames stacked together as input.
- `gray_scale`: Whether to use grayscale images.
- `scale`: Whether to scale the input data.
- `clip_rewards`: Whether to clip reward values.
- `episode_life`: If True, the game ends when the agent loses a life, otherwise, the game only ends when all lives are lost.
- `env_type`: The type of environment.
- `frame_skip`: The number of frames to repeat the same action.
- `stop_value`: The target score that stops the training.
- `replay_path`: Path to store the replay.
- `save_replay`: Whether to save the replay video.
- `channel_last`: Whether to put the channel dimension in the last dimension of the input data.
- `warp_frame`: Whether to crop each frame of the picture.
- `manager`: Specifies the type of environment manager, mainly used to control the parallelization mode of the environment.

### 1.2 Main Parameters in the `policy` Part

- `model`: Specifies the neural network model used by the policy.
    - `model_type`: The type of model to use.
    - `observation_shape`: The dimensions of the observation space.
    - `action_space_size`: The size of the action space.
    - `continuous_action_space`: Whether the action space is continuous.
    - `num_res_blocks`: The number of residual blocks in the model.
    - `downsample`: Whether to downsample the input.
    - `norm_type`: The type of normalization used.
    - `num_channels`: The number of channels in the convolutional layers (number of features extracted).
    - `reward_support_range`: The range of the reward support set (`(start, stop, step)`).
    - `value_support_range`: The range of the value support set (`(start, stop, step)`).
    - `bias`: Whether to use bias terms in the layers.
    - `discrete_action_encoding_type`: How discrete actions are encoded.
    - `self_supervised_learning_loss`: Whether to use a self-supervised learning loss (as in EfficientZero).
    - `image_channel`: The number of channels in the input image.
    - `frame_stack_num`: Number of frames stacked.
    - `gray_scale`: Whether to use gray images.
    - `use_sim_norm`: Whether to use SimNorm after the Latent State.
    - `use_sim_norm_kl_loss`: Whether the obs_loss corresponding to the Latent State after SimNorm uses KL divergence loss, which is often used together with SimNorm.
    - `res_connection_in_dynamics`: Whether to use the residual connection in the dynamics model.
- `learn`: Configuration for the learning process.
    - `learner`: Configuration for the learner (dictionary type), including train iterations and checkpoint saving.
    - `resume_training`: Whether to resume training.
- `collect`: Configuration for the collect process.
    - `collector`: Collector configuration (dictionary type), including type and print frequency.
- `eval`: Configuration for the evaluation process
    - `evaluator`: Evaluator configuration (dictionary type), including evaluation frequency, number of episodes to evaluate, and path to save images.
- `other`: Other configurations.
    - `replay_buffer`: Replay buffer configuration (dictionary type), including buffer size, maximum usage and staleness of experiences, and parameters for throughput control and monitoring.
- `cuda`: Whether to use CUDA (GPU) for training.
- `multi_gpu`: Whether to enable multi-GPU training.
- `use_wandb`: Whether to use Weights & Biases (wandb) for logging.
- `mcts_ctree`: Whether to use the C++ version of Monte Carlo Tree Search.
- `collector_env_num`: The number of collection environments.
- `evaluator_env_num`: The number of evaluation environments.
- `env_type`: The type of environment (board game or non-board game).
- `action_type`: The type of action space (fixed or other).
- `game_segment_length`: The length corresponding to the basic unit game segment during collection.
- `cal_dormant_ratio`: Whether to calculate the ratio of dormant neurons.
- `use_augmentation`: Whether to use data augmentation.
- `augmentation`: The data augmentation methods to use.
- `update_per_collect`: The number of model updates after each data collection phase.
- `batch_size`: The batch size used for training updates.
- `optim_type`: The type of optimizer.
- `reanalyze_ratio`: The reanalyze ratio, which controls the probability to conduct reanalyze.
- `reanalyze_noise`: Whether to introduce noise during MCTS reanalysis (for exploration).
- `reanalyze_batch_size`: Reanalyze batch size.
- `reanalyze_partition`: The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
-`random_collect_episode_num`: Number of episodes of random collection, to provide initial exploration data.
- `eps`: Parameters for exploration control, including whether to use epsilon-greedy, update schedules, start/end values, and decay rate.
- `piecewise_decay_lr_scheduler`: Whether to use piecewise constant learning rate decay.
- `learning_rate`: The initial learning rate.
- `num_simulations`: The number of simulations used in the MCTS algorithm.
- `reward_loss_weight`: Weight for the reward loss.
- `policy_loss_weight`: Weight for the policy loss.
- `value_loss_weight`: Weight for the value loss.
- `ssl_loss_weight`: The weight of the self-supervised learning loss.
- `n_episode`: The number of episodes in parallel collector.
- `eval_freq`: The frequency of policy evaluation (in terms of training steps).
- `replay_buffer_size`: The capacity of the replay buffer.
- `target_update_freq`: How often to update the target network.
- `grad_clip_value`: Value to clip gradient.
- `discount_factor`: Discount factor.
- `td_steps`: TD steps.
- `num_unroll_steps`: The number of rollout steps during MuZero training.

Two frequently changed parameter setting areas are also specially mentioned here, annotated by comments:

```python
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# These are parameters that need to be adjusted frequently based on the actual situation
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
```

to remind users that these parameters often need to be adjusted, such as `collector_env_num`, `num_simulations`, `update_per_collect`, `batch_size`, `max_env_step`, etc. Adjusting these parameters can optimize the algorithm performance and accelerate the training speed.

## 2. `create_config`

The `create_config` dictionary specifies the specific environment class and policy class to be used and their reference paths, mainly containing two parts: `env` and `policy`.

### 2.1 Settings in the `env` Part

```python
env=dict(
    type='atari_lightzero',
    import_names=['zoo.atari.envs.atari_lightzero_env'],
),
```

Here, `type` specifies the environment name to be used, and `env_name` specifies the reference path where the environment class is located. The predefined `atari_lightzero_env` is used here. If you want to use a custom environment class, you need to change `type` to the custom environment class name and modify the `import_names` parameter accordingly.

### 2.2 Settings in the `policy` Part

```python
policy=dict(
    type='muzero',
    import_names=['lzero.policy.muzero'],
),
```

Here, `type` specifies the policy name to be used, and `import_names` specifies the reference path where the policy class is located. The predefined MuZero algorithm in LightZero is used here. If you want to use a custom policy class, you need to change `type` to the custom policy class and modify the `import_names` parameter to the reference path where the custom policy is located.

## 3. Running the Algorithm

After completing the configuration, call the following in the `main` function:

```python
if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
```

This will run the MuZero algorithm on the configured environment for training. `[main_config, create_config]` specifies the configuration used for training, `seed` specifies the random number seed, and `max_env_step` specifies the maximum number of environment interaction steps.

## 4. Notes

The above briefly introduces the methods for configuring algorithms for custom environments under the LightZero framework, and hopes to be helpful to you. Please pay attention to the following points during the configuration process:

- When using a custom environment, be sure to write the environment class according to the environment interface standards defined by the LightZero framework, otherwise errors may occur.
- Different algorithms and environments require different configuration parameters. Before configuring, you need to thoroughly understand the principles of the algorithm and the characteristics of the environment, and you can refer to relevant academic papers to set parameters reasonably.
- If you want to run an algorithm supported by LightZero on a custom environment, you can first use the default policy configuration of the algorithm, and then optimize and adjust according to the actual training situation.
- When configuring the number of parallel environments, the number should be set reasonably according to your computing resources to avoid problems of insufficient memory due to too many parallel environments.
- You can use tools such as tensorboard to monitor the training situation and solve problems in time. For details, please refer to the [Log System Documentation](https://github.com/opendilab/LightZero/tree/main/docs/source/tutorials/logs/logs.md).

- Wish you a smooth experience using the LightZero framework!