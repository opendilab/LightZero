# How to Set Configuration Files in LightZero

In the LightZero framework, to run a specific algorithm in a specific environment, you need to set the corresponding configuration files. The configuration files mainly consist of two parts: `main_config` and `create_config`. Among them, `main_config` defines the main parameters for running the algorithm, such as environment settings and policy settings, while `create_config` specifies the specific environment class and policy class to be used and their reference paths.

To run a specific algorithm in a custom environment, you can find the default `config` file corresponding to different algorithms `<algo>` for the existing environment `<env>` under the path `zoo/<env>/config/<env>_<algo>_config`. Then, based on this, you can mainly modify the part corresponding to `env` and then perform debugging and optimization.

Below, we use [atari_muzero_config.py](https://github.com/opendilab/LightZero/blob/main/zoo/atari/config/atari_muzero_config.py) as an example to explain the configuration file settings in detail.

## 1. `main_config`

The `main_config` dictionary contains the main parameter settings for running the algorithm, which are mainly divided into two parts: `env` and `policy`.

### 1.1 Main Parameters in the `env` Part

- `env_id`: Specifies the environment to be used.
- `obs_shape`: The dimension of the environment observation.
- `collector_env_num`: The number of parallel environments used to collect data in the experience replay collector.
- `evaluator_env_num`: The number of parallel environments used to evaluate policy performance in the evaluator.
- `n_evaluator_episode`: The number of episodes run by each environment in the evaluator.
- `manager`: Specifies the type of environment manager, mainly used to control the parallelization mode of the environment.

### 1.2 Main Parameters in the `policy` Part

- `model`: Specifies the neural network model used by the policy, including the input dimension of the model, the number of frame stacking, the action space dimension of the model output, whether the model needs to use downsampling, whether to use self-supervised learning auxiliary loss, the action encoding type, the Normalization mode used in the network, etc.
- `cuda`: Specifies whether to migrate the model to the GPU for training.
- `reanalyze_noise`: Whether to introduce noise during MCTS reanalysis, which can increase exploration.
- `env_type`: Marks the environment type faced by the MuZero algorithm. According to different environment types, the MuZero algorithm will have some differences in detail processing.
- `game_segment_length`: The length of the sequence (game segment) used for self-play.
- `random_collect_episode_num`: The number of randomly collected episodes, providing initial data for exploration.
- `eps`: Exploration control parameters, including whether to use the epsilon-greedy method for control, the update method of control parameters, the starting value, the termination value, the decay rate, etc.
- `use_augmentation`: Whether to use data augmentation.
- `update_per_collect`: The number of updates after each data collection.
- `batch_size`: The batch size sampled during the update.
- `optim_type`: Optimizer type.
- `lr_piecewise_constant_decay`: Whether to use piecewise constant learning rate decay.
- `learning_rate`: Initial learning rate.
- `num_simulations`: The number of simulations used in the MCTS algorithm.
- `reanalyze_ratio`: Reanalysis coefficient, controlling the probability of reanalysis.
- `ssl_loss_weight`: The weight of the self-supervised learning loss function.
- `n_episode`: The number of episodes run by each environment in the parallel collector.
- `eval_freq`: Policy evaluation frequency (measured by training steps).
- `replay_buffer_size`: The capacity of the experience replay buffer.

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