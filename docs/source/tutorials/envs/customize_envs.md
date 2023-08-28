# **How to Customize Your Environments in LightZero?**

When conducting reinforcement learning research or applications with LightZero, you may need to create a custom environment. Creating a custom environment can better adapt to specific problems or tasks, allowing the reinforcement learning algorithms to be effectively trained in those specific environments.

For a typical environment in LightZero, please refer to `atari_lightzero_env.py`. The environment design of LightZero is largely based on the BaseEnv class in DI-engine. When creating a custom environment, we follow similar basic steps as in [DI-engine](https://di-engine-docs.readthedocs.io/en/latest/04_best_practice/ding_env.html).

## Major Differences from BaseEnv

To make the algorithms in LightZero compatible with both board game environments and non-board game environments, the `LightZeroEnv` class needs to support both types of environments. Board game environments, due to the alternating actions of players and changing legal actions, require the inclusion of information such as `action_mask` and `to_play` in the environment's `obs`. For compatibility, for non-board game environments, LightZero also requires that the 'obs' returned by the environment contain corresponding `action_mask`, `to_play`, etc.

The differences between LightZero environments and DI-engine environments mainly reflect in the following points:

- In the `reset()` method, `LightZeroEnv` returns a dictionary `lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}`.
  - For non-board game environments
    - Regarding the setting of `to_play`: Since non-board game environments generally only have one player, `to_play` is set to `-1`. (In our algorithm, we judge whether to execute the single player algorithm logic (`to_play=-1`), or the multiple player algorithm logic (`to_play=N`) based on this value.)
    - Regarding the setting of `action_mask`:
      - Discrete action space: `action_mask= np.ones(self.env.action_space.n, 'int8')` is a numpy array of ones, indicating that all actions are legal actions.
      - Continuous action space: `action_mask= None`, the special `None` indicates that the environment is a continuous action space.
- In the `step()` method, `BaseEnvTimestep(lightzero_obs_dict, rew, done, info)` is returned, where `lightzero_obs_dict` contains the updated observation results.

## Basic Steps

Here are the basic steps to create a custom LightZero environment:

1. **Create the Environment Class**<br>
   First, you need to create a new environment class that inherits from the `BaseEnv` class in DI-engine. For example:

```python
from ding.envs import BaseEnv
```

2. **Initialization Method**<br>
   In your custom environment class, you need to define an initialization method `__init__`. In this method, you need to set some basic properties of the environment, such as observation space, action space, reward space, etc. For example:

```python
def __init__(self, cfg=None):
    self.cfg = cfg
    self._init_flag = False
    # set other properties...
```

3. **Reset Method**<br>
   The reset method is used to reset the environment to an initial state. This method should return the initial observation of the environment. For example:

```python
def reset(self):
    # reset the environment...
    obs = self._env.reset()
    # get the action_mask according to the legal action
    ...
    lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
    return lightzero_obs_dict
```

4. **Step Method**<br>
   The step method takes an action as input, executes this action, and returns a tuple containing the new observation, reward, whether it's done, and other information. For example:

```python
def step(self, action):
  # The core original env step.
    obs, rew, done, info = self.env.step(action)
    
    if self.cfg.continuous:
        action_mask = None
    else:
        # get the action_mask according to the legal action
        action_mask = np.ones(self.env.action_space.n, 'int8')
    
    lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
    
    self._eval_episode_return += rew
    if done:
        info['eval_episode_return'] = self._eval_episode_return
    
    return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)
```

5. **Observation Space and Action Space**<br>
   In a custom environment, you need to provide properties for observation space and action space. These properties are `gym.Space` objects that describe the shape and type of observations and actions. For example:

```python
@property
defobservation_space(self):
    return self.env.observation_space

@property
def action_space(self):
    return self.env.action_space
```

6. **Other Methods**<br>
   Depending on the requirement, you might also need to define other methods, such as `render` (for displaying the current state of the environment), `close` (for closing the environment and performing cleanup), etc.

7. **Register the Environment**<br>
   Lastly, you need to use the `ENV_REGISTRY.register` decorator to register your new environment so that it can be used in the configuration file. For example:

```python
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('my_custom_env')
class MyCustomEnv(BaseEnv):
    # ...
```

Creating a custom environment may require a deep understanding of the specific task and reinforcement learning. When implementing a custom environment, you may need to experiment and adjust to make the environment effectively support reinforcement learning training.

## **LightZeroEnvWrapper**

- We provide a `LightZeroEnvWrapper` in `lzero/envs/wrappers`. It can wrap `classic_control`, `box2d` environments into the env format required by LightZero.
- Specifically, use the following function to wrap a gym environment into the env format required by LightZero through `LightZeroEnvWrapper`:

```python
def get_wrappered_env(wrapper_cfg: EasyDict, env_name: str):
    # overview comments
    ...
    if wrapper_cfg.manually_discretization:
        return lambda: DingEnvWrapper(
            gym.make(env_name),
            cfg={
                'env_wrapper': [
                    lambda env: ActionDiscretizationEnvWrapper(env, wrapper_cfg), lambda env:
                    LightZeroEnvWrapper(env, wrapper_cfg)
                ]
            }
        )
    else:
        return lambda: DingEnvWrapper(
            gym.make(env_name), cfg={'env_wrapper': [lambda env: LightZeroEnvWrapper(env, wrapper_cfg)]}
        )
```

Then call the `train_muzero_with_gym_env` method in the main entry point of the algorithm, and you can use the wrapped env for training:

```python
if __name__ == "__main__":
    """
    Overview:
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
    """
    from lzero.entry import train_muzero_with_gym_env
    train_muzero_with_gym_env([main_config, create_config], seed=0, max_env_step=max_env_step)
```

## **Considerations**

1. **State Representation**: Consider how to represent the environment state as an observation space. For simple environments, you can directly use low-dimensional continuous states; for complex environments, you might need to use images or other high-dimensional discrete states.
2. **Preprocessing Observation Space**: Depending on the type of the observation space, perform appropriate preprocessing operations on the input data, such as scaling, cropping, graying, normalization, etc. Preprocessing can reduce the dimension of input data and accelerate the learning process.
3. **Reward Design**: Design a reasonable reward function that aligns with the goal. For example, try to normalize the extrinsic reward given by the environment to \[0, 1\]. By normalizing the extrinsic reward given by the environment, you can better determine the weight of the intrinsic reward and other hyperparameters in the RND algorithm.