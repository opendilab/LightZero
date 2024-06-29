# How to Customize Your Environments in LightZero?

When conducting reinforcement learning research or applications with LightZero, you may need to create a custom environment. Creating a custom environment can better adapt to specific problems or tasks, allowing the reinforcement learning algorithms to be effectively trained in those specific environments.

For a typical environment in LightZero, please refer to `atari_lightzero_env.py`. The environment design of LightZero is largely based on the BaseEnv class in DI-engine. When creating a custom environment, we follow similar basic steps as in [DI-engine](https://di-engine-docs.readthedocs.io/en/latest/04_best_practice/ding_env.html).

## Major Differences from BaseEnv

In LightZero, there are many board game environments. Due to the alternating actions of players and the changing set of legal moves, the observation state of the environment in board game environments should include not only the board information but also action masks and current player information. Therefore, in LightZero, the `obs` is no longer an array like in DI-engine but a dictionary. The `observation` key in the dictionary corresponds to `obs` in DI-engine, and in addition, the dictionary contains information such as `action_mask` and `to_play`. For the sake of code compatibility, LightZero also requires the environment to return `obs` that include `action_mask`, `to_play`, and similar information for non-board game environments.

In the specific implementation, these differences are primarily manifested in the following aspects:

- In the `reset()` method, `LightZeroEnv` returns a dictionary `lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}`.
  - For non-board game environments
    - Regarding the setting of `to_play`: Since non-board game environments generally only have one player, `to_play` is set to `-1`. (In our algorithm, we judge whether to execute the single player algorithm logic (`to_play=-1`), or the multiple player algorithm logic (`to_play=N`) based on this value.)
    - Regarding the setting of `action_mask`:
      - Discrete action space: `action_mask= np.ones(self.env.action_space.n, 'int8')` is a numpy array of ones, indicating that all actions are legal actions.
      - Continuous action space: `action_mask= None`, the special `None` indicates that the environment is a continuous action space.
  - For board game environments: To facilitate the subsequent MCTS process, the `lightzero_obs_dict` may also include variables such as the board information `board` and the index of the current player `current_player_index`.
- In the `step` method, `BaseEnvTimestep(lightzero_obs_dict, rew, done, info)` is returned, where `lightzero_obs_dict` contains the updated observation results.

## Basic Steps

Here are the basic steps to create a custom LightZero environment:

### 1. Create the Environment Class
First, you need to create a new environment class that inherits from the `BaseEnv` class in DI-engine. For example:

```python
from ding.envs import BaseEnv
```

### 2. **__init__ Method**<br>
In your custom environment class, you need to define an initialization method `__init__`. In this method, you need to set some basic properties of the environment, such as observation space, action space, reward space, etc. For example:

```python
def __init__(self, cfg=None):
    self.cfg = cfg
    self._init_flag = False
    # set other properties...
```

### 3. **Reset Method**<br>
The `reset` method is used to reset the environment to an initial state. This method should return the initial observation of the environment. For example:

```python
def reset(self):
    # reset the environment...
    obs = self._env.reset()
    # get the action_mask according to the legal action
    ...
    lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
    return lightzero_obs_dict
```

### 4. **Step Method**<br>
The `step` method takes an action as input, executes this action, and returns a tuple containing the new observation, reward, whether it's done, and other information. For example:

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

### 5. **Observation Space and Action Space**<br>
In a custom environment, you need to provide properties for observation space and action space. These properties are `gym.Space` objects that describe the shape and type of observations and actions. For example:

```python
@property
defobservation_space(self):
    return self.env.observation_space

@property
def action_space(self):
    return self.env.action_space
```

### 6. **Render Method**<br>
The `render` method displays the gameplay of the game for users to observe. For environments that have implemented the `render` method, users can choose whether to call `render` during the execution of the `step` function to render the game state at each step.

```python
def render(self, mode: str = 'image_savefile_mode') -> None:
        """
        Overview:
            Renders the game environment.
        Arguments:
            - mode (:obj:`str`): The rendering mode. Options are 
            'state_realtime_mode', 
            'image_realtime_mode', 
            or 'image_savefile_mode'.
        """
        # In 'state_realtime_mode' mode, print the current game board for rendering.
        if mode == "state_realtime_mode":
            ...
        # In other two modes, use a screen for rendering. 
        # Draw the screen.
        ...
        if mode == "image_realtime_mode":
            # Render the picture to user's window.
            ...
        elif mode == "image_savefile_mode":
            # Save the picture to frames.
            ...
            self.frames.append(self.screen)
        return None
```

In the `render` function, there are three different modes available:

- In the `state_realtime_mode`, `render` directly prints the current state.
- In the `image_realtime_mode`, `render` uses graphical assets to `render` the environment state, creating a visual interface and displaying it in a real-time window.
- In the `image_savefile_mode`, `render` saves the rendered images in `self.frames` and converts them into files using `save_render_output` at the end of the game.

During runtime, the mode used by render depends on the value of `self.render_mode`. If `self.render_mode` is set to None, the environment will not call the `render` method.

### 7. **Other Methods**<br>
Depending on the requirement, you might also need to define other methods, such as `close` (for closing the environment and performing cleanup), etc.

### 8. **Register the Environment**<br>
Lastly, you need to use the `ENV_REGISTRY.register` decorator to register your new environment so that it can be used in the configuration file. For example:

```python
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('my_custom_env')
class MyCustomEnv(BaseEnv):
    # ...
```

Once the environment is registered, you can specify the creation of the corresponding environment in the `create_config` section of the configuration file:

```python
create_config = dict(
    env=dict(
        type='my_custom_env',
        import_names=['zoo.board_games.my_custom_env.envs.my_custom_env'],
    ),
    ...
)
```

In the configuration, the `type` should be set to the registered environment name, while the `import_names` should be set to the location of the environment package.

Creating a custom environment may require a deep understanding of the specific task and reinforcement learning. When implementing a custom environment, you may need to experiment and adjust to make the environment effectively support reinforcement learning training.

## **Special Methods for Board Game Environments**

Here are the additional steps for creating custom board game environments in LightZero:

1. There are three different modes for board game environments in LightZero: `self_play_mode`, `play_with_bot_mode`, and `eval_mode`. Here is an explanation of these modes:
    - `self_play_mode`: In this mode, the environment follows the classical setup of board games. Each call to the `step` function places a move in the environment based on the provided action. At the time step when the game is decided, a reward of +1 is returned. In all other time steps where the game is not decided, the reward is 0.
    - `play_with_bot_mode`: In this mode, each call to the `step` function places a move in the environment based on the provided action, followed by the bot generating an action and placing a move based on that action. In other words, the agent plays as player 1, and the bot plays as player 2 against the agent. At the end of the game, if the agent wins, a reward of +1 is returned. If the bot wins, a reward of -1 is returned. In case of a draw, the reward is 0. In all other time steps where the game is not decided, the reward is 0.
    - `eval_mode`: This mode is used to evaluate the level of the current agent. There are two evaluation methods: bot evaluation and human evaluation. In bot evaluation, similar to play_with_bot_mode, the bot plays as player 2 against the agent, and the agent's win rate is calculated based on the results. In human evaluation, the user plays as player 2 and interacts with the agent by entering actions in the command line.

    In each mode, at the end of the game, the `eval_episode_return` information from the perspective of player 1 is recorded (if player 1 wins, `eval_episode_return` is 1; if player 1 loses, it is -1; if it's a draw, it is 0), and it is logged in the last time step.

2. In board game environments, as the game progresses, the available actions may decrease. Therefore, it is necessary to implement the `legal_action` method. This method can be used to validate the actions provided by the players and generate child nodes during the MCTS process. Taking the Connect4 environment as an example, this method checks if each column on the game board is full and returns a list. The value in the list is 1 for columns where a move can be made and 0 for other positions.

```python
def legal_actions(self) -> List[int]:
    return [i for i in range(7) if self.board[i] == 0]
```

3. In LightZero's board game environments, additional action generation methods need to be implemented, such as `bot_action` and `random_action`. The `bot_action` method retrieves the corresponding type of bot based on the value of `self.bot_action_type` and generates an action using the pre-implemented algorithm in the bot. On the other hand, `random_action` selects a random action from the current list of legal actions. `bot_action` is used in the `play_with_bot_mode` to implement the interaction with the bot, while `random_action` is called with a certain probability during action selection by the agent and the bot to increase the randomness of the game samples.

```python
def bot_action(self) -> int:
    if np.random.rand() < self.prob_random_action_in_bot:
        return self.random_action()
    else:
        if self.bot_action_type == 'rule':
            return self.rule_bot.get_rule_bot_action(self.board, self._current_player)
        elif self.bot_action_type == 'mcts':
            return self.mcts_bot.get_actions(self.board, player_index=self.current_player_index)
```

## **LightZeroEnvWrapper**

We provide a [LightZeroEnvWrapper](https://github.com/opendilab/LightZero/blob/main/lzero/envs/wrappers/lightzero_env_wrapper.py) in the lzero/envs/wrappers directory. It wraps `classic_control` and `box2d` environments into the format required by LightZero. During initialization, an original environment is passed to the LightZeroEnvWrapper instance, which is initialized using the parent class `gym.Wrapper`. This allows the instance to call methods like `render`, `close`, and `seed` from the original environment. Based on this, the `LightZeroEnvWrapper` class overrides the `step` and `reset` methods to wrap their outputs into a dictionary `lightzero_obs_dict` that conforms to the requirements of LightZero. As a result, the wrapped environment instance meets the requirements of LightZero's custom environments.

```python
class LightZeroEnvWrapper(gym.Wrapper):
    # overview comments
    def __init__(self, env: gym.Env, cfg: EasyDict) -> None:
        # overview comments
        super().__init__(env)
        ...
```
Specifically, use the following function to wrap a gym environment into the format required by LightZero using `LightZeroEnvWrapper`. The `get_wrappered_env` function returns an anonymous function that generates a `DingEnvWrapper` instance each time it is called. This instance takes `LightZeroEnvWrapper` as an anonymous function and internally wraps the original environment into the format required by LightZero.

```python
def get_wrappered_env(wrapper_cfg: EasyDict, env_id: str):
    # overview comments
    ...
    if wrapper_cfg.manually_discretization:
        return lambda: DingEnvWrapper(
            gym.make(env_id),
            cfg={
                'env_wrapper': [
                    lambda env: ActionDiscretizationEnvWrapper(env, wrapper_cfg), lambda env:
                    LightZeroEnvWrapper(env, wrapper_cfg)
                ]
            }
        )
    else:
        return lambda: DingEnvWrapper(
            gym.make(env_id), cfg={'env_wrapper': [lambda env: LightZeroEnvWrapper(env, wrapper_cfg)]}
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