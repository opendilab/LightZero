# LightZero 中如何自定义环境?

- 在使用 LightZero 进行强化学习的研究或应用时，可能需要创建自定义的环境。创建自定义环境可以更好地适应特定的问题或任务，使得强化学习算法能够在特定环境中进行有效的训练。
- 一个典型的 LightZero 中的环境，请参考 [atari_lightzero_env.py](https://github.com/opendilab/LightZero/blob/main/zoo/atari/envs/atari_lightzero_env.py)。LightZero的环境设计大致基于DI-engine的`BaseEnv`类。在创建自定义环境时，我们遵循了与DI-engine相似的基本步骤。以下是DI-engine中创建自定义环境的文档
  - https://di-engine-docs.readthedocs.io/zh_CN/latest/04_best_practice/ding_env_zh.html 

## 与 BaseEnv 的主要差异

在LightZero中，有很多棋类环境。棋类环境由于存在玩家交替执行动作，合法动作在变化的情况，所以环境的观测状态除了棋面信息，还应包含动作掩码，当前玩家等信息。因此，LightZero 中的 obs 不再像 DI-engine 中那样是一个数组，而是一个字典。字典中的 'observation' 对应于DI-engine中的 obs，此外字典中还包含了 'action_mask'、'to_play' 等信息。为了代码的兼容性，对于非棋类环境，LightZero 同样要求环境返回的 obs 包含'action_mask'、'to_play'  等信息。

在具体的方法实现中，这种差异主要体现在下面几点：

- 在 `reset()` 方法中，LightZeroEnv  返回的是一个字典 `lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}`。
  - 对于非棋类环境
    - `to_play`的设置：由于非棋类环境一般只有一个玩家，因此设置 `to_play`=-1。(我们在算法中根据该值，判断执行单player的算法逻辑(`to_play`=-1)，还是多player的算法逻辑(`to_play`=N))
    - 对于action_mask的设置
      - 离散动作空间：`action_mask`= np.ones(self.env.action_space.n, 'int8') 是一个全1的numpy数组，表示所有动作都是合法动作。
      - 连续动作空间：`action_mask`= None，特殊的None表示环境是连续动作空间。
  - 对于棋类环境：为了方便后续 MCTS 流程,`lightzero_obs_dict `中可能还会增加棋面信息`board`和当前玩家 `curren_player_index`等变量。
- 在 `step()` 方法中，返回的是 `BaseEnvTimestep(lightzero_obs_dict, rew, done, info)`，其中的 `lightzero_obs_dict` 包含了更新后的观察结果。

## 基本步骤

以下是创建自定义 LightZero 环境的基本步骤：

### 1. 创建环境类

首先，需要创建一个新的环境类，该类需要继承自 DI-engine 的 BaseEnv 类。例如：

```Python
from ding.envs import BaseEnv

class MyCustomEnv(BaseEnv):
    pass
```

### 2. __init__方法

在自定义环境类中，需要定义一个初始化方法`__init__`。在这个方法中，需要设置一些环境的基本属性，例如观察空间、动作空间、奖励空间等。例如：

```Python
def __init__(self, cfg=None):
    self.cfg = cfg
    self._init_flag = False
    # set other properties...
```

### 3. Reset 方法

`reset`方法用于重置环境到一个初始状态。这个方法应该返回环境的初始观察。例如：

```Python
def reset(self):
    # reset the environment...
    obs = self._env.reset()
    # get the action_mask according to the legal action
    ...
    lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
    return lightzero_obs_dict
```

### 4. Step 方法

`step`方法接受一个动作作为输入，执行这个动作，并返回一个元组，包含新的观察、奖励、是否完成和其他信息。例如：

```Python
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

### 5. 观察空间和动作空间

在自定义环境中，需要提供观察空间和动作空间的属性。这些属性是 `gym.Space` 对象，描述了观察和动作的形状和类型。例如：

```Python
@property
def observation_space(self):
    return self._observation_space

@property
def action_space(self):
    return self._action_space
    
@property
def legal_actions(self):
    # get the actual legal actions
    return np.arange(self._action_space.n)
```

### 6. render 方法

`render`方法会将游戏的对局演示出来，供用户查看。对于实现了`render`方法的环境，`render`会在每一次`step`函数执行时被调用。

```Python
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

在`render`中，有三种不同的模式。
- 在`state_realtime_mode`下，`render`会直接打印当前状态。
- 在`image_realtime_mode`下，`render`会根据一些图形素材将环境状态渲染出来，形成可视化的界面，并弹出实时的窗口展示。
- 在`image_savefile_mode`下，`render`会将渲染的图像保存在`self.frames`中，并在对局结束时通过`save_render_output`将其转化为文件保存下来。
在运行时，`render`所采取的模式取决于`self.render_mode`的取值。当`self.render_mode`取值为`None`时，环境不会调用`render`方法。

### 7. 其他方法

根据需要，可能还需要定义其他方法，例如`close`（用于关闭环境并进行清理）等。

### 8. 注册环境

最后，需要使用 `ENV_REGISTRY.register` 装饰器来注册新的环境，使得可以在配置文件中使用它。例如：

```Python
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('my_custom_env')
class MyCustomEnv(BaseEnv):
    # ...
```

当环境注册好之后，可以在配置文件中的`create_config`部分指定生成相应的环境：

```Python
create_config = dict(
    env=dict(
        type='my_custom_env',
        import_names=['zoo.board_games.my_custom_env.envs.my_custom_env'],
    ),
    ...
)
```

其中`type`要设定为所注册的环境名，`import_names`则设置为环境包的位置。

创建自定义环境可能需要对具体的任务和强化学习有深入的理解。在实现自定义环境时，可能需要进行一些试验和调整，以使环境能够有效地支持强化学习的训练。

## 棋类环境的特殊方法

以下是创建自定义 LightZero 棋类环境的额外步骤：
1. LightZero中的棋类环境有三种不同的模式：  `self_play_mode`, `play_with_bot_mode`, `eval_mode`。这三种模式的说明如下：
    - `self_play_mode`：该模式下，采取棋类环境的经典设置，每调用一次`step`函数，会根据传入的动作在环境中落子一次。在分出胜负的时间步，会返回+1的reward。在没有分出胜负的所有时间步，reward均为0。
    - `play_with_bot_mode`：该模式下，每调用一次`step`函数，会根据传入的动作在环境中落子一次，随后调用环境中的bot产生一个动作，并根据bot的动作再落子一次。也就是说，agent扮演了1号玩家的角色，而bot扮演了2号玩家的角色和agent对抗。在对局结束时，如果agent胜利，则返回+1的reward，如果bot胜利，则返回-1的reward，平局则reward为0。在其余没有分出胜负的时间步，reward均为0。
    - `eval_mode`：该模式用于评估当前的agent的水平。具体有bot和human两种评估方法。采取bot评估时，和play_with_bot_mode中一样，会让bot扮演2号玩家和agent对抗，并根据结果计算agent的胜率。采取human模式时，则让用户扮演2号玩家，在命令行输入动作和agent对打。  

    每种模式下，在棋局结束后，都会从1号玩家的视角记录本局的`eval_episode_return`信息（如果1号玩家赢了，则`eval_episode_return`为1，如果输了为-1，平局为0），并记录在最后一个时间步中。
2. 在棋类环境中，随着对局的推进，可以采取的动作会不断变少，因此还需要实现`legal_action`方法。该方法可以用于检验玩家输入的动作是否合法，以及在MCTS过程中根据合法动作生成子节点。以Connect4环境为例，该方法会检查棋盘中的每一列是否下满，然后返回一个列表。该列表在可以落子的列取值为1，其余位置取值为0。

```Python
def legal_actions(self) -> List[int]:
        return [i for i in range(7) if self.board[i] == 0]
```

3. LightZero的棋类环境中，还需要实现一些动作生成方法，例如`bot_action`和`random_action`。其中`bot_action`会根据`self.bot_action_type`的值调取相应种类的bot，通过bot中预实现的算法生成一个动作。而`random_action`则会从当前的合法动作列表中随机选取一个动作返回。`bot_action`用于实现环境的`play_with_bot_mode`，而`random_action`则会在agent和bot选取动作时依一定概率被调用，来增加对局样本的随机性。

```Python
def bot_action(self) -> int:
        if np.random.rand() < self.prob_random_action_in_bot:
            return self.random_action()
        else:
            if self.bot_action_type == 'rule':
                return self.rule_bot.get_rule_bot_action(self.board, self._current_player)
            elif self.bot_action_type == 'mcts':
                return self.mcts_bot.get_actions(self.board, player_index=self.current_player_index)
```

## LightZeroEnvWrapper

我们在 lzero/envs/wrappers 中提供了一个 [LightZeroEnvWrapper](https://github.com/opendilab/LightZero/blob/main/lzero/envs/wrappers/lightzero_env_wrapper.py)。它能够将经典的 classic_control, box2d 环境包装成 LightZero 所需要的环境格式。在初始化实例时，会传入一个原始环境，这个原始环境通过父类`gym.Wrapper`被初始化，这使得实例可以调用原始环境中的`render`，`close`，`seed`等方法。在此基础上，`LightZeroEnvWrapper` 类重写了`step`和`reset`方法，将其输出封装成符合 LightZero 要求的字典`lightzero_obs_dict`。这样一来，封装后的新环境实例就满足了LightZero自定义环境的要求。

```Python
class LightZeroEnvWrapper(gym.Wrapper):
    # overview comments
    def __init__(self, env: gym.Env, cfg: EasyDict) -> None:
        # overview comments
        super().__init__(env)
        ...
```

具体使用时，使用下面的函数，将一个 gym 环境，通过`LightZeroEnvWrapper`包装成 LightZero 所需要的环境格式。`get_wrappered_env`会返回一个匿名函数，该匿名函数每次调用都会产生一个`DingEnvWrapper`实例，该实例会将`LightZeroEnvWrapper`作为匿名函数传入，并在实例内部将原始环境封装成 LightZero 所需的格式。

```Python
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

然后在算法的主入口处中调用 `train_muzero_with_gym_env` 方法，即可使用上述包装后的 env 用于训练：

```Python
if __name__ == "__main__":
    """
    Overview:
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
    """
    from lzero.entry import train_muzero_with_gym_env
    train_muzero_with_gym_env([main_config, create_config], seed=0, max_env_step=max_env_step)
```

## 注意事项

- 状态表示：思考如何将环境状态表示为观察空间。对于简单的环境，可以直接使用低维连续状态；对于复杂的环境，可能需要使用图像或其他高维离散状态表示。
- 观察空间预处理：根据观察空间的类型，对输入数据进行适当的预处理操作，例如缩放、裁剪、灰度化、归一化等。预处理可以减少输入数据的维度，加速学习过程。
- 奖励设计：设计合理的符合目标的的奖励函数。例如，环境给出的外在奖励尽量归一化在[0, 1]。通过归一化环境给出的外在奖励，能更好的确定 RND 算法中的内在奖励权重等超参数。
