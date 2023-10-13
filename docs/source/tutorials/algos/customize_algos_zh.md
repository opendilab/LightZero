# LightZero 中如何自定义算法?

LightZero 是一个 MCTS+RL 强化学习框架，它提供了一组高级 API，使得用户可以在其中自定义自己的算法。以下是一些关于如何在 LightZero 中自定义算法的步骤和注意事项。

## 基本步骤

### 1. 理解框架结构

在开始编写自定义算法之前，你需要对 LightZero 的框架结构有一个基本的理解，LightZero 的流程如图所示。

<p align="center">
  <img src="assets/lightzero_pipeline.svg" alt="Image" width="50%" height="auto" style="margin: 0 1%;">
</p>

仓库的文件夹主要由 `lzero` 和 `zoo` 这两部分组成。`lzero`中实现了LightZero框架流程所需的核心模块。而 `zoo` 提供了一系列预定义的环境（`envs`）以及对应的配置（`config`）文件。
`lzero`文件夹下包括多个核心模块，包括策略（`policy`）、模型（`model`）、工作器（`worker`）以及入口（`entry`）等。这些模块在一起协同工作，实现复杂的强化学习算法。
- 在此架构中，`policy`模块负责实现算法的决策逻辑，如在智能体与环境交互时的动作选择，以及如何根据收集到的数据更新策略。`model`模块则负责实现算法所需的神经网络结构。
- `worker`模块包含 Collector 和 Evaluator 两个类。Collector 实例负责执行智能体与环境的交互，以收集训练所需的数据，而 Evaluator 实例则负责评估当前策略的性能。
- `entry` 模块负责初始化环境、模型、策略等，并在其主循环中负责实现数据收集、模型训练以及策略评估等核心过程。
- 在这些模块之间，存在着紧密的交互关系。具体来说，`entry`模块会调用`worker`模块的Collector和Evaluator来完成数据收集和算法评估。同时，`policy`模块的决策函数会被Collector和Evaluator调用，以决定智能体在特定环境中的行动。而`model`模块实现的神经网络模型，则被嵌入到`policy`对象中，用于在交互过程中生成动作，以及在训练过程中进行更新。
- 在`policy`模块中，你可以找到多种算法的实现，例如，MuZero策略就在`muzero.py`文件中实现。

### 2. 创建新的策略文件

在 `lzero/policy` 目录下创建一个新的 Python 文件。这个文件将包含你的算法实现。例如，如果你的算法名为 `MyAlgorithm`，你可以创建一个名为 `my_algorithm.py` 的文件。

### 3. 实现你的策略

在你的策略文件中，你需要定义一个类来实现你的策略。这个类应该继承自 DI-engine中的 `Policy` 类，并实现所需的方法。

以下是一个基本的策略类的框架：

```Python
@POLICY_REGISTRY.register('my_algorithm')
class MyAlgorithmPolicy(Policy):
    """
    Overview:
        The policy class for MyAlgorithm.
    """
    
    config = dict(
        # Add your config here
    )
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Initialize your policy here

    def default_model(self) -> Tuple[str, List[str]]:
        # Return this algorithm default model setting for demonstration.
    
    def _init_learn(self):
        # Initialize the learn mode here
    
    def _forward_learn(self, data):
        # Implement the forward function for learning mode here
    
    def _init_collect(self):
        # Initialize the collect mode here
    
    def _forward_collect(self, data, **kwargs):
        # Implement the forward function for collect mode here
    
    def _init_eval(self):
        # Initialize the eval mode here
    
    def _forward_eval(self, data, **kwargs):
        # Implement the forward function for eval mode here
```

#### 收集数据与评估模型

- 在 `default_model`中设置当前策略使用的默认模型的类名和相应的引用路径。
- _`init_collect`和`_init_eval`函数均负责实例化动作选取策略，相应的策略实例会被`_forward_collect`和`_forward_eval`函数调用。
- `_forward_collect`函数会接收当前环境的状态，并通过调用`_init_collect`中实例化的策略来选择一步动作。函数会返回所选的动作列表以及其他相关信息。在训练期间，该函数会通过由Entry文件创建的Collector对象的`collector.collect`方法进行调用。
- `_forward_eval`函数的逻辑与`_forward_collect`函数基本一致。唯一的区别在于，`_forward_collect`中采用的策略更侧重于探索，以收集尽可能多样的训练信息；而在`_forward_eval`函数中，所采用的策略更侧重于利用，以获取当前策略的最优性能。在训练期间，该函数会通过由Entry文件创建的Evaluator对象的`evaluator.eval`方法进行调用。

#### 策略的学习

- `_init_learn`函数会利用config文件传入的学习率、更新频率、优化器类型等策略的关联参数初始化网络模型、优化器以及训练过程中所需的其他对象。
- `_forward_learn`函数则负责实现网络的更新。通常，`_forward_learn`函数会接收Collector所收集的数据，根据这些数据计算损失函数并进行梯度更新。函数会返回更新过程中的各项损失以及更新所采用的相关参数，以便进行实验记录。在训练期间，该函数会通过由Entry文件创建的Learner对象的`learner.train`方法进行调用。

### 4. 注册你的策略

为了让 LightZero 能够识别你的策略，你需要在你的策略类上方使用`@POLICY_REGISTRY.register('my_algorithm')` 这个装饰器来注册你的策略。这样，LightZero 就可以通过 'my_algorithm' 这个名字来引用你的策略了。
具体而言，在实验的配置文件中，通过`create_config`部分来指定相应的算法：

```Python
create_config = dict(
    ...
    policy=dict(
        type='my_algorithm',
        import_names=['lzero.policy.my_algorithm'],
    ),
    ...
)
```

其中`type`要设定为所注册的策略名，`import_names`则设置为策略包的位置。

### 5. **可能的其他更改**
- 模型：在LightZero的`model.common`包中提供了一些通用的网络结构，例如将2D图像映射到隐空间中的表征网络`RepresentationNetwork`，在MCTS中用于预测概率和节点价值的预测网络`PredictionNetwork`等。如果自定义的策略需要专门的网络模型，则需要自行在`model`文件夹下实现相应的模型。例如Muzero算法的模型保存在`muzero_model.py`文件中，该文件实现了Muzero算法所需要的`DynamicsNetwork`，并通过调用`model.common`包中现成的网络结构最终实现了`MuZeroModel`。
- 工作件：在LightZero中实现了AlphaZero和MuZero的相应工作件。后续的EfficientZero和GumbelMuzero等算法沿用了MuZero的工作件。如果你的算法在数据采集的逻辑上有所不同，则需要自行实现相应的工作件。例如，如果你的算法需要筛选符合条件的episode数据再加入到buffer中，则需要修改collector文件中的`collect`函数。下面这段代码通过调用`get_train_sample`函数实现这一功能：

```Python
if timestep.done:
    # Prepare trajectory data.
    transitions = to_tensor_transitions(self._traj_buffer[env_id])
    # Use ``get_train_sample`` to process the data.
    train_sample = self._policy.get_train_sample(transitions)
    return_data.extend(train_sample)
    self._traj_buffer[env_id].clear()
```

### 6. **测试你的策略**

在你实现你的策略之后，确保策略的正确性和有效性是非常重要的。为此，你应该编写一些单元测试来验证你的策略是否正常工作。比如，你可以测试策略是否能在特定的环境中执行，策略的输出是否符合预期等。单元测试的编写及意义可以参考DI-engine 中的[说明文档](https://di-engine-docs.readthedocs.io/zh_CN/latest/22_test/index_zh.html) ,你可以在 `lzero/policy/tests` 目录下添加你的测试。在编写测试时，尽可能考虑到所有可能的场景和边界条件，确保你的策略在各种情况下都能正常运行。
下面是一个LightZero中单元测试的例子。在这个例子中，所测试的对象是`inverse_scalar_transform`和`InverseScalarTransform`方法。这两个方法都将经过变换的value逆变换为原本的值，但是采取了不同的实现。单元测试时，用这两个方法对同一组数据进行处理，并比较输出的结果是否相同。如果相同，则会通过测试。

```Python
import pytest
import torch
from lzero.policy.scaling_transform import inverse_scalar_transform, InverseScalarTransform

@pytest.mark.unittest
def test_scaling_transform():
    import time
    logit = torch.randn(16, 601)
    start = time.time()
    output_1 = inverse_scalar_transform(logit, 300)
    print('t1', time.time() - start)
    handle = InverseScalarTransform(300)
    start = time.time()
    output_2 = handle(logit)
    print('t2', time.time() - start)
    assert output_1.shape == output_2.shape == (16, 1)
    assert (output_1 == output_2).all()
```

在单元测试文件中，要将测试通过`@pytest.mark.unittest`标记到python的测试框架中，这样就可以通过在命令行输入`pytest -sv xxx.py`直接运行单元测试文件。

### 7. **完整测试与运行**

在确保策略的基本功能正常之后，你需要利用如 cartpole 等经典环境，对你的策略进行完整的正确性和收敛性测试。这是为了验证你的策略不仅能在单元测试中工作，而且能在实际游戏环境中有效工作。

你可以仿照  [cartpole_muzero_config.py](https://github.com/opendilab/LightZero/blob/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py)  编写相关的配置文件和入口程序。在测试过程中，注意记录策略的性能数据，如每轮的得分、策略的收敛速度等，以便于分析和改进。

### 8. **贡献**

在你完成了所有以上步骤后，如果你希望把你的策略贡献到 LightZero 仓库中，你可以在官方仓库上提交 Pull Request。在提交之前，请确保你的代码符合仓库的编码规范，所有测试都已通过，并且已经有足够的文档和注释来解释你的代码和策略。

在 PR 的描述中，详细说明你的策略，包括它的工作原理，你的实现方法，以及在测试中的表现。这会帮助其他人理解你的贡献，并加速 PR 的审查过程。

### 9. **分享讨论，反馈改进**

完成策略实现和测试后，考虑将你的结果和经验分享给社区。你可以在论坛、博客或者社交媒体上发布你的策略和测试结果，邀请其他人对你的工作进行评价和讨论。这不仅可以得到其他人的反馈，还能帮助你建立专业网络，并可能引发新的想法和合作。

基于你的测试结果和社区的反馈，不断改进和优化你的策略。这可能涉及到调整策略的参数，改进代码的性能，或者解决出现的问题和bug。记住，策略的开发是一个迭代的过程，永远有提升的空间。

## 注意事项

- 请确保你的代码符合 python PEP8 编码规范。
- 当你在实现 `_forward_learn`、`_forward_collect` 和 `_forward_eval` 等方法时，请确保正确处理输入和返回的数据。
- 在编写策略时，请确保考虑到不同的环境类型。你的策略应该能够处理不同的环境。
- 在实现你的策略时，请尽可能使你的代码模块化，以便于其他人理解和重用你的代码。
- 请编写清晰的文档和注释，描述你的策略如何工作，以及你的代码是如何实现这个策略的。