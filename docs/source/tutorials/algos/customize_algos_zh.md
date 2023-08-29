# LightZero 中如何自定义算法?

LightZero 是一个 MCTS+RL 强化学习框架，它提供了一组高级 API，使得用户可以在其中自定义自己的算法。以下是一些关于如何在 LightZero 中自定义算法的步骤和注意事项。

## 基本步骤

### 1. 理解框架结构

在开始编写自定义算法之前，你需要对 LightZero 的框架结构有一个基本的理解。框架主要由 `lzero` 和 [zoo](https://github.com/opendilab/LightZero/tree/main/zoo) 这两部分组成。`lzero` 包含了一些基础的模块，如策略（`policy`）、模型（`model`）等。[zoo](https://github.com/opendilab/LightZero/tree/main/zoo) 提供了一系列预定义的环境（`envs`）。在 `policy` 目录中，你可以找到各种算法的实现，如 MuZero 策略在 `muzero.py` 中实现。

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

你需要根据你的策略的需求来实现这些方法。

### 4. 注册你的策略

为了让 LightZero 能够识别你的策略，你需要在你的策略类上方使用`@POLICY_REGISTRY.register('my_algorithm')` 这个装饰器来注册你的策略。这样，LightZero 就可以通过 'my_algorithm' 这个名字来引用你的策略了。

### 5. **测试你的策略**

在你实现你的策略之后，确保策略的正确性和有效性是非常重要的。为此，你应该编写一些单元测试来验证你的策略是否正常工作。比如，你可以测试策略是否能在特定的环境中执行，策略的输出是否符合预期等。

你可以在 `lzero/policy/tests` 目录下添加你的测试。在编写测试时，尽可能考虑到所有可能的场景和边界条件，确保你的策略在各种情况下都能正常运行。

### 6. **完整测试与运行**

在确保策略的基本功能正常之后，你需要利用如 cartpole 等经典环境，对你的策略进行完整的正确性和收敛性测试。这是为了验证你的策略不仅能在单元测试中工作，而且能在实际游戏环境中有效工作。

你可以仿照  [cartpole_muzero_config.py](https://github.com/opendilab/LightZero/blob/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py)  编写相关的配置文件和入口程序。在测试过程中，注意记录策略的性能数据，如每轮的得分、策略的收敛速度等，以便于分析和改进。

### 7. **贡献**

在你完成了所有以上步骤后，如果你希望把你的策略贡献到 LightZero 仓库中，你可以在官方仓库上提交 Pull Request。在提交之前，请确保你的代码符合仓库的编码规范，所有测试都已通过，并且已经有足够的文档和注释来解释你的代码和策略。

在 PR 的描述中，详细说明你的策略，包括它的工作原理，你的实现方法，以及在测试中的表现。这会帮助其他人理解你的贡献，并加速 PR 的审查过程。

### 8. **分享讨论，反馈改进**

完成策略实现和测试后，考虑将你的结果和经验分享给社区。你可以在论坛、博客或者社交媒体上发布你的策略和测试结果，邀请其他人对你的工作进行评价和讨论。这不仅可以得到其他人的反馈，还能帮助你建立专业网络，并可能引发新的想法和合作。

基于你的测试结果和社区的反馈，不断改进和优化你的策略。这可能涉及到调整策略的参数，改进代码的性能，或者解决出现的问题和bug。记住，策略的开发是一个迭代的过程，永远有提升的空间。

## 注意事项

- 请确保你的代码符合 python PEP8 编码规范。
- 当你在实现 `_forward_learn`、`_forward_collect` 和 `_forward_eval` 等方法时，请确保正确处理输入和返回的数据。
- 在编写策略时，请确保考虑到不同的环境类型。你的策略应该能够处理不同的环境。
- 在实现你的策略时，请尽可能使你的代码模块化，以便于其他人理解和重用你的代码。
- 请编写清晰的文档和注释，描述你的策略如何工作，以及你的代码是如何实现这个策略的。