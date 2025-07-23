# How to Customize Your Algorithms in LightZero?

LightZero is an MCTS+RL reinforcement learning framework that provides a set of high-level APIs, enabling users to customize their algorithms within it. Here are some steps and considerations on how to customize an algorithm in LightZero.

## **Basic Steps**

### 1. Understand the Framework Structure

Before you start coding your custom algorithms, you need to have a basic understanding of the LightZero framework's structure. The LightZero pipeline is illustrated in the following diagram.
<p align="center"> <img src="assets/lightzero_pipeline.svg" alt="Image" width="50%" height="auto" style="margin: 0 1%;"> 
</p>

The repository's folder consists primarily of two parts: `lzero` and `zoo`. The `lzero` folder contains the core modules required for the LightZero framework's workflow. The `zoo` folder provides a set of predefined environments (`envs`) and their corresponding configuration (`config`) files. The `lzero` folder includes several core modules, including the `policy`, `model`, `worker`, and `entry`. These modules work together to implement complex reinforcement learning algorithms.

- In this architecture, the `policy` module is responsible for implementing the algorithm's decision-making logic, such as action selection during agent-environment interaction and how to update the policy based on collected data. The `model` module is responsible for implementing the neural network structures required by the algorithm.

- The `worker` module consists of two classes: Collector and Evaluator. An instance of the Collector class handles the agent-environment interaction to collect the necessary data for training, while an instance of the Evaluator class evaluates the performance of the current policy.

- The `entry` module is responsible for initializing the environment, model, policy, etc., and its main loop implements core processes such as data collection, model training, and policy evaluation.

- There are close interactions among these modules. Specifically, the `entry` module calls the Collector and Evaluator of the `worker` module to perform data collection and algorithm evaluation. The decision functions of the `policy` module are called by the Collector and Evaluator to determine the agent's actions in a specific environment. The neural network models implemented in the `model` module are embedded in the `policy` object for action generation during interaction and for updates during the training process.

- In the `policy` module, you can find implementations of various algorithms. For example, the MuZero policy is implemented in the `muzero.py` file.


### 2. Create a New Policy File
Create a new Python file under the `lzero/policy` directory. This file will contain your algorithm implementation. For example, if your algorithm is called MyAlgorithm, you can create a file named `my_algorithm.py`.

### 3. Implement Your Policy

Within your policy file, you need to define a class to implement your strategy. This class should inherit from the `Policy` class in DI-engine and implement required methods. Below is a basic framework for a policy class:


```python
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
        # Set the default model name and the import path so that the default model can be loaded during policy initialization
    
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

#### Data Collection and Model Evaluation

- In `default_model`, set the class name of the default model used by the current policy and the corresponding reference path.
- The `_init_collect` and `_init_eval` functions are responsible for instantiating the action selection policy, and the respective policy instances will be called by the _forward_collect and _forward_eval functions.
- The `_forward_collect` function takes the current state of the environment and selects a step action by calling the instantiated policy in `_init_collect`. The function returns the selected action list and other relevant information. During training, this function is called through the `collector.collect` method of the Collector object created by the Entry file.
- The logic of the `_forward_eval` function is similar to that of the `_forward_collect`function. The only difference is that the policy used in `_forward_collect` is more focused on exploration to collect diverse training information, while the policy used in `_forward_eval` is more focused on exploitation to obtain the optimal performance of the current policy. During training, this function is called through the `evaluator.eval` method of the Evaluator object created by the Entry file.

#### Policy Learning

- The `_init_learn` function initializes the network model, optimizer, and other objects required during training using the associated parameters of the strategy, such as learning rate, update frequency, optimizer type, passed in from the config file.
- The `_forward_learn` function is responsible for updating the network. Typically, the `_forward_learn` function receives the data collected by the Collector, calculates the loss function based on this data, and performs gradient updates. The function returns the various losses during the update process and the relevant parameters used for the update, for experimental recording purposes. During training, this function is called through the `learner.train` method of the Learner object created by the Entry file.

### 4. Register Your Policy
To make LightZero recognize your policy, you need to use the `@POLICY_REGISTRY.register('my_algorithm')` decorator above your policy class to register your policy. This way, LightZero can refer to your policy by the name 'my_algorithm'. Specifically, in the experiment's configuration file, the corresponding algorithm is specified through the `create_config` section:

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

Here, `type` should be set to the registered policy name, and `import_names` should be set to the location of the policy package.

### 5. Possible Other Modifications
- **Model**: The LightZero `model.common` package provides some common network structures, such as the `RepresentationNetwork` that maps 2D images to a latent space representation and the `PredictionNetwork` used in MCTS for predicting probabilities and node values. If a custom policy requires a specific network model, you need to implement the corresponding model under the `model` folder. For example, the model for the MuZero algorithm is saved in the `muzero_model.py` file, which implements the `DynamicsNetwork` required by the MuZero algorithm and ultimately creates the `MuZeroModel` by calling the existing network structures in the `model.common` package.
- **Worker**: LightZero provides corresponding `worker` for AlphaZero and MuZero. Subsequent algorithms like EfficientZero and GumbelMuzero inherit the `worker` from MuZero. If your algorithm has different logic for data collection, you need to implement the corresponding `worker`. For example, if your algorithm requires preprocessing of collected transitions, you can add this segment under the `collect` function of the collector, in which the `get_train_sample` function implements the specific data processing process.

```Python
if timestep.done:
    # Prepare trajectory data.
    transitions = to_tensor_transitions(self._traj_buffer[env_id])
    # Use ``get_train_sample`` to process the data.
    train_sample = self._policy.get_train_sample(transitions)
    return_data.extend(train_sample)
    self._traj_buffer[env_id].clear()
```

### 6. Test Your Policy
After implementing your strategy, it is crucial to ensure its correctness and effectiveness. To do so, you should write some unit tests to verify that your strategy is functioning correctly. For example, you can test if the strategy can execute in a specific environment and if the output of the strategy matches the expected results. You can refer to the [documentation](https://di-engine-docs.readthedocs.io/zh_CN/latest/22_test/index_zh.html) in the DI-engine for guidance on how to write unit tests. You can add your tests in the `lzero/policy/tests`. When writing tests, try to consider all possible scenarios and boundary conditions to ensure your strategy can run properly in various situations.

Here is an example of unit testing in LightZero. In this example, we test the `inverse_scalar_transform` and `InverseScalarTransform`methods. Both methods reverse the transformation of a given value, but they have different implementations. In the unit test, we apply these two methods to the same set of data and compare the output results. If the results are the same, the test passes.

```Python
import pytest
import torch
from lzero.policy.scaling_transform import DiscreteSupport, inverse_scalar_transform, InverseScalarTransform

@pytest.mark.unittest
def test_scaling_transform():
    import time
    logit = torch.randn(16, 601)
    discrete_support = DiscreteSupport(-300., 301., 1.)
    start = time.time()
    output_1 = inverse_scalar_transform(logit, discrete_support)
    print('t1', time.time() - start)
    handle = InverseScalarTransform(discrete_support)
    start = time.time()
    output_2 = handle(logit)
    print('t2', time.time() - start)
    assert output_1.shape == output_2.shape == (16, 1)
    assert (output_1 == output_2).all()
```

In the unit test file, you need to mark the tests with `@pytest.mark.unittest` to include them in the Python testing framework. This allows you to run the unit test file directly by entering `pytest -sv xxx.py` in the command line. `-sv` is a command option that, when used, prints detailed information to the terminal during the test execution for easier inspection.

### 7. Comprehensive Testing and Running

- After ensuring the basic functionality of the policy, you need to use classic environments like cartpole to conduct comprehensive correctness and convergence tests on your policy. This is to verify that your policy can work effectively not only in unit tests but also in real game environments.
- You can write related configuration files and entry programs by referring to `cartpole_muzero_config.py`. During the testing process, pay attention to record performance data of the policy, such as the score of each round, the convergence speed of the policy, etc., for analysis and improvement.

### 8. Contribution

- After completing all the above steps, if you wish to contribute your policy to the LightZero repository, you can submit a Pull Request on the official repository. Before submission, ensure your code complies with the repository's coding standards, all tests have passed, and there are sufficient documents and comments to explain your code and policy.

- In the description of the PR, explain your policy in detail, including its working principle, your implementation method, and its performance in tests. This will help others understand your contribution and speed up the PR review process.

### 9. Share, Discuss, and Improve

- After implementing and testing the policy, consider sharing your results and experiences with the community. You can post your policy and test results on forums, blogs, or social media and invite others to review and discuss your work. This not only allows you to receive feedback from others but also helps you build a professional network and may trigger new ideas and collaborations.
- Based on your test results and community feedback, continuously improve and optimize your policy. This may involve adjusting policy parameters, improving code performance, or solving problems and bugs that arise. Remember, policy development is an iterative process, and there's always room for improvement.

## **Considerations**

- Ensure that your code complies with the Python PEP8 coding standards.
- When implementing methods like `_forward_learn`, `_forward_collect`, and `_forward_eval`, ensure that you correctly handle input and returned data.
- When writing your policy, ensure that you consider different types of environments. Your policy should be able to handle various environments.
- When implementing your policy, try to make your code as modular as possible, facilitating others to understand and reuse your code.
- Write clear documentation and comments describing how your policy works and how your code implements this policy. Strive to maintain the core meaning of the content while enhancing its professionalism and fluency.