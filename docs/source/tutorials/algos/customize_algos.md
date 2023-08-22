# **How to Customize Your Algorithms in LightZero?**

LightZero is an MCTS+RL reinforcement learning framework that provides a set of high-level APIs, enabling users to customize their algorithms within it. Here are some steps and considerations on how to customize an algorithm in LightZero.

## **Basic Steps**

1. **Understand the Framework Structure**<br>
   Before you start coding your custom algorithm, you need to have a basic understanding of LightZero’s framework structure. The framework mainly consists of two parts: `lzero` and `zoo`. `lzero` includes some basic modules, such as policy, model, etc. `zoo` provides a series of predefined environments (envs) and games. In the `policy` directory, you can find implementations of various algorithms, like the MuZero policy implemented in  `muzero.py`.

2. **Create a New Policy File**<br>
   Create a new Python file under the `lzero/policy` directory. This file will contain your algorithm implementation. For example, if your algorithm is called MyAlgorithm, you can create a file named `my_algorithm.py`.

3. **Implement Your Policy**<br>

   - Within your policy file, you need to define a class to implement your strategy. This class should inherit from the `Policy`  class in DI-engine and implement required methods.

   - Below is a basic framework for a policy class:


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

You need to implement these methods according to the needs of your policy.

4. **Register Your Policy**<br>
   To make LightZero recognize your policy, you need to use the `@POLICY_REGISTRY.register('my_algorithm')` decorator above your policy class to register your policy. This way, LightZero can refer to your policy by the name 'my_algorithm'.

5. **Test Your Policy**<br>

   - After implementing your policy, it's crucial to ensure the correctness and effectiveness of your policy. For this, you should write some unit tests to verify whether your policy is working as expected. For example, you can test whether the policy can execute in a specific environment and whether the policy's output meets the expectations.
   - You can add your test under the `lzero/policy/tests` directory. When writing tests, consider all possible scenarios and boundary conditions to ensure your policy can run normally under various circumstances.

6. **Comprehensive Testing and Running**<br>

   - After ensuring the basic functionality of the policy, you need to use classic environments like cartpole to conduct comprehensive correctness and convergence tests on your policy. This is to verify that your policy can work effectively not only in unit tests but also in real game environments.
   - You can write related configuration files and entry programs by referring to `cartpole_muzero_config.py`. During the testing process, pay attention to record performance data of the policy, such as the score of each round, the convergence speed of the policy, etc., for analysis and improvement.

7. **Contribution**<br>

   - After completing all the above steps, if you wish to contribute your policy to the LightZero repository, you can submit a Pull Request on the official repository. Before submission, ensure your code complies with the repository's coding standards, all tests have passed, and there are sufficient documents and comments to explain your code and policy.

   - In the description of the PR, explain your policy in detail, including its working principle, your implementation method, and its performance in tests. This will help others understand your contribution and speed up the PR review process.

8. **Share, Discuss, and Improve**<br>

   - After implementing and testing the policy, consider sharing your results and experiences with the community. You can post your policy and test results on forums, blogs, or social media and invite others to review and discuss your work. This not only allows you to receive feedback from others but also helps you build a professional network and may trigger new ideas and collaborations.
   - Based on your test results and community feedback, continuously improve and optimize your policy. This may involve adjusting policy parameters, improving code performance, or solving problems and bugs that arise. Remember, policy development is an iterative process, and there's always room for improvement.

## **Considerations**

- Ensure that your code complies with the Python PEP8 coding standards.
- When implementing methods like `_forward_learn`, `_forward_collect`, and `_forward_eval`, ensure that you correctly handle input and returned data.
- When writing your policy, ensure that you consider different types of environments. Your policy should be able to handle various environments.
- When implementing your policy, try to make your code as modular as possible, facilitating others to understand and reuse your code.
- Write clear documentation and comments describing how your policy works and how your code implements this policy. Strive to maintain the core meaning of the content while enhancing its professionalism and fluency.