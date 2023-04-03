
## Environment Versatility

The following is a brief introduction to the environment supported by our zoo：

<details open><summary>Click to collapse</summary>


|  No  |                Environment               |                 Label               |         Visualization            |                   Doc Links                   |
| :--: | :--------------------------------------: | :---------------------------------: | :--------------------------------:|:---------------------------------------------------------: |
|  1   |       [atari](https://github.com/openai/gym/tree/master/gym/envs/atari)    | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)   | ![original](./dizoo/atari/atari.gif)     | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/atari.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/atari_zh.html)        |
|  2   |       [box2d/lunarlander](https://github.com/openai/gym/tree/master/gym/envs/box2d)      | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![continuous](https://img.shields.io/badge/-continous-green)  | ![original](./dizoo/box2d/lunarlander/lunarlander.gif)   | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/lunarlander_zh.html)  |
|  3   |       [box2d/bipedalwalker](https://github.com/openai/gym/tree/master/gym/envs/box2d)    | ![continuous](https://img.shields.io/badge/-continous-green) ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  | ![original](./dizoo/box2d/bipedalwalker/original.gif)        | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/bipedalwalker.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/bipedalwalker_zh.html) |
|  4   |       [classic_control/cartpole](https://github.com/openai/gym/tree/master/gym/envs/classic_control)       | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)   | ![original](./dizoo/classic_control/cartpole/cartpole.gif)    | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/cartpole.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/cartpole_zh.html) |
|  5   |       [classic_control/pendulum](https://github.com/openai/gym/tree/master/gym/envs/classic_control)       | ![continuous](https://img.shields.io/badge/-continous-green) ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  | ![original](./dizoo/classic_control/pendulum/pendulum.gif)    |  [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/pendulum.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/pendulum_zh.html) |
|  6   |       [board_games/tictactoe]()       | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)   | ![original]()    | [env tutorial](https://en.wikipedia.org/wiki/Tic-tac-toe) |
|  7   |       [board_games/gomoku]()       | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  | ![original]()    |  [env tutorial](https://en.wikipedia.org/wiki/Gomoku)|

![discrete](https://img.shields.io/badge/-discrete-brightgreen) means discrete action space

![continuous](https://img.shields.io/badge/-continous-green) means continuous action space

P.S. The LunarLander environment has both continuous and discrete action spaces. Continuous action space environments, such as BipedalWalker and Pendulum, can be discretized manually to obtain discrete action spaces, 
please refer to lzero/envs/wrappers/action_discretization_env_wrapper.py for details.
</details>