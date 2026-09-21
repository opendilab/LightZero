# 中国象棋（Xiangqi）

[English](README.md)

本目录提供一个不依赖第三方象棋库的中国象棋环境，以及可直接运行的 AlphaZero、MuZero 训练和终端对战入口。规则核心、动作编码和环境都属于 LightZero 代码的一部分，适合用于算法实验、规则回归和 checkpoint 评估。

## 快速开始

在 LightZero 项目根目录执行以下命令。默认配置使用内置随机 bot、CTree 搜索、CUDA（如可用）和 200 万环境步；训练输出目录由 `exp_name` 决定。

```bash
# AlphaZero
python -m zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config

# MuZero
python -m zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config
```

更适合本地实验或 rjob 的统一入口如下：

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot alphazero \
    --max-env-step 2000000 --num-simulations 50
python -m zoo.board_games.chinese_chess.entry.train_random_bot muzero \
    --max-env-step 2000000 --num-simulations 50
```

常用覆盖参数包括 `--seed`、`--exp-name`、`--collector-env-num`、`--evaluator-env-num`、`--n-episode`、`--update-per-collect`、`--batch-size`、`--eval-freq`、`--num-res-blocks`、`--num-channels`、`--max-episode-steps`、`--stop-value` 和 `--cpu`。完整列表：

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot --help
```

注意：如果训练时修改了 `--num-res-blocks` 或 `--num-channels`，加载 checkpoint 进行评估或终端对战时必须传入相同值。

## MDP 定义

环境注册名为 `chinese_chess`，动作空间和观测空间符合 DI-engine 环境接口。

| 部件 | 定义与实现 |
| --- | --- |
| 状态 `s` | `XiangqiState` 保存 10×9 棋盘、轮到谁、半回合无吃子计数、总 ply、重复局面信息和稳定位置哈希。环境另外保存最近 4 帧棋盘历史。 |
| 观测 `obs['observation']` | 默认形状 `(57, 10, 9)`、`float32`。4 帧 × 14 个二值棋子平面（当前方 7 类棋子、对手 7 类棋子）再加 1 个 side-to-move 平面。黑方行棋时棋盘旋转 180°，因此网络始终从当前行棋方视角看“向前”。`channel_last=True` 时形状为 `(10, 9, 57)`。 |
| 合法动作掩码 | `obs['action_mask']` 是长度 2,086 的 `int8` 向量；1 表示当前局面合法，0 表示非法。策略和搜索应使用该掩码，不应直接在全部动作中采样。 |
| 动作 `a` | `Discrete(2086)`。2,086 个标签覆盖几何上可能的车/炮/帅/兵直线着法、马着法，以及受九宫和河界约束的士/象着法；具体合法性由规则引擎判定。标签与 `action_encoding.py` 中的 `action_to_move`、`move_to_action` 双向转换。 |
| 奖励 `r` | 非终局为 0；终局从 agent 视角返回胜利 `+1`、失败 `-1`；将死、困毙或吃掉将/帅按输棋处理；三次重复、60 回合无吃子或达到步数上限为和棋 0。 |
| 终止 `done` | 由规则结果或 `max_episode_steps` 触发。默认上限为 500 个物理 ply；`info['eval_episode_return']` 在终局提供整局结果。 |
| `to_play` | `self_play_mode` 中为当前玩家索引；`play_with_bot_mode` 中固定为 `-1`，表示 agent 视角的单玩家 bot 对局。`current_player_index` 仍记录规则引擎当前轮到红方还是黑方。 |

### 两种主要 battle mode

- `self_play_mode`：一次 `env.step(action)` 只执行一个物理着法，执行后轮换行棋方。这是 AlphaZero 自博弈和模拟环境使用的模式。
- `play_with_bot_mode`：agent 先走一步；若未终局，内置 bot 再走一步；一次环境 step 返回 bot 走完后的局面和 agent 视角奖励。因此一个 agent episode 最多约有 250 次决策（默认物理上限仍是 500 ply）。bot 使用 NumPy RNG 随机选择当前合法着法，可通过 `env.seed(seed)` 复现实验序列。
- `eval_mode`：默认仍使用 bot；当 `agent_vs_human=True` 时，第二个半步由终端人类输入。

`play_with_bot_mode` 把对手回合封装进 agent transition，便于训练“对随机 bot 作战”的单玩家策略；但对 MuZero 而言，bot 着法是 agent action 之后发生的未观测随机转移，动力学网络只能学习其近似分布。若要检查逐 ply 的确定性规则转移，应使用 `self_play_mode` 或直接调用 `simulate_action`。

## 实现结构

- `envs/xiangqi.py`：独立规则核心。实现棋子走法、蹩马腿、塞象眼、炮架、九宫和过河限制、将帅照面、自将过滤、将死/困毙、重复局面及 60 回合无吃子判和。
- `envs/action_encoding.py`：生成固定的 2,086 个动作标签，提供动作/棋盘起终点/ICCS 字符串之间的转换，以及黑方视角所需的 180° 镜像动作映射。
- `envs/chinese_chess_env.py`：维护观测历史、视角规范化、动作掩码、bot/human 回合、奖励和终止逻辑。`board` 字段将棋盘、历史、回合计数和位置哈希压缩为 `int16`，供 CTree reset/clone 使用。
- `config/*_bot_mode_config.py`：AlphaZero 和 MuZero 的默认实验配置。
- `entry/train_random_bot.py`：统一训练入口；`entry/evaluate_random_bot.py`：多 seed checkpoint 评估；两个 `*_bot_human.py`：终端对战入口。
- `entry/chinese_chess_random_bot_human.py`：最小终端示例，人类执红、内置随机 bot 执黑。

## 当前默认训练设置

两份 preset 都使用 `play_with_bot_mode`、8 个 collector 环境、4 个 evaluator 环境、AdamW、学习率 `3e-4`、batch size `256`、每次收集更新 `100` 次、最多 `2e6` 环境步、每局最多 `500` 个物理 ply。默认搜索模拟次数为 `50`，提前停止阈值为平均回报 `0.9`。

算法差异如下：

- AlphaZero：6 个残差块、128 个通道；策略直接从真实观测预测并用 CTree MCTS 改进策略目标。
- MuZero：6 个残差块、128 个通道；使用 57×10×9 表示网络、动力学网络和预测网络，`num_unroll_steps=5`、`td_steps=100`、折扣因子 `1.0`，动作动力学编码当前为 `not_one_hot`（把离散动作 id 归一化为一个动作平面）。当前配置传入 `reward_support_range=(-1, 1, 1)` 和 `value_support_range=(-1, 1, 1)`；由于 `torch.arange` 的上界不包含在内，实际离散支撑是 `{-1, 0}`，而环境终局奖励仍可能为 `+1`。若要让分类支撑覆盖 `{-1, 0, +1}`，上界应至少设为 `2` 并重新训练。

具体字段以对应配置文件为准；本文不替代配置文件本身。

训练目录通常包含 `ckpt/ckpt_best.pth.tar`、按迭代保存的 `ckpt/iteration_*.pth.tar`、训练日志和配置快照。使用 `--exp-name` 时可将这些产物写到指定目录；评估和终端对战应优先使用验证集表现最好的 `ckpt_best.pth.tar`。

## 收敛评估

训练过程中的 evaluator 使用平均回报：

```text
mean_return = win_rate - loss_rate
```

默认 `mean_return >= 0.9` 会触发提前停止。单次 evaluator 局数较少，不能单独作为稳健的收敛结论。建议对 `ckpt_best.pth.tar` 使用至少 3 个固定 seed、每个 seed 至少 100 局：

```bash
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    alphazero /path/to/ckpt_best.pth.tar \
    --episodes 100 --seeds 0 1 2 --simulations 200

python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    muzero /path/to/ckpt_best.pth.tar \
    --episodes 100 --seeds 0 1 2 --simulations 200
```

脚本最后输出总局数、胜率、和率和负率。评估 checkpoint 时，模型深度和宽度必须与训练时一致；AlphaZero 的 `--simulations` 必须是大于等于 4 的正整数倍数，MuZero 则使用正整数。

## 终端人机对战

两个入口都加载 checkpoint 作为红方 bot，人类执黑。默认每个 bot 决策使用 200 次搜索：

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_alphazero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200

python -m zoo.board_games.chinese_chess.entry.chinese_chess_muzero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200
```

终端显示棋盘并等待人类输入 ICCS 坐标。列为 `a`–`i`，行为 `0`–`9`，例如 `h9g7` 表示从 h9 走到 g7。输入 `q`、`quit` 或 `exit` 退出。棋盘字符为：`K/k` 将、`A/a` 士、`E/e` 象、`H/h` 马、`R/r` 车、`C/c` 炮、`P/p` 兵/卒。若训练使用了非默认模型规模，请同步传入 `--num-res-blocks` 和 `--num-channels`。

## 终端人类 vs 内置随机 bot

如果只想了解环境接口而不加载 checkpoint，可以运行：

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_random_bot_human --seed 0
```

人类执红，在终端输入 ICCS 走法；每次人类走子后，`play_with_bot_mode`
会自动让黑方执行一个均匀随机的合法动作。这是随机基线示例，不是 Xiangqi MCTS 引擎。

## 直接使用环境

```python
from zoo.board_games.chinese_chess.envs.chinese_chess_env import ChineseChessEnv

env = ChineseChessEnv({'battle_mode': 'play_with_bot_mode'})
obs = env.reset()
action = env.legal_actions[0]
timestep = env.step(action)
print(obs['observation'].shape, obs['action_mask'].sum())
```

需要逐个物理着法时：

```python
env = ChineseChessEnv({'battle_mode': 'self_play_mode'})
env.reset()
next_env = env.simulate_action(env.legal_actions[0])  # 不修改 env 本身
```

## 规则边界

当前实现将三次重复统一判为和棋；复杂赛事中的长将、长捉等裁定不在范围内。该取舍让训练环境的终止和奖励定义稳定、可复现，但不等同于所有正式比赛裁判规则。
