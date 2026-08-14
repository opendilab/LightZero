# 中国象棋（Xiangqi）

[English](README.md)

本目录提供一个不依赖外部象棋库的中国象棋环境，包含 2,086 维紧凑动作空间、当前行棋方视角的规范化观测、内置随机对手，以及兼容 CTree 的状态序列化。

动作标签布局与 [ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero) 使用的策略表示兼容；规则引擎为独立实现，可按 LightZero 的 Apache-2.0 许可证使用。

## 训练

直接使用默认的 random-bot 配置：

```bash
python -m zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config
python -m zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config
```

需要覆盖训练预算、搜索次数或模型规模时，使用统一入口。例如：

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot alphazero \
    --max-env-step 2000000 --num-simulations 50
python -m zoo.board_games.chinese_chess.entry.train_random_bot muzero \
    --max-env-step 2000000 --num-simulations 50
```

可通过 `python -m zoo.board_games.chinese_chess.entry.train_random_bot --help` 查看全部覆盖参数。

## 收敛验证

训练过程中的 evaluator 与内置 random bot 对战，指标为平均回报：

```text
mean_return = win_rate - loss_rate
```

默认配置在 `mean_return >= 0.9` 时提前停止。最终报告不应只使用单次 evaluator 结果；建议对 `ckpt_best.pth.tar` 使用至少 3 个固定种子、每个种子至少 100 局：

```bash
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    alphazero /path/to/ckpt_best.pth.tar --episodes 100 --seeds 0 1 2
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    muzero /path/to/ckpt_best.pth.tar --episodes 100 --seeds 0 1 2
```

脚本会输出总局数以及胜率、和率和负率。训练速度和最终结果会受硬件及随机种子影响，因此修改超参数后应保留 checkpoint 和完整评估记录。

## 在终端与模型对战

以下命令加载训练 checkpoint。模型执红先行，玩家执黑：

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_alphazero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200
python -m zoo.board_games.chinese_chess.entry.chinese_chess_muzero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200
```

轮到玩家时，终端会显示棋盘并提示输入 ICCS 坐标。棋盘列为 `a` 到 `i`、行为 `0` 到 `9`，例如黑马从 `h9` 走到 `g7`：

```text
h9g7
```

输入 `q`、`quit` 或 `exit` 可退出。大写字母表示红方棋子，小写字母表示黑方棋子：`K/k` 将、`A/a` 士、`E/e` 象、`H/h` 马、`R/r` 车、`C/c` 炮、`P/p` 兵/卒。

## 规则范围

规则引擎实现了全部棋子走法、蹩马腿、塞象眼、炮架、九宫与过河限制、将帅照面、自将过滤、将死/困毙、三次重复以及 60 回合无吃子和棋。复杂的赛事长将/长捉裁定不在当前范围内；重复局面统一判和，以保持训练 MDP 确定。
