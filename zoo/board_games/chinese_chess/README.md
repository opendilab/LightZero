# Chinese Chess (Xiangqi)

[中文说明](README_zh.md)

This directory provides a dependency-free Xiangqi environment with a 2,086-action compact policy head, canonical
current-player observations, a random training opponent, and CTree-compatible state serialization.

The compact label layout is compatible with the policy representation used by
[ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero), while the rules implementation here is
independent and Apache-2.0-compatible.

Train the random-bot presets:

```bash
python -m zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config
python -m zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config
```

For reproducible local or rjob overrides, use the common training entry:

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot alphazero --max-env-step 2000000
python -m zoo.board_games.chinese_chess.entry.train_random_bot muzero --max-env-step 2000000
```

Play against a trained checkpoint as Black (moves use ICCS coordinates such as `h9g7`):

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_alphazero_bot_human /path/to/ckpt_best.pth.tar
python -m zoo.board_games.chinese_chess.entry.chinese_chess_muzero_bot_human /path/to/ckpt_best.pth.tar
```

Measure convergence over several deterministic random-bot seeds:

```bash
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot alphazero /path/to/ckpt_best.pth.tar
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot muzero /path/to/ckpt_best.pth.tar
```

The training evaluator uses mean return (`win_rate - loss_rate`) and the presets stop at `0.9`. For a final convergence
claim, evaluate `ckpt_best.pth.tar` on at least 100 games for each of three seeds and report win/draw/loss rates. The
provided hyperparameters are random-bot training presets; full convergence remains hardware- and seed-dependent.

The rules core covers standard piece movement, self-check, flying generals, checkmate/stalemate, threefold repetition and the 60-move no-capture rule. Tournament adjudication of complex long-check/long-chase sequences is intentionally outside the environment; repeated positions are scored as draws to keep the training MDP deterministic.
