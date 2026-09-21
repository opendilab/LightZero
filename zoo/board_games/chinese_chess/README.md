# Chinese Chess (Xiangqi)

[中文说明](README_zh.md)

This directory contains a dependency-free Xiangqi environment plus runnable AlphaZero, MuZero, checkpoint-evaluation, and terminal-play entry points. The rules core, action encoding, and environment are implemented in LightZero and are suitable for algorithm experiments and regression tests.

## Quick start

Run these commands from the LightZero repository root. The presets use the built-in random bot, CTree search, CUDA when enabled, and a budget of two million environment steps.

```bash
# AlphaZero
python -m zoo.board_games.chinese_chess.config.chinese_chess_alphazero_bot_mode_config

# MuZero
python -m zoo.board_games.chinese_chess.config.chinese_chess_muzero_bot_mode_config
```

For local experiments or rjob overrides, use the common entry point:

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot alphazero \
    --max-env-step 2000000 --num-simulations 50
python -m zoo.board_games.chinese_chess.entry.train_random_bot muzero \
    --max-env-step 2000000 --num-simulations 50
```

Useful overrides include `--seed`, `--exp-name`, `--collector-env-num`, `--evaluator-env-num`, `--n-episode`, `--update-per-collect`, `--batch-size`, `--eval-freq`, `--num-res-blocks`, `--num-channels`, `--max-episode-steps`, `--stop-value`, and `--cpu`. See the complete list with:

```bash
python -m zoo.board_games.chinese_chess.entry.train_random_bot --help
```

If depth or width is overridden during training, pass the same `--num-res-blocks` and `--num-channels` values when evaluating or playing the checkpoint.

## MDP definition

The environment is registered as `chinese_chess` and follows the DI-engine environment interface.

| Component | Definition and implementation |
| --- | --- |
| State `s` | `XiangqiState` stores the 10×9 board, side to move, no-capture halfmove clock, ply count, repetition data, and a stable position hash. The environment also keeps the latest four board frames. |
| Observation | `obs['observation']` is `(57, 10, 9)` `float32` by default: four frames × 14 binary piece planes (seven piece types for the side to move and seven for the opponent), plus one side-to-move plane. Black positions are rotated 180° so the network always sees the current player moving “forward”. With `channel_last=True`, the shape is `(10, 9, 57)`. |
| Action mask | `obs['action_mask']` is an `int8` vector of length 2,086. A one marks a legal action in the current position. Policies and search must apply this mask before selecting an action. |
| Action `a` | `Discrete(2086)`. The labels enumerate geometrically possible rook/cannon/king/pawn line moves, horse moves, and palace/river-restricted advisor/elephant moves. Position legality is checked by the rules core. `action_encoding.py` provides action↔move and action↔ICCS conversions. |
| Reward `r` | Zero before termination; `+1` for an agent win, `-1` for an agent loss, and `0` for a draw. Checkmate, stalemate, or a captured general is a loss; threefold repetition, 60 no-capture moves, and the step limit are draws. |
| Termination | A rules result or `max_episode_steps` ends the episode. The default limit is 500 physical plies. `info['eval_episode_return']` contains the final game return. |
| `to_play` | In `self_play_mode` it identifies the current player. In `play_with_bot_mode` it is always `-1`, because the transition is expressed from the single agent's perspective; `current_player_index` still records the rules engine's side to move. |

### Battle modes

- `self_play_mode`: one `env.step(action)` executes one physical move and then changes the side to move. This is used by AlphaZero self-play and simulation environments.
- `play_with_bot_mode`: the agent moves first; if the game is not over, the built-in bot moves once. The returned observation is after both physical plies, and the reward is from the agent's perspective. With the default 500-ply limit, an agent episode has at most about 250 decisions. The bot samples a legal move using NumPy's RNG; `env.seed(seed)` makes the sequence reproducible.
- `eval_mode`: uses the bot by default; with `agent_vs_human=True`, the second half-move is read from the terminal.

Wrapping the bot reply inside one transition is convenient for single-agent training, but it also makes the transition stochastic from the agent's `(state, action)` alone: MuZero does not observe the bot's intermediate move. Use `self_play_mode` or `simulate_action` when testing one-ply deterministic dynamics.

## Implementation map

- `envs/xiangqi.py`: standalone rules core covering piece movement, horse-leg and elephant-eye blocking, cannon screens, palaces, river restrictions, flying generals, self-check filtering, checkmate/stalemate, repetition, and the 60-move no-capture rule.
- `envs/action_encoding.py`: builds the fixed 2,086-label action space, conversion helpers, ICCS formatting, and the 180° mirrored action mapping used for the Black canonical view.
- `envs/chinese_chess_env.py`: handles history planes, canonical views, masks, bot/human turns, rewards, termination, and CTree state serialization. The serialized `int16` payload contains board history, turn/clock metadata, and recent position hashes.
- `config/*_bot_mode_config.py`: default AlphaZero and MuZero experiments.
- `entry/train_random_bot.py`: common training CLI; `entry/evaluate_random_bot.py`: multi-seed checkpoint evaluation; `entry/*_bot_human.py`: terminal play.
- `entry/chinese_chess_random_bot_human.py`: minimal terminal example for a human (Red) versus the built-in random bot (Black).

## Current presets

Both presets use `play_with_bot_mode`, eight collector environments, four evaluator environments, AdamW with learning rate `3e-4`, batch size `256`, 100 updates per collection, a two-million-environment-step budget, and 500 physical plies per game. The default search budget is 50 simulations and the early-stop threshold is mean return `0.9`.

AlphaZero uses six residual blocks with 128 channels and improves policy targets with CTree MCTS. MuZero uses the same 57×10×9 representation with representation, dynamics, and prediction networks; its current preset uses five unroll steps, `td_steps=100`, discount `1.0`, and scalar normalized (`not_one_hot`) action encoding. The config passes `reward_support_range=(-1, 1, 1)` and `value_support_range=(-1, 1, 1)`; because `torch.arange` excludes its upper bound, the actual categorical support is `{-1, 0}` even though the environment can return terminal reward `+1`. To cover `{-1, 0, +1}`, set the upper bound to at least `2` and retrain. The exact source of truth is the corresponding config file.

Training directories normally contain `ckpt/ckpt_best.pth.tar`, per-iteration `ckpt/iteration_*.pth.tar`, logs, and a config snapshot. Use `--exp-name` to choose the output directory; use `ckpt_best.pth.tar` for evaluation and terminal play.

## Checkpoint evaluation

The evaluator reports mean return:

```text
mean_return = win_rate - loss_rate
```

The preset stops early at `mean_return >= 0.9`. Because an internal evaluator uses relatively few games, a single evaluator result is not a robust convergence claim. Evaluate the best checkpoint on at least 100 games for each of three fixed seeds:

```bash
python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    alphazero /path/to/ckpt_best.pth.tar \
    --episodes 100 --seeds 0 1 2 --simulations 200

python -m zoo.board_games.chinese_chess.entry.evaluate_random_bot \
    muzero /path/to/ckpt_best.pth.tar \
    --episodes 100 --seeds 0 1 2 --simulations 200
```

The command prints aggregate win, draw, and loss rates. AlphaZero simulation counts must be positive multiples of four; MuZero accepts any positive integer.

## Terminal play against a checkpoint

Both commands load the checkpoint as the Red bot; the human plays Black. The default bot search budget is 200 simulations per move:

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_alphazero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200

python -m zoo.board_games.chinese_chess.entry.chinese_chess_muzero_bot_human \
    /path/to/ckpt_best.pth.tar --simulations 200
```

The terminal prints the board and accepts ICCS coordinates: files `a`–`i`, ranks `0`–`9`; for example, `h9g7` moves from h9 to g7. Enter `q`, `quit`, or `exit` to leave. Pieces are rendered as `K/k` king, `A/a` advisor, `E/e` elephant, `H/h` horse, `R/r` rook, `C/c` cannon, and `P/p` pawn. If training used non-default model dimensions, pass the same `--num-res-blocks` and `--num-channels` overrides.

## Terminal play against the built-in random bot

To learn the environment API without a checkpoint, run:

```bash
python -m zoo.board_games.chinese_chess.entry.chinese_chess_random_bot_human --seed 0
```

You play Red and enter ICCS moves in the terminal; after each human move,
`play_with_bot_mode` automatically applies one uniformly sampled legal Black
move. This is a random baseline, not an Xiangqi MCTS engine.

## Direct environment use

```python
from zoo.board_games.chinese_chess.envs.chinese_chess_env import ChineseChessEnv

env = ChineseChessEnv({'battle_mode': 'play_with_bot_mode'})
obs = env.reset()
action = env.legal_actions[0]
timestep = env.step(action)
print(obs['observation'].shape, obs['action_mask'].sum())
```

For one physical move at a time:

```python
env = ChineseChessEnv({'battle_mode': 'self_play_mode'})
env.reset()
next_env = env.simulate_action(env.legal_actions[0])  # leaves env unchanged
```

## Rule boundaries

Threefold repetition is scored as a draw. Complex tournament adjudication of long-check and long-chase cases is outside the current scope. This keeps termination and reward semantics stable and reproducible, but is not a complete implementation of every competition rule.
