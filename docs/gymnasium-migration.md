# Gymnasium environment migration

LightZero's generic environment factory now uses `gymnasium.make()` with Gymnasium
1.0 or later. Install `gymnasium[box2d]` for LunarLander/BipedalWalker and
`gymnasium[atari]` for Atari. MountainCar recording also requires MoviePy.

## Wrapper compatibility

`LightZeroEnvWrapper` and `ActionDiscretizationEnvWrapper` require Gymnasium
observation and action spaces. Passing an old `gym.make()` environment now raises
a `TypeError` at construction with a migration instruction. Change such callers
to `gymnasium.make()`; merely changing the reset/step tuple is insufficient because
old Gym and Gymnasium spaces are different types.

These wrappers accept native Gymnasium reset/step results, as well as the legacy
four-value API produced by DI-engine's adapter around a Gymnasium environment.
LightZero still returns its observation dictionary and `BaseEnvTimestep`, with
`done = terminated or truncated`. Wrapped spaces are initialized at construction
so DI-engine caches the correct dictionary observation and discretized action
spaces. Atari retains its explicit Gymnasium-to-four-value adapter for DI-engine's
wrapper chain. Delegating wrapper classes allow this mixed chain without
requiring every wrapper to inherit the same Gym base class.

DI-engine 0.5.3 still depends on old Gym, so this migration does not remove Gym
from the installation. The old-Gym-native `zoo/dmc2gym` and vendored
`zoo/metadrive` integrations are outside this migration. The PriorZero BabyAI
and TextCraft integrations added on main also retain their existing Gym imports
and are not covered by this PR.

## LunarLander v3

LunarLander configs now use `LunarLander-v3` / `LunarLanderContinuous-v3` because
Gymnasium 1.0+ rejects the deprecated v2 IDs. The v3 changes include resetting wind
and turbulence offsets between episodes, fully destroying the previous Box2D world
to fix nondeterministic resets, and widening observation bounds for positions,
velocities and angles. The observation still has eight components and action
dimensions remain unchanged. Historical v2 learning curves should record their
environment version and should not be treated as directly equivalent v3 baselines.
See the [Gymnasium LunarLander version history](https://gymnasium.farama.org/environments/box2d/lunar_lander/#version-history).

## Seeding and recording

With DI-engine 0.5.3, `dynamic_seed=False` repeats the same initial state on reset
in both the previous Gym path and the Gymnasium path (checked with Pendulum).
This is existing static-seed behavior, not a regression introduced by this PR.
Use dynamic seeding when episode initial states should vary. MountainCar creates
one `RecordVideo` wrapper per environment instance, reuses it across episodes,
and closes/recreates the environment when the recording directory changes.

## Behavioral checks

Run the regression tests explicitly; `pytest lzero/` does not collect `zoo/` or
the top-level `tests/` directory:

```bash
python -m pytest -q tests/test_gymnasium_migration.py \
  lzero/envs/tests/test_lightzero_env_wrapper.py \
  zoo/atari/tests/test_atari_wrappers.py \
  zoo/classic_control/mountain_car/envs/test_mtcar_env.py
```

The regression tests cover continuous/discretized episodes and resets, direct
Gymnasium truncation, explicit old-Gym rejection, LunarLander config creation,
the Atari wrapper chain, MountainCar truncation and two recorded episodes, and
optional Jericho/Pooltool space construction. Optional dependency tests skip when
their dependencies are absent; the Atari chain test uses deterministic frames
and does not claim to benchmark ALE gameplay.

Validated on 2026-11-25 with Python 3.11, DI-engine 0.5.3, NumPy 1.26.4,
Box2D 2.3.10 and ALE 0.12.1: the command above plus
`lzero/envs/tests/test_ding_env_wrapper.py` and
`zoo/atari/tests/test_atari_reset_seed.py` produced **38 passed, 2 skipped**
on both Gymnasium 1.0.0 and 1.4.0. The two skips are optional Jericho and
Pooltool checks, whose packages were not installed. Actual MountainCar MP4
recording passed on both versions. Combined statement coverage for the two
LightZero wrappers, Atari wrappers and MountainCar adapter was 81% (individual
coverage 95%, 96%, 70% and 91%). This is focused migration validation, not a
repository-wide training benchmark.
