# NumPy 1 and 2 training compatibility

This branch accepts `numpy>=1.24.1,<3` and pins DI-engine to
[`ce54f6ba`](https://github.com/opendilab/DI-engine/pull/882), which removes its
independent NumPy `<2` restriction and requires a NumPy-2-compatible PyTorch
on Python 3.9+. A compatible DI-engine release should replace the Git commit
pin once available. The pin is temporary but immutable; installing this
branch requires Git and access to GitHub as well as the package index.

NumPy 2.5.5 needs Python 3.12; Python 3.11 resolves NumPy 2.4.6 instead.
The CI matrix covers Python 3.11 / NumPy 1.26.4, Python 3.11 / latest NumPy 2,
and Python 3.12 / latest NumPy 2. It performs a normal full-stack installation,
`pip check`, native extension compilation, environment interaction, replay
sampling, MCTS, and a real MuZero learner update. A changed, finite model is
required for the training smoke test to pass.

## Installation

Use a fresh environment; existing applications may impose conflicting NumPy
constraints. For a CPU-only setup:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . 'numpy>=2,<3' pytest-timeout 'opencv-python-headless>=4.10'
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q tests/test_numpy_training_compatibility.py \
  tests/test_gymnasium_migration.py \
  lzero/mcts/tests/test_game_buffer_index_alignment.py \
  lzero/worker/tests/test_muzero_evaluator.py \
  zoo/minigrid/tests/test_minigrid_muzero_rnd.py
```

For NumPy 1, use Python 3.11 and substitute `numpy==1.26.4` for the NumPy
requirement. The native-only ABI matrix in [PR #515](https://github.com/opendilab/LightZero/pull/515)
is complementary; it does not replace the full-stack installation/training gate.

## Environment and validation scope

The generic environment factory now creates Gymnasium environments. DI-engine
adapts them to the existing LightZero timestep API; the wrappers require Gymnasium spaces,
and discretized action spaces are available before the
environment manager snapshots them. MountainCar also uses Gymnasium and
combines termination/truncation correctly. NumPy arrays containing a single
discretized action are converted with `.item()` instead of implicit scalar
conversion, which NumPy 2 rejects.

The end-to-end smoke test also covers the evaluator's standard
`eval_episode_return` fallback; environments providing `score` retain that
metric. This was a pre-existing training blocker discovered during validation.

The tested paths include CartPole, continuous/discretized Pendulum,
MountainCar, MiniGrid observation/horizon contracts, replay priorities, and a
short CPU MuZero training run. Long-run learning quality, GPU training, Atari
ROMs, and optional third-party simulators are not certified by these tests.
Raw legacy Gym environments still use removed NumPy APIs; create Gymnasium
environments or keep those integrations on NumPy 1. No global NumPy aliases,
metadata patches in site-packages, or `--no-deps` installation bypass is used.
