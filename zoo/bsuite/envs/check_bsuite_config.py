try:
    import bsuite
    from bsuite import sweep
except ImportError:
    raise ImportError("Please install the bsuite package: pip install bsuite")


# List the configurations for the given experiment
for bsuite_id in sweep.BANDIT_NOISE:
    env = bsuite.load_from_id(bsuite_id)
    print('bsuite_id={}, settings={}, num_episodes={}'
          .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

for bsuite_id in sweep.CARTPOLE_SWINGUP:
    env = bsuite.load_from_id(bsuite_id)
    print('bsuite_id={}, settings={}, num_episodes={}'
          .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

for bsuite_id in sweep.MEMORY_LEN:
    env = bsuite.load_from_id(bsuite_id)
    print('bsuite_id={}, settings={}, num_episodes={}'
          .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

for bsuite_id in sweep.MEMORY_SIZE:
    env = bsuite.load_from_id(bsuite_id)
    print('bsuite_id={}, settings={}, num_episodes={}'
          .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))
