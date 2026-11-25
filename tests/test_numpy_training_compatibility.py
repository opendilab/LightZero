"""Exercise environment, replay, search and learning with the installed NumPy."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from easydict import EasyDict


@pytest.mark.parametrize('env_id,continuous,discretize', [
    ('CartPole-v1', False, False),
    ('Pendulum-v1', True, False),
    ('Pendulum-v1', False, True),
])
def test_gymnasium_entry(env_id, continuous, discretize):
    from lzero.envs.get_wrapped_env import get_wrappered_env

    cfg = EasyDict(env_id=env_id, continuous=continuous, manually_discretization=discretize,
                   each_dim_disc_size=5, is_train=True)
    env = get_wrappered_env(cfg, env_id)()
    try:
        env.seed(0, dynamic_seed=False)
        obs = env.reset()
        assert np.isfinite(obs['observation']).all()
        if not continuous:
            assert obs['action_mask'].sum() == env.action_space.n
        for _ in range(5):
            timestep = env.step(env.random_action())
            assert np.isfinite(timestep.reward).all()
            assert np.isfinite(timestep.obs['observation']).all()
            if timestep.done:
                env.reset()
    finally:
        env.close()


def test_mountain_car_seed_and_truncation():
    from zoo.classic_control.mountain_car.envs.mtcar_lightzero_env import MountainCarEnv

    env = MountainCarEnv(EasyDict())
    try:
        env.seed(3, dynamic_seed=False)
        obs = env.reset()['observation']
        np.testing.assert_array_equal(obs, env.reset()['observation'])
        # A zero-throttle action never reaches the goal; the 200-step horizon must terminate.
        for _ in range(200):
            timestep = env.step(np.array([1], dtype=np.int64))
        assert timestep.done
        assert timestep.info['eval_episode_return'] == -200
        assert np.isfinite(timestep.obs['observation']).all()
    finally:
        env.close()


@pytest.mark.timeout(180)
def test_muzero_collect_replay_and_learn(tmp_path, monkeypatch):
    from lzero.entry import train_muzero
    from lzero.model.muzero_model_mlp import MuZeroModelMLP
    from zoo.classic_control.cartpole.config.cartpole_muzero_config import main_config, create_config

    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg, create = deepcopy(main_config), deepcopy(create_config)
    cfg.exp_name = 'numpy-smoke'
    cfg.env.update(collector_env_num=2, evaluator_env_num=1, n_evaluator_episode=1, stop_value=float('inf'))
    cfg.policy.update(cuda=False, collector_env_num=2, evaluator_env_num=1, n_episode=2,
                      batch_size=4, game_segment_length=8, num_unroll_steps=3, td_steps=3,
                      num_simulations=3, update_per_collect=1, replay_buffer_size=1000,
                      reanalyze_ratio=0, ssl_loss_weight=0, use_wandb=False)
    cfg.policy.model.update(latent_state_dim=32, self_supervised_learning_loss=False)
    create.env_manager.type = 'base'
    model = MuZeroModelMLP(**cfg.policy.model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    policy = train_muzero([cfg, create], seed=0, model=model, max_train_iter=1, max_env_step=500)
    parameters = dict(model.named_parameters())
    assert all(torch.isfinite(value).all() for value in parameters.values())
    assert any(not torch.equal(before[name], value) for name, value in parameters.items())
    assert policy.learn_mode is not None
