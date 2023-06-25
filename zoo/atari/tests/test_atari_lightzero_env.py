import pytest
from zoo.atari.envs.atari_lightzero_env import AtariLightZeroEnv
from easydict import EasyDict

config = EasyDict(dict(
    collector_env_num=8,
    evaluator_env_num=3,
    n_evaluator_episode=3,
    env_name='PongNoFrameskip-v4',
    env_type='Atari',
    obs_shape=(4, 96, 96),
    collect_max_episode_steps=int(1.08e5),
    eval_max_episode_steps=int(1.08e5),
    gray_scale=True,
    frame_skip=4,
    episode_life=True,
    clip_rewards=True,
    channel_last=True,
    render_mode_human=False,
    scale=True,
    warp_frame=True,
    save_video=False,
    transform2string=False,
    game_wrapper=True,
    manager=dict(shared_memory=False, ),
    stop_value=int(1e6),
))

config.max_episode_steps = config.eval_max_episode_steps

@pytest.mark.envtest
class TestAtariLightZeroEnv:
    def test_naive(self):
        env = AtariLightZeroEnv(config)
        env.reset()
        while True:
            action = env.random_action()
            obs, reward, done, info = env.step(action)
            if done:
                print(info)
                break