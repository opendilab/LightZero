import pytest
from zoo.atari.envs.atari_lightzero_env import AtariLightZeroEnv
from easydict import EasyDict


cfg = EasyDict(
    env_name='PongNoFrameskip-v4',   # action_space_size=6
    # env_name='QbertNoFrameskip-v4',   # action_space_size=6
    # env_name='BreakoutNoFrameskip-v4',   # action_space_size=4
    # env_name='MsPacmanNoFrameskip-v4',  # action_space_size=9
    # render_mode_human=True,
    render_mode_human=False,
    frame_skip=4,
    frame_stack=4,
    episode_life=True,
    channel_last=True,
    obs_shape=(3, 96, 96),
    discount_factor=0.997,
    max_episode_steps=1.08e5,
    game_wrapper=True,
    dqn_expert_data=False,
    clip_rewards=True,
    scale=True,
    warp_frame=True,
    save_video=False,
    gray_scale=True,
    # gray_scale=False,
    # trade memory for speed
    transform2string=False,
    stop_value=int(1e6),
)


@pytest.mark.envtest
class TestAtariLightZeroEnv:

    def test_naive(self):
        env = AtariLightZeroEnv(cfg)
        env.reset()
        # env.render()
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            obs, reward, done, info = env.step(action)
            # print(obs['observation'].shape)
            if done:
                print(info)
                break


TestAtariLightZeroEnv().test_naive()