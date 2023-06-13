
import pytest
from easydict import EasyDict
from gobigger_env import GoBiggerLightZeroEnv
from gobigger_rule_bot import BotAgent


env_cfg=EasyDict(dict(
    env_name='gobigger',
    team_num=2,
    player_num_per_team=2,
    direction_num=12,
    step_mul=8,
    map_width=64,
    map_height=64,
    frame_limit=3600,
    action_space_size=27,
    use_action_mask=False,
    reward_div_value=0.1,
    reward_type='log_reward',
    contain_raw_obs=True, # False on collect mode, True on eval vsbot mode, because bot need raw obs
    start_spirit_progress=0.2,
    end_spirit_progress=0.8,
    manager_settings=dict(
            food_manager=dict(
                num_init=260,
                num_min=260,
                num_max=300,
            ),
            thorns_manager=dict(
                num_init=3,
                num_min=3,
                num_max=4,
            ),
            player_manager=dict(
                ball_settings=dict(
                    score_init=13000,
                ),
            ),
    ),
    playback_settings=dict(
        playback_type='by_frame',
        by_frame=dict(
            # save_frame=False,
            save_frame=True,
            save_dir='./',
            save_name_prefix='test',
        ),
    ),
))

@pytest.mark.envtest
class TestGoBiggerLightZeroEnv:

    def test_env(self):
        env = GoBiggerLightZeroEnv(env_cfg)
        obs = env.reset()
        from gobigger_rule_bot import BotAgent
        bot = [BotAgent(i) for i in range(4)]
        while True:
            actions = {}
            for i in range(4):
                # bot[i].step(obs['raw_obs'] is dict
                actions.update(bot[i].step(obs['raw_obs']))
            obs, rew, done, info = env.step(actions)
            print(rew, info)
            if done:
                break
