import pytest
from lzero.entry import eval_muzero
from test_atari_sampled_efficientzero_config import create_config, main_config
from gym.wrappers import RecordVideo

@pytest.mark.envtest
class TestAtariLightZeroEnvVisualization:

    def test_naive_env(self):
        import gym, random
        env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
        env = RecordVideo(env, video_folder='./', name_prefix='navie')
        env.reset()
        score=0
        while True:
            action = random.choice([0,1,2,3])
            obs, reward, done, info = env.step(action)       
            score+=reward
            if done:
                break
        print('Score:{}'.format(score))
        env.close()

    def test_lightzero_env(self):
        create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
        main_config.env.evaluator_env_num = 1    # Visualization requires the 'env_num' to be set as 1
        main_config.env.n_evaluator_episode = 2
        main_config.env.render_mode_human = True
        main_config.env.save_video = True
        main_config.env.save_path = './'
        main_config.env.eval_max_episode_steps=int(1e2) # Set as needed
        model_path = "/path/ckpt/ckpt_best.pth.tar"

        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=0,
            num_episodes_each_seed=1,
            print_seed_details=False,
            model_path=model_path
        )
        print(returns_mean, returns)

TestAtariLightZeroEnvVisualization().test_naive_env()
TestAtariLightZeroEnvVisualization().test_lightzero_env()