import argparse
import numpy as np
from easydict import EasyDict
from zoo.memory.envs.memory_lightzero_env import MemoryEnvLightZero


def main():
    parser = argparse.ArgumentParser(description='Test MemoryEnvLightZero with random or human actions')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'human'],
                        help='Action mode (default: random)')
    # parser.add_argument('--save_replay', action='store_true', help='Enable saving GIF replay')
    # parser.add_argument('--render', action='store_true', help='Enable real-time rendering')
    parser.add_argument('--save_replay', type=bool, default=True, help='Whether to save GIF replay')
    parser.add_argument('--render', type=bool, default=True, help='Whether to enable real-time rendering')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to run')

    args = parser.parse_args()

    config = dict(
        env_name='visual_match',  # The name of the environment, options: 'visual_match', 'key_to_door'
        # max_step=60,  # The maximum number of steps for each episode
        num_apples=10,  # Number of apples in the distractor phase
        # apple_reward=(1, 10),  # Range of rewards for collecting an apple
        # apple_reward=(1, 1),  # Range of rewards for collecting an apple
        apple_reward=(0, 0),  # Range of rewards for collecting an apple
        fix_apple_reward_in_episode=False,  # Whether to fix apple reward (DEFAULT_APPLE_REWARD) within an episode
        final_reward=10.0,  # Reward for choosing the correct door in the final phase
        respawn_every=300,  # Respawn interval for apples
        crop=True,  # Whether to crop the observation
        max_frames={
            "explore": 15,
            "distractor": 30,
            "reward": 15
        },  # Maximum frames per phase
        save_replay=args.save_replay,
        render=args.render,
        scale_observation=True,
    )

    for i in range(args.num_episodes):
        env = MemoryEnvLightZero(EasyDict(config))
        env.seed(i+args.seed)
        obs = env.reset()
        done = False
        episode_return = 0

        while not done:
            action = env.random_action() if args.mode == 'random' else get_human_action(env)
            timestep = env.step(action)
            obs = timestep.obs
            reward = timestep.reward
            done = timestep.done
            info = timestep.info
            episode_return += reward
            print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        print(f"Episode {i} finished with return: {episode_return}")


def get_human_action(env):
    action = None
    while action is None:
        try:
            action_input = input("Enter an action (0-3): ")
            action = np.array([int(action_input)], dtype=np.int64)
            if action < 0 or action >= env.action_space.n:
                raise ValueError
        except ValueError:
            print("Invalid action. Please enter a number between 0 and 3.")
            action = None
    return action


if __name__ == '__main__':
    main()
    # python test_render.py --save_replay --render --mode human
    # python test_render.py --save_replay --render --mode random
