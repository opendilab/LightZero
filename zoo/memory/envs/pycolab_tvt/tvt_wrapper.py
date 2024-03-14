import numpy as np
import gym
from envs.key_to_door import env, key_to_door
from envs.key_to_door import env, visual_match


class VisualMatch(gym.Env):
    def __init__(
        self,
        num_apples=10,
        apple_reward=1.0,
        fix_apple_reward_in_episode=True,
        final_reward=10.0,
        default_reward=0,
        respawn_every=20,
        passive=True,
        REWARD_GRID=visual_match.REWARD_GRID,
        max_frames=visual_match.MAX_FRAMES_PER_PHASE,
        crop=True,
        flatten_img=True,
        one_hot_actions=False,
    ):
        explore_grid = (
            visual_match.PASSIVE_EXPLORE_GRID if passive else visual_match.EXPLORE_GRID
        )
        super().__init__()
        self.pycolab_env = env.PycolabEnvironment(
            game="visual_match",
            num_apples=num_apples,
            apple_reward=apple_reward,
            fix_apple_reward_in_episode=fix_apple_reward_in_episode,
            final_reward=final_reward,
            respawn_every=respawn_every,
            crop=crop,
            default_reward=default_reward,
            REWARD_GRID=REWARD_GRID,
            EXPLORE_GRID=explore_grid,
            max_frames=max_frames,
        )

        self.action_space = gym.spaces.Discrete(4)  # 4 directions
        self.one_hot_actions = one_hot_actions

        # original agent uses HWC size, but pytorch uses CHW size, so we transpose below
        self.img_size = (3, 5, 5)
        self.image_space = gym.spaces.Box(
            shape=self.img_size, low=0, high=255, dtype=np.uint8
        )
        # NOTE: uint8 is important
        # the pixel normalization should be done in image encoder, not here

        self.flatten_img = flatten_img
        if flatten_img:
            self.observation_space = gym.spaces.Box(
                shape=(np.array(self.img_size).prod(),), low=0, high=255, dtype=np.uint8
            )
        else:
            self.observation_space = self.image_space

    def _convert_obs(self, obs):
        new_obs = np.transpose(obs, (-1, 0, 1))  # (H,W,C) -> (C,H,W)
        if self.flatten_img:
            new_obs = new_obs.flatten()  # -> (C*H*W)
        return new_obs

    def step(self, action):
        if self.one_hot_actions:
            action = np.argmax(action)
        obs, r = self.pycolab_env.step(action)
        self._ret += r

        info = {}

        if self.pycolab_env._episode.game_over:
            done = True
            info["success"] = self.pycolab_env.last_phase_reward() > 0.0
        else:
            done = False

        return self._convert_obs(obs), r, done, info

    def reset(self):
        obs, _ = self.pycolab_env.reset()
        self._ret = 0.0

        return self._convert_obs(obs)


class KeyToDoor(gym.Env):
    def __init__(
        self,
        num_apples=10,
        apple_reward=1.0,
        fix_apple_reward_in_episode=True,
        final_reward=10.0,
        default_reward=0,
        respawn_every=20,
        REWARD_GRID=key_to_door.REWARD_GRID_SR,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_SR,
        crop=True,
        flatten_img=True,
        one_hot_actions=False,
    ):
        super().__init__()
        self.pycolab_env = env.PycolabEnvironment(
            game="key_to_door",
            num_apples=num_apples,
            apple_reward=apple_reward,
            fix_apple_reward_in_episode=fix_apple_reward_in_episode,
            final_reward=final_reward,
            respawn_every=respawn_every,
            crop=crop,
            default_reward=default_reward,
            REWARD_GRID=REWARD_GRID,
            max_frames=max_frames,
        )

        self.action_space = gym.spaces.Discrete(4)  # 4 directions
        self.one_hot_actions = one_hot_actions

        # original agent uses HWC size, but pytorch uses CHW size, so we transpose below
        self.img_size = (3, 5, 5)
        self.image_space = gym.spaces.Box(
            shape=self.img_size, low=0, high=255, dtype=np.uint8
        )
        # NOTE: uint8 is important
        # the pixel normalization should be done in image encoder, not here

        self.flatten_img = flatten_img
        if flatten_img:
            self.observation_space = gym.spaces.Box(
                shape=(np.array(self.img_size).prod(),), low=0, high=255, dtype=np.uint8
            )
        else:
            self.observation_space = self.image_space

    def _convert_obs(self, obs):
        new_obs = np.transpose(obs, (-1, 0, 1))  # (H,W,C) -> (C,H,W)
        if self.flatten_img:
            new_obs = new_obs.flatten()  # -> (C*H*W)
        return new_obs

    def step(self, action):
        if self.one_hot_actions:
            action = np.argmax(action)
        obs, r = self.pycolab_env.step(action)
        self._ret += r

        info = {}

        if self.pycolab_env._episode.game_over:
            done = True
            info["success"] = self.pycolab_env.last_phase_reward() > 0.0
        else:
            done = False

        return self._convert_obs(obs), r, done, info

    def reset(self):
        obs, _ = self.pycolab_env.reset()
        self._ret = 0.0

        return self._convert_obs(obs)


if __name__ == "__main__":
    env = KeyToDoor()
    obs = env.reset()
    done = False
    t = 0
    while not done:
        t += 1
        obs, rew, done, info = env.step(env.action_space.sample())
        print(t, rew, info)

    # import ipdb; ipdb.set_trace()
