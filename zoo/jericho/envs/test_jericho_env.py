from easydict import EasyDict
from .jericho_env import JerichoEnv
import numpy as np
import pytest


@pytest.mark.unittest
class TestJerichoEnv():
    def setup(self) -> None:
        # Configuration for the Jericho environment
        cfg = EasyDict(
            dict(
                game_path="z-machine-games-master/jericho-game-suite/zork1.z5",
                max_action_num=50,
                tokenizer_path="google-bert/bert-base-uncased",
                max_seq_len=512
            )
        )
        # Create a Jericho environment that will be used in the following tests.
        self.env = JerichoEnv(cfg)

    # Test the initialization of the Jericho environment.
    def test_initialization(self):
        assert isinstance(self.env, JerichoEnv)

    # Test the reset method of the Jericho environment.
    # Ensure that the shape of the observation is as expected.
    def test_reset(self):
        obs = self.env.reset()
        assert obs['observation'].shape == (512,)

    # Test the step method of the Jericho environment.
    # Ensure that the shape of the observation, the type of the reward,
    # the type of the done flag and the type of the info are as expected.
    def test_step_shape(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(1)
        assert obs['observation'].shape == (512,)
        assert isinstance(reward, np.ndarray)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
