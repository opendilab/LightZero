from lzero.policy.alphazero import AlphaZeroPolicy
from ding.policy.base_policy import POLICY_REGISTRY
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

@POLICY_REGISTRY.register('MyAlphaZeroPolicy')
class MyAlphaZeroPolicy(AlphaZeroPolicy):
    def _get_simulation_env(self):
        # Create a simulation environment using the merged environment config.
        sim_cfg = EasyDict(self.cfg.get('env', {}))
        sim_env = ENV_REGISTRY.get(self.cfg.simulation_env_id)(cfg=sim_cfg)
        return sim_env

    @property
    def simulate_env(self):
        # Lazily initialize and cache the simulation environment.
        if not hasattr(self, '_simulate_env'):
            self._simulate_env = self._get_simulation_env()
        return self._simulate_env
