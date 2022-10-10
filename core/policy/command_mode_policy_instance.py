from ding.rl_utils import get_epsilon_greedy_fn

from .efficientzero import EfficientZeroPolicy
from .efficientzero_expert_data import EfficientZeroExertDataPolicy

from ding.utils import POLICY_REGISTRY
from ding.policy.base_policy import CommandModePolicy

# from core.utils import POLICY_REGISTRY  # NOTE: use customized POLICY_REGISTRY
# from .base_policy import CommandModePolicy


class EpsCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Set the eps_greedy rule according to the config for command
        """
        eps_cfg = self._cfg.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_train_iter', 'collector_envstep']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        # Decay according to `learner_train_iter`
        # step = command_info['learner_train_iter']
        # Decay according to `envstep`
        step = command_info['envstep']
        return {'eps': self.epsilon_greedy(step)}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


class DummyCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        pass

    def _get_setting_collect(self, command_info: dict) -> dict:
        return {}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


@POLICY_REGISTRY.register('efficientzero_command')
class EfficientZeroCommandModePolicy(EfficientZeroPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('efficientzero_expert_data_command')
class EfficientZeroExpertDataCommandModePolicy(EfficientZeroExertDataPolicy, DummyCommandModePolicy):
    pass
