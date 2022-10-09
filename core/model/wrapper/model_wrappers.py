from abc import ABC
from typing import Any, Tuple, Callable, Optional, List, Dict

import numpy as np
import torch
from ding.rl_utils import create_noise_generator
from torch.distributions import Categorical, Independent, Normal


class IModelWrapper(ABC):
    r"""
    Overview:
        the base class of Model Wrappers
    Interfaces:
        register
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, key: str) -> Any:
        r"""
        Overview:
            Get the attrbute in model.
        Arguments:
            - key (:obj:`str`): The key to query.
        Returns:
            - ret (:obj:`Any`): The queried attribute.
        """
        return getattr(self._model, key)

    def info(self, attr_name):
        r"""
        Overview:
            get info of attr_name
        """
        if attr_name in dir(self):
            if isinstance(self._model, IModelWrapper):
                return '{} {}'.format(self.__class__.__name__, self._model.info(attr_name))
            else:
                if attr_name in dir(self._model):
                    return '{} {}'.format(self.__class__.__name__, self._model.__class__.__name__)
                else:
                    return '{}'.format(self.__class__.__name__)
        else:
            if isinstance(self._model, IModelWrapper):
                return '{}'.format(self._model.info(attr_name))
            else:
                return '{}'.format(self._model.__class__.__name__)


class BaseModelWrapper(IModelWrapper):
    r"""
    Overview:
        the base class of Model Wrappers
    Interfaces:
        register
    """

    def reset(self, data_id: List[int] = None) -> None:
        r"""
        Overview
            the reset function that the Model Wrappers with states should implement
            used to reset the stored states
        """
        pass


def zeros_like(h):
    if isinstance(h, torch.Tensor):
        return torch.zeros_like(h)
    elif isinstance(h, (list, tuple)):
        return [zeros_like(t) for t in h]
    elif isinstance(h, dict):
        return {k: zeros_like(v) for k, v in h.items()}
    else:
        raise TypeError("not support type: {}".format(h))


def sample_action(logit=None, prob=None):
    if prob is None:
        prob = torch.softmax(logit, dim=-1)
    shape = prob.shape
    prob += 1e-8
    prob = prob.view(-1, shape[-1])
    # prob can also be treated as weight in multinomial sample
    action = torch.multinomial(prob, 1).squeeze(-1)
    action = action.view(*shape[:-1])
    return action


class ArgmaxSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to help the model to sample argmax action
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class MultiArgmaxSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to help the model to sample argmax action
    """

    def forward(self, *args, **kwargs):
        output_list = self._model.forward(*args, **kwargs)
        assert isinstance(output_list, list), "model output must be list, but find {}".format(type(output_list))
        ans = []
        for output in output_list:
            logit = output['logit']
            assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
            if isinstance(logit, torch.Tensor):
                logit = [logit]
            if 'action_mask' in output:
                mask = output['action_mask']
                if isinstance(mask, torch.Tensor):
                    mask = [mask]
                logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
            action = [l.argmax(dim=-1) for l in logit]
            if len(action) == 1:
                action, logit = action[0], logit[0]
            output['action'] = action

            ans.append(output)

        return ans


class MultiAverageArgmaxSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to help the model to sample argmax action
    """

    def forward(self, *args, **kwargs):
        output_list = self._model.forward(*args, **kwargs)
        assert isinstance(output_list, list), "model output must be list, but find {}".format(type(output_list))

        ans = {}
        ans['logit'] = output_list
        # get the average q value for each action
        output_tensor = torch.stack([output['logit'] for output in output_list], dim=-1)
        logit = output_tensor.mean(-1)

        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        ans['action'] = action
        return ans


class EpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
        The type of eps can vary from different algorithms, such as:
        - float (i.e. python native scalar): for almost normal case
        - Dict[str, float]: for algorithm NGU
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        if isinstance(eps, dict):
            # for NGU policy, eps is a dict, each collect env has a different eps
            for i, l in enumerate(logit[0]):
                eps_tmp = eps[i]
                if np.random.random() > eps_tmp:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(
                            sample_action(prob=mask[0][i].float().unsqueeze(0)).to(logit[0].device).squeeze(0)
                        )
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]).to(logit[0].device))
            action = torch.stack(action, dim=-1)  # shape torch.size([env_num])
        else:
            for i, l in enumerate(logit):
                if np.random.random() > eps:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(sample_action(prob=mask[i].float()))
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
            if len(action) == 1:
                action, logit = action[0], logit[0]
        output['action'] = action
        return output


class MultiEpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
        The type of eps can vary from different algorithms, such as:
        - float (i.e. python native scalar): for almost normal case
        - Dict[str, float]: for algorithm NGU
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output_list = self._model.forward(*args, **kwargs)
        assert isinstance(output_list, list), "model output must be list, but find {}".format(type(output_list))
        ans = []
        for output in output_list:
            logit = output['logit']
            assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
            if isinstance(logit, torch.Tensor):
                logit = [logit]
            if 'action_mask' in output:
                mask = output['action_mask']
                if isinstance(mask, torch.Tensor):
                    mask = [mask]
                logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
            else:
                mask = None
            action = []
            if isinstance(eps, dict):
                # for NGU policy, eps is a dict, each collect env has a different eps
                for i, l in enumerate(logit[0]):
                    eps_tmp = eps[i]
                    if np.random.random() > eps_tmp:
                        action.append(l.argmax(dim=-1))
                    else:
                        if mask is not None:
                            action.append(
                                sample_action(prob=mask[0][i].float().unsqueeze(0)).to(logit[0].device).squeeze(0)
                            )
                        else:
                            action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]).to(logit[0].device))
                action = torch.stack(action, dim=-1)  # shape torch.size([env_num])
            else:
                for i, l in enumerate(logit):
                    if np.random.random() > eps:
                        action.append(l.argmax(dim=-1))
                    else:
                        if mask is not None:
                            action.append(sample_action(prob=mask[i].float()))
                        else:
                            action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
                if len(action) == 1:
                    action, logit = action[0], logit[0]
            output['action'] = action

            ans.append(output)
        return ans


class MultiAverageEpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
        The type of eps can vary from different algorithms, such as:
        - float (i.e. python native scalar): for almost normal case
        - Dict[str, float]: for algorithm NGU
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output_list = self._model.forward(*args, **kwargs)
        assert isinstance(output_list, list), "model output must be list, but find {}".format(type(output_list))

        ans = {}
        # get the average q value for each action
        output_tensor = torch.stack([output['logit'] for output in output_list], dim=-1)
        logit = output_tensor.mean(-1)

        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        mask = None
        action = []
        if isinstance(eps, dict):
            # for NGU policy, eps is a dict, each collect env has a different eps
            for i, l in enumerate(logit[0]):
                eps_tmp = eps[i]
                if np.random.random() > eps_tmp:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(
                            sample_action(prob=mask[0][i].float().unsqueeze(0)).to(logit[0].device).squeeze(0)
                        )
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]).to(logit[0].device))
            action = torch.stack(action, dim=-1)  # shape torch.size([env_num])
        else:
            for i, l in enumerate(logit):
                if np.random.random() > eps:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(sample_action(prob=mask[i].float()))
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
            if len(action) == 1:
                action, logit = action[0], logit[0]
        ans['action'] = action
        return ans


class EpsGreedyMultinomialSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        if 'alpha' in kwargs.keys():
            alpha = kwargs.pop('alpha')
        else:
            alpha = None
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                if alpha is None:
                    action = [sample_action(logit=l) for l in logit]
                else:
                    # Note that if alpha is passed in here, we will divide logit by alpha.
                    action = [sample_action(logit=l / alpha) for l in logit]
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class ReparamSample(IModelWrapper):
    """
    Overview:
        Reparameterization gaussian sampler used in collector_model.
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        mu, sigma = output['logit']['mu'], output['logit']['sigma']
        dist = Independent(Normal(mu, sigma), 1)
        output['action'] = dist.sample()
        return output


class ActionNoiseWrapper(IModelWrapper):
    r"""
    Overview:
        Add noise to collector's action output; Do clips on both generated noise and action after adding noise.
    Interfaces:
        register, __init__, add_noise, reset
    Arguments:
        - model (:obj:`Any`): Wrapped model class. Should contain ``forward`` method.
        - noise_type (:obj:`str`): The type of noise that should be generated, support ['gauss', 'ou'].
        - noise_kwargs (:obj:`dict`): Keyword args that should be used in noise init. Depends on ``noise_type``.
        - noise_range (:obj:`Optional[dict]`): Range of noise, used for clipping.
        - action_range (:obj:`Optional[dict]`): Range of action + noise, used for clip, default clip to [-1, 1].
    """

    def __init__(
            self,
            model: Any,
            noise_type: str = 'gauss',
            noise_kwargs: dict = {},
            noise_range: Optional[dict] = None,
            action_range: Optional[dict] = {
                'min': -1,
                'max': 1
            }
    ) -> None:
        super().__init__(model)
        self.noise_generator = create_noise_generator(noise_type, noise_kwargs)
        self.noise_range = noise_range
        self.action_range = action_range

    def forward(self, *args, **kwargs):
        # if noise sigma need decay, update noise kwargs.
        if 'sigma' in kwargs:
            sigma = kwargs.pop('sigma')
            if sigma is not None:
                self.noise_generator.sigma = sigma
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        if 'action' in output or 'action_args' in output:
            key = 'action' if 'action' in output else 'action_args'
            action = output[key]
            assert isinstance(action, torch.Tensor)
            action = self.add_noise(action)
            output[key] = action
        return output

    def add_noise(self, action: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Generate noise and clip noise if needed. Add noise to action and clip action if needed.
        Arguments:
            - action (:obj:`torch.Tensor`): Model's action output.
        Returns:
            - noised_action (:obj:`torch.Tensor`): Action processed after adding noise and clipping.
        """
        noise = self.noise_generator(action.shape, action.device)
        if self.noise_range is not None:
            noise = noise.clamp(self.noise_range['min'], self.noise_range['max'])
        action += noise
        if self.action_range is not None:
            action = action.clamp(self.action_range['min'], self.action_range['max'])
        return action

    def reset(self) -> None:
        r"""
        Overview:
            Reset noise generator.
        """
        pass


class TargetNetworkWrapper(IModelWrapper):
    r"""
    Overview:
        Maintain and update the target network
    Interfaces:
        update, reset
    """

    def __init__(self, model: Any, update_type: str, update_kwargs: dict):
        super().__init__(model)
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = update_kwargs
        self._update_count = 0

    def reset(self, *args, **kwargs):
        target_update_count = kwargs.pop('target_update_count', None)
        self.reset_state(target_update_count)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def update(self, state_dict: dict, direct: bool = False) -> None:
        r"""
        Overview:
            Update the target network state dict

        Arguments:
            - state_dict (:obj:`dict`): the state_dict from learner model
            - direct (:obj:`bool`): whether to update the target network directly, \
                if true then will simply call the load_state_dict method of the model
        """
        if direct:
            self._model.load_state_dict(state_dict, strict=True)
            self._update_count = 0
        else:
            if self._update_type == 'assign':
                if (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                    self._model.load_state_dict(state_dict, strict=True)
                self._update_count += 1
            elif self._update_type == 'momentum':
                theta = self._update_kwargs['theta']
                for name, p in self._model.named_parameters():
                    # default theta = 0.001
                    p.data = (1 - theta) * p.data + theta * state_dict[name]

    def reset_state(self, target_update_count: int = None) -> None:
        r"""
        Overview:
            Reset the update_count
        Arguments:
            target_update_count (:obj:`int`): reset target update count value.
        """
        if target_update_count is not None:
            self._update_count = target_update_count


wrapper_name_map = {
    'base': BaseModelWrapper,
    'argmax_sample': ArgmaxSampleWrapper,
    'multi_argmax_sample': MultiArgmaxSampleWrapper,
    'multi_average_argmax_sample': MultiAverageArgmaxSampleWrapper,
    'eps_greedy_sample': EpsGreedySampleWrapper,
    'multi_eps_greedy_sample': MultiEpsGreedySampleWrapper,
    'multi_average_eps_greedy_sample': MultiAverageEpsGreedySampleWrapper,
    'action_noise': ActionNoiseWrapper,
    # model wrapper
    'target': TargetNetworkWrapper,
}


def model_wrap(model, wrapper_name: str = None, **kwargs):
    if wrapper_name in wrapper_name_map:
        if not isinstance(model, IModelWrapper):
            model = wrapper_name_map['base'](model)
        model = wrapper_name_map[wrapper_name](model, **kwargs)
    else:
        raise TypeError("not support model_wrapper type: {}".format(wrapper_name))
    return model


def register_wrapper(name: str, wrapper_type: type):
    r"""
    Overview:
        Register new wrapper to wrapper_name_map
    Arguments:
        - name (:obj:`str`): the name of the wrapper
        - wrapper_type (subclass of :obj:`IModelWrapper`): the wrapper class added to the plguin_name_map
    """
    assert isinstance(name, str)
    assert issubclass(wrapper_type, IModelWrapper)
    wrapper_name_map[name] = wrapper_type
