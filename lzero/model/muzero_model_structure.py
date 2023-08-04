from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, PredictionNetworkMLP
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean
from lzero.model.muzero_model_mlp import MuZeroModelMLP, DynamicsNetwork



@MODEL_REGISTRY.register('MuZeroModelStructure')
class MuZeroModelMLPStructure(MuZeroModelMLP):

    def __init__(
        self,
        env_name: str,
        action_space_size: int = 6,
        latent_state_dim: int = 256,
        fc_reward_layers: SequenceType = [32],
        fc_value_layers: SequenceType = [32],
        fc_policy_layers: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        discrete_action_encoding_type: str = 'one_hot',
        norm_type: Optional[str] = 'BN',
        res_connection_in_dynamics: bool = False,
        *args,
        **kwargs
    ):
        """
        Overview:
            The definition of the network model of MuZero, which is a generalization version for 1D vector obs.
            The networks are mainly built on fully connected layers.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP network which predicts the next latent state, and reward given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, e.g. 4 for Lunarlander.
            - latent_state_dim (:obj:`int`): The dimension of latent state, such as 256.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            - discrete_action_encoding_type (:obj:`str`): The encoding type of discrete action, which can be 'one_hot' or 'not_one_hot'.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
        """
        super(MuZeroModelMLP, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.action_space_size = action_space_size
        self.continuous_action_space = False
        # The dim of action space. For discrete action space, it is 1.
        # For continuous action space, it is the dimension of continuous action.
        self.action_space_dim = action_space_size if self.continuous_action_space else 1
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.continuous_action_space:
            self.action_encoding_dim = action_space_size
        else:
            if self.discrete_action_encoding_type == 'one_hot':
                self.action_encoding_dim = action_space_size
            elif self.discrete_action_encoding_type == 'not_one_hot':
                self.action_encoding_dim = 1

        self.latent_state_dim = latent_state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics

        if env_name == 'gobigger':
            from lzero.model.gobigger.gobigger_encoder import GoBiggerEncoder as Encoder
        elif env_name == 'ptz_simple_spread':
            from lzero.model.petting_zoo.encoder import PettingZooEncoder as Encoder
        else:
            raise NotImplementedError
        self.representation_network = Encoder()

        self.dynamics_network = DynamicsNetwork(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=self.latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            fc_reward_layers=fc_reward_layers,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )

        self.prediction_network = PredictionNetworkMLP(
            action_space_size=action_space_size,
            num_channels=latent_state_dim,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type
        )

        if self.self_supervised_learning_loss:
            # self_supervised_learning_loss related network proposed in EfficientZero
            self.projection_input_dim = latent_state_dim

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of MuZero model, which is the first step of the MuZero model.
            To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
            Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state", and
            also prepare the zeros-like ``reward`` for the next step of the MuZero model.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The hidden state of LSTM about reward. In initial inference, \
                we set it to the zeros-like hidden state (H and C).
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        batch_size = obs['action_mask'].shape[0]
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )
