from typing import Optional, Dict

import torch
import torch.nn as nn
from ding.torch_utils import fc_block, noise_block, NoiseLinearLayer, MLP


class DuelingHeadM(nn.Module):
    """
        Overview:
            The ``DuelingHead`` used to output discrete actions logit. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        a_layer_num: Optional[int] = None,
        v_layer_num: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
        ensemble_num: Optional[int] = 1,
    ) -> None:
        """
        Overview:
            Init the ``DuelingHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``DuelingHead``.
            - output_size (:obj:`int`): The number of outputs.
            - a_layer_num (:obj:`int`): The number of layers used in the network to compute action output.
            - v_layer_num (:obj:`int`): The number of layers used in the network to compute value output.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
        """
        super(DuelingHeadM, self).__init__()

        self.output_size = output_size
        self.ensemble_num = ensemble_num

        if a_layer_num is None:
            a_layer_num = layer_num
        if v_layer_num is None:
            v_layer_num = layer_num
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.A = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                a_layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, output_size * self.ensemble_num)
        )
        self.V = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                v_layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, 1 * self.ensemble_num)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``DuelingHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keyword ``logit`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.

        Examples:
            >>> head = DuelingHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
        """
        # a = self.A(x)
        # v = self.V(x)
        # q_value = a - a.mean(dim=-1, keepdim=True) + v
        # return {'logit': q_value}

        a = self.A(x)  # shape: (B, A*self.ensemble_num)
        v = self.V(x)  # shape: (B, self.ensemble_num)

        a = a.view(-1, self.output_size, self.ensemble_num)  # shape: (B, A, self.ensemble_num)
        v = v.view(-1, self.ensemble_num)  # shape: (B, self.ensemble_num)
        # v.unsqueeze(1).repeat(1, self.output_size, 1) shape: (B, A, self.ensemble_num)
        q_value = a - a.mean(
            dim=1, keepdim=True
        ) + v.unsqueeze(1).repeat(1, self.output_size, 1)  # shape:  (B, A, self.ensemble_num)

        output = []
        for i in range(self.ensemble_num):
            output.append({'logit': q_value[:, :, i]})
        return output
