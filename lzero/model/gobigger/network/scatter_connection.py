from typing import Tuple

import torch
import torch.nn as nn


class ScatterConnection(nn.Module):
    r"""
        Overview:
            Scatter feature to its corresponding location
            In alphastar, each entity is embedded into a tensor, these tensors are scattered into a feature map
            with map size
    """

    def __init__(self, scatter_type='add') -> None:
        r"""
            Overview:
                Init class
            Arguments:
                - scatter_type (:obj:`str`): add or cover, if two entities have same location, scatter type decides the
                    first one should be covered or added to second one
        """
        super(ScatterConnection, self).__init__()
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

    def xy_forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], coord_x: torch.Tensor,coord_y) -> torch.Tensor:
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = (coord_x * W + coord_y).long()
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize,
                                                                                         H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)
        return output

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        """
            Overview:
                scatter x into a spatial feature map
            Arguments:
                - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                  the dimension of entity attributes
                - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into
                - location (:obj:`tensor`): :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
            Returns:
                - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the\
                    scattered feature map
            Shapes:
                - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                  the dimension of entity attributes
                - Size: Tuple[H, W]
                - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
                - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size

            .. note::
                when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
                use the addition as temporal substitute.
        """
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = location[:, :, 1] + location[:, :, 0] * W
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize,
                                                                                         H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)

        # device = x.device
        # B, M, N = x.shape
        # H, W = spatial_size
        # index = location.view(-1, 2)
        # bias = torch.arange(B).mul_(H * W).unsqueeze(1).repeat(1, M).view(-1).to(device)
        # index = index[:, 0] * W + index[:, 1]
        # index += bias
        # index = index.repeat(N, 1)
        # x = x.view(-1, N).permute(1, 0)
        # output = torch.zeros(N, B * H * W, device=device)
        # if self.scatter_type == 'cover':
        #     output.scatter_(dim=1, index=index, src=x)
        # elif self.scatter_type == 'add':
        #     output.scatter_add_(dim=1, index=index, src=x)
        # output = output.reshape(N, B, H, W)
        # output = output.permute(1, 0, 2, 3).contiguous()

        return output


if __name__ == '__main__':
    scatter_conn = ScatterConnection()
    BatchSize, Num, EmbeddingSize = 10, 20, 3
    SpatialSize = (13, 17)
    for _ in range(10):
        x = torch.randn(size=(BatchSize, Num, EmbeddingSize))
        locations = torch.randint(low=0, high=12, size=(BatchSize, Num, 2))
        scatter_conn.forward(x, SpatialSize, location=locations)
