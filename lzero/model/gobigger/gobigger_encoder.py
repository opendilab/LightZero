from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .network import sequence_mask, ScatterConnection
from .network.encoder import SignBinaryEncoder, BinaryEncoder, OnehotEncoder, TimeEncoder, UnsqueezeEncoder
from .network.nn_module import fc_block, conv2d_block, MLP
from .network.res_block import ResBlock
from .network.transformer import Transformer
from typing import Any, List, Tuple, Union, Optional, Callable
from easydict import EasyDict
from ding.utils.default_helper import deep_merge_dicts

def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] =None):
    r"""
        Overview:
            create a mask for a batch sequences with different lengths
        Arguments:
            - lengths (:obj:`tensor`): lengths in each different sequences, shape could be (n, 1) or (n)
            - max_len (:obj:`int`): the padding size, if max_len is None, the padding size is the
                max length of sequences
        Returns:
            - masks (:obj:`torch.BoolTensor`): mask has the same device as lengths
    """
    if len(lengths.shape) == 1:
        lengths = lengths.unsqueeze(dim=1)
    bz = lengths.numel()
    if max_len is None:
        max_len = lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(bz, 1).lt(lengths).to(lengths.device)


class ScalarEncoder(nn.Module):
    def __init__(self, cfg):
        super(ScalarEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.scalar_encoder
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'time':
                self.encode_modules[k] = TimeEncoder(embedding_dim=item['embedding_dim'])
            elif item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.layers = MLP(in_channels=self.cfg.mlp.input_dim, hidden_channels=self.cfg.mlp.hidden_dim,
                          out_channels=self.cfg.mlp.output_dim,
                          layer_num=self.cfg.mlp.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.mlp.activation,
                          norm_type=self.cfg.mlp.norm_type,
                          use_dropout=False
                          )

    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = self.layers(out)
        return out


class TeamEncoder(nn.Module):
    def __init__(self, cfg):
        super(TeamEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.team_encoder
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.encode_layers = MLP(in_channels=self.cfg.mlp.input_dim,
                                 hidden_channels=self.cfg.mlp.hidden_dim,
                                 out_channels=self.cfg.mlp.output_dim,
                                 layer_num=self.cfg.mlp.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.cfg.mlp.activation,
                                 norm_type=self.cfg.mlp.norm_type,
                                 use_dropout=False)

        self.transformer = Transformer(
            n_heads=self.cfg.transformer.head_num,
            embedding_size=self.cfg.transformer.embedding_dim,
            ffn_size=self.cfg.transformer.ffn_size,
            n_layers=self.cfg.transformer.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.cfg.transformer.activation,
            variant=self.cfg.transformer.variant,
        )
        self.output_fc = fc_block(self.cfg.fc_block.input_dim,
                                  self.cfg.fc_block.output_dim,
                                  norm_type=self.cfg.fc_block.norm_type,
                                  activation=self.cfg.fc_block.activation)

    def forward(self, x):
        embeddings = []
        player_num = x['player_num']
        mask = sequence_mask(player_num, max_len=x['view_x'].shape[1])
        for key, item in self.cfg.modules.items():
            assert key in x, f"{key} not implemented"
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))

        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)
        team_info = self.output_fc(x.sum(dim=1) / player_num.unsqueeze(dim=-1))
        return team_info


class BallEncoder(nn.Module):
    def __init__(self, cfg):
        super(BallEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.ball_encoder
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder()
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError
        self.encode_layers = MLP(in_channels=self.cfg.mlp.input_dim,
                                 hidden_channels=self.cfg.mlp.hidden_dim,
                                 out_channels=self.cfg.mlp.output_dim,
                                 layer_num=self.cfg.mlp.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.cfg.mlp.activation,
                                 norm_type=self.cfg.mlp.norm_type,
                                 use_dropout=False)

        self.transformer = Transformer(
            n_heads=self.cfg.transformer.head_num,
            embedding_size=self.cfg.transformer.embedding_dim,
            ffn_size=self.cfg.transformer.ffn_size,
            n_layers=self.cfg.transformer.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.cfg.transformer.activation,
            variant=self.cfg.transformer.variant,
        )
        self.output_fc = fc_block(self.cfg.fc_block.input_dim,
                                  self.cfg.fc_block.output_dim,
                                  norm_type=self.cfg.fc_block.norm_type,
                                  activation=self.cfg.fc_block.activation)

    def forward(self, x):
        ball_num = x['ball_num']
        embeddings = []
        mask = sequence_mask(ball_num, max_len=x['x'].shape[1])
        for key, item in self.cfg.modules.items():
            assert key in x, key
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))
        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)

        ball_info = x.sum(dim=1) / ball_num.unsqueeze(dim=-1)
        ball_info = self.output_fc(ball_info)
        return x, ball_info


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super(SpatialEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.spatial_encoder

        # scatter related
        self.spatial_x = 64
        self.spatial_y = 64
        self.scatter_fc = fc_block(in_channels=self.cfg.scatter.input_dim, 
                                   out_channels=self.cfg.scatter.output_dim,
                                   activation=self.cfg.scatter.activation, 
                                   norm_type=self.cfg.scatter.norm_type)
        self.scatter_connection = ScatterConnection(self.cfg.scatter.scatter_type)

        # resnet related
        self.get_resnet_blocks()

        self.output_fc = fc_block(
            in_channels=self.spatial_x // 8 * self.spatial_y // 8 * self.cfg.resnet.down_channels[-1],
            out_channels=self.cfg.fc_block.output_dim,
            norm_type=self.cfg.fc_block.norm_type,
            activation=self.cfg.fc_block.activation)

    def get_resnet_blocks(self):
        # 2 means food/spore embedding
        project = conv2d_block(in_channels=self.cfg.scatter.output_dim + 2,
                               out_channels=self.cfg.resnet.project_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               activation=self.cfg.resnet.activation,
                               norm_type=self.cfg.resnet.norm_type,
                               bias=False,
                               )

        layers = [project]
        dims = [self.cfg.resnet.project_dim] + self.cfg.resnet.down_channels
        for i in range(len(dims) - 1):
            layer = conv2d_block(in_channels=dims[i],
                                 out_channels=dims[i + 1],
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 activation=self.cfg.resnet.activation,
                                 norm_type=self.cfg.resnet.norm_type,
                                 bias=False,
                                 )
            layers.append(layer)
            layers.append(ResBlock(in_channels=dims[i + 1],
                                   activation=self.cfg.resnet.activation,
                                   norm_type=self.cfg.resnet.norm_type))
        self.resnet = torch.nn.Sequential(*layers)


    def get_background_embedding(self, coord_x, coord_y, num, ):

        background_ones = torch.ones(size=(coord_x.shape[0], coord_x.shape[1]), device=coord_x.device)
        background_mask = sequence_mask(num, max_len=coord_x.shape[1])
        background_ones = (background_ones * background_mask).unsqueeze(-1)
        background_embedding = self.scatter_connection.xy_forward(background_ones,
                                                                  spatial_size=[self.spatial_x, self.spatial_y],
                                                                  coord_x=coord_x,
                                                                  coord_y=coord_y)

        return background_embedding

    def forward(self, inputs, ball_embeddings, ):
        spatial_info = inputs['spatial_info']
        # food and spore
        food_embedding = self.get_background_embedding(coord_x=spatial_info['food_x'],
                                                       coord_y=spatial_info['food_y'],
                                                       num=spatial_info['food_num'], )

        spore_embedding = self.get_background_embedding(coord_x=spatial_info['spore_x'],
                                                        coord_y=spatial_info['spore_y'],
                                                        num=spatial_info['spore_num'], )
        # scatter ball embeddings
        ball_info = inputs['ball_info']
        ball_num = ball_info['ball_num']
        ball_mask = sequence_mask(ball_num, max_len=ball_embeddings.shape[1])
        ball_embedding = self.scatter_fc(ball_embeddings) * ball_mask.unsqueeze(dim=2)

        ball_embedding = self.scatter_connection.xy_forward(ball_embedding,
                                                            spatial_size=[self.spatial_x, self.spatial_y],
                                                            coord_x=spatial_info['ball_x'],
                                                            coord_y=spatial_info['ball_y'])

        x = torch.cat([food_embedding, spore_embedding, ball_embedding], dim=1)

        x = self.resnet(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.output_fc(x)
        return x


class GoBiggerEncoder(nn.Module):
    config=dict(
        scalar_encoder=dict(
            modules=dict(
                view_x=dict(arc='sign_binary', num_embeddings=7),
                view_y=dict(arc='sign_binary', num_embeddings=7),
                view_width=dict(arc='binary', num_embeddings=7),
                score=dict(arc='one_hot', num_embeddings=10),
                team_score=dict(arc='one_hot', num_embeddings=10),
                rank=dict(arc='one_hot', num_embeddings=4),
                time=dict(arc='time', embedding_dim=8),
                last_action_type=dict(arc='one_hot', num_embeddings=27),
                ),
            mlp=dict(input_dim=80, hidden_dim=64, layer_num=2, norm_type='none', output_dim=32, activation='relu'),
        ),
        team_encoder=dict(
            modules=dict(
                alliance=dict(arc='one_hot', num_embeddings=2),
                view_x=dict(arc='sign_binary', num_embeddings=7),
                view_y=dict(arc='sign_binary', num_embeddings=7),
                ),
            mlp=dict(input_dim=16, hidden_dim=32, layer_num=2, norm_type='none', output_dim=16, activation='relu'),
            transformer=dict(head_num=4, ffn_size=32, layer_num=2, embedding_dim=16, activation='relu', variant='postnorm'),
            fc_block=dict(input_dim=16, output_dim=16, activation='relu', norm_type='none'),
        ),
        ball_encoder=dict(
            modules=dict(
                alliance=dict(arc='one_hot', num_embeddings=4),
                score=dict(arc='one_hot', num_embeddings=50),
                radius=dict(arc='unsqueeze',),
                rank=dict(arc='one_hot', num_embeddings=5),
                x=dict(arc='sign_binary', num_embeddings=8),
                y=dict(arc='sign_binary', num_embeddings=8),
                next_x=dict(arc='sign_binary', num_embeddings=8),
                next_y=dict(arc='sign_binary', num_embeddings=8),
            ),
            mlp=dict(input_dim=92, hidden_dim=128, layer_num=2, norm_type='none', output_dim=64, activation='relu'),
            transformer=dict(head_num=4, ffn_size=64, layer_num=3,  embedding_dim=64, activation='relu', variant='postnorm'),
            fc_block=dict(input_dim=64, output_dim=64, activation='relu', norm_type='none'),
        ),
        spatial_encoder=dict(
            scatter=dict(input_dim=64, output_dim=16, scatter_type='add', activation='relu', norm_type='none'),
            resnet=dict(project_dim=12, down_channels=[32, 32, 16 ], activation='relu', norm_type='none'),
            fc_block=dict(output_dim=64, activation='relu', norm_type='none'),
        ),
    )

    def __init__(self, cfg=None):
        super(GoBiggerEncoder, self).__init__()
        self._cfg = deep_merge_dicts(self.config, cfg)
        self._cfg = EasyDict(self._cfg)
        self.scalar_encoder = ScalarEncoder(self._cfg)
        self.team_encoder = TeamEncoder(self._cfg)
        self.ball_encoder = BallEncoder(self._cfg)
        self.spatial_encoder = SpatialEncoder(self._cfg)

    def forward(self, x):
        scalar_info = self.scalar_encoder(x['scalar_info'])
        team_info = self.team_encoder(x['team_info'])
        ball_embeddings, ball_info = self.ball_encoder(x['ball_info'])
        spatial_info = self.spatial_encoder(x, ball_embeddings)
        x = torch.cat([scalar_info, team_info, ball_info, spatial_info], dim=1)
        return x
