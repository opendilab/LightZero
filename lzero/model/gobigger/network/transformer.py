import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_NORM_EPS = 1e-5
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


class MultiHeadAttention(nn.Module):
    r"""
    Overview:
        For each entry embedding, compute individual attention across all entries, add them up to get output attention
    """

    def __init__(self, n_heads: int = None, dim: int = None, dropout: float = 0):
        r"""
        Overview:
            Init attention
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): head num for multihead attention
            - dropout (:obj:`nn.Module`): dropout layer
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)

        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

        # self.attention_pre = fc_block(self.dim, self.dim * 3)  # query, key, value
        # self.project = fc_block(self.dim,self.dim)

    def split(self, x, T=False):
        r"""
        Overview:
            Split input to get multihead queries, keys, values
        Arguments:
            - x (:obj:`tensor`): query or key or value
            - T (:obj:`bool`): whether to transpose output
        Returns:
            - x (:obj:`list`): list of output tensors for each head
        """
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                mask: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, query_len, dim = query.size()
        assert (
                dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                    .contiguous()
                    .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))
        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
                .view(batch_size, 1, -1, full_key_len)
                .repeat(1, n_heads, 1, 1)
                .expand(batch_size, n_heads, query_len, full_key_len)
                .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(
            dot_prod, dim=-1, dtype=torch.float  # type: ignore
        ).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out, dot_prod
    #
    # def forward(self, x, mask=None):
    #     r"""
    #     Overview:
    #        Compute attention
    #     Arguments:
    #         - x (:obj:`tensor`): input tensor
    #         - mask (:obj:`tensor`): mask out invalid entries
    #     Returns:
    #         - attention (:obj:`tensor`): attention tensor
    #     """
    #     assert (len(x.shape) == 3)
    #     B, N = x.shape[:2]
    #     x = self.attention_pre(x)
    #     query, key, value = torch.chunk(x, 3, dim=2)
    #     query, key, value = self.split(query), self.split(key, T=True), self.split(value)
    #
    #     score = torch.matmul(query, key)  # B, head_num, N, N
    #     score /= math.sqrt(self.head_dim)
    #     if mask is not None:
    #         score.masked_fill_(~mask, value=-1e9)
    #
    #     score = F.softmax(score, dim=-1)
    #     score = self.dropout(score)
    #     attention = torch.matmul(score, value)  # B, head_num, N, head_dim
    #
    #     attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
    #     attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
    #     return attention


class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(
            self,
            dim: int = None,
            dim_hidden: int = None,
            dropout: float = 0,
            activation: str = 'relu',
            **kwargs,
    ):
        super(TransformerFFN, self).__init__(**kwargs)
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.dropout_ratio = dropout
        self.relu_dropout = nn.Dropout(p=self.dropout_ratio)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = F.gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(self.dim, self.dim_hidden)
        self.lin2 = nn.Linear(self.dim_hidden, self.dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x


class TransformerLayer(nn.Module):
    r"""
    Overview:
        In transformer layer, first computes entries's attention and applies a feedforward layer
    """

    def __init__(self,
                 n_heads: int = None,
                 embedding_size: int = None,
                 ffn_size: int = None,
                 attention_dropout: float = 0.0,
                 relu_dropout: float = 0.0,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 variant: Optional[str] = None,
                 ):
        r"""
        Overview:
            Init transformer layer
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers
            - dropout (:obj:`nn.Module`): dropout layer
            - activation (:obj:`nn.Module`): activation function
        """
        super(TransformerLayer, self).__init__()
        self.n_heads = n_heads
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads=self.n_heads,
            dim=embedding_size,
            dropout=attention_dropout)
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(dim=embedding_size,
                                  dim_hidden=ffn_size,
                                  dropout=relu_dropout,
                                  activation=activation,
                                  )
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Overview:
            transformer layer forward
        Arguments:
            - inputs (:obj:`tuple`): x and mask
        Returns:
            - output (:obj:`tuple`): x and mask
        """
        residual = x

        if self.variant == 'prenorm':
            x = self.norm1(x)
        attended_tensor = self.attention(x, mask=mask)[0]
        x = residual + self.dropout(attended_tensor)
        if self.variant == 'postnorm':
            x = self.norm1(x)

        residual = x
        if self.variant == 'prenorm':
            x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        if self.variant == 'postnorm':
            x = self.norm2(x)

        x *= mask.unsqueeze(-1).type_as(x)
        return x


class Transformer(nn.Module):
    '''
    Overview:
        Transformer implementation

        Note:
            For details refer to Attention is all you need: http://arxiv.org/abs/1706.03762
    '''

    def __init__(
            self,
            n_heads=8,
            embedding_size: int = 128,
            ffn_size: int = 128,
            n_layers: int = 3,
            attention_dropout: float = 0.0,
            relu_dropout: float = 0.0,
            dropout: float = 0.0,
            activation: Optional[str] = 'relu',
            variant: Optional[str] = 'prenorm',
    ):
        r"""
        Overview:
            Init transformer
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers
            - layer_num (:obj:`int`): number of transformer layers
            - dropout_ratio (:obj:`float`): dropout ratio
            - activation (:obj:`nn.Module`): activation function
        """
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.dim = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers

        self.dropout_ratio = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.activation = activation
        self.variant = variant

        # build the model
        self.layers = self.build_layers()
        self.norm_embedding = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)

    def build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = TransformerLayer(
                n_heads=self.n_heads,
                embedding_size=self.dim,
                ffn_size=self.ffn_size,
                attention_dropout=self.attention_dropout,
                relu_dropout=self.relu_dropout,
                dropout=self.dropout_ratio,
                variant=self.variant,
                activation=self.activation,
            )
            layers.append(layer)
        return layers

    def forward(self, x, mask=None):
        r"""
        Overview:
            Transformer forward
        Arguments:
            - x (:obj:`tensor`): input tensor, shape (B, N, C), B is batch size, N is number of entries,
                C is feature dimension
            - mask (:obj:`tensor` or :obj:`None`): bool tensor, can be used to mask out invalid entries in attention,
                shape (B, N), B is batch size, N is number of entries
        Returns:
            - x (:obj:`tensor`): transformer output
        """
        if self.variant == 'postnorm':
            x = self.norm_embedding(x)
        if mask is not None:
            x *= mask.unsqueeze(-1).type_as(x)
        else:
            mask = torch.ones(size=x.shape[:2],dtype=torch.bool, device=x.device)
        if self.variant == 'postnorm':
            x = self.norm_embedding(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        if self.variant == 'prenorm':
            x = self.norm_embedding(x)
        return x

if __name__  == '__main__':
    transformer = Transformer(n_heads=8,embedding_size=128)
    from bigrl.core.torch_utils.network.rnn import sequence_mask
    mask = sequence_mask(lengths=torch.tensor([1,2,3,4,5,6,2,3,0,0]),max_len=20)
    y = transformer.forward(x = torch.randn(size=(10,20,128)),mask=mask)
    print(y)