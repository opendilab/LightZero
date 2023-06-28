import numpy as np
import torch
import torch.nn as nn


class OnehotEncoder(nn.Module):
    """
    Overview:
        For encoding integers into one-hot vectors using PyTorch's Embedding layer.
    """
    def __init__(self, num_embeddings: int):
        """
        Overview:
            The initializer for OnehotEncoder. It initializes an Embedding layer with an identity matrix as its weights.
        Arguments:
            - num_embeddings (int): The size of the dictionary of embeddings, i.e., the number of rows in the embedding matrix.
        """
        super(OnehotEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.main = nn.Embedding.from_pretrained(torch.eye(self.num_embeddings), freeze=True, padding_idx=None)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the OnehotEncoder. It encodes the input tensor into one-hot vectors.
        Arguments:
            - x (torch.Tensor): The input tensor should be integers and its maximum value should be less than the 'num_embeddings' specified in the initializer.
        Returns:
            - (torch.Tensor): Return the x-th row of the embedding layer.
        """
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class OnehotEmbedding(nn.Module):
    """
    Overview:
        For encoding integers into higher-dimensional embedding vectors using PyTorch's Embedding layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Overview:
            The initializer for OnehotEmbedding. It initializes an Embedding layer with 'num_embeddings' rows and 'embedding_dim' columns.
        Arguments:
            - num_embeddings (int): The size of the dictionary of embeddings, i.e., the number of rows in the embedding matrix.
            - embedding_dim (int): The size of each embedding vector, i.e., the number of columns in the embedding matrix.
        """
        super(OnehotEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.main = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the OnehotEmbedding. 
            It encodes the input tensor into higher-dimensional embedding vectors.
        Arguments:
            - x (torch.Tensor): The input tensor should be integers, and its maximum value should be less than the 'num_embeddings' specified in the initializer.
        Returns:
            - (torch.Tensor): Return the x-th row of the embedding layer.
        """
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class BinaryEncoder(nn.Module):
    """
    Overview:
        For encoding integers into binary vectors using PyTorch's Embedding layer.
    """
    def __init__(self, num_embeddings: int):
        """
        Overview:
            The initializer for BinaryEncoder. It initializes an Embedding layer with a binary embedding matrix
            as its weights. The binary embedding matrix is constructed by representing each integer (from 0 to 2^bit_num-1)
            as a binary vector.
        Arguments:
            - num_embeddings (int): The number of bits in the binary representation. It determines the size of the dictionary of embeddings, 
            i.e., the number of rows in the embedding matrix (2^bit_num), and the size of each embedding vector, 
            i.e., the number of columns in the embedding matrix (bit_num).
        """
        super(BinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(
            self.get_binary_embed_matrix(self.bit_num), freeze=True, padding_idx=None
        )

    @staticmethod
    def get_binary_embed_matrix(bit_num):
        """
        Overview:
            A helper function that generates the binary embedding matrix.
        Arguments:
            - bit_num (int): The number of bits in the binary representation.
        Returns:
            - (torch.Tensor): A tensor of shape (2^bit_num, bit_num), where each row is the binary representation of the row index.
        """
        embedding_matrix = []
        for n in range(2 ** bit_num):
            embedding = [n >> d & 1 for d in range(bit_num)][::-1]
            embedding_matrix.append(embedding)
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the BinaryEncoder. It encodes the input tensor into binary vectors.
        Arguments:
            - x (torch.Tensor): The input tensor should be integers, and its maximum value should be less than 2^bit_num.
        Returns:
            - (torch.Tensor): Return the x-th row of the embedding layer.
        """
        x = x.long().clamp_(max=2 ** self.bit_num - 1)
        return self.main(x)


class SignBinaryEncoder(nn.Module):
    """
    Overview:
        For encoding integers into signed binary vectors using PyTorch's Embedding layer.
    """
    def __init__(self, num_embeddings: int):
        """
        Overview:
            The initializer for SignBinaryEncoder. It initializes an Embedding layer with a signed binary embedding matrix
            as its weights. The signed binary embedding matrix is constructed by representing each integer (from -2^(bit_num-1) to 2^(bit_num-1)-1)
            as a signed binary vector. The first bit is the sign bit, with 1 representing negative and 0 representing nonnegative.
        Arguments:
            - num_embeddings (int): The number of bits in the signed binary representation. It determines the size of the dictionary of embeddings, 
            i.e., the number of rows in the embedding matrix (2^bit_num), and the size of each embedding vector, 
            i.e., the number of columns in the embedding matrix (bit_num).
        """
        super(SignBinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(
            self.get_sign_binary_matrix(self.bit_num), freeze=True, padding_idx=None
        )
        self.max_val = 2 ** (self.bit_num - 1) - 1

    @staticmethod
    def get_sign_binary_matrix(bit_num):
        """
        Overview:
            A helper function that generates the signed binary embedding matrix.
        Arguments:
            - bit_num (int): The number of bits in the signed binary representation.
        Returns:
            - (torch.Tensor): A tensor of shape (2^bit_num, bit_num), where each row is the signed binary representation of the row index minus 2^(bit_num-1).
            The first column is the sign bit, with 1 representing negative and 0 representing nonnegative.
        """
        neg_embedding_matrix = []
        pos_embedding_matrix = []
        for n in range(1, 2 ** (bit_num - 1)):
            embedding = [n >> d & 1 for d in range(bit_num - 1)][::-1]
            neg_embedding_matrix.append([1] + embedding)
            pos_embedding_matrix.append([0] + embedding)
        embedding_matrix = neg_embedding_matrix[::-1] + [[0 for _ in range(bit_num)]] + pos_embedding_matrix
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the SignBinaryEncoder. It encodes the input tensor into signed binary vectors.
        Arguments:
            - x (torch.Tensor): The input tensor. Its data type should be integers, and its maximum absolute value should be less than 2^(bit_num-1).
        Returns:
            - (torch.Tensor): Return the x-th row of the embedding layer.
        """
        x = x.long().clamp_(max=self.max_val, min=-self.max_val)
        return self.main(x + self.max_val)


class PositionEncoder(nn.Module):
    """
    Overview:
        For encoding the position of elements into higher-dimensional vectors using PyTorch's Embedding layer.
        This is typically used in Transformer models to add positional information to the input tokens.
        The position encoding is initialized using a sinusoidal formula, as proposed in the "Attention is All You Need" paper.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int = None):
        """
        Overview:
            The initializer for PositionEncoder. It initializes an Embedding layer with a sinusoidal position encoding matrix
            as its weights. 
        Arguments:
            - num_embeddings (int): The maximum number of positions to be encoded, i.e., the number of rows in the position encoding matrix.
            - embedding_dim (int, optional): The size of each position encoding vector, i.e., the number of columns in the position encoding matrix. 
            If not provided, it is set equal to 'num_embeddings'.
        """
        super(PositionEncoder, self).__init__()
        self.n_position = num_embeddings
        self.embedding_dim = self.n_position if embedding_dim is None else embedding_dim
        self.position_enc = nn.Embedding.from_pretrained(
            self.position_encoding_init(self.n_position, self.embedding_dim), freeze=True, padding_idx=None
        )

    @staticmethod
    def position_encoding_init(n_position, embedding_dim):
        """
        Overview:
            A helper function that generates the sinusoidal position encoding matrix.
        Arguments:
            - n_position (int): The maximum number of positions to be encoded.
            - embedding_dim (int): The size of each position encoding vector.
        Returns:
            - (torch.Tensor): A tensor of shape (n_position, embedding_dim), where each row is the sinusoidal position encoding of the row index.
        """
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
                for pos in range(n_position)
            ]
        )
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on 0th,2nd,4th...embedding_dim
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on 1st,3rd,5th...embedding_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the PositionEncoder. It encodes the input tensor into positional vectors.
        Arguments:
            - x (torch.Tensor): The input tensor should be integers, and its maximum value should be less than 'num_embeddings' specified in the initializer. Each value in 'x' represents a position to be encoded.
        Returns:
            - (torch.Tensor): Return the x-th row of the embedding layer.
        """
        return self.position_enc(x)


class TimeEncoder(nn.Module):
    """
    Overview:
        For encoding temporal or sequential data into higher-dimensional vectors using the sinusoidal position 
        encoding mechanism used in Transformer models. This is useful when working with time series data or sequences where 
        the position of each element (in this case time) is important.
    """
    def __init__(self, embedding_dim: int):
        """
        Overview:
            The initializer for TimeEncoder. It initializes the position array which is used to scale the input data in the sinusoidal encoding function.
        Arguments:
        - embedding_dim (int): The size of each position encoding vector, i.e., the number of features in the encoded representation.
        """
        super(TimeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_array = torch.nn.Parameter(self.get_position_array(), requires_grad=False)

    def get_position_array(self):
        """
        Overview:
            A helper function that generates the position array used in the sinusoidal encoding function. Each element in the array is 1 / (10000^(2i/d)), 
            where i is the position in the array and d is the embedding dimension. This array is used to scale the input data in the encoding function.
        Returns:
            - (torch.Tensor): A tensor of shape (embedding_dim,) containing the position array.
        """
        x = torch.arange(0, self.embedding_dim, dtype=torch.float)
        x = x // 2 * 2
        x = torch.div(x, self.embedding_dim)
        x = torch.pow(10000., x)
        x = torch.div(1., x)
        return x

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the TimeEncoder. It encodes the input tensor into temporal vectors.
        Arguments:
            - x (torch.Tensor): The input tensor. Its data type should be one-dimensional, and each value in 'x' represents a timestamp or a position in sequence to be encoded.
        Returns:
            - (torch.Tensor): A tensor containing the temporal encoded vectors of the input tensor 'x'. 
        """
        v = torch.zeros(size=(x.shape[0], self.embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # apply sin on even-indexed positions
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # apply cos on odd-indexed positions
        return v


class UnsqueezeEncoder(nn.Module):
    """
    Overview:
        For unsqueezes a tensor along the specified dimension and then optionally normalizes the tensor.
        This is useful when we want to add an extra dimension to the input tensor and potentially scale its values.
    """
    def __init__(self, unsqueeze_dim: int = -1, norm_value: float = 1):
        """
        Overview:
            The initializer for UnsqueezeEncoder.
        Arguments:
            - unsqueeze_dim (int, optional): The dimension to unsqueeze. Default is -1, which unsqueezes at the last dimension.
            - norm_value (float, optional): The value to normalize the tensor by. Default is 1, which means no normalization.
        """
        super(UnsqueezeEncoder, self).__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.norm_value = norm_value

    def forward(self, x: torch.Tensor):
        """
        Overview:
            The common computation graph of the UnsqueezeEncoder. It unsqueezes the input tensor along the specified dimension and then normalizes the tensor.
        Arguments:
            - x (torch.Tensor): The input tensor.
        Returns:
            - (torch.Tensor): The unsqueezed and normalized tensor. Its shape is the same as the input tensor, but with an extra dimension at the position specified by 'unsqueeze_dim'. Its values are the values of the input tensor divided by 'norm_value'.
        """
        x = x.float().unsqueeze(dim=self.unsqueeze_dim)
        if self.norm_value != 1:
            x = x / self.norm_value
        return x
