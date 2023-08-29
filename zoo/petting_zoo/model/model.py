import torch.nn as nn
from ding.model.common import FCEncoder

class PettingZooEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = FCEncoder(obs_shape=18, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)

    def forward(self, x):
        x = x['agent_state']
        x = self.encoder(x)
        return x