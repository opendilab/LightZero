import torch.nn as nn

class PettingZooEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()

    def forward(self, x):
        x = x['agent_state']
        x = self.encoder(x)
        return x