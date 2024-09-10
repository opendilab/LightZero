"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.model import ReparameterizationHead
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import RepresentationNetwork

import torch

from ding.utils import MODEL_REGISTRY
from torch import nn
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    import sys
    from ditk import logging
    logging.warning("not found transformer, please install it using: pip install transformers")
    sys.exit(1)

from FlagEmbedding import BGEM3FlagModel


@MODEL_REGISTRY.register('AlphaZeroModel')
class AlphaZeroModel(nn.Module):

    def __init__(
            self,
            model_name: str = "BAAI/bge-m3",
            add_linear: bool = True,
            embedding_size: int = 1024,
            freeze_encoder: bool = True,
            action_space_size: int = 9
    ) -> None:
        super().__init__()
        self.model = BGEM3FlagModel('BAAI/bge-m3',  
                            use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        # Freeze transformer encoder and only train the linear layer
        # if freeze_encoder:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(
                1024, embedding_size
            )  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

        value_support_size= 1  # TODO
        # action_space_size = 9
        self.value_head = nn.Linear(self.embedding_size, value_support_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        self.policy_head = nn.Linear(self.embedding_size, action_space_size)  # 768 for bert-base-uncased, distilbert-base-uncased


    def _calc_embedding(self, x: list) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        sentence_embedding = self.model.encode(x, 
                            batch_size=32, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
        sentence_embedding = torch.from_numpy(sentence_embedding).to(self.value_head.weight.device).float()
        
        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding

    def forward(self, train_samples: list, candidate_samples: list = None) -> dict:
        prompt_embedding = self._calc_embedding(train_samples)
        # cands_embedding = self._calc_embedding(candidate_samples).detach()  # TODO: detach
        # scores = torch.mm(prompt_embedding, cands_embedding.t())
        
        policy_logits = self.policy_head(prompt_embedding)  # len(input_list) x embedding_size

        if self.value_head:
            value = self.value_head(prompt_embedding)  # len(input_list) x embedding_size

        # return {'dist': torch.distributions.Categorical(logits=scores), 'logit': scores, 'value': value}
        return policy_logits, value

    def compute_policy_value(self, train_samples: list, candidate_samples: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logit, value = self.forward(train_samples, candidate_samples)
        prob = torch.nn.functional.softmax(logit, dim=-1)
        return prob, value

    def compute_logp_value(self,  train_samples: list, candidate_samples: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logit, value = self.forward(train_samples, candidate_samples)
        # use log_softmax to calculate log probability
        log_prob = F.log_softmax(logit, dim=-1)
        return log_prob, value


