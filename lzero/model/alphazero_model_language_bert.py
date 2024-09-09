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


# @MODEL_REGISTRY.register('language_transformer')
# class LanguageTransformer(nn.Module):
@MODEL_REGISTRY.register('AlphaZeroModel')
class AlphaZeroModel(nn.Module):

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            add_linear: bool = True,
            embedding_size: int = 128,
            freeze_encoder: bool = True,
            action_space_size: int = 9
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(
                self.model.config.hidden_size, embedding_size
            )  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

        value_support_size= 1  # TODO
        # action_space_size = 9
        self.value_head = nn.Linear(self.embedding_size, value_support_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        self.policy_head = nn.Linear(self.embedding_size, action_space_size)  # 768 for bert-base-uncased, distilbert-base-uncased

        # self.policy_head = MLP(
        #         in_channels=self.model.config.hidden_size,
        #         hidden_channels=256,
        #         out_channels=action_space_size,
        #         layer_num=2,
        #         activation=nn.GELU(),
        #         norm_type='LN',
        #         output_activation=False,
        #         output_norm=False,
        #         last_linear_layer_init_zero=True,
        #     )

    def _calc_embedding(self, x: list) -> torch.Tensor:
        # ``truncation=True`` means that if the length of the prompt exceed the ``max_length`` of the tokenizer,
        # the exceeded part will be truncated. ``padding=True`` means that if the length of the prompt does not reach
        # the ``max_length``, the latter part will be padded. These settings ensure the length of encoded tokens is
        # exactly ``max_length``, which can enable batch-wise computing.
        input = self.tokenizer(x, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size

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

    # def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     encoded_state = self.representation_network(state_batch)
    #     logit, value = self.prediction_network(encoded_state)
    #     return logit, value

    def compute_policy_value(self, train_samples: list, candidate_samples: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logit, value = self.forward(train_samples, candidate_samples)
        prob = torch.nn.functional.softmax(logit, dim=-1)
        return prob, value

    def compute_logp_value(self,  train_samples: list, candidate_samples: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logit, value = self.forward(train_samples, candidate_samples)
        # use log_softmax to calculate log probability
        log_prob = F.log_softmax(logit, dim=-1)
        return log_prob, value


