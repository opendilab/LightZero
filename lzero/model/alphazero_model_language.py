"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY
from FlagEmbedding import BGEM3FlagModel

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    import sys
    from ditk import logging
    logging.warning("not found transformer, please install it using: pip install transformers")
    sys.exit(1)


@MODEL_REGISTRY.register('AlphaZeroModel')
class AlphaZeroModel(nn.Module):

    def __init__(
            self,
            add_linear: bool = True,
            embedding_size: int = 1024,
            action_space_size: int = 9
    ) -> None:
        super().__init__()
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(1024, embedding_size)
        else:
            self.linear = None

        value_support_size= 1  # TODO
        self.value_head = nn.Linear(self.embedding_size, value_support_size) 
        self.policy_head = nn.Linear(self.embedding_size, action_space_size)

    def _calc_embedding(self, x: list) -> torch.Tensor:
        sentence_embedding = self.model.encode(x, batch_size=32, max_length=8192, )['dense_vecs'] # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        sentence_embedding = torch.from_numpy(sentence_embedding).to(self.value_head.weight.device).float()
        
        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding

    def forward(self, train_samples: list, candidate_samples: list = None) -> dict:
        state_embedding = self._calc_embedding(train_samples)
        
        policy_logits = self.policy_head(state_embedding)  # len(input_list) x embedding_size

        if self.value_head:
            value = self.value_head(state_embedding)  # len(input_list) x embedding_size

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



@MODEL_REGISTRY.register('AlphaZeroModelBert')
class AlphaZeroModelBert(nn.Module):

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
        self.value_head = nn.Linear(self.embedding_size, value_support_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        self.policy_head = nn.Linear(self.embedding_size, action_space_size)  # 768 for bert-base-uncased, distilbert-base-uncased

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
        state_embedding = self._calc_embedding(train_samples)
        policy_logits = self.policy_head(state_embedding)  # len(input_list) x embedding_size

        if self.value_head:
            value = self.value_head(state_embedding)  # len(input_list) x embedding_size

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




