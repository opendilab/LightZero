import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

# 定义MoeArgs数据类，用于存储MoE的配置参数
@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int

# 定义Mixture of Experts（MoE）层
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 1:
            # 只有一个专家时，直接使用该专家
            return self.experts[0](inputs)
        
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, nth_expert][:, None] * expert(inputs[batch_idx, token_idx])
        return results

# 定义一个简单的Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )
        
        if config.moe_in_transformer:
            self.feed_forward = MoeLayer(
                experts=[self.mlp for _ in range(config.num_experts_of_moe_in_transformer)],
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=1,
            )
            print("="*20)
            print('使用MoE在Transformer的feed_forward中')
            print("="*20)
        else:
            self.feed_forward = self.mlp

    def forward(self, x):
        return self.feed_forward(x)

# 定义配置类
class Config:
    def __init__(self, embed_dim, resid_pdrop, num_experts_of_moe_in_transformer, moe_in_transformer):
        self.embed_dim = embed_dim
        self.resid_pdrop = resid_pdrop
        self.num_experts_of_moe_in_transformer = num_experts_of_moe_in_transformer
        self.moe_in_transformer = moe_in_transformer

# 测试代码
def test_transformer_block():
    # 初始化配置
    embed_dim = 64
    resid_pdrop = 0.1
    num_experts_of_moe_in_transformer = 1

    # 创建输入数据
    inputs = torch.randn(10, 5, embed_dim)  # (batch_size, seq_len, embed_dim)

    # 初始化两个输出变量
    outputs_true = None
    outputs_false = None

    # 对于moe_in_transformer为True和False分别进行测试
    for moe_in_transformer in [True, False]:
        config = Config(embed_dim, resid_pdrop, num_experts_of_moe_in_transformer, moe_in_transformer)
        transformer_block = TransformerBlock(config)
        
        outputs = transformer_block(inputs)
        print(f"moe_in_transformer={moe_in_transformer}: outputs={outputs}")

        if moe_in_transformer:
            outputs_true = outputs
        else:
            outputs_false = outputs

    # 计算输出的差异
    mse_difference = None
    if outputs_true is not None and outputs_false is not None:
        mse_difference = F.mse_loss(outputs_true, outputs_false).item()
    
    print(f"输出差异的均方误差（MSE）: {mse_difference}")

if __name__ == "__main__":
    test_transformer_block()