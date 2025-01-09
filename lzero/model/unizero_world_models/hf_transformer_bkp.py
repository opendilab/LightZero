from typing import Optional, List
import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel

from .kv_caching import KeysValues
from transformers.cache_utils import DynamicCache


def kv2dc(cache: KeysValues) -> DynamicCache:
    """
    将自定义的 KeysValues 缓存转换为 Huggingface 的 DynamicCache 格式。

    Args:
        cache (KeysValues): 自定义的键值缓存。

    Returns:
        DynamicCache: Huggingface 的动态缓存对象。
    """
    res = DynamicCache()
    for kv_cache in cache:
        k_tensor = kv_cache._k_cache.get()
        v_tensor = kv_cache._v_cache.get()
        res.key_cache.append(k_tensor)
        res.value_cache.append(v_tensor)
    return res


def update_kv(cache: KeysValues, new_cache: DynamicCache) -> None:
    """
    更新自定义的 KeysValues 缓存。

    Args:
        cache (KeysValues): 自定义的键值缓存。
        new_cache (DynamicCache): Huggingface 的动态缓存对象。
    """
    for i in range(len(new_cache.key_cache)):
        # 更新时使用当前最新的 key 和 value
        cache[i].update(new_cache.key_cache[i], new_cache.value_cache[i])


class HuggingfaceLlamaTransformer(LlamaForCausalLM):
    """
    使用预训练的 Huggingface Llama 模型作为主干的 Transformer 类。

    继承自 LlamaForCausalLM，并扩展自定义的缓存与投影层。
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        # 假设需要一个自定义的投影层，如果不需要可以移除
        self.projection_layer = nn.Linear(config.hidden_size, config.hidden_size)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs):
        """
        从预训练模型加载权重，并初始化自定义的 Transformer 类。

        Args:
            model_name_or_path (str): 预训练模型的名称或路径。

        Returns:
            HuggingfaceLlamaTransformer: 初始化后的模型实例。
        """
        model = super(HuggingfaceLlamaTransformer, cls).from_pretrained(model_name_or_path, *args, **kwargs)
        return model

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        生成键值缓存的占位符。

        Args:
            n (int): 批量大小。
            max_tokens (int): 序列的最大长度。

        Returns:
            KeysValues: 包含空键值的对象。
        """
        device = self.device  # 使用模型所在的设备
        return KeysValues(
            n=n,
            num_heads=self.config.num_attention_heads,
            max_tokens=max_tokens,
            embed_dim=self.config.hidden_size,
            num_layers=self.config.num_hidden_layers,
            device=device
        )

    def _get_positional_embedding(self, layer: int, attn_type: str, pos_emb) -> torch.Tensor:
        """
        获取指定层和注意力类型的位置信息嵌入。

        Args:
            layer (int): 层索引。
            attn_type (str): 注意力类型，'key' 或 'value'。
            pos_emb: 位置信息嵌入对象。

        Returns:
            torch.Tensor: 位置信息嵌入张量。
        """
        if attn_type == 'key':
            module_name = 'k_proj'
        elif attn_type == 'value':
            module_name = 'v_proj'
        else:
            raise ValueError("attn_type 必须是 'key' 或 'value'")
        
        # 获取对应层的注意力投影模块
        attn_module = self.model.layers[layer].self_attn
        attn_func = getattr(attn_module, module_name)
        return attn_func(pos_emb.weight)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Transformer 模型的前向传播。

        Args:
            input_ids (Optional[torch.Tensor]): 输入的 token IDs，形状为 (batch_size, seq_length)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，形状为 (batch_size, seq_length)。
            past_key_values (Optional[KeysValues]): 预计算的键值缓存，用于加速生成。
            valid_context_lengths (Optional[torch.Tensor]): 有效的上下文长度，用于掩码。
            inputs_embeds (Optional[torch.Tensor]): 输入的嵌入，形状为 (batch_size, seq_length, embed_dim)。

        Returns:
            torch.Tensor: 模型的输出。
        """
        # 将自定义的键值缓存转换为 Huggingface 的格式
        if past_key_values is not None:
            kv_cache = kv2dc(past_key_values)
            use_cache = True
        else:
            kv_cache = None
            use_cache = True  # 根据需求，可以设置为 False

        # 如果提供了有效上下文长度，则构建 attention_mask
        if valid_context_lengths is not None:
            B, T = input_ids.shape
            # 创建一个全为 1 的 attention_mask
            attention_mask = torch.ones((B, T), dtype=torch.long, device=self.device)
            # 根据 valid_context_lengths 设置无效部分为 0
            for i in range(B):
                attention_mask[i, :T - valid_context_lengths[i]] = 0
        else:
            if attention_mask is None:
                # 默认情况下，创建一个全为 1 的 attention_mask
                if input_ids is not None:
                    attention_mask = torch.ones_like(input_ids, device=self.device)
                elif inputs_embeds is not None:
                    attention_mask = torch.ones(inputs_embeds.size()[:2], device=self.device)
                else:
                    raise ValueError("输入缺少 input_ids 或 inputs_embeds")

        # 调用 Huggingface 的前向方法
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )

        # 更新自定义的 KeysValues 缓存
        if past_key_values is not None:
            update_kv(past_key_values, outputs.past_key_values)

        # 如果需要，可以添加自定义的投影层
        if hasattr(self, 'projection_layer') and self.projection_layer is not None:
            # 确保最后一个隐藏状态的形状正确
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
            output_projection = self.projection_layer(last_hidden_state)  # (batch_size, seq_length, hidden_size)
            return output_projection
        else:
            return outputs.last_hidden_state