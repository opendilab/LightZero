import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

def visualize_attention_map(model: Transformer,  input_embeddings: torch.Tensor, kv_cache: Optional[KeysValues] = None, valid_context_lengths: Optional[torch.Tensor] = None, layer_id: int = 0, head_id: int = 0, suffix='visual_match_memlen1-0-15_v2/attn_map'):
    """
    可视化attention map
    
    参数:
        model: Transformer模型
        input_embeddings: 输入的token embdding序列,shape为(B, T)
        kv_cache: 缓存的keys和values,用于支持长序列的推断
        valid_context_lengths: 有效的上下文长度,用于处理变长上下文
        layer_id: 要可视化的层的编号,从0开始
        head_id: 要可视化的头的编号,从0开始
        
    返回:
        None
    """
    assert 0 <= layer_id < len(model.blocks)
    assert 0 <= head_id < model.config.num_heads
    
    # B, T = input_embeddings.shape
    B, T, C = input_embeddings.shape
    if kv_cache is not None:
        _, _, L, _ = kv_cache[layer_id].shape
    else:
        L = 0
    
    with torch.no_grad():
        model.eval()
        # hidden_states = model.drop(model.embed(input_ids))
        hidden_states = input_embeddings
        input_ids = torch.arange(T).expand(B, T)

        for i, block in enumerate(model.blocks):
            if i < layer_id:
                hidden_states = block(hidden_states, None if kv_cache is None else kv_cache[i], valid_context_lengths)
            elif i == layer_id:
                attention_map = block.attn.get_attention_map(block.ln1(hidden_states), None if kv_cache is None else kv_cache[i], valid_context_lengths)
                break
    
    attention_map = attention_map[0, head_id].cpu().numpy()  # 取第一个样本的attention map
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_map, cmap='coolwarm', square=True, cbar_kws={"shrink": 0.5}, xticklabels=input_ids[0].cpu().numpy(), yticklabels=input_ids[0, -T:].cpu().numpy())
    plt.xticks(rotation=90)  
    plt.yticks(rotation=0)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(f'Attention Map of Layer {layer_id} Head {head_id}')
    plt.show()
    directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
    # 检查路径是否存在，不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/attn_map_layer_{layer_id}_head_{head_id}.png')
    plt.close()

if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # 加载预训练的GPT-2模型和tokenizer
    model = Transformer(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 准备输入
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # 可视化第0层第0个头的attention map
    visualize_attention_map(model, input_ids, layer_id=0, head_id=0)