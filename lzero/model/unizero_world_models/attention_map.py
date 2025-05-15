import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def visualize_attention_map(model, input_embeddings: torch.Tensor, kv_cache: Optional[dict] = None,
                            valid_context_lengths: Optional[torch.Tensor] = None, layer_id: int = 0, head_id: int = 0,
                            suffix='visual_match/attn_map'):
    """
    Overview:
        Visualize the attention map for a specific layer and head in the Transformer model.

    Arguments:
        - model (:obj:`Transformer`): Transformer model.
        - input_embeddings (:obj:`torch.Tensor`): Input token embeddings of shape (B, T, C).
        - kv_cache (:obj:`Optional[dict]`): Cached keys and values for long sequence inference.
        - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for variable-length contexts.
        - layer_id (:obj:`int`): The index of the layer to visualize (0-based).
        - head_id (:obj:`int`): The index of the head to visualize (0-based).
        - suffix (:obj:`str`): Directory suffix for saving the attention map image.

    Returns:
        None
    """
    assert 0 <= layer_id < len(model.blocks), "Invalid layer_id"
    assert 0 <= head_id < model.config.num_heads, "Invalid head_id"

    B, T, C = input_embeddings.shape
    L = kv_cache[layer_id].shape[2] if kv_cache is not None else 0

    with torch.no_grad():
        model.eval()
        hidden_states = input_embeddings
        input_ids = torch.arange(T).expand(B, T)

        for i, block in enumerate(model.blocks):
            if i < layer_id:
                hidden_states = block(hidden_states, None if kv_cache is None else kv_cache[i], valid_context_lengths)
            elif i == layer_id:
                attention_map = block.attn.get_attention_map(block.ln1(hidden_states),
                                                             None if kv_cache is None else kv_cache[i],
                                                             valid_context_lengths)
                break

    attention_map = attention_map[0, head_id].cpu().numpy()  # Select the attention map of the first sample

    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_map, cmap='coolwarm', square=True, cbar_kws={"shrink": 0.5},
                xticklabels=input_ids[0].cpu().numpy(), yticklabels=input_ids[0, -T:].cpu().numpy())
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(f'Attention Map of Layer {layer_id} Head {head_id}')
    plt.show()

    directory = f'/home/ddediosallegue/projects/UniZero/{suffix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/attn_map_layer_{layer_id}_head_{head_id}.png')
    plt.close()


def visualize_attention_maps(model, input_embeddings: torch.Tensor, kv_cache: Optional[dict] = None,
                             valid_context_lengths: Optional[torch.Tensor] = None,
                             suffix='visual_match/attn_map_all_head_and_layer', nhead_each_row=4):
    """
    Overview:
        Visualize all attention maps for all layers and heads, arranging them in a single figure.

    Arguments:
        - model (:obj:`Transformer`): Transformer model.
        - input_embeddings (:obj:`torch.Tensor`): Input token embeddings of shape (B, T, C).
        - kv_cache (:obj:`Optional[dict]`): Cached keys and values for long sequence inference.
        - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for variable-length contexts.
        - suffix (:obj:`str`): Directory suffix for saving the attention map image.
        - nhead_each_row (:obj:`int`): Number of heads to display per row.

    Returns:
        None
    """
    B, T, C = input_embeddings.shape
    num_layers = len(model.blocks)
    num_heads = model.config.num_heads

    num_cols = min(num_heads, nhead_each_row)
    num_rows = math.ceil(num_layers * num_heads / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    with torch.no_grad():
        model.eval()
        hidden_states = input_embeddings
        input_ids = torch.arange(T).expand(B, T)

        head_count = 0
        for layer_id, block in enumerate(model.blocks):
            hidden_states = block(hidden_states, None if kv_cache is None else kv_cache[layer_id],
                                  valid_context_lengths)
            attention_maps = block.attn.get_attention_map(block.ln1(hidden_states),
                                                          None if kv_cache is None else kv_cache[layer_id],
                                                          valid_context_lengths)

            for head_id in range(num_heads):
                row_id = head_count // num_cols
                col_id = head_count % num_cols
                ax = axs[row_id, col_id] if num_rows > 1 else axs[col_id]

                attention_map = attention_maps[0, head_id].cpu().numpy()  # Select the attention map of the first sample
                sns.heatmap(attention_map, cmap='coolwarm', square=True, cbar=False, ax=ax)

                ax.tick_params(labelsize=8)
                ax.tick_params(axis='x', rotation=90)
                ax.tick_params(axis='y', rotation=0)
                ax.set_xlabel(f'Key - Head {head_id + 1}', fontsize=10)
                ax.set_ylabel(f'Query - Layer {layer_id + 1}', fontsize=10)

                head_count += 1

    plt.tight_layout()
    directory = f'/home/ddediosallegue/projects/UniZero/{suffix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/attn_maps_{nhead_each_row}-each-row.png', dpi=300)
    print(f'Attention maps saved to {directory}/attn_maps_{nhead_each_row}-each-row.png')
    plt.close()
