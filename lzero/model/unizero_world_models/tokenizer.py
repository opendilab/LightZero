"""
Modified from https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing import Optional, List

class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        """Initialize with various loss components."""
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        """Divide all loss components by a given value."""
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    """
    Overview:
        Tokenizer model that encodes and decodes observations.
    """
    def __init__(self, encoder=None, decoder_network=None, with_lpips: bool = False, projection: list = None) -> None:
        """Initialize the Tokenizer.

        Arguments:
            encoder (nn.Module, optional): Encoder network. Defaults to None.
            decoder_network (nn.Module, optional): Decoder network. Defaults to None.
            with_lpips (bool, optional): Whether to use LPIPS for perceptual loss. Defaults to False.
        """
        super().__init__()
        if with_lpips:
            from lzero.model.unizero_world_models.lpips import LPIPS
            self.lpips = LPIPS().eval()
        else:
            self.lpips = None

        self.encoder = encoder
        self.decoder_network = decoder_network

        # ---- weight tying ----
        vocab_size = self.decoder_network.embed_tokens.weight.size(0)
        self.decoder_network.lm_head = nn.Linear(
            self.decoder_network.config.d_model, vocab_size, bias=False
        )
        self.decoder_network.lm_head.weight = self.decoder_network.embed_tokens.weight

        if projection is None:
            self.projection_layer = nn.Identity()
        else:
            self.projection_layer = nn.Linear(projection[0], projection[1])

    def encode_to_obs_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, ...).

        Returns:
            torch.Tensor: Encoded embeddings of shape (B, 1, E).
        """
        shape = x.shape
        # Process input tensor based on its dimensionality
        if len(shape) == 2:
            # Case when input is 2D (B, E)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 3:
            # Case when input is 3D (B, T, E)
            x = x.contiguous().view(-1, shape[-1])  # Flatten the last two dimensions (B * T, E)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 4:
            # Case when input is 4D (B, C, H, W)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 5:
            # Case when input is 5D (B, T, C, H, W)
            x = x.contiguous().view(-1, *shape[-3:])  # Flatten the first two dimensions (B * T, C, H, W)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        else:
            raise ValueError(f"Invalid input shape: {shape}")

        return obs_embeddings

    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to observations.

        Arguments:
            embeddings (:obj:`torch.Tensor`): Input embeddings.

        Returns:
            torch.Tensor: Decoded observations.
        """
        return self.decoder_network(embeddings)

    @staticmethod
    def _shift_right(labels: torch.LongTensor,
                     pad_token_id: int,
                     start_token_id: int) -> torch.LongTensor:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def decode_to_language_logits(
        self, embeddings: torch.Tensor, target_ids: torch.Tensor,
        pad_token_id: int = 0, decoder_start_token_id: int = 0) -> torch.Tensor:

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.dim() == 3:
            B,T,E = embeddings.shape
            embeddings = embeddings.reshape(B*T,1,E)
            target_ids = target_ids.reshape(B*T, -1)

        embeddings = self.projection_layer(embeddings)     # (B',1,E)

        # shift-right
        decoder_input_ids = self._shift_right(
            target_ids, pad_token_id, decoder_start_token_id
        )
        dec_attn_mask = decoder_input_ids.ne(pad_token_id)
        enc_attn_mask = torch.ones(embeddings.size(0),1,
                                   dtype=torch.long, device=embeddings.device)

        outputs = self.decoder_network(
            input_ids = decoder_input_ids,
            attention_mask = dec_attn_mask,
            encoder_hidden_states = embeddings,
            encoder_attention_mask = enc_attn_mask,
        )
        logits = self.decoder_network.lm_head(outputs.last_hidden_state)
        return logits

    # for Train
    # def decode_to_language_logits(self, embeddings: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    #     # embeddings: [B, T, H] -> [B * T, 1, H]
    #     if embeddings.dim() == 3:
    #         embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], 1, -1)
    #     elif embeddings.dim() == 2:
    #         embeddings = embeddings.unsqueeze(1)
    #     # target_ids: [B, T, L] -> [B * T, L]
    #     target_ids = target_ids.reshape(target_ids.shape[0] * target_ids.shape[1], -1)
    #     # For each decision transformer token (encoding for one observation),
    #     # the embedding serves as the initial hidden state for t5 to decode.
    #     # Hence, the sequence dimension can be paralleled, i.e. should be merged to the batch dimension.
    #     embeddings = self.projection_layer(embeddings)
    #     outputs = self.decoder_network(
    #         input_ids=target_ids,
    #         encoder_hidden_states=embeddings,
    #     )
    #     logits = self.decoder_network.lm_head(outputs.last_hidden_state)
    #     return logits
    
    def decode_to_language_logits_for_inference(
            self, embeddings: torch.Tensor,
            max_length: int = 512,
            pad_token_id: int = 0,
            decoder_start_token_id: int = 0,
            eos_token_id: int = 1,
            sampling: bool = False,         # 是否采用采样解码（True 为采样解码，False 为贪心解码）
            top_k: Optional[int] = None,      # 采样时采用 top-k 过滤（可选）
            top_p: Optional[float] = None     # 采样时采用 nucleus 过滤（可选）
        ) -> List[List[int]]:
        """
        将给定的编码器或嵌入表示 embeddings 翻译成 token 序列。
        
        参数:
            embeddings (torch.Tensor): 编码器或其他隐层表示，形状可能为 (B, E) 或 (B, L, E)。
            max_length (int): 最大解码步数。
            pad_token_id (int): 当序列生成结束后，用此 token 补全剩下的位置。
            decoder_start_token_id (int): 解码时的起始 token id。
            eos_token_id (int): 序列结束 token 的 id。
            sampling (bool): 是否采用采样解码；默认 False，采用贪心解码。
            top_k (Optional[int]): 使用 top-k 采样时的 k 值，如果提供则会对 logits 进行 top-k 过滤。
            top_p (Optional[float]): 使用 nucleus（top-p）采样时的 p 值，如果提供则会对 logits 进行 top-p 过滤。
        
        返回:
            List[List[int]]: 生成的 token 序列，每个子列表代表一个 batch 内序列 (剔除起始 token)。
        """
        
        # 设置 decoder_network 与 projection_layer 为评估模式，关闭 dropout 等训练行为
        self.decoder_network.eval()
        self.projection_layer.eval()

        # 如果 embeddings 不是 Tensor，则转换为 torch.Tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # 尝试从 decoder_network 获取设备信息，如果没有则从模型参数中获取
        try:
            device = self.decoder_network.device
        except AttributeError:
            device = next(self.decoder_network.parameters()).device
            
        # 将 embeddings 移动到正确的设备上
        embeddings = embeddings.to(device)

        with torch.no_grad():  # 在推理过程中关闭梯度计算，节约显存和计算
            # 如果 embeddings 是二维 (B, E)，则在第2维扩展，变成 (B, 1, E)
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)

            # 通过 projection_layer 投影得到新的 embeddings 表示，预期输出形状 (B, 1, E)
            embeddings = self.projection_layer(embeddings)
            B = embeddings.size(0)  # 获取 batch 大小

            # 初始化生成序列，每个序列第一个 token 为 decoder_start_token_id
            generated = torch.full(
                (B, 1), decoder_start_token_id, dtype=torch.long, device=device
            )
            # 用于标记每个序列是否已经生成 EOS token 的标志向量
            is_finished = torch.zeros(B, dtype=torch.bool, device=device)
            past_key_values = None  # 初始化过去状态，用于加速自回归解码

            # 开始逐步生成 token，最大不超过 max_length 步
            for _ in range(max_length):
                # 调用 decoder_network，只输入生成序列的最后一个 token，
                # 同时将 encoder 隐层表示以及 past_key_values 传入，并开启缓存
                outputs = self.decoder_network(
                    input_ids=generated[:, -1:],  # 仅传入最后一个 token
                    encoder_hidden_states=embeddings,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # 获取 decoder 输出的最后隐藏状态
                hidden_states = outputs.last_hidden_state
                # 通过语言模型头计算 logits（注意：这里假设 decoder_network 有 lm_head 属性）
                logits = self.decoder_network.lm_head(hidden_states)
                # 取出当前时间步的 logits，形状为 (B, vocab_size)
                next_token_logits = logits[:, -1, :]

                # 判断是否采用采样解码
                if sampling:
                    # 若配置了 top_k 或 top_p，则可以基于 logits 做过滤再采样；
                    # 这里简单示范使用 softmax 后的采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心解码，直接选择概率最高的 token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 更新 past_key_values，供下一步解码时使用
                past_key_values = outputs.past_key_values

                # 对于已经结束生成（即之前生成了 EOS）的序列，设置当前 token 为 pad_token_id
                next_token = torch.where(
                    is_finished.unsqueeze(-1),
                    torch.full_like(next_token, pad_token_id),
                    next_token
                )
                
                # 将新生成的 token 拼接到当前生成序列上
                generated = torch.cat([generated, next_token], dim=1)
                
                # 更新 is_finished 标志：若当前 token 为 eos_token_id，则标记该序列结束
                is_finished |= next_token.squeeze(-1).eq(eos_token_id)
                
                # 如果所有序列均已生成 EOS，则提前退出循环
                if is_finished.all():
                    break

            # 返回结果时去掉起始的 decoder_start_token_id，并将结果转换为 CPU 上的 list 类型
            return generated[:, 1:].cpu().tolist()

    # @torch.no_grad() 
    # def decode_to_language_logits_for_inference(self, embeddings: torch.Tensor, max_length: int = 512, pad_token_id: int = 0, eos_token_id: int = 102) -> torch.Tensor:
    #     self.decoder_network.eval()
    #     self.projection_layer.eval()
        
    #     if not isinstance(embeddings, torch.Tensor):
    #         embeddings = torch.tensor(embeddings, dtype=torch.float32) 

    #     embeddings = embeddings.to(self.decoder_network.device)

    #     if embeddings.dim() == 3:
    #         embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], 1, -1)
    #     elif embeddings.dim() == 2:
    #         embeddings = embeddings.unsqueeze(1)

    #     embeddings = self.projection_layer(embeddings)

    #     batch_size = embeddings.shape[0]
        
    #     device = embeddings.device
    #     current_input_ids = torch.full(
    #         (batch_size, 1),
    #         pad_token_id,
    #         dtype=torch.long,
    #         device=device
    #     )

    #     # generated_ids = [1, 2, 3, 4]
    #     generated_ids = [current_input_ids]
    #     past_key_values = None

    #     is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    #     for step in range(max_length):
    #         outputs = self.decoder_network(
    #             input_ids=current_input_ids,
    #             encoder_hidden_states=embeddings,
    #             past_key_values=past_key_values,
    #             use_cache=True,
    #             return_dict=True
    #         )

    #         hidden_states = outputs.last_hidden_state      
    #         logits = self.decoder_network.lm_head(hidden_states)  

    #         next_token_logits = logits[:, -1, :]            
    #         next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) 

    #         past_key_values = outputs.past_key_values

    #         next_token = torch.where(is_finished.unsqueeze(-1),
    #                                  torch.full_like(next_token, pad_token_id),
    #                                  next_token)
    #         generated_ids.append(next_token)

    #         just_finished = ~is_finished & (next_token.squeeze(-1) == eos_token_id)
    #         is_finished |= just_finished
    #         current_input_ids = next_token

    #         if is_finished.all():
    #             break

    #     all_generated_ids = torch.cat(generated_ids, dim=1)

    #     return all_generated_ids.cpu().tolist()
    
    # def decode_to_language_logits_for_inference(self, embeddings: torch.Tensor, max_length: int = 512, pad_token_id: int = 0, eos_token_id: int = 102) -> torch.Tensor:
    #     return [0]

    @staticmethod
    def reconstruction_loss(original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """Calculate the reconstruction loss.

        Arguments:
            - original_images (:obj:`torch.Tensor`): Original images.
            - reconstructed_images (:obj:`torch.Tensor`): Reconstructed images.

        Returns:
            - torch.Tensor: Computed reconstruction loss.
        """
        if len(original_images.shape) == 2:
            # For memory environment vector observations
            loss = F.mse_loss(original_images, reconstructed_images)  # L2 loss
        else:
            # For Atari image environment
            loss = torch.abs(original_images - reconstructed_images).mean()  # L1 loss
        return loss

    def lm_reconstruction_loss(self, labels: torch.Tensor, logits: torch.Tensor, ignore_index: int) -> torch.Tensor:
        total_dims = 1
        for i in labels.shape:
            total_dims *= i
        logits = logits.reshape(total_dims, -1)
        labels = labels.reshape(total_dims).long()
        if ignore_index is None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        return loss

    def perceptual_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """Calculate the perceptual loss using LPIPS.

        Arguments:
            original_images (:obj:`torch.Tensor`): Original images.
            reconstructed_images (:obj:`torch.Tensor`): Reconstructed images.

        Returns:
            torch.Tensor: Computed perceptual loss.
        """
        return torch.mean(self.lpips(original_images, reconstructed_images))

    

    def __repr__(self) -> str:
        return "Tokenizer"