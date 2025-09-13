import logging
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical, Independent, Normal, TransformedDistribution, TanhTransform

from lzero.model.common import SimNorm, L2Norm
from lzero.model.utils import cal_dormant_ratio
from .kv_caching import KeysValues
from .slicer import Head, PolicyHeadCont
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights, WorldModelOutput, hash_state
from collections import OrderedDict 
logging.getLogger().setLevel(logging.DEBUG)

from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
# In unizero_world_model.py

import torch
import torch.nn as nn

# --- HOOK FUNCTION FOR DEBUGGING ---
def print_intermediate_activation_hook(module, input, output):
    """
    A PyTorch hook that prints the mean and std of a module's output.
    This function will be registered to a specific layer (e.g., the first Linear layer in a Head).
    
    Args:
        module: The module the hook is registered on.
        input: The input to the module's forward pass.
        output: The output from the module's forward pass.
    """
    # output is the tensor we want to inspect
    mean = output.mean().item()
    std = output.std().item()
    # We add the module name for clarity, to know which layer's output we are seeing.
    print(f"  [HOOK DEBUG] Layer '{module.__class__.__name__}' Output -> mean: {mean:.6f}, std: {std:.6f}")
    


            
class LRUCache(OrderedDict):
    """
    一个固定容量的、遵循LRU（最近最少使用）原则的有序字典。
    非常适合用于管理与环形缓冲区同步的缓存映射。
    """
    def __init__(self, capacity: int=2):
        """
        初始化LRU缓存。
        参数:
            - capacity (int): 缓存的最大容量。
        """
        self.capacity = capacity
        super().__init__()

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        重写设置条目的方法，以实现LRU逻辑。
        """
        # 如果键已存在，先删除旧条目，以确保后续添加时它会成为最新项。
        if key in self:
            self.move_to_end(key)
        
        # 调用父类的方法来实际设置键值对。
        super().__setitem__(key, value)

        # 检查是否超出容量。如果超出，则删除最旧的条目。
        # popitem(last=False) 会移除并返回字典中第一个（最旧的）条目。
        if len(self) > self.capacity:
            self.popitem(last=False)

class WorldModel(nn.Module):
    """
    Overview:
        The WorldModel class is responsible for the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667),
        which is used to predict the next latent state, rewards, policy, and value based on the current latent state and action.
        The world model consists of three main components:
            - a tokenizer, which encodes observations into embeddings,
            - a transformer, which processes the input sequences,
            - and heads, which generate the logits for observations, rewards, policy, and value.
    """

    def __init__(self, config: TransformerConfig, tokenizer) -> None:
        """
        Overview:
            Initialize the WorldModel class.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration for the transformer.
            - tokenizer (:obj:`Tokenizer`): The tokenizer.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.transformer = Transformer(self.config)

        if self.config.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move all modules to the specified device
        logging.info(f"self.device: {self.device}")
        self.to(self.device)

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        self.hidden_size = config.embed_dim // config.num_heads

        # Position embedding
        if not self.config.rotary_emb:
            # self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim, device=self.device)
            # TODO(pu)
            self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim, device=self.device, max_norm=1.0)
            self.precompute_pos_emb_diff_kv()
            print(f"self.pos_emb.weight.device: {self.pos_emb.weight.device}")

        self.continuous_action_space = self.config.continuous_action_space

        # Initialize action embedding table
        if self.continuous_action_space:
            # TODO: check the effect of SimNorm
            self.act_embedding_table = nn.Sequential(
                nn.Linear(config.action_space_size, config.embed_dim, device=self.device, bias=False),
                SimNorm(simnorm_dim=self.group_size))
        else:
            # for discrete action space
            # self.act_embedding_table = nn.Embedding(config.action_space_size, config.embed_dim, device=self.device)
            # TODO(pu)
            self.act_embedding_table = nn.Embedding(config.action_space_size, config.embed_dim, device=self.device, max_norm=1.0)

            logging.info(f"self.act_embedding_table.weight.device: {self.act_embedding_table.weight.device}")

        self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', 'SimNorm')

        # Head modules
        self.head_rewards = self._create_head(self.act_tokens_pattern, self.support_size, use_norm_in_head=True) # TODO
        self.head_observations = self._create_head(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim, \
                                                    self._get_final_norm(self.final_norm_option_in_obs_head)  # NOTE: using the specified normalization method for observations head
                                                   )
        if self.continuous_action_space:
            self.sigma_type = self.config.sigma_type
            self.bound_type = self.config.bound_type
            self.head_policy = self._create_head_cont(self.value_policy_tokens_pattern, self.action_space_size)
        else:
            self.head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_space_size)
        self.head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size, use_norm_in_head=True)

        # # ==================== NEW DEBUGGING CODE VIA HOOKS ====================
        # # We will attach our hook to the first Linear layer inside the head_value and head_rewards modules.
        # # The head_module is an nn.Sequential, so its layers can be accessed by index.
        # # Index 0: First nn.Linear
        # # Index 1: nn.GELU
        # # Index 2: Second nn.Linear

        # # Get the first linear layer from the sequential module
        # first_linear_layer_value = self.head_value.head_module[0]
        # first_linear_layer_rewards = self.head_rewards.head_module[0]

        # # Register the forward hook
        # print("--- Attaching DEBUG hooks to head_value and head_rewards ---")
        # self.value_hook_handle = first_linear_layer_value.register_forward_hook(print_intermediate_activation_hook)
        # self.rewards_hook_handle = first_linear_layer_rewards.register_forward_hook(print_intermediate_activation_hook)

        # # NOTE: It's good practice to store the hook handle so you can remove it later if needed, e.g., during evaluation or after debugging.
        # # To remove the hook: self.value_hook_handle.remove()
        # # ====================================================================


        # Build the set of modules to skip during re-initialization.
        # This is compatible with cases where self.tokenizer.encoder does not have 'pretrained_model',
        # or self.tokenizer does not have 'decoder_network'.
        # NOTE: This step is crucial — without skipping, pretrained modules (e.g., encoder/decoder) would be unintentionally re-initialized
        skip_modules = set()
        if hasattr(self.tokenizer.encoder, 'pretrained_model'):
            skip_modules.update(self.tokenizer.encoder.pretrained_model.modules())
        if hasattr(self.tokenizer, 'decoder_network'):
            if self.tokenizer.decoder_network is not None:
                skip_modules.update(self.tokenizer.decoder_network.modules())

        def custom_init(module):
            # If the current module is part of the skip list, return without reinitializing
            if module in skip_modules:
                return
            # Otherwise, apply the specified initialization method
            init_weights(module, norm_type=self.config.norm_type)

        # Recursively apply `custom_init` to all submodules of the model
        self.apply(custom_init)

        # self.apply(init_weights)

        self._initialize_last_layer()

        # for self.kv_cache_init_infer
        # In contrast, init_infer only needs to retain the results of the most recent step.
        # self.shared_pool_size_init = int(2)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?
        # 先设置为game_segment_length，以保持self.shared_pool_init_infer都是有效的kv
         # TODO: 非常重要，应该改为和segment_length一样
        self.shared_pool_size_init = int(self.config.game_segment_length)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?

        # self.shared_pool_size_init = int(20)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?
        # self.shared_pool_size_init = int(200)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?
       
        self.num_simulations = getattr(self.config, 'num_simulations', 50)

        # TODO: recur kv pool是否应该分成不同的环境有不同的pool呢
        self.shared_pool_size_recur = int(self.num_simulations*self.env_num)

        # self.shared_pool_size_init = int(50)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?
        self.stale_pointer_detections = 0
        self.stale_pointer_detections_recur = 0
        # Cache structures
        self._initialize_cache_structures()

        # Projection input dimension
        self._initialize_projection_input_dim()

        # Hit count and query count statistics
        self._initialize_statistics()

        # Initialize keys and values for transformer
        self._initialize_transformer_keys_values()

        self.latent_recon_loss = torch.tensor(0., device=self.device)
        self.perceptual_loss = torch.tensor(0., device=self.device)

        # TODO: check the size of the shared pool
        # for self.kv_cache_recurrent_infer
        # If needed, recurrent_infer should store the results of the one MCTS search.

        # self.shared_pool_size_recur = int(2)

        self.shared_pool_recur_infer = [None] * self.shared_pool_size_recur
        self.shared_pool_index = 0

        # for self.kv_cache_init_infer
        # In contrast, init_infer only needs to retain the results of the most recent step.
        # self.shared_pool_size_init = int(2)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?
        # TODO
        # self.shared_pool_size_init = int(50)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?

        # TODO: 分析self.env_num>1的情况，不同env之间的相同latent-state hash对应的kv_cache可以公用吗
        self.shared_pool_init_infer = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        self.shared_pool_index_init_envs = [0 for _ in range(self.env_num)]

        # for self.kv_cache_wm
        self.shared_pool_size_wm = int(self.env_num)
        self.shared_pool_wm = [None] * self.shared_pool_size_wm
        self.shared_pool_index_wm = 0

        self.reanalyze_phase = False

        # 用于t-SNE可视化的计数器
        self.tsne_visualization_step = 0

        # 用于存储梯度hook的handle
        self._grad_hooks = []
        
        # ======================= 注册梯度Hooks =======================
        # self.register_gradient_hooks(self.tokenizer.representation_network)
        # =============================================================


    def register_gradient_hooks(self, model_to_hook: nn.Module):
        """
        递归地为模型中的可学习参数注册梯度hook。
        """
        
        def hook_fn(grad):
            # 这个hook会在该参数的梯度被计算出来后立即执行
            if grad is not None:
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                # 为了避免信息过载，我们可以只打印非零梯度的统计信息
                if grad_norm > 1e-9:
                    print(f"    [GRAD HOOK] Param: {name}, Shape: {grad.shape} | Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

        # 遍历模型的所有命名参数
        for name, param in model_to_hook.named_parameters():
            if param.requires_grad:
                # 使用 .register_hook() 为张量注册hook
                handle = param.register_hook(hook_fn)
                self._grad_hooks.append(handle)
                print(f"  [INFO] Registered gradient hook for: {name}")

    def remove_gradient_hooks(self):
        """
        移除所有已注册的梯度hook，在评估或部署时调用。
        """
        for handle in self._grad_hooks:
            handle.remove()
        self._grad_hooks.clear()
        print("[INFO] All gradient hooks removed.")

    def _analyze_latent_representation(
        self, 
        latent_states: torch.Tensor, 
        timesteps: torch.Tensor, 
        game_states: torch.Tensor, 
        predicted_values: torch.Tensor,
        predicted_rewards: torch.Tensor,
        step_counter: int
    ):
        """
        分析并记录 latent states 的统计信息和t-SNE可视化。
        【新功能】：在t-SNE图上显示对应的游戏图像，并标注预测的Value和Reward。
        
        Args:
            latent_states (torch.Tensor): Encoder的输出, shape (B*L, 1, E)
            timesteps (torch.Tensor): 对应的时间步, shape (B, L)
            game_states (torch.Tensor): 原始的游戏观测, shape (B, L, C, H, W)
            predicted_values (torch.Tensor): 预测的标量Value, shape (B*L,)
            predicted_rewards (torch.Tensor): 预测的标量Reward, shape (B*L,)
            step_counter (int): 全局训练步数
        """
        # ... (统计分析部分保持不变) ...
        # (确保 latent_states 和 game_states 的形状为 (N, ...))
        if latent_states.dim() > 2:
            latent_states = latent_states.reshape(-1, latent_states.shape[-1])
        num_c, num_h, num_w = game_states.shape[-3:]
        game_states = game_states.reshape(-1, num_c, num_h, num_w)
        
        with torch.no_grad():
            l2_norm = torch.norm(latent_states, p=2, dim=1).mean()
            mean = latent_states.mean()
            std = latent_states.std()
            print(f"[Step {step_counter}] Latent Stats | L2 Norm: {l2_norm:.4f}, Mean: {mean:.4f}, Std: {std:.4f}")

        # 带图像和V/R值的 t-SNE 可视化
        if step_counter >= 0:
        # if step_counter > 0 and step_counter % 200 == 0:
        
            print(f"[Step {step_counter}] Performing t-SNE analysis with images, values, and rewards...")

            # 将数据转换到CPU
            latents_np = latent_states.detach().cpu().numpy()
            images_np = game_states.detach().cpu().numpy()
            values_np = predicted_values.detach().cpu().numpy()
            rewards_np = predicted_rewards.detach().cpu().numpy()
            
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
            tsne_results = tsne.fit_transform(latents_np)
            
            # --- 绘制带图像和标注的散点图 ---
            
            # 减少图像数量以保持清晰
            num_points_to_plot = min(len(latents_np), 70) # 减少到70个点
            indices = np.random.choice(len(latents_np), num_points_to_plot, replace=False)
            
            fig, ax = plt.subplots(figsize=(20, 18)) # 增大画布尺寸
            
            # 先画出所有点的散点图作为背景
            ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=values_np, cmap='viridis', alpha=0.3, s=10)
            
            for i in indices:
                x, y = tsne_results[i]
                img = images_np[i].transpose(1, 2, 0)
                img = np.clip(img, 0, 1)

                # 放置图像
                im = OffsetImage(img, zoom=0.7) # 稍微放大图像
                ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.0, bboxprops=dict(edgecolor='none'))
                ax.add_artist(ab)
                
                # 在图像下方添加文字标注
                text_label = f"V:{values_np[i]:.1f} R:{rewards_np[i]:.1f}"
                ax.text(x, y - 1.0, text_label, ha='center', va='top', fontsize=8, color='red',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

            ax.update_datalim(tsne_results)
            ax.autoscale()
            
            ax.set_title(f't-SNE of Latent States (Value as Color) at Step {step_counter}', fontsize=16)
            ax.set_xlabel('t-SNE dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE dimension 2', fontsize=12)
            
            # 添加colorbar来解释背景点的颜色
            norm = plt.Normalize(values_np.min(), values_np.max())
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Predicted Value')

            # save_path = f'zoo/atari/unizero_mspacman_analyze/tsne_with_vr_{self.config.optim_type}_lr{self.config.learning_rate}_step_{step_counter}.png'
            save_path = f'/mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/unizero_mspacman_analyze/tsne_with_vr_{self.config.optim_type}_lr{self.config.learning_rate}_obs96_step_{step_counter}.png'

            plt.savefig(save_path)
            plt.close()
            print(f"t-SNE plot with V/R annotations saved to {save_path}")

    def _debug_check_for_stale_pointers(self, env_id: int, current_key: Any, index_to_be_written: int):
        """
        调试函数：检查即将被写入的索引是否存在过时的指针。
        """
        # 获取对应环境的指针映射表
        cache_map = self.past_kv_cache_init_infer_envs[env_id]
        
        # 遍历映射表中的所有条目 (旧哈希 -> 旧索引)
        for old_key, old_index in cache_map.items():
            # 检查条件：
            # 1. 旧索引 == 即将被覆盖的索引
            # 2. 旧哈希 != 当前要写入的新哈希
            if old_index == index_to_be_written and old_key != current_key:
                # 如果条件满足，说明我们找到了一个过时指针
                self.stale_pointer_detections += 1
                
                # 打印详细的调试信息
                print("="*60)
                print(f"!!! INIT BUG CONDITION DETECTED (Detection #{self.stale_pointer_detections}) !!!")
                print(f"    Environment ID: {env_id}")
                print(f"    Pool Index to be overwritten: {index_to_be_written}")
                print(f"    New state hash being written: '{current_key}'")
                print(f"    Stale pointer found in cache_map: '{old_key}' also points to index {old_index}.")
                print(f"    This means the data for '{old_key}' is about to be lost, but its pointer remains.")
                print(f"    Current cache_map size: {len(cache_map)}")
                print("="*60)
                
                # 找到一个就足够了，可以提前退出循环以提高效率
                return

    def _debug_check_for_stale_pointers_recur(self, current_key: Any, index_to_be_written: int):
        """
        调试函数：检查 recurrent cache 中是否存在过时的指针。
        """
        cache_map = self.past_kv_cache_recurrent_infer
        
        for old_key, old_index in cache_map.items():
            if old_index == index_to_be_written and old_key != current_key:
                self.stale_pointer_detections_recur += 1
                print("="*60)
                print(f"!!! RECURRENT BUG DETECTED (Detection #{self.stale_pointer_detections_recur}) !!!")
                print(f"    Pool Index to be overwritten: {index_to_be_written}")
                print(f"    New state hash being written: '{current_key}'")
                print(f"    Stale pointer found: '{old_key}' also points to index {old_index}.")
                print("="*60)
                return

    def _get_final_norm(self, norm_option: str) -> nn.Module:
        """
        Return the corresponding normalization module based on the specified normalization option.
        """
        if norm_option == 'LayerNorm':
            return nn.LayerNorm(self.config.embed_dim, eps=1e-5)
        elif norm_option == 'SimNorm':
            return SimNorm(simnorm_dim=self.config.group_size)
        elif norm_option == 'L2Norm':
            return L2Norm(eps=1e-6)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_obs_head: {norm_option}")

    def custom_copy_kv_cache_to_shared_init_envs(self, src_kv: KeysValues, env_id) -> int:
        """
        Overview:
            Efficiently copies the contents of a KeysValues object to the shared pool for a specific environment in the init_infer stage.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object from which data is copied.
            - env_id (:obj:`int`): The identifier of the environment for which the cache is being copied.
        Returns:
            - index (:obj:`int`): The index in the shared pool where the KeysValues object is stored.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]] is None:
            self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]] = KeysValues(
                src_kv_shape[0],  # Number of elements (n)
                src_kv_shape[1],  # Number of attention heads (num_heads)
                src_kv_shape[2],  # Maximum number of tokens (max_tokens)
                src_kv_shape[3] * src_kv_shape[1],  # Embedding dimension (embed_dim)
                len(src_kv),  # Number of layers (num_layers)
                src_kv._keys_values[0]._k_cache._cache.device,  # Device where the cache is stored
            )
        
        dst_kv = self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_() for efficient data transfer
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = self.shared_pool_index_init_envs[env_id]
        self.shared_pool_index_init_envs[env_id] = (self.shared_pool_index_init_envs[env_id] + 1) % self.shared_pool_size_init
        
        return index

    def custom_copy_kv_cache_to_shared_wm(self, src_kv: KeysValues) -> int:
        """
        Overview:
            Efficiently copies the contents of a KeysValues object to the shared pool for world model usage.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object from which data is copied.
        Returns:
            - index (:obj:`int`): The index in the shared pool where the KeysValues object is stored.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_wm[self.shared_pool_index_wm] is None:
            self.shared_pool_wm[self.shared_pool_index_wm] = KeysValues(
                src_kv_shape[0],  # Number of elements (n)
                src_kv_shape[1],  # Number of attention heads (num_heads)
                src_kv_shape[2],  # Maximum number of tokens (max_tokens)
                src_kv_shape[3] * src_kv_shape[1],  # Embedding dimension (embed_dim)
                len(src_kv),  # Number of layers (num_layers)
                src_kv._keys_values[0]._k_cache._cache.device,  # Device where the cache is stored
            )
        
        dst_kv = self.shared_pool_wm[self.shared_pool_index_wm]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_() for efficient data transfer
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        self.shared_pool_index_wm = (self.shared_pool_index_wm + 1) % self.shared_pool_size_wm
        
        return dst_kv

    def custom_copy_kv_cache_to_shared_recur(self, src_kv: KeysValues) -> int:
        """
        Overview:
            Efficiently copies the contents of a KeysValues object to the shared pool for recurrent inference.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object from which data is copied.
        Returns:
            - index (:obj:`int`): The index in the shared pool where the KeysValues object is stored.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_recur_infer[self.shared_pool_index] is None:
            self.shared_pool_recur_infer[self.shared_pool_index] = KeysValues(
                src_kv_shape[0],  # Number of elements (n)
                src_kv_shape[1],  # Number of attention heads (num_heads)
                src_kv_shape[2],  # Maximum number of tokens (max_tokens)
                src_kv_shape[3] * src_kv_shape[1],  # Embedding dimension (embed_dim)
                len(src_kv),  # Number of layers (num_layers)
                src_kv._keys_values[0]._k_cache._cache.device,  # Device where the cache is stored
            )
        
        dst_kv = self.shared_pool_recur_infer[self.shared_pool_index]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_() for efficient data transfer
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = self.shared_pool_index
        self.shared_pool_index = (self.shared_pool_index + 1) % self.shared_pool_size_recur
        
        return index

    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
        self.policy_entropy_weight = self.config.policy_entropy_weight
        self.predict_latent_loss_type = self.config.predict_latent_loss_type
        self.group_size = self.config.group_size
        self.num_groups = self.config.embed_dim // self.group_size
        self.obs_type = self.config.obs_type
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.gamma = self.config.gamma
        self.context_length = self.config.context_length
        self.dormant_threshold = self.config.dormant_threshold
        self.analysis_dormant_ratio = self.config.analysis_dormant_ratio
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.obs_per_embdding_dim = self.config.embed_dim
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)

    def _initialize_patterns(self) -> None:
        """Initialize patterns for block masks."""
        self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
        self.all_but_last_latent_state_pattern[-2] = 0
        self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.act_tokens_pattern[-1] = 1
        self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.value_policy_tokens_pattern[-2] = 1

    # def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
    #     """Create head modules for the transformer."""
    #     modules = [
    #         nn.Linear(self.config.embed_dim, self.config.embed_dim),
    #         nn.GELU(approximate='tanh'),
    #         nn.Linear(self.config.embed_dim, output_dim)
    #     ]
    #     if norm_layer:
    #         modules.append(norm_layer)
    #     return Head(
    #         max_blocks=self.config.max_blocks,
    #         block_mask=block_mask,
    #         head_module=nn.Sequential(*modules)
    #     )

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None, use_norm_in_head: bool = False) -> Head:
        """Create head modules for the transformer."""
        # modules = [
        #     nn.Linear(self.config.embed_dim, self.config.embed_dim),
        # ]
        
        # ==================== 头部优化：防御性设计 ====================
        # 在头部入口处增加一个LayerNorm，以防止输入饱和。
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
        ]
        # =============================================================


        # ==================== PROPOSED FIX ====================
        # Add a LayerNorm after the first linear layer and before the activation.
        # This stabilizes the activations within the head, preventing drift.
        if use_norm_in_head:
            modules.append(nn.LayerNorm(self.config.embed_dim))
        # ======================================================

        modules.extend([
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim, output_dim),
            # 最后的LayerNorm可以保留，也可以视情况移除，因为它主要影响输出的尺度
            # nn.LayerNorm(output_dim) 
        ])

        if norm_layer:
            modules.append(norm_layer)
            
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_head_cont(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        from ding.model.common import ReparameterizationHead
        self.fc_policy_head = ReparameterizationHead(
            input_size=self.config.embed_dim,
            output_size=output_dim,
            layer_num=2,  # TODO: check the effect of layer_num
            sigma_type=self.sigma_type,
            activation=nn.GELU(approximate='tanh'),
            fixed_sigma_value=self.config.fixed_sigma_value if self.sigma_type == 'fixed' else 0.5,
            norm_type=None,
            bound_type=self.bound_type
        )
        return PolicyHeadCont(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=self.fc_policy_head
        )

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            if self.continuous_action_space:
                module_to_initialize = [self.head_value, self.head_rewards, self.head_observations]
            else:
                module_to_initialize = [self.head_policy, self.head_value, self.head_rewards, self.head_observations]
            for head in module_to_initialize:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_cache_structures(self) -> None:
        """Initialize cache structures for past keys and values."""
        from collections import defaultdict
        # self.past_kv_cache_recurrent_infer = defaultdict(dict)
        # 使用 LRUCache 替换 defaultdict，并同步容量

        # ========================= 核心修复与注释 (Recurrent Infer) =========================
        # 问题: recurrent_infer 缓存同样存在 LRUCache 与环形缓冲区逻辑不匹配的问题。
        #
        # 修复方案:
        # 1. 将 past_kv_cache_recurrent_infer 从 LRUCache 改为标准字典。
        # 2. 引入辅助列表 pool_idx_to_key_map_recur_infer 来维护反向映射。
        #    这确保了在覆写 recurrent 数据池中的条目时，可以同步删除旧的指针。
        
        self.past_kv_cache_recurrent_infer = {}
        self.pool_idx_to_key_map_recur_infer = [None] * self.shared_pool_size_recur
        # ========================== 修复结束 ==========================

        # self.past_kv_cache_init_infer_envs = [defaultdict(dict) for _ in range(self.env_num)]

        # TODO(pu): 非常重要  self.past_kv_cache_init_infer_envs应该改成和(shared_pool_size_init)完全一致，
        # 目前是将shared_pool_size_init设置为segment_length以在一次collect后 清空self.past_kv_cache_init_infer_envs
        # 来避免self.past_kv_cache_init_infer_envs里面存有kv索引过期的问题

        # ========================= 核心修复与注释 =========================
        # 原来的实现:
        # self.past_kv_cache_init_infer_envs = [defaultdict(dict) for _ in range(self.env_num)]
        #
        # 问题: defaultdict 会无限增长，并且不会自动删除与环形缓冲区中
        #       被覆盖数据相关的旧“指针”，导致Episode内部的缓存污染。
        #
        # 修复方案:
        # 使用我们定义的LRUCache，其容量与环形缓冲区的大小(shared_pool_size_init)完全一致。
        #
        # 效果:
        # 1. 自动淘汰: 当添加第 N+1 个新条目时，LRUCache会自动删除最旧的那个条目。
        # 2. 生命周期同步: 这确保了“指针字典”中的映射关系，与“数据池”中实际存储的数据
        #    完全同步。当数据池的索引0被新数据覆盖时，指向旧索引0的指针也已被自动清除。
        # 3. 杜绝污染: 从根本上解决了Episode内部的状态哈希碰撞问题。
        
        # self.past_kv_cache_init_infer_envs = [LRUCache(self.shared_pool_size_init-1) for _ in range(self.env_num)]
        # ========================== 修复结束 ==========================

        # ========================= 核心修复与注释 =========================
        # 问题: LRUCache 的淘汰逻辑（基于访问顺序）与环形缓冲区的覆写逻辑（基于写入顺序）不匹配，导致指针过时。
        #
        # 修复方案:
        # 1. 使用一个标准的字典 `past_kv_cache_init_infer_envs` 来存储 {state_hash -> pool_index}。
        # 2. 引入一个辅助列表 `pool_idx_to_key_map_init_envs` 来维护反向映射 {pool_index -> state_hash}。
        #
        # 效果:
        # 在向环形缓冲区的某个索引写入新数据之前，我们可以通过辅助列表立即找到即将被覆盖的旧 state_hash，
        # 并从主字典中精确地删除这个过时的条目。这确保了字典和数据池的完全同步。
        
        self.past_kv_cache_init_infer_envs = [{} for _ in range(self.env_num)]
        # 辅助数据结构，用于反向查找：pool_index -> key
        self.pool_idx_to_key_map_init_envs = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        # ========================== 修复结束 ==========================

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []

    def _initialize_projection_input_dim(self) -> None:
        """Initialize the projection input dimension based on the number of observation tokens."""
        if self.num_observations_tokens == 16:
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:
            self.projection_input_dim = self.obs_per_embdding_dim

    def _initialize_statistics(self) -> None:
        """Initialize counters for hit count and query count statistics."""
        self.recur_hit_count = 0
        self.recur_total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.length_largethan_contextminus3_cnt = 0

        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

    def _initialize_transformer_keys_values(self) -> None:
        """Initialize keys and values for the transformer."""
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1,
                                                                                     max_tokens=self.context_length)
        self.keys_values_wm_single_env_tmp = self.transformer.generate_empty_keys_values(n=1,
                                                                                     max_tokens=self.context_length)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num,
                                                                          max_tokens=self.context_length)

    def precompute_pos_emb_diff_kv(self):
        """ Precompute positional embedding differences for key and value. """
        if self.context_length <= 2:
            # If context length is 2 or less, no context is present
            return
        # Precompute positional embedding matrices for inference in collect/eval stages, not for training
        self.positional_embedding_k = [
            self._get_positional_embedding(layer, 'key')
            for layer in range(self.config.num_layers)
        ]
        self.positional_embedding_v = [
            self._get_positional_embedding(layer, 'value')
            for layer in range(self.config.num_layers)
        ]

        # Precompute all possible positional embedding differences
        self.pos_emb_diff_k = []
        self.pos_emb_diff_v = []

        for layer in range(self.config.num_layers):
            layer_pos_emb_diff_k = {}
            layer_pos_emb_diff_v = {}

            for start in [2]:
                for end in [self.context_length - 1]:
                    original_pos_emb_k = self.positional_embedding_k[layer][:, :, start:end, :]
                    new_pos_emb_k = self.positional_embedding_k[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_k[(start, end)] = new_pos_emb_k - original_pos_emb_k

                    original_pos_emb_v = self.positional_embedding_v[layer][:, :, start:end, :]
                    new_pos_emb_v = self.positional_embedding_v[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_v[(start, end)] = new_pos_emb_v - original_pos_emb_v

            self.pos_emb_diff_k.append(layer_pos_emb_diff_k)
            self.pos_emb_diff_v.append(layer_pos_emb_diff_v)

    def _get_positional_embedding(self, layer, attn_type) -> torch.Tensor:
        """
         Helper function to get positional embedding for a given layer and attention type.

         Arguments:
         - layer (:obj:`int`): Layer index.
         - attn_type (:obj:`str`): Attention type, either 'key' or 'value'.

         Returns:
         - torch.Tensor: The positional embedding tensor.
         """
        attn_func = getattr(self.transformer.blocks[layer].attn, attn_type)
        if torch.cuda.is_available():
            return attn_func(self.pos_emb.weight).view(
                1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
            ).transpose(1, 2).to(self.device).detach()
        else:
            return attn_func(self.pos_emb.weight).view(
                1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
            ).transpose(1, 2).detach()

    def forward(
        self,
        obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, Tuple]],
        past_keys_values: Optional[torch.Tensor] = None,
        kvcache_independent: bool = False,
        is_init_infer: bool = True,
        valid_context_lengths: Optional[torch.Tensor] = None,
        start_pos: Union[int, List[int]] = 0,
        search_depth: Optional[List[int]] = None
    ) -> "WorldModelOutput":
        """
        Overview:
            Forward pass for the world model. This method processes observation embeddings and/or action tokens,
            optionally adds position encodings (with or without rotary position embeddings), passes the resulting
            sequences through the transformer, and finally generates logits for observations, rewards, policy, and value.
        
        Arguments:
            - obs_embeddings_or_act_tokens (dict): Dictionary containing one or more of the following keys:
                - 'obs_embeddings': torch.Tensor representing observation embeddings.
                - 'act_tokens': torch.Tensor representing action tokens.
                - 'obs_embeddings_and_act_tokens': Combined data for both observations and actions.
            - past_keys_values (Optional[torch.Tensor]): Cached key-value pairs for the transformer. Defaults to None.
            - kvcache_independent (bool): Flag to indicate whether key-value caching is independent. Defaults to False.
            - is_init_infer (bool): Flag to indicate if this is the initial inference step. Defaults to True.
            - valid_context_lengths (Optional[torch.Tensor]): Valid lengths for the context. Defaults to None.
            - start_pos (int or List[int]): Starting positional index for the current sequence (or batch). Defaults to 0.
            - search_depth (Optional[List[int]]): List representing the search depth for each batch element, used for
                position encoding adjustment. Defaults to None.
        
        Returns:
            WorldModelOutput: An output instance containing:
                - x: Output features from the transformer.
                - logits for observations.
                - logits for rewards.
                - logits_ends (None).
                - logits for policy.
                - logits for value.
        """

        # Calculate previous steps based on key-value caching configuration
        if kvcache_independent:
            # If kv caching is independent, compute previous steps for each past key-value pair.
            prev_steps = torch.tensor(
                [0 if past_keys_values is None else past_kv.size for past_kv in past_keys_values],
                device=self.device
            )
        else:
            # Otherwise, use a single value for previous steps.
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # Reset valid context lengths during initial inference phase.
        if is_init_infer:
            valid_context_lengths = None

        # sequences: torch.Tensor  # Output sequence to feed into transformer
        # num_steps: int           # Number of timesteps in the sequence
        # start_pos_adjusted: Union[int, List[int]]  # Adjusted starting position index for positional encoding

        if not self.config.rotary_emb:
            start_pos_adjusted = None

        # Process observation embeddings if available.
        if "obs_embeddings" in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens["obs_embeddings"]
            # If the observation embeddings have 2 dimensions, expand them to include a time dimension.
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)
            
            if not self.config.rotary_emb:
                # Add traditional position embeddings if not using rotary embeddings.
                sequences = self._add_position_embeddings(
                    obs_embeddings, prev_steps, num_steps, kvcache_independent,
                    is_init_infer, valid_context_lengths
                )
            else:
                # Keep the observation embeddings unchanged when using rotary embeddings.
                sequences = obs_embeddings

                if is_init_infer:
                    if self.reanalyze_phase:
                        # During reanalyze phase in initial inference, adjust start_pos:
                        # Multiply by 2 because timestep only counts observations,
                        # but the sequence contains both observations and actions.
                        start_pos_adjusted = start_pos * 2
                        if not isinstance(start_pos_adjusted, (int, float)):
                            # Pad zero if start_pos_adjusted is not a scalar.
                            padding = np.zeros((start_pos_adjusted.shape[0], 1), dtype=start_pos_adjusted.dtype)
                            start_pos_adjusted = np.concatenate([start_pos_adjusted, padding], axis=1).reshape(-1)
                    else:
                        # For regular initial inference, adjust start_pos accordingly.
                        if isinstance(start_pos, (int, float)):
                            start_pos_adjusted = start_pos * 2
                        else:
                            start_pos_adjusted = [pos * 2 for pos in start_pos]
                else:
                    # For recurrent inference (non-init), calculate the correct positional index.
                    if self.reanalyze_phase:
                        # In reanalyze phase, start_pos for batch mode might be an array that needs padding.
                        if not isinstance(start_pos, (int, float)):
                            padding = np.zeros((start_pos.shape[0], 1), dtype=start_pos.dtype)
                            start_pos_adjusted = np.concatenate([start_pos, padding], axis=1).reshape(-1)
                        # Ensure search_depth length matches adjusted start_pos.
                        assert len(search_depth) == len(start_pos_adjusted)
                        start_pos_adjusted = [
                            (search_depth[i] + pos + 1) * 2 + 1 for i, pos in enumerate(start_pos_adjusted)
                        ]
                    else:
                        start_pos_adjusted = [
                            (search_depth[i] + pos) * 2 + 2 for i, pos in enumerate(start_pos)
                        ]

        # Process action tokens if available.
        elif "act_tokens" in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens["act_tokens"]
            if self.continuous_action_space:
                num_steps = 1
                act_tokens = act_tokens.float()
                if len(act_tokens.shape) == 2:
                    act_tokens = act_tokens.unsqueeze(1)
            else:
                if len(act_tokens.shape) == 3:
                    act_tokens = act_tokens.squeeze(1)
                num_steps = act_tokens.size(1)
            # Convert action tokens to embeddings using the action embedding table.
            act_embeddings = self.act_embedding_table(act_tokens)
            if not self.config.rotary_emb:
                sequences = self._add_position_embeddings(
                    act_embeddings, prev_steps, num_steps, kvcache_independent,
                    is_init_infer, valid_context_lengths
                )
            else:
                sequences = act_embeddings

                if is_init_infer:
                    if self.reanalyze_phase:
                        # In reanalyze phase during initial inference, the action tokens represent the current timestep.
                        start_pos_adjusted = start_pos * 2 + 1
                        if not isinstance(start_pos_adjusted, (int, float)):
                            padding = np.zeros((start_pos_adjusted.shape[0], 1), dtype=start_pos_adjusted.dtype)
                            start_pos_adjusted = np.concatenate([start_pos_adjusted, padding], axis=1).reshape(-1)
                    else:
                        # For regular initial inference using action tokens, adjust start_pos by subtracting 1.
                        if isinstance(start_pos, (int, float)):
                            start_pos_adjusted = start_pos * 2 - 1
                        else:
                            start_pos_adjusted = [pos * 2 - 1 for pos in start_pos]
                else:
                    # During recurrent inference for action tokens.
                    if self.reanalyze_phase:
                        if not isinstance(start_pos, (int, float)):
                            padding = np.zeros((start_pos.shape[0], 1), dtype=start_pos.dtype)
                            start_pos_adjusted = np.concatenate([start_pos, padding], axis=1).reshape(-1)
                        assert len(search_depth) == len(start_pos_adjusted)
                        start_pos_adjusted = [
                            (search_depth[i] + pos + 1) * 2 + 1 for i, pos in enumerate(start_pos_adjusted)
                        ]
                    else:
                        start_pos_adjusted = [
                            (search_depth[i] + pos) * 2 + 1 for i, pos in enumerate(start_pos)
                        ]

        # Process combined observation embeddings and action tokens.
        elif "obs_embeddings_and_act_tokens" in obs_embeddings_or_act_tokens:
            # Process combined inputs to calculate either the target value (for training)
            # or target policy (for reanalyze phase).
            if self.continuous_action_space:
                sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens, prev_steps)
            else:
                sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens, prev_steps)
            # Adjust start positions: multiply by 2 as the sequence has both obs and act.
            start_pos_adjusted = [pos * 2 for pos in start_pos]
        else:
            raise ValueError("Input dictionary must contain one of 'obs_embeddings', 'act_tokens', or 'obs_embeddings_and_act_tokens'.")

        # Pass the sequence through the transformer.
        x = self._transformer_pass(
            sequences, past_keys_values, kvcache_independent, valid_context_lengths, start_pos=start_pos_adjusted
        )

        # print(f"x.mean(): {x.mean().item():.6f}, x.std(): {x.std().item():.6f}")

        # Generate logits for various components.
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # print(f"logits_observations.mean(): {logits_observations.mean().item():.6f}")
        # print(f"logits_rewards.mean(): {logits_rewards.mean().item():.6f}")
        # print(f"logits_policy.mean(): {logits_policy.mean().item():.6f}")
        # print(f"logits_value.mean(): {logits_value.mean().item():.6f}")

        # The 'logits_ends' is intentionally set to None.
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    def _add_position_embeddings(self, embeddings, prev_steps, num_steps, kvcache_independent, is_init_infer,
                                 valid_context_lengths):
        """
        Add position embeddings to the input embeddings.

        Arguments:
            - embeddings (:obj:`torch.Tensor`): Input embeddings.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
            - num_steps (:obj:`int`): Number of steps.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - is_init_infer (:obj:`bool`): Initialize inference.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Embeddings with position information added.
        """
        if kvcache_independent:
            steps_indices = prev_steps + torch.arange(num_steps, device=embeddings.device)
            position_embeddings = self.pos_emb(steps_indices).view(-1, num_steps, embeddings.shape[-1])
            return embeddings + position_embeddings
        else:
            if is_init_infer:
                return embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device))
            else:
                valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)
                position_embeddings = self.pos_emb(
                    valid_context_lengths + torch.arange(num_steps, device=self.device)).unsqueeze(1)
                return embeddings + position_embeddings

    def _process_obs_act_combined_cont(self, obs_embeddings_or_act_tokens, prev_steps):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        if self.continuous_action_space:
            act_tokens = act_tokens.float()
            if len(act_tokens.shape) == 2:  # TODO
                act_tokens = act_tokens.unsqueeze(-1)

        # B, L, E
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        # B, L*2, E
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return_result = obs_act_embeddings
        if not self.config.rotary_emb:
            return_result += self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device))
        return return_result, num_steps

    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens, prev_steps):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act
            
        return_result = obs_act_embeddings
        if not self.config.rotary_emb:
            return_result += self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device))
        return return_result, num_steps

    def _transformer_pass(self, sequences, past_keys_values, kvcache_independent, valid_context_lengths, start_pos: int = 0):
        """
        Pass sequences through the transformer.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input sequences.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): Previous keys and values for transformer.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Transformer output.
        """
        if kvcache_independent:
            x = [self.transformer(sequences[k].unsqueeze(0), past_kv,
                                  valid_context_lengths=valid_context_lengths[k].unsqueeze(0), start_pos=start_pos) for k, past_kv in
                 enumerate(past_keys_values)]
            return torch.cat(x, dim=0)
        else:
            return self.transformer(sequences, past_keys_values, valid_context_lengths=valid_context_lengths, start_pos=start_pos)

    @torch.no_grad()
    def reset_for_initial_inference(self, obs_act_dict: torch.FloatTensor, start_pos: int = 0) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Arguments:
            - obs_act_dict (:obj:`torch.FloatTensor`): A dictionary containing 'obs', 'action', and 'current_obs'.
        Returns:
            - torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(obs_act_dict, dict):
            batch_obs = obs_act_dict['obs']  # obs_act_dict['obs'] is at timestep t
            batch_action = obs_act_dict['action'] # obs_act_dict['action'] is at timestep t
            batch_current_obs = obs_act_dict['current_obs'] # obs_act_dict['current_obs'] is at timestep t+1

        # Encode observations to latent embeddings.
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_obs)

        if batch_current_obs is not None:
            # ================ Collect and Evaluation Phase ================
            # Encode current observations to latent embeddings
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_current_obs)
            # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
            self.latent_state = current_obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action,
                                                                                   current_obs_embeddings, start_pos)
        else:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            self.latent_state = obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action, None, start_pos)

        return outputs_wm, self.latent_state

    @torch.no_grad()
    def wm_forward_for_initial_infererence(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None, start_pos: int = 0) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - last_obs_embeddings (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = last_obs_embeddings.shape
        if n <= self.env_num and current_obs_embeddings is not None:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                 # Determine whether it is the first step in an episode.
                if self.continuous_action_space:
                    first_step_flag = not isinstance(batch_action[0], np.ndarray)
                else:
                    first_step_flag = max(batch_action) == -1
                if first_step_flag:
                    # ------------------------- First Step of an Episode -------------------------
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=current_obs_embeddings.shape[0],
                                                                                      max_tokens=self.context_length)
                    # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, start_pos=start_pos)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)
                else:
                    # --------------------- Continuing an Episode (Multi-environment) ---------------------
                    # current_obs_embeddings is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []

                    for i in range(ready_env_num):
                        # Retrieve latent state for a single environment
                        # TODO: len(last_obs_embeddings) may smaller than len(current_obs_embeddings), because some environments may have done

                        state_single_env = last_obs_embeddings[i]
                        # Compute hash value using latent state for a single environment
                        cache_key = hash_state(state_single_env.view(-1).cpu().numpy())  # last_obs_embeddings[i] is torch.Tensor

                        # Retrieve cached value
                        cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                        if cache_index is not None:
                            matched_value = self.shared_pool_init_infer[i][cache_index]
                        else:
                            matched_value = None

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # If a matching value is found, add it to the list
                            self.root_hit_cnt += 1
                            # if self.root_total_query_cnt > 0 and self.root_total_query_cnt % 50 == 0:
                            #     self.root_hit_freq = self.root_hit_cnt / self.root_total_query_cnt
                            #     print('root total_query_count:', self.root_total_query_cnt)
                            #     print('root root_hit_freq:', self.root_hit_freq)

                            # NOTE: deepcopy is needed because forward modifies matched_value in place
                            self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # Reset using zero values
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                            # If using RoPE positional encoding, then at reset, the pos_embed should use the absolute position start_pos[i].
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)},
                                                      past_keys_values=self.keys_values_wm_single_env,
                                                      is_init_infer=True, start_pos=start_pos[i].item())
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # Input self.keys_values_wm_list, output self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    start_pos = start_pos[:ready_env_num]
                    # TODO: len(last_obs_embeddings) may smaller than len(current_obs_embeddings), because some environments may have done
                    # TODO: the order may be not correct?  len(batch_action) may smaller than len(current_obs_embeddings), because some environments may have done
                    batch_action = batch_action[:ready_env_num]
                    
                    # TODO: only for debug
                    # if ready_env_num < self.env_num:
                    #     print(f'init inference ready_env_num: {ready_env_num} < env_num: {self.env_num}')
                    #     print(f"ready_env_num: {ready_env_num}")
                    #     print(f"start_pos: {start_pos}")
                    #     print(f"batch_action: {batch_action}")
                    #     print(f"len(last_obs_embeddings): {len(last_obs_embeddings)}")
                    #     print(f"len(batch_action): {len(batch_action)}")
                    #     print(f"len(current_obs_embeddings): {len(current_obs_embeddings)}")

                    if self.continuous_action_space:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(1)
                    else:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)
                    
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm,
                                              is_init_infer=True, start_pos=start_pos)
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, start_pos=start_pos)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, num_observations_tokens,
                                                          self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

            last_obs_embeddings = last_obs_embeddings[:, :-1, :]
            batch_action = torch.from_numpy(batch_action).to(last_obs_embeddings.device)
            if self.continuous_action_space:
                act_tokens = batch_action
            else:
                act_tokens = rearrange(batch_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            # Each sample in the batch (last_obs_embeddings, act_tokens) corresponds to the same time step, and start_pos also corresponds to each sample's respective t.
            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)}, start_pos=start_pos)

            # select the last timestep for each sample
            last_steps_value = outputs_wm.logits_value[:, -1:, :]
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            # outputs_wm.logits_value.shape (B, H, 101) = (B*H, 101)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')

        return outputs_wm

    @torch.no_grad()
    def forward_initial_inference(self, obs_act_dict, start_pos: int = 0):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict, start_pos)
        # TODO(pu): 由于预测误差的存在，不clear，也很可能不能检索到上次mcts 树搜索中的节点
        # 所有collect env公用应该也是合理的，不同环境很难遇到完全一致的预测的latent state？
        # self.past_kv_cache_recurrent_infer.clear()

        # ==================== 正确的修复位置 ====================
        # 在每次新的MCTS搜索（即调用initial_inference）开始时，
        # 清除上一次搜索遗留的 recurrent (MCTS) 缓存。
        self.past_kv_cache_recurrent_infer.clear()
        if hasattr(self, 'pool_idx_to_key_map_recur_infer'):
            # 同时也要清理辅助映射表
            self.pool_idx_to_key_map_recur_infer = [None] * self.shared_pool_size_recur
        # =========================================================

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    search_depth=[], start_pos: int = 0):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of state and action history.
            - simulation_index (:obj:`int`, optional): Index of the current simulation. Defaults to 0.
            - search_depth (:obj:`list`, optional): List containing depth of latent states in the search tree. 
        Returns:
            - tuple: A tuple containing output sequence, updated latent state, reward, logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index, start_pos)

        latent_state_list = []
        if not self.continuous_action_space:
            token = action.reshape(-1, 1)
        else:
            token = action.reshape(-1, self.action_space_size)

        # ======= Print statistics for debugging =============
        min_size = min(self.keys_values_wm_size_list)
        # # if min_size >= self.config.max_tokens - 5:
        # #     self.length_largethan_maxminus5_context_cnt += len(self.keys_values_wm_size_list)
        # # if min_size >= self.config.max_tokens - 7:
        # #     self.length_largethan_maxminus7_context_cnt += len(self.keys_values_wm_size_list)
        # if min_size >= self.config.context_length - 3:
        #     self.length_largethan_contextminus3_cnt += len(self.keys_values_wm_size_list)
        # # if self.recur_total_query_count > 0 and self.recur_total_query_count % 10000 == 0:
        # if self.recur_total_query_count > 0 and self.recur_total_query_count % 1000 == 0:
        #     self.hit_freq = self.recur_hit_count / self.recur_total_query_count
        #     print('recur total_query_count:', self.recur_total_query_count)
        #     # length_largethan_maxminus5_context_cnt_ratio = self.length_largethan_maxminus5_context_cnt / self.recur_total_query_count
        #     # print('recurrent largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
        #     # print('recurrent largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
        #     # length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.recur_total_query_count
        #     # print('recurrent largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
        #     # print('recurrent largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)
        #     length_largethan_contextminus3_cnt_ratio = self.length_largethan_contextminus3_cnt / self.recur_total_query_count
        #     print('recurrent length_largethan_contextminus3_cnt_ratio:', length_largethan_contextminus3_cnt_ratio)
        #     print('recurrent length_largethan_contextminus3_cnt:', self.length_largethan_contextminus3_cnt)

        # Trim and pad kv_cache: modify self.keys_values_wm in-place
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False)
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list

        for k in range(2):
            # action_token obs_token
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            # Perform forward pass
            outputs_wm = self.forward(
                obs_embeddings_or_act_tokens,
                past_keys_values=self.keys_values_wm,
                kvcache_independent=False,
                is_init_infer=False,
                start_pos=start_pos,
                search_depth=search_depth # List containing depth of latent states in the search tree. 
            )

            self.keys_values_wm_size_list_current = [i + 1 for i in self.keys_values_wm_size_list_current]

            if k == 0:
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        del self.latent_state  # Very important to minimize cuda memory usage
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(
            self.latent_state,
            is_init_infer=False,
            simulation_index=simulation_index,
        )

        return (outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value)

    # TODO: precompute_pos_emb_diff_kv 与 update_cache_context 的硬编码不匹配,collect_env_num=1应该没有问题
    def trim_and_pad_kv_cache(self, is_init_infer=True) -> list:
        """
        Adjusts the key-value cache for each environment to ensure they all have the same size.

        In a multi-environment setting, the key-value cache (kv_cache) for each environment is stored separately.
        During recurrent inference, the kv_cache sizes may vary across environments. This method pads each kv_cache
        to match the largest size found among them, facilitating batch processing in the transformer forward pass.

        Arguments:
            - is_init_infer (:obj:`bool`): Indicates if this is an initial inference. Default is True.
        Returns:
            - list: Updated sizes of the key-value caches.
        """
        # Find the maximum size among all key-value caches
        max_size = max(self.keys_values_wm_size_list)

        # Iterate over each layer of the transformer
        for layer in range(self.num_layers):
            kv_cache_k_list = []
            kv_cache_v_list = []

            # Enumerate through each environment's key-value pairs
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                effective_size = self.keys_values_wm_size_list[idx]
                pad_size = max_size - effective_size

                # If padding is required, trim the end and pad the beginning of the cache
                if pad_size > 0:
                    k_cache_trimmed = k_cache[:, :, :-pad_size, :]
                    v_cache_trimmed = v_cache[:, :, :-pad_size, :]
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # Stack the caches along a new dimension and remove any extra dimensions
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # Update the cache size to the maximum size
            self.keys_values_wm._keys_values[layer]._k_cache._size = max_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = max_size

        return self.keys_values_wm_size_list

    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             search_depth=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state tensor.
            - is_init_infer (:obj:`bool`): Flag to indicate if this is the initial inference.
            - simulation_index (:obj:`int`): Index of the simulation.
            - search_depth (:obj:`list`): List of depth indices in the search tree.
            - valid_context_lengths (:obj:`list`): List of valid context lengths.
        """
        if self.context_length <= 2:
            # No context to update if the context length is less than or equal to 2.
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            cache_key = hash_state(latent_state[i].view(-1).cpu().numpy())  # latent_state[i] is torch.Tensor
            context_length = self.context_length

            if not is_init_infer:


                # ============ Internal Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding
                current_max_context_length = max(self.keys_values_wm_size_list_current)
                trim_size = current_max_context_length - self.keys_values_wm_size_list_current[i]
                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============
                    # cache shape [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                    if trim_size > 0:
                        # Trim invalid leading zeros as per effective length
                        # Remove the first trim_size zero kv items
                        k_cache_trimmed = k_cache_current[:, trim_size:, :]
                        v_cache_trimmed = v_cache_current[:, trim_size:, :]
                        # If effective length < current_max_context_length, pad the end of cache with 'trim_size' zeros
                        k_cache_padded = F.pad(k_cache_trimmed, (0, 0, 0, trim_size), "constant",
                                               0)  # Pad with 'trim_size' zeros at end of cache
                        v_cache_padded = F.pad(v_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)
                    else:
                        k_cache_padded = k_cache_current
                        v_cache_padded = v_cache_current

                    # Update cache of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    # Update size of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm_size_list_current[i]
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm_size_list_current[i]

                    # ============ NOTE: Very Important ============
                    if self.keys_values_wm_single_env._keys_values[layer]._k_cache._size >= context_length - 1:
                        # Keep only the last self.context_length-3 timesteps of context
                        # For memory environments, training is for H steps, recurrent_inference might exceed H steps
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length - 1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length - 1, :].squeeze(0)

                        if not self.config.rotary_emb:
                            # Index pre-computed positional encoding differences
                            pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                            pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                            # ============ NOTE: Very Important ============
                            # Apply positional encoding correction to k and v
                            k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                            v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update single environment cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)

                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            else:
                # ============ Root Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding

                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============

                    if self.keys_values_wm._keys_values[layer]._k_cache._size < context_length - 1:  # Keep only the last self.context_length-1 timesteps of context
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(
                            0)  # Shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm._keys_values[layer]._k_cache._size
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm._keys_values[layer]._v_cache._size
                    else:
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                        v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, 2:context_length - 1, :]
                        v_cache_trimmed = v_cache_current[:, 2:context_length - 1, :]

                        if not self.config.rotary_emb:
                            # Index pre-computed positional encoding differences
                            pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                            pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                            # ============ NOTE: Very Important ============
                            # Apply positional encoding correction to k and v
                            k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                            v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update cache of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        # Update size of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            if is_init_infer:
                # TODO
                # ==================== 主动淘汰修复逻辑 ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index_init_envs[i]
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_init_envs[i][index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    # 确保要删除的键确实存在，避免意外错误
                    if old_key_to_evict in self.past_kv_cache_init_infer_envs[i]:
                        del self.past_kv_cache_init_infer_envs[i][old_key_to_evict]

                # 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
                
                # 4. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
                self.pool_idx_to_key_map_init_envs[i][index_to_write] = cache_key

                # 调用调试函数进行检查
                self._debug_check_for_stale_pointers(env_id=i, current_key=cache_key, index_to_be_written=index_to_write)
                # ============================================================

                # Store the latest key-value cache for initial inference
                # cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
                # self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
            else:
                # TODO 获取要存入的cache的某个唯一标识，例如tensor的和
                # cache_to_store = self.keys_values_wm_single_env._keys_values[0]._k_cache._cache
                # cache_sum = torch.sum(cache_to_store).item()
                # cache_shape = cache_to_store.shape
                # print(f"[CACHE WRITE] Storing for key={cache_key}, cache_shape={cache_shape}, cache_sum={cache_sum:.4f}")
                
                # ==================== RECURRENT INFER FIX ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_recur_infer[index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    if old_key_to_evict in self.past_kv_cache_recurrent_infer:
                        del self.past_kv_cache_recurrent_infer[old_key_to_evict]

                # 4. 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)

                # 5. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_recurrent_infer[cache_key] = cache_index
                self.pool_idx_to_key_map_recur_infer[index_to_write] = cache_key
                # ============================================================

                # ==================== DEBUG CODE INSERTION ====================
                # 调用调试函数进行检查
                self._debug_check_for_stale_pointers_recur(current_key=cache_key, index_to_be_written=index_to_write)
                # ============================================================

                # Store the latest key-value cache for recurrent inference
                # cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)
                # self.past_kv_cache_recurrent_infer[cache_key] = cache_index


    def retrieve_or_generate_kvcache(self, latent_state: list, ready_env_num: int,
                                     simulation_index: int = 0, start_pos: int = 0) -> list:
        """
        Retrieves or generates key-value caches for each environment based on the latent state.

        For each environment, this method either retrieves a matching cache from the predefined
        caches if available, or generates a new cache if no match is found. The method updates
        the internal lists with these caches and their sizes.

        Arguments:
            - latent_state (:obj:`list`): List of latent states for each environment.
            - ready_env_num (:obj:`int`): Number of environments ready for processing.
            - simulation_index (:obj:`int`, optional): Index for simulation tracking. Default is 0.
        Returns:
            - list: Sizes of the key-value caches for each environment.
        """
        for index in range(ready_env_num):
            self.recur_total_query_count += 1
            state_single_env = latent_state[index]  # latent_state[i] is np.array
            cache_key = hash_state(state_single_env)

            if self.reanalyze_phase:
                # TODO: check if this is correct
                matched_value = None
            else:
                # Try to retrieve the cached value from past_kv_cache_init_infer_envs
                cache_index = self.past_kv_cache_init_infer_envs[index].get(cache_key)
                if cache_index is not None:
                    matched_value = self.shared_pool_init_infer[index][cache_index]

                    # TODO
                    # retrieved_cache = matched_value._keys_values[0]._k_cache._cache
                    # retrieved_sum = torch.sum(retrieved_cache).item()
                    # retrieved_shape = retrieved_cache.shape
                    # print(f"[CACHE HIT]   Found for key={cache_key}, retrieved_shape={retrieved_shape}, retrieved_sum={retrieved_sum:.4f}")


                else:
                    matched_value = None

                # If not found, try to retrieve from past_kv_cache_recurrent_infer
                # if matched_value is None:
                #     matched_value = self.shared_pool_recur_infer[self.past_kv_cache_recurrent_infer.get(cache_key)]

                # ==================== 核心修复 ====================
                # 步骤 2: 仅当在 init_infer 中未找到时，才尝试从 recurrent_infer 缓存中查找
                if matched_value is None:
                    # 2.1 安全地从字典中获取索引，它可能返回 None
                    recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
                    # 2.2 只有在索引有效（不是 None）的情况下，才使用它来从物理池中检索值
                    if recur_cache_index is not None:
                        matched_value = self.shared_pool_recur_infer[recur_cache_index]
                    
                    if recur_cache_index is None:
                        print(f"[CACHE MISS]  Not found for key={cache_key} in recurrent infer. Generating new cache.")

                # =================================================
                    # # TODO
                    # retrieved_cache = matched_value._keys_values[0]._k_cache._cache
                    # retrieved_sum = torch.sum(retrieved_cache).item()
                    # retrieved_shape = retrieved_cache.shape
                    # print(f"[CACHE HIT]   Found for key={cache_key}, retrieved_shape={retrieved_shape}, retrieved_sum={retrieved_sum:.4f}")


            if matched_value is not None:
                # If a matching cache is found, add it to the lists
                self.recur_hit_count += 1
                # Perform a deep copy because the transformer's forward pass might modify matched_value in-place
                self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # print(f"[CACHE MISS]  Not found for key={cache_key}. Generating new cache.")

                # If no matching cache is found, generate a new one using zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(
                    n=1, max_tokens=self.context_length
                )
                
                # Determine the absolute start position based on the reanalyze phase flag.
                if self.reanalyze_phase:
                    num_rows, num_cols = start_pos.shape  # Original start_pos shape is (batch, num_columns)
                    total_cols = num_cols + 1             # Each logical row is extended by one column.
                    row_idx = index // total_cols
                    col_idx = index % total_cols
                    # If the column index equals the original number of columns, this indicates the added column; set to 0.
                    start_pos_adjusted: int = 0 if col_idx == num_cols else int(start_pos[row_idx, col_idx])
                else:
                    start_pos_adjusted = int(start_pos[index].item())

                self.forward(
                    {'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)},
                    past_keys_values=self.keys_values_wm_single_env, is_init_infer=True, start_pos=start_pos_adjusted
                )
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

        return self.keys_values_wm_size_list


    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        start_pos = batch['timestep']
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])


        # # ======================= 在这里插入分析代码 =======================
        # # 从kwargs获取全局step，假设您在训练循环中传入了它
        global_step = kwargs.get('global_step', 0)

        # # 为了避免影响训练，可以控制调用频率
        # if global_step % 10 == 0: # 每100个training step分析一次
        #     self._analyze_latent_representation(
        #         latent_states=obs_embeddings,
        #         timesteps=batch['timestep'],
        #         game_states=batch['observations'], # 传入原始图像
        #         step_counter=global_step
        #     )
        # # =================================================================

        # ========= for visual analysis =========
        # Uncomment the lines below for visual analysis in Pong
        # self.plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong_H10_H4_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='pong_H10_H4_tsne')
        # Uncomment the lines below for visual analysis in visual match
        # self.plot_latent_tsne_each_and_all(obs_embeddings, suffix='visual_match_memlen1-60-15_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='visual_match_memlen1-60-15_tsne')


        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            dormant_ratio_encoder = cal_dormant_ratio(self.tokenizer.representation_network, inputs.detach(),
                                                      percentage=self.dormant_threshold)
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)

        # Action tokens
        if self.continuous_action_space:
            act_tokens = batch['actions']
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        with torch.no_grad():
            # Calculate the L2 norm of the latent state roots
            latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()
            # Calculate the L2 norm of the latent action
            latent_action_l2_norms = torch.norm(self.act_embedding_table(act_tokens), p=2, dim=2).mean()

        if self.config.latent_norm_loss:
            # ==================== L2惩罚损失计算（最终修复版 v2） ====================
            # 1. 计算每个 latent_state 向量的L2范数的平方。
            #    根据调试信息，obs_embeddings shape: (B*L, 1, E)
            #    所以 latent_norm_sq shape: (B*L, 1)
            latent_norm_sq = torch.norm(obs_embeddings, p=2, dim=-1).pow(2)
            # 2. 获取源掩码。
            #    根据调试信息，mask_source shape: (B, L)
            mask_source = batch['mask_padding']
            # 3. 将源掩码从 (B, L) reshape 为 (B*L, 1)，以匹配 latent_norm_sq 的形状。
            #    这是解决维度不匹配错误的关键。
            #    我们使用 view(-1, 1) 来实现这个变形。
            correct_mask = mask_source.contiguous().view(-1, 1)
            # 4. 检查变形后的形状是否匹配。
            #    这是一个防御性编程，确保两个张量的第一个维度是相同的。
            if latent_norm_sq.shape[0] != correct_mask.shape[0]:
                # 如果形状不匹配，打印错误信息并抛出异常，这能帮助我们更快地定位未来可能出现的新问题。
                raise RuntimeError(
                    f"Shape mismatch for L2 norm loss calculation! "
                    f"latent_norm_sq shape: {latent_norm_sq.shape}, "
                    f"but correct_mask shape after reshape is: {correct_mask.shape}. "
                    f"Original mask_source shape was: {mask_source.shape}"
                )
            # 5. 直接进行逐元素乘法。因为现在它们的形状都是 (B*L, 1)，所以可以安全相乘。
            masked_latent_norm_sq = latent_norm_sq * correct_mask
            # 6. 计算平均损失。分母是掩码中所有“1”的总和，代表有效的元素数量。
            #    增加一个极小值 epsilon (1e-8) 防止分母为零。
            latent_norm_loss = masked_latent_norm_sq.sum() / (correct_mask.sum() + 1e-8)
            # =================================================================
        else:
            latent_norm_loss = torch.tensor(0.)


        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, start_pos=start_pos)

        # TODO============
        # ======================= 在这里插入分析代码 =======================
        # if global_step > 0 and global_step % 1000 == 0:
        # if global_step > 0 and global_step % 5000 == 0:
        # if global_step >= 0 and global_step % 5000 == 0: # 5k
        if global_step >= 0 and global_step % 10000 == 0: # 10k
        
            with torch.no_grad():
                # 将logits转换为标量值
                # 注意：outputs的形状是(B, L, E)，我们需要reshape
                batch_size, seq_len = batch['actions'].shape[0], batch['actions'].shape[1]
                
                pred_val_logits = outputs.logits_value.view(batch_size * seq_len, -1)
                pred_rew_logits = outputs.logits_rewards.view(batch_size * seq_len, -1)
                
                scalar_values = inverse_scalar_transform_handle(pred_val_logits).squeeze(-1)
                scalar_rewards = inverse_scalar_transform_handle(pred_rew_logits).squeeze(-1)

                self._analyze_latent_representation(
                    latent_states=obs_embeddings,
                    timesteps=batch['timestep'],
                    game_states=batch['observations'],
                    predicted_values=scalar_values, # 传入预测的Value
                    predicted_rewards=scalar_rewards, # 传入预测的Reward
                    step_counter=global_step
                )
        # =================================================================

        if self.config.use_priority:
            # ==================== START MODIFICATION 5 ====================
            # Calculate value_priority, similar to MuZero.
            with torch.no_grad():
                # 1. Get the predicted value logits for the first step of the sequence (t=0).
                # The shape is (B, support_size).
                predicted_value_logits_step0 = outputs.logits_value[:, 0, :]

                # 2. Convert the categorical prediction to a scalar value.
                # The shape becomes (B, 1).
                predicted_scalar_value_step0 = inverse_scalar_transform_handle(predicted_value_logits_step0)

                # 3. Get the target scalar value for the first step from the batch.
                # The shape is (B, num_unroll_steps), so we take the first column.
                target_scalar_value_step0 = batch['scalar_target_value'][:, 0]

                # 4. Calculate the L1 loss (absolute difference) between prediction and target.
                # This is the priority. We use reduction='none' to get per-sample priorities.
                value_priority = F.l1_loss(predicted_scalar_value_step0.squeeze(-1), target_scalar_value_step0, reduction='none')
            # ===================== END MODIFICATION 5 =====================
        else:
            value_priority = torch.tensor(0.)

        if self.obs_type == 'image':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # original_images, reconstructed_images = batch['observations'], reconstructed_images
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # ========== Calculate reconstruction loss and perceptual loss ============
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            # perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))

            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = self.latent_recon_loss

        elif self.obs_type == 'text':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=torch.float32)
            decode_loss_mode = self.config.decode_loss_mode 

            # Reconstruction loss for predicting the next latent (via backbone)
            # input -> encoder -> backbone(unizero) -> decoder -> latent_recon_loss
            if decode_loss_mode == "after_backbone":
                next_latent_state = outputs.logits_observations[:, :-1, :]
                next_target_ids = batch['observations'][:, 1:, :] 

                latent_recon_loss = self.tokenizer.decode_to_reconstruction_outputs(
                    embeddings=next_latent_state,
                    target_ids=next_target_ids,
                ).loss

            #Reconstruction loss for predicting the current latent (without using the backbone)
            # input -> encoder -> decoder -> latent_recon_loss
            elif decode_loss_mode == "before_backbone":
                latent_recon_loss = self.tokenizer.decode_to_reconstruction_outputs(
                    embeddings=obs_embeddings,
                    target_ids=batch['observations'],
                ).loss

            else:
                latent_recon_loss = self.latent_recon_loss

        elif self.obs_type == 'image_memory':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            # original_images, reconstructed_images = batch['observations'], reconstructed_images

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
            #                                                        reconstructed_images)
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the world model
            dormant_ratio_world_model = cal_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          percentage=self.dormant_threshold)
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_world_model = torch.tensor(0.)

        #  ========== for visualization ==========
        # Uncomment the lines below for visualization
        # predict_policy = outputs.logits_policy
        # predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        # predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # import pdb; pdb.set_trace()
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=[], suffix='pong_H10_H4_0613')

        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_success_episode')
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_fail_episode')
        #  ========== for visualization ==========

        with torch.no_grad():
            # For training stability, use target_tokenizer to compute the true next latent state representations
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])
            # target_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, _ = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        # Compute prediction loss for observations. Options: MSE and Group KL
        if self.predict_latent_loss_type == 'mse':
            # MSE loss, directly compare logits and labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(
                -1)
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL loss, group features and calculate KL divergence within each group
            batch_size, num_features = logits_observations.shape
            epsilon = 1e-6
            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)

            #  ========== for debugging ==========
            # print('loss_obs:', loss_obs.mean())
            # assert not torch.isnan(loss_obs).any(), "loss_obs contains NaN values"
            # assert not torch.isinf(loss_obs).any(), "loss_obs contains Inf values"
            # for name, param in self.tokenizer.encoder.named_parameters():
            #     print('name, param.mean(), param.std():', name, param.mean(), param.std())

        # Apply mask to loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # Compute labels for policy and value
        labels_value, labels_policy = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        # Compute losses for rewards, policy, and value
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')

        if not self.continuous_action_space:
            loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy,
                                                                                            batch,
                                                                                            element='policy')
        else:
            # NOTE: for continuous action space
            if self.config.policy_loss_type == 'simple':
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont_simple(outputs, batch)
            else:
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(outputs, batch)
            
            loss_policy = orig_policy_loss + self.policy_entropy_weight * policy_entropy_loss
            policy_entropy = - policy_entropy_loss

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # ==== TODO: calculate the new priorities for each transition. ====
        # value_priority = L1Loss(reduction='none')(labels_value.squeeze(-1), outputs['logits_value'][:, 0])
        # value_priority = value_priority.data.cpu().numpy() + 1e-6

        # Compute timesteps
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # Compute discount coefficients for each timestep
        discounts = self.gamma ** timesteps

        if batch['mask_padding'].sum() == 0:
            assert False, "mask_padding is all zeros"

        # Group losses into first step, middle step, and last step
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        # batch['mask_padding'] indicates mask status for future H steps, exclude masked losses to maintain accurate mean statistics
        # Group losses for each loss item
        for loss_name, loss_tmp in zip(
                ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
                [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]

            # Adjust loss shape to (batch_size, seq_len)
            loss_tmp = loss_tmp.view(-1, seq_len)

            # First step loss
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()

            # Middle step loss
            middle_timestep = seq_len // 2
            middle_step_mask = mask_padding[:, middle_timestep]
            middle_step_losses[loss_name] = loss_tmp[:, middle_timestep][middle_step_mask].mean()

            # Last step loss
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # Discount reconstruction loss and perceptual loss
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

        # Calculate overall discounted loss
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).sum()/ batch['mask_padding'][:,1:].sum()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()

        # 为了让外部的训练循环能够获取encoder的输出，我们将其加入返回字典
        # 使用 .detach() 是因为这个张量仅用于后续的clip操作，不应影响梯度计算
        detached_obs_embeddings = obs_embeddings.detach()

        if self.continuous_action_space:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                continuous_action_space=True,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
                latent_action_l2_norms=latent_action_l2_norms,
                policy_mu=mu,
                policy_sigma=sigma,
                target_sampled_actions=target_sampled_actions,
                latent_norm_loss=latent_norm_loss, # 新增
                value_priority=value_priority,
                obs_embeddings=detached_obs_embeddings,  # <-- 新增
            )
        else:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                continuous_action_space=False,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
                latent_action_l2_norms=latent_action_l2_norms,
                latent_norm_loss=latent_norm_loss, # 新增
                value_priority=value_priority,
                obs_embeddings=detached_obs_embeddings,  # <-- 新增

            )

    
    # TODO: test correctness
    def _calculate_policy_loss_cont_simple(self, outputs, batch: dict):
        """
        Simplified policy loss calculation for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.

        Returns:
            - policy_loss (:obj:`torch.Tensor`): The simplified policy loss.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        # Get the policy logits and batch data
        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding'].contiguous().view(-1)
        target_policy = batch['target_policy'].contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = batch['child_sampled_actions'].contiguous().view(batch_size * num_unroll_steps, -1, action_space_size)

        # Flatten for vectorized computation
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        
        # Extract mean and standard deviation from logits
        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        dist = Independent(Normal(mu, sigma), 1)  # Create the normal distribution

        # Find the indices of the maximum values in the target policy
        target_best_action_idx = torch.argmax(target_policy, dim=1)

        # Select the best actions based on the indices
        target_best_action = target_sampled_actions[torch.arange(target_best_action_idx.size(0)), target_best_action_idx]

        # Clip the target actions to prevent numerical issues during arctanh
        # target_best_action_clamped = torch.clamp(target_best_action, -1 + 1e-6, 1 - 1e-6)
        target_best_action_clamped = torch.clamp(target_best_action, -0.999, 0.999)
        target_best_action_before_tanh = torch.arctanh(target_best_action_clamped)

        # Calculate the log probability of the best action
        log_prob_best_action = dist.log_prob(target_best_action_before_tanh)

        # Mask the log probability with the padding mask
        log_prob_best_action = log_prob_best_action * mask_batch

        # Return the negative log probability as the policy loss (we want to maximize log_prob)
        # policy_loss = -log_prob_best_action.mean()
        policy_loss = -log_prob_best_action

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -policy_entropy * mask_batch
        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def _calculate_policy_loss_cont(self, outputs, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the policy loss for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.
        Returns:
            - policy_loss (:obj:`torch.Tensor`): The calculated policy loss.
            - policy_entropy_loss (:obj:`torch.Tensor`): The entropy loss of the policy.
            - target_policy_entropy (:obj:`float`): The entropy of the target policy distribution.
            - target_sampled_actions (:obj:`torch.Tensor`): The actions sampled from the target policy.
            - mu (:obj:`torch.Tensor`): The mean of the normal distribution.
            - sigma (:obj:`torch.Tensor`): The standard deviation of the normal distribution.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding']
        child_sampled_actions_batch = batch['child_sampled_actions']
        target_policy = batch['target_policy']

        # Flatten the unroll step dimension for easier vectorized operations
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        mask_batch = mask_batch.contiguous().view(-1)
        child_sampled_actions_batch = child_sampled_actions_batch.contiguous().view(batch_size * num_unroll_steps, -1,
                                                                                    action_space_size)

        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        mu = mu.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        dist = Independent(Normal(mu, sigma), 1)

        target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = child_sampled_actions_batch

        policy_entropy = dist.entropy().mean(dim=1)
        policy_entropy_loss = -policy_entropy * mask_batch

        # NOTE： Alternative way to calculate the log probability of the target actions
        # y = 1 - target_sampled_actions.pow(2)
        # target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -1 + 1e-6, 1 - 1e-6)
        # target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)
        # log_prob = dist.log_prob(target_sampled_actions_before_tanh)
        # log_prob = log_prob - torch.log(y + 1e-6).sum(-1)
        # log_prob_sampled_actions = log_prob

        base_dist = Normal(mu, sigma)
        tanh_transform = TanhTransform()
        dist = TransformedDistribution(base_dist, [tanh_transform])
        dist = Independent(dist, 1)
        target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -0.999, 0.999)
        # assert torch.all(target_sampled_actions_clamped < 1) and torch.all(target_sampled_actions_clamped > -1), "Actions are not properly clamped."
        log_prob = dist.log_prob(target_sampled_actions_clamped)
        log_prob_sampled_actions = log_prob

        # KL as projector
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        policy_loss = -torch.sum(
            torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
        ) * mask_batch

        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs is an object with logits attributes like 'rewards', 'policy', and 'value'.
        # labels is a target tensor for comparison. batch is a dictionary with a mask indicating valid timesteps.

        logits = getattr(outputs, f'logits_{element}')

        if torch.isnan(logits).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")
        
        if torch.isnan(labels).any():
            raise ValueError(f"NaN detected in labels_value for batch {batch} and element '{element}'")

        # Reshape your tensors
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assume labels initially have shape [batch, time, dim]

        # Reshape your mask. True indicates valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # Compute cross-entropy loss
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if torch.isnan(loss).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if element == 'policy':
            # Compute policy entropy loss
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # Combine losses with specified weight
            # print(f"self.policy_entropy_weight:{self.policy_entropy_weight}")
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    def compute_policy_entropy_loss(self, logits, mask):
        # Compute entropy of the policy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # Apply mask and return average entropy loss
        entropy_loss = (entropy * mask)
        return entropy_loss

    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # Each sequence sample should have at most one 'done' flag
        mask_fill = torch.logical_not(mask_padding)

        # Prepare observation labels
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]

        # Fill the masked areas of rewards
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        # Fill the masked areas of ends
        # labels_endgs = ends.masked_fill(mask_fill, -100)

        # return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)
        return labels_observations, labels_rewards.view(-1, self.support_size), None


    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute labels for value and policy predictions. """
        mask_fill = torch.logical_not(mask_padding)

        # Fill the masked areas of policy
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        # Fill the masked areas of value
        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)

        if self.continuous_action_space:
            return labels_value.reshape(-1, self.support_size), None
        else:
            return labels_value.reshape(-1, self.support_size), labels_policy.reshape(-1, self.action_space_size)

    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        self.past_kv_cache_recurrent_infer.clear()
        self.keys_values_wm_list.clear()
        print(f'Cleared {self.__class__.__name__} past_kv_cache.')

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"
