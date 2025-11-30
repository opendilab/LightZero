from typing import Optional, Union

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from easydict import EasyDict

from .common import MZNetworkOutput, RepresentationNetworkUniZero, RepresentationNetworkMLP, LatentDecoder, \
    VectorDecoderForMemoryEnv, LatentEncoderForMemoryEnv, LatentDecoderForMemoryEnv, FeatureAndGradientHook, \
    HFLanguageRepresentationNetwork
from .unizero_world_models.tokenizer import Tokenizer
from .unizero_world_models.world_model_multitask import WorldModelMT

from line_profiler import line_profiler
from .vit import ViT
# from .vit_efficient import VisionTransformer as ViT


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('UniZeroMTModel')
class UniZeroMTModel(nn.Module):
    #@profile
    def __init__(
            self,
            observation_shape: SequenceType = (4, 64, 64),
            action_space_size: Union[int, list] = 0,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            downsample: bool = True,
            norm_type: Optional[str] = 'BN',
            world_model_cfg: EasyDict = None,
            task_num: int = 1,
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of data procession in the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667), including two main parts:
                - initial_inference, which is used to predict the value, policy, and latent state based on the current observation.
                - recurrent_inference, which is used to predict the value, policy, reward, and next latent state based on the current latent state and action.
            The world model consists of three main components:
                - a tokenizer, which encodes observations into embeddings,
                - a transformer, which processes the input sequences,
                - and heads, which generate the logits for observations, rewards, policy, and value.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[3, 64, 64] for Atari.
            - action_space_size: (:obj:`[int, list]`): Action space size. For discrete or fixed action spaces, this is usually an integer. For multi-task environments where the action spaces are different, this is a list.
            - num_res_blocks (:obj:`int`): The number of res blocks in UniZero model.
            - num_channels (:obj:`int`): The channels of hidden states in representation network.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - world_model_cfg (:obj:`EasyDict`): The configuration of the world model, including the following keys:
                - obs_type (:obj:`str`): The type of observation, which can be 'image', 'vector', or 'image_memory'.
                - embed_dim (:obj:`int`): The dimension of the embedding.
                - group_size (:obj:`int`): The group size of the transformer.
                - max_blocks (:obj:`int`): The maximum number of blocks in the transformer.
                - max_tokens (:obj:`int`): The maximum number of tokens in the transformer.
                - context_length (:obj:`int`): The context length of the transformer.
                - device (:obj:`str`): The device of the model, which can be 'cuda' or 'cpu'.
                - action_space_size (:obj:`int`): The shape of the action.
                - num_layers (:obj:`int`): The number of layers in the transformer.
                - num_heads (:obj:`int`): The number of heads in the transformer.
                - policy_entropy_weight (:obj:`float`): The weight of the policy entropy.
                - analysis_sim_norm (:obj:`bool`): Whether to analyze the similarity of the norm.
        """
        super(UniZeroMTModel, self).__init__()

        print(f'==========UniZeroMTModel, num_res_blocks:{num_res_blocks}, num_channels:{num_channels}===========')

        self.action_space_size = action_space_size

        # for multi-task
        self.task_num = task_num

        self.activation = activation
        self.downsample = downsample
        world_model_cfg.norm_type = norm_type
        assert world_model_cfg.max_tokens == 2 * world_model_cfg.max_blocks, 'max_tokens should be 2 * max_blocks, because each timestep has 2 tokens: obs and action'

        if world_model_cfg.task_embed_option == "concat_task_embed":
            obs_act_embed_dim = world_model_cfg.embed_dim - world_model_cfg.task_embed_dim if hasattr(world_model_cfg, "task_embed_dim") else 96
        else:
            obs_act_embed_dim = world_model_cfg.embed_dim

        if world_model_cfg.obs_type == 'vector':
            self.representation_network = RepresentationNetworkMLP(
                observation_shape,
                hidden_channels=obs_act_embed_dim,
                layer_num=2,
                activation=self.activation,
                group_size=world_model_cfg.group_size,
            )
            # TODO: only for MemoryEnv now
            self.decoder_network = VectorDecoderForMemoryEnv(embedding_dim=world_model_cfg.embed_dim, output_shape=25)
            self.tokenizer = Tokenizer(encoder=self.representation_network,
                                       decoder_network=self.decoder_network, with_lpips=False, obs_type=world_model_cfg.obs_type)
            self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image':
            self.representation_network = nn.ModuleList()
            if world_model_cfg.encoder_type == "resnet":
                # for task_id in range(self.task_num):  # TODO: N independent encoder
                for task_id in range(1):  # TODO: one share encoder
                    self.representation_network.append(RepresentationNetworkUniZero(
                        observation_shape,
                        num_res_blocks,
                        num_channels,
                        self.downsample,
                        activation=self.activation,
                        norm_type=norm_type,
                        embedding_dim=obs_act_embed_dim,
                        group_size=world_model_cfg.group_size,
                        final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
                    ))
            elif world_model_cfg.encoder_type == "vit":
                for task_id in range(1):  # TODO: one share encoder
                    if world_model_cfg.task_num <=8: 
                        # # vit base
                        # self.representation_network.append(ViT(
                        #     image_size =observation_shape[1],
                        #     patch_size = 8,
                        #     num_classes = obs_act_embed_dim,
                        #     dim = 768,
                        #     depth = 12,
                        #     heads = 12,
                        #     mlp_dim = 3072,
                        #     dropout = 0.1,
                        #     emb_dropout = 0.1,
                        #     final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
                        # ))
                        # vit small
                        self.representation_network.append(ViT(
                            image_size =observation_shape[1],
                            patch_size = 8,
                            num_classes = obs_act_embed_dim,
                            dim = 768,
                            depth = 6,
                            heads = 6,
                            mlp_dim = 2048,
                            dropout = 0.1,
                            emb_dropout = 0.1,
                            final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
                            # ==================== 新增/修改部分 开始 ====================
                            config=world_model_cfg # <--- 将包含LoRA参数的配置传递给ViT
                            # ==================== 新增/修改部分 结束 ====================
                        
                        ))
                    elif world_model_cfg.task_num > 8: 
                        # vit base
                        self.representation_network.append(ViT(
                            image_size =observation_shape[1],
                            patch_size = 8,
                            num_classes = obs_act_embed_dim,
                            dim = 768,
                            depth = 12,
                            heads = 12,
                            mlp_dim = 3072,
                            dropout = 0.1,
                            emb_dropout = 0.1,
                            final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
                            # ==================== 新增/修改部分 开始 ====================
                            config=world_model_cfg # <--- 将包含LoRA参数的配置传递给ViT
                            # ==================== 新增/修改部分 结束 ====================

                        ))
                        # # vit large # TODO======
                        # self.representation_network.append(ViT(
                        #     image_size =observation_shape[1],
                        #     # patch_size = 32,
                        #     patch_size = 8,
                        #     num_classes = obs_act_embed_dim,
                        #     dim = 1024,
                        #     depth = 24,
                        #     heads = 16,
                        #     mlp_dim = 4096,
                        #     dropout = 0.1,
                        #     emb_dropout = 0.1
                        # ))


            # TODO: we should change the output_shape to the real observation shape
            # self.decoder_network = LatentDecoder(embedding_dim=world_model_cfg.embed_dim, output_shape=(3, 64, 64))

            # ====== for analysis ======
            if world_model_cfg.analysis_sim_norm:
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.tokenizer = Tokenizer(encoder=self.representation_network, decoder_network=None, with_lpips=False, obs_type=world_model_cfg.obs_type)
            self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image_memory':
            # todo for concat_task_embed
            self.representation_network = LatentEncoderForMemoryEnv(
                image_shape=(3, 5, 5),
                embedding_size=world_model_cfg.embed_dim,
                channels=[16, 32, 64],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
                activation=self.activation,
                group_size=world_model_cfg.group_size,
            )
            self.decoder_network = LatentDecoderForMemoryEnv(
                image_shape=(3, 5, 5),
                embedding_size=world_model_cfg.embed_dim,
                channels=[64, 32, 16],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
                activation=self.activation,
            )

            if world_model_cfg.analysis_sim_norm:
                # ====== for analysis ======
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.tokenizer = Tokenizer(with_lpips=True, encoder=self.representation_network,
                                       decoder_network=self.decoder_network, obs_type=world_model_cfg.obs_type)
            self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print(f'{sum(p.numel() for p in self.world_model.parameters()) - sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.world_model - (decoder_network and lpips)')

            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'text':
            self.representation_network = nn.ModuleList()
            for task_id in range(1):  # TODO: one share encoder
                self.representation_network.append(
                    HFLanguageRepresentationNetwork(
                        model_path=kwargs['encoder_url'], 
                        embedding_size=world_model_cfg.embed_dim,
                        final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder
                    )
                )

            self.tokenizer = Tokenizer(encoder=self.representation_network, decoder_network=None, with_lpips=False, obs_type=world_model_cfg.obs_type)
            self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)
        self._log_model_parameters(world_model_cfg.obs_type)

    def _log_model_parameters(self, obs_type: str) -> None:
        """
        Overview:
            Logs detailed parameter counts for all model components with a comprehensive breakdown.
            Includes encoder, transformer, prediction heads, and other components.
        Arguments:
            - obs_type (:obj:`str`): The type of observation ('vector', 'image', or 'image_memory').
        """
        from ding.utils import get_rank

        # Only print from rank 0 to avoid duplicate logs in DDP
        if get_rank() != 0:
            return

        print('=' * 80)
        print('MODEL PARAMETER STATISTICS'.center(80))
        print('=' * 80)

        # --- Total Model Parameters ---
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\n{"TOTAL MODEL":<40} {total_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {total_trainable:>15,} parameters')
        print(f'{"  └─ Frozen":<40} {total_params - total_trainable:>15,} parameters')

        # --- World Model Components ---
        print(f'\n{"-" * 80}')
        print(f'{"WORLD MODEL BREAKDOWN":<40}')
        print(f'{"-" * 80}')

        wm_params = sum(p.numel() for p in self.world_model.parameters())
        wm_trainable = sum(p.numel() for p in self.world_model.parameters() if p.requires_grad)
        print(f'{"World Model Total":<40} {wm_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {wm_trainable:>15,} parameters ({100*wm_trainable/wm_params:.1f}%)')

        # --- Encoder ---
        encoder_params = sum(p.numel() for p in self.tokenizer.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.tokenizer.encoder.parameters() if p.requires_grad)
        print(f'\n{"1. ENCODER (Tokenizer)":<40} {encoder_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {encoder_trainable:>15,} parameters ({100*encoder_trainable/encoder_params:.1f}%)')

        # --- Transformer Backbone ---
        transformer_params = sum(p.numel() for p in self.world_model.transformer.parameters())
        transformer_trainable = sum(p.numel() for p in self.world_model.transformer.parameters() if p.requires_grad)
        print(f'\n{"2. TRANSFORMER BACKBONE":<40} {transformer_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {transformer_trainable:>15,} parameters ({100*transformer_trainable/transformer_params:.1f}%)')

        # --- Prediction Heads (Detailed Breakdown) ---
        print(f'\n{"3. PREDICTION HEADS":<40}')

        # Access head_dict from world_model
        if hasattr(self.world_model, 'head_dict'):
            head_dict = self.world_model.head_dict

            # Calculate total heads parameters
            total_heads_params = sum(p.numel() for module in head_dict.values() for p in module.parameters())
            total_heads_trainable = sum(p.numel() for module in head_dict.values() for p in module.parameters() if p.requires_grad)
            print(f'{"  Total (All Heads)":<40} {total_heads_params:>15,} parameters')
            print(f'{"  └─ Trainable":<40} {total_heads_trainable:>15,} parameters ({100*total_heads_trainable/total_heads_params:.1f}%)')

            # Breakdown by head type
            head_names_map = {
                'head_policy_multi_task': 'Policy Head',
                'head_value_multi_task': 'Value Head',
                'head_rewards_multi_task': 'Reward Head',
                'head_observations_multi_task': 'Next Latent (Obs) Head'
            }

            print(f'\n{"  Breakdown by Head Type:":<40}')
            for head_key, head_name in head_names_map.items():
                if head_key in head_dict:
                    head_module = head_dict[head_key]
                    head_params = sum(p.numel() for p in head_module.parameters())
                    head_trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)

                    # Count number of task-specific heads (for ModuleList)
                    if isinstance(head_module, nn.ModuleList):
                        num_heads = len(head_module)
                        params_per_head = head_params // num_heads if num_heads > 0 else 0
                        print(f'{"    ├─ " + head_name:<38} {head_params:>15,} parameters')
                        print(f'{"      └─ " + f"{num_heads} task-specific heads":<38} {params_per_head:>15,} params/head')
                    else:
                        print(f'{"    ├─ " + head_name:<38} {head_params:>15,} parameters')
                        print(f'{"      └─ Shared across tasks":<38}')

        # --- Positional & Task Embeddings ---
        print(f'\n{"4. EMBEDDINGS":<40}')

        if hasattr(self.world_model, 'pos_emb'):
            pos_emb_params = sum(p.numel() for p in self.world_model.pos_emb.parameters())
            pos_emb_trainable = sum(p.numel() for p in self.world_model.pos_emb.parameters() if p.requires_grad)
            print(f'{"  ├─ Positional Embedding":<40} {pos_emb_params:>15,} parameters')
            if pos_emb_trainable == 0:
                print(f'{"    └─ (Frozen)":<40}')

        if hasattr(self.world_model, 'task_emb') and self.world_model.task_emb is not None:
            task_emb_params = sum(p.numel() for p in self.world_model.task_emb.parameters())
            task_emb_trainable = sum(p.numel() for p in self.world_model.task_emb.parameters() if p.requires_grad)
            print(f'{"  ├─ Task Embedding":<40} {task_emb_params:>15,} parameters')
            print(f'{"    └─ Trainable":<40} {task_emb_trainable:>15,} parameters')

        if hasattr(self.world_model, 'act_embedding_table'):
            act_emb_params = sum(p.numel() for p in self.world_model.act_embedding_table.parameters())
            act_emb_trainable = sum(p.numel() for p in self.world_model.act_embedding_table.parameters() if p.requires_grad)
            print(f'{"  └─ Action Embedding":<40} {act_emb_params:>15,} parameters')
            print(f'{"    └─ Trainable":<40} {act_emb_trainable:>15,} parameters')

        # --- Decoder (if applicable) ---
        if obs_type in ['vector', 'image_memory'] and self.tokenizer.decoder_network is not None:
            print(f'\n{"5. DECODER":<40}')
            decoder_params = sum(p.numel() for p in self.tokenizer.decoder_network.parameters())
            decoder_trainable = sum(p.numel() for p in self.tokenizer.decoder_network.parameters() if p.requires_grad)
            print(f'{"  Decoder Network":<40} {decoder_params:>15,} parameters')
            print(f'{"  └─ Trainable":<40} {decoder_trainable:>15,} parameters')

            if obs_type == 'image_memory' and hasattr(self.tokenizer, 'lpips'):
                lpips_params = sum(p.numel() for p in self.tokenizer.lpips.parameters())
                print(f'{"  LPIPS Loss Network":<40} {lpips_params:>15,} parameters')

                # Calculate world model params excluding decoder and LPIPS
                params_without_decoder = wm_params - decoder_params - lpips_params
                print(f'\n{"  World Model (exc. Decoder & LPIPS)":<40} {params_without_decoder:>15,} parameters')

        # --- Summary Table ---
        print(f'\n{"=" * 80}')
        print(f'{"SUMMARY":<40}')
        print(f'{"=" * 80}')
        print(f'{"Component":<30} {"Total Params":>15} {"Trainable":>15} {"% of Total":>15}')
        print(f'{"-" * 80}')

        components = [
            ("Encoder", encoder_params, encoder_trainable),
            ("Transformer", transformer_params, transformer_trainable),
        ]

        if hasattr(self.world_model, 'head_dict'):
            components.append(("Prediction Heads", total_heads_params, total_heads_trainable))

        for name, total, trainable in components:
            pct = 100 * total / total_params if total_params > 0 else 0
            print(f'{name:<30} {total:>15,} {trainable:>15,} {pct:>14.1f}%')

        print(f'{"=" * 80}')
        print(f'{"TOTAL":<30} {total_params:>15,} {total_trainable:>15,} {"100.0%":>15}')
        print(f'{"=" * 80}\n')
    
    #@profile
    def initial_inference(self, obs_batch: torch.Tensor, action_batch=None, current_obs_batch=None, task_id=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of UniZero model, which is the first step of the UniZero model.
            To perform the initial inference, we first use the representation network to obtain the ``latent_state``.
            Then we use the prediction network to predict ``value`` and ``policy_logits`` of the ``latent_state``.
        Arguments:
            - obs_batch (:obj:`torch.Tensor`): The 3D image observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        batch_size = obs_batch.size(0)
        # print('=here 5='*20)
        # import ipdb; ipdb.set_trace()
        obs_act_dict = {'obs': obs_batch, 'action': action_batch, 'current_obs': current_obs_batch}
        _, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict, task_id=task_id)
        latent_state, reward, policy_logits, value = obs_token, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)

        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    #@profile
    def recurrent_inference(self, state_action_history: torch.Tensor, simulation_index=0,
                            search_depth=[], task_id=None) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of UniZero model.To perform the recurrent inference, we concurrently predict the latent dynamics (reward/next_latent_state)
            and decision-oriented quantities (value/policy) conditioned on the learned latent history in the world_model.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        _, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, search_depth, task_id=task_id)
        next_latent_state, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        reward = reward.squeeze(1)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)