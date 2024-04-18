cfg = {}
cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    'vocab_size': 128,  # TODO: for atari debug
                    'embed_dim': 128,  # z_channels
                    'encoder':
                        {'resolution': 64, 'in_channels': 3, 'z_channels': 128, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 3, 'dropout': 0.0},  # TODO：for atari debug
                    'decoder':
                        {'resolution': 64, 'in_channels': 3, 'z_channels': 128, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 3, 'dropout': 0.0}}  # TODO：for atari debug
cfg['world_model'] = {
    'tokens_per_block': 2,


    # 'max_blocks': 16,
    # "max_tokens": 2 * 16,  # 1+0+15 memory_length = 0
    # "context_length": 2 * 16,
    # "context_length_for_recurrent": 2 * 16,

    'max_blocks': 16+5, # for memory env 我们必须保存第一帧的obs, 而在最后一帧进行MCTS时肯定会超出epidoe_length，因此设置context_length比训练的更长以保证不会把第一帧去掉。
    "max_tokens": 2 * (16+5),  # 1+0+15 memory_length = 0
    "context_length": 2 * (16+5),
    "context_length_for_recurrent": 2 * (16+5),


    # 'max_blocks': 21,
    # "max_tokens": 2 * 21,  # 1+0+15 memory_length = 0
    # "context_length": 2 * 21,
    # "context_length_for_recurrent": 2 * 21,
    # "recurrent_keep_deepth": 100,

    # 'max_blocks': 17, # TODO
    # "max_tokens": 2 * 17,  # 1+0+15 memory_length = 0
    # "context_length": 2 * 17,
    # "context_length_for_recurrent": 2 * 17,
    # "recurrent_keep_deepth": 100,


    # 'max_blocks': 46, 
    # "max_tokens": 2 * 46, # 1+30+15=76 memory_length = 30
    # "context_length": 2 * 46,
    # "context_length_for_recurrent": 2 * 46,
    # "recurrent_keep_deepth": 100,

    'max_blocks': 76+5, 
    "max_tokens": 2 * (76+5), # 1+60+15=76 memory_length = 60
    "context_length": 2 * (76+5),
    "context_length_for_recurrent": 2 * (76+5),
    "recurrent_keep_deepth": 100,

    #     'max_blocks': 116, 
    # "max_tokens": 2 * 116, # 1+100+15=116 memory_length = 100
    # "context_length": 2 * 116,
    # "context_length_for_recurrent": 2 * 116,
    # "recurrent_keep_deepth": 100,

    # 'max_blocks': 80, # memory_length = 50
    # "max_tokens": 2 * 80,

    #   'max_blocks': 90, 
    # "max_tokens": 2 * 90, # 15+60+15 memory_length = 60

    #   'max_blocks': 130,
    # "max_tokens": 2 * 130, # 15+100+15 memory_length = 100

    #   'max_blocks': 150, 
    # "max_tokens": 2 * 150,  # 15+120+15 memory_length = 120

    #   'max_blocks': 280, 
    # "max_tokens": 2 * 280, # 15+250+15 memory_length = 250

    #    'max_blocks': 530, #  memory_length = 500
    #   "max_tokens": 2 * 530,

    "device": 'cuda:1',

    'group_size': 8,  # NOTE
    # 'group_size': 768,  # NOTE
    'attention': 'causal',
    # 'num_layers': 1,
    # 'num_layers': 2,  # same as <Transformer shine in RL> paper
    # 'num_layers': 4,

    'num_layers': 8,
    'num_heads': 8,
    # 'embed_dim': 96,  # TODO：for memory # same as <Transformer shine in RL> paper
    # 'embed_dim': 768, # TODO：Gpt2 Base
    'embed_dim': 256, # TODO：

    # 'num_layers': 12, # TODO：Gpt2 Base
    # 'num_heads': 12, # TODO：Gpt2 Base
    # 'embed_dim': 768, # TODO：Gpt2 Base

    # 'num_layers': 12, # TODO：Gato Medium
    # 'num_heads': 12, # TODO：Gato Medium
    # 'embed_dim': 1536, # TODO：Gato Medium

    # 'num_layers': 8, # TODO：Gato Base
    # 'num_heads': 24, # TODO：Gato Base
    # 'embed_dim': 768, # TODO：Gato Base

    # 'num_layers': 24, # TODO：Gato Large
    # 'num_heads': 16, # TODO：Gato Large
    # 'embed_dim': 2048, # TODO：Gato  Large

    'gru_gating': False,

    'embed_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'attn_pdrop': 0.1,

    'support_size': 101, # TODO
    'action_shape': 4,  # NOTE：for memory
    'max_cache_size': 5000,
    "env_num": 8,
    #   "env_num": 20,

    'latent_recon_loss_weight': 0.05,
    # 'latent_recon_loss_weight': 0.0,
    # 'latent_recon_loss_weight':0.5,
    # 'latent_recon_loss_weight':10,

    'perceptual_loss_weight': 0.,
    'policy_entropy_weight': 1e-4,  # NOTE：for key_to_door
    # 'policy_entropy_weight': 1e-1,  # NOTE：for visual_match

    'predict_latent_loss_type': 'group_kl',
    # 'predict_latent_loss_type': 'mse',

    'obs_type': 'image_memory',  # 'vector', 'image'
     'gamma': 1, # 0.5, 0.9, 0.99, 0.999
    #  'gamma': 1.2, # 0.5, 0.9, 0.99, 0.999


}
from easydict import EasyDict

cfg = EasyDict(cfg)
