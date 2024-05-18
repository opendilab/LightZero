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


    # ===================
    # for memory env 我们必须保存第一帧的obs, 而在最后一帧进行MCTS时肯定会超出epidoe_length，因此设置context_length比训练的更长以保证不会把第一帧去掉。
    # ===================
    
    # 'max_blocks': 16+5, 
    # "max_tokens": 2 * (16+5),  # 1+0+15 memory_length = 0
    # "context_length": 2 * (16+5),
    # "context_length_for_recurrent": 2 * (16+5),

    # 'max_blocks': 18+5, 
    # "max_tokens": 2 * (18+5),  # 1+2+15 memory_length = 2
    # "context_length": 2 * (18+5),
    # "context_length_for_recurrent": 2 * (18+5),

    'max_blocks': 76+5, 
    "max_tokens": 2 * (76+5),  # 1+60+15 memory_length = 60
    "context_length": 2 * (76+5),
    "context_length_for_recurrent": 2 * (76+5),

    # 'max_blocks': 116+5, 
    # "max_tokens": 2 * (116+5),  # 1+100+15 memory_length = 100
    # "context_length": 2 * (116+5),
    # "context_length_for_recurrent": 2 * (116+5),

    # 'max_blocks': 136+5, 
    # "max_tokens": 2 * (136+5),  # 1+120+15 memory_length = 120
    # "context_length": 2 * (136+5),
    # "context_length_for_recurrent": 2 * (136+5),

    # 'max_blocks': 266+5, 
    # "max_tokens": 2 * (266+5),  # 1+250+15 memory_length = 250
    # "context_length": 2 * (266+5),
    # "context_length_for_recurrent": 2 * (266+5),

    # 'max_blocks': 516+5, 
    # "max_tokens": 2 * (516+5),  # 1+500+15 memory_length = 500
    # "context_length": 2 * (516+5),
    # "context_length_for_recurrent": 2 * (516+5),

    # 'max_blocks': 1016+5, 
    # "max_tokens": 2 * (1016+5),  # 1+1000+15 memory_length = 1000
    # "context_length": 2 * (1016+5),
    # "context_length_for_recurrent": 2 * (1016+5),

    # Key2Door
    
    # 'max_blocks': 32+5, 
    # "max_tokens": 2 * (32+5),  # 15+2+15 memory_length = 2
    # "context_length": 2 * (32+5),
    # "context_length_for_recurrent": 2 * (32+5),

    # 'max_blocks': 90+5, 
    # "max_tokens": 2 * (90+5),  # 15+60+15 memory_length = 60
    # "context_length": 2 * (90+5),
    # "context_length_for_recurrent": 2 * (90+5),

    # 'max_blocks': 150+5, 
    # "max_tokens": 2 * (150+5),  # 15+120+15 memory_length = 120
    # "context_length": 2 * (150+5),
    # "context_length_for_recurrent": 2 * (150+5),

    # 'max_blocks': 280+5, 
    # "max_tokens": 2 * (280+5),  # 15+250+15 memory_length = 250
    # "context_length": 2 * (280+5),
    # "context_length_for_recurrent": 2 * (280+5),

    # 'max_blocks': 530+5, 
    # "max_tokens": 2 * (530+5),  # 15+500+15 memory_length = 500
    # "context_length": 2 * (530+5),
    # "context_length_for_recurrent": 2 * (530+5),


    "device": 'cuda:0',
    'analysis_sim_norm': False,
    'analysis_dormant_ratio': False,

    'group_size': 8,  # NOTE
    # 'group_size': 64,  # NOTE

    # 'group_size': 768,  # NOTE
    'attention': 'causal',

    # 'num_layers': 8,
    # 'num_heads': 8,
    # 'embed_dim': 64, # TODO： for for visual_match [1000]

    # 'num_layers': 4,
    'num_layers': 12,
    'num_heads': 4,
    'embed_dim': 64, # TODO： for for visual_match [250, 500]
    # 'embed_dim': 96, # TODO： for for visual_match [250, 500]

    # 'embed_dim': 32, # TODO： for for visual_match [250, 500]

    # 'num_layers': 4,
    # 'num_heads': 4,
    # 'embed_dim': 64, # TODO：  for visual_match [2, 60, 100]
    
    # 'num_layers': 8,
    # 'num_heads': 8,
    # 'embed_dim': 64, # TODO： for memlen=500,1000

    # 'num_layers': 8,
    # 'num_heads': 8,
    # 'embed_dim': 128, # TODO： for memlen=250
    # 'embed_dim': 256, # TODO：for memlen=0/60/100


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
    # "env_num": 20, # for memeory_length=1000 TODO
      "env_num": 10,

    # 'latent_recon_loss_weight': 0.05,
    'latent_recon_loss_weight': 0.0, # TODO
    # 'latent_recon_loss_weight':0.5,
    # 'latent_recon_loss_weight':10,

    'perceptual_loss_weight': 0.,
    # 'policy_entropy_weight': 1e-4,  # NOTE：for visual_match 
    'policy_entropy_weight': 1e-3,  # NOTE：for key_to_door

    'predict_latent_loss_type': 'group_kl',
    # 'predict_latent_loss_type': 'mse',

    'obs_type': 'image_memory',  # 'vector', 'image'
     'gamma': 1, # 0.5, 0.9, 0.99, 0.999
     'dormant_threshold': 0.025,
}
from easydict import EasyDict

cfg = EasyDict(cfg)
