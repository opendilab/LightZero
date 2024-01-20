# small net for cartpole
cfg = {}
cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    # 'vocab_size': 512, # TODO: for atari
                    # 'embed_dim': 512,
                    # 'vocab_size': 128,  # TODO: for atari debug
                    # 'embed_dim': 128,
                    # 'vocab_size': 64,  # TODO: for cartpole
                    # 'embed_dim': 64,
                    # 'encoder':
                    #     {'resolution': 1, 'in_channels': 4, 'z_channels': 64, 'ch': 64,
                    #      'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #      'out_ch': 4, 'dropout': 0.0},# TODO: for cartpole
                    # 'decoder':
                    #     {'resolution': 1, 'in_channels': 4, 'z_channels': 64, 'ch': 64,
                    #      'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #      'out_ch': 4, 'dropout': 0.0}} # TODO: for cartpole
                    'vocab_size': 64,  # TODO: for cartpole
                    # 'embed_dim': 64,
                    'embed_dim': 16,

                    'encoder':
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 16, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 4, 'dropout': 0.0},# TODO: for cartpole
                    'decoder':
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 16, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 4, 'dropout': 0.0}} # TODO: for cartpole
cfg['world_model'] = {
                      'tokens_per_block': 2,
                      # 'max_blocks': 10,
                      #   "max_tokens": 17 * 10,  # TODO： horizon

                      # 'max_blocks': 5,
                      #   "max_tokens": 17 * 5,  # TODO： horizon

                        'max_blocks': 5,
                        "max_tokens": 2 * 5,  # TODO： horizon

                      'attention': 'causal',
                      'num_heads': 2,
                      'num_layers': 2, # TODO：for debug
                      
                      # 'embed_dim': 64, # TODO: for cartpole
                      # 'embed_dim': 1024, # TODO: for cartpole
                      'embed_dim': 256, # TODO: for cartpole

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:6',
                      # "device": 'cpu',
                      'support_size': 601,
                      'action_shape': 4, # TODO: for lunarlander
                      # 'action_shape': 2, # TODO: for memory_length

                      'max_cache_size': 500,
                      # 'max_cache_size': 50,
                      "env_num":8,
                      'latent_recon_loss_weight':0.0,
                      'perceptual_loss_weight':0.0,
                      }
from easydict import EasyDict
cfg = EasyDict(cfg)