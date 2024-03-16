cfg = {}
cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    'vocab_size': 128,  # TODO: for atari debug
                    'embed_dim': 128, # z_channels
                    'encoder':
                               {'resolution': 64, 'in_channels': 3, 'z_channels': 128, 'ch': 64,
                                'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                                'out_ch': 3, 'dropout': 0.0},# TODO：for atari debug
                            'decoder':
                    {'resolution': 64, 'in_channels': 3, 'z_channels': 128, 'ch': 64,
                     'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                     'out_ch': 3, 'dropout': 0.0}}  # TODO：for atari debug
cfg['world_model'] = {
                      'tokens_per_block': 2,
                      'max_blocks': 32,
                      "max_tokens": 2 * 32,  # TODO： horizon

                    #   'tokens_per_block': 2,
                    #   'max_blocks': 80,
                    #   "max_tokens": 2 * 80,  # TODO： horizon

                      'embed_dim': 128, # TODO：for memory
                      'group_size': 8,  # NOTE

                      'attention': 'causal',
                      'num_layers': 2, 
                      'num_heads': 2,

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:6',
                      'support_size': 21,
                      'action_shape': 4, # NOTE：for memory
                      'max_cache_size':5000,
                      "env_num": 8,

                      'latent_recon_loss_weight':0.,
                      'perceptual_loss_weight':0.,

                    #   'policy_entropy_weight': 1e-4,  # NOTE：for memory
                      'policy_entropy_weight': 1e-3,  # NOTE：for memory

                      'predict_latent_loss_type': 'group_kl', # 'mse'
                    #   'predict_latent_loss_type': 'mse', # 'mse'
                      'obs_type': 'vector', # 'vector', 'image'
                      }
from easydict import EasyDict
cfg = EasyDict(cfg)