# large net
cfg = {}

cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    'vocab_size': 512, # TODO: for atari
                    'embed_dim': 512,
                    # 'vocab_size': 128,  # TODO: for atari debug
                    # 'embed_dim': 128,
                         # 'vocab_size': 64,  # TODO: for atari debug
                    # 'embed_dim': 64,
                    'encoder':
                               {'resolution': 64, 'in_channels': 3, 'z_channels': 512, 'ch': 64,
                                'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                                'out_ch': 3, 'dropout': 0.0},# TODO：for atari debug
                            'decoder':
                    {'resolution': 64, 'in_channels': 3, 'z_channels': 512, 'ch': 64,
                     'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                     'out_ch': 3, 'dropout': 0.0}}  # TODO：for atari debug
                    # {'resolution': 64, 'in_channels': 1, 'z_channels': 512, 'ch': 64,
                    #             'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #             'out_ch': 3, 'dropout': 0.0},# TODO：for atari
                    #         'decoder':
                    # {'resolution': 64, 'in_channels': 1, 'z_channels': 512, 'ch': 64,
                    #  'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #  'out_ch': 3, 'dropout': 0.0}}  # TODO：for atari
cfg['world_model'] = {
                        'tokens_per_block': 17,
                      # 'max_blocks': 20,
                      #   "max_tokens": 17 * 20,  # TODO： horizon
                      'max_blocks': 5,
                      "max_tokens": 17 * 5,  # TODO： horizon
                      'attention': 'causal',
                      # 'num_layers': 10,# TODO：for atari
                      'num_layers': 2, # TODO：for atari debug
                      'num_heads': 4,
                      'embed_dim': 256, # TODO：for atari
                      # 'embed_dim': 128, # TODO：for atari
                      # 'embed_dim': 64, # TODO：for atari debug
                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:3',
                    #   "device": 'cpu',
                      'support_size': 601,
                      'action_shape': 6,# TODO：for atari
                      # 'max_cache_size':5000,
                      'max_cache_size':500,

                      "env_num":8,

                      }

from easydict import EasyDict
cfg = EasyDict(cfg)