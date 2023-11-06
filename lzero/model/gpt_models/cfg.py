cfg = {}

cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    'vocab_size': 512,
                    'embed_dim': 512,
                    'encoder':
                    # {'resolution': 64, 'in_channels': 3, 'z_channels': 512, 'ch': 64,
                    #             'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #             'out_ch': 3, 'dropout': 0.0},
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 512, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 3, 'dropout': 0.0},
                    'decoder':
                    # {'resolution': 64, 'in_channels': 3, 'z_channels': 512, 'ch': 64,
                    #  'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                    #  'out_ch': 3, 'dropout': 0.0}}
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 512, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 3, 'dropout': 0.0}}

cfg['world_model'] = {'device': "cpu",  # TODO：
                      'tokens_per_block': 17,
                      'max_blocks': 20,
                      "max_tokens": 17 * 20,  # TODO： horizon
                      # 'max_blocks': 5,
                      # "max_tokens": 17 * 5,  # TODO： horizon
                      'attention': 'causal',
                      'num_layers': 10,
                      'num_heads': 4,
                      'embed_dim': 256,
                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      }

from easydict import EasyDict

cfg = EasyDict(cfg)
