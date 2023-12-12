# # small net
cfg = {}
cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    # 'vocab_size': 512, # TODO: for atari
                    # 'embed_dim': 512,
                    # 'vocab_size': 128,  # TODO: for atari debug
                    # 'embed_dim': 128,
                    'vocab_size': 64,  # TODO: for cartpole
                    'embed_dim': 64,
                    'encoder':
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 64, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 4, 'dropout': 0.0},# TODO: for cartpole
                    'decoder':
                        {'resolution': 1, 'in_channels': 4, 'z_channels': 64, 'ch': 64,
                         'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
                         'out_ch': 4, 'dropout': 0.0}} # TODO: for cartpole
cfg['world_model'] = {
                      'tokens_per_block': 17,
                      # 'max_blocks': 20,
                      #   "max_tokens": 17 * 20,  # TODO： horizon
                      'max_blocks': 5,
                        "max_tokens": 17 * 5,  # TODO： horizon
                      'attention': 'causal',
                      'num_heads': 2,
                      # 'num_layers': 10,# TODO：for atari
                      'num_layers': 2, # TODO：for debug
                      # 'embed_dim': 128, # TODO: for cartpole
                      'embed_dim': 64, # TODO: for cartpole
                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:0',
                      # "device": 'cpu',
                      # 'support_size': 601,
                      'support_size': 21,
                      'action_shape': 2,# TODO: for cartpole
                      'max_cache_size': 500,
                      # 'max_cache_size':25,
                      "env_num":8,
                      }
from easydict import EasyDict
cfg = EasyDict(cfg)



# # mediumnet
# cfg = {}
# cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
#                     # 'vocab_size': 512, # TODO: for atari
#                     # 'embed_dim': 512,
#                     'vocab_size': 128,  # TODO: for atari debug
#                     'embed_dim': 128,
#                     # 'vocab_size': 64,  # TODO: for cartpole
#                     # 'embed_dim': 64,
#                     'encoder':
#                         {'resolution': 1, 'in_channels': 4, 'z_channels': 128, 'ch': 64,
#                          'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
#                          'out_ch': 4, 'dropout': 0.0},# TODO: for cartpole
#                     'decoder':
#                         {'resolution': 1, 'in_channels': 4, 'z_channels': 128, 'ch': 64,
#                          'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
#                          'out_ch': 4, 'dropout': 0.0}} # TODO: for cartpole
# cfg['world_model'] = {
#                       'tokens_per_block': 17,
#                       # 'max_blocks': 20,
#                       #   "max_tokens": 17 * 20,  # TODO： horizon
#                       'max_blocks': 5,
#                         "max_tokens": 17 * 5,  # TODO： horizon
#                       'attention': 'causal',
#                       'num_heads': 4,
#                       # 'num_layers': 10,# TODO：for atari
#                       'num_layers': 2, # TODO：for debug
#                       'embed_dim': 128, # TODO: for cartpole
#                       # 'embed_dim': 64, # TODO: for cartpole
#                       'embed_pdrop': 0.1,
#                       'resid_pdrop': 0.1,
#                       'attn_pdrop': 0.1,
#                       "device": 'cuda:0',
#                       # "device": 'cpu',
#                       'support_size': 601,
#                       # 'support_size': 21,
#                       'action_shape': 2,# TODO: for cartpole
#                       # 'max_cache_size': 500,
#                       'max_cache_size':25,
#                       "env_num":8,
#                       }
# from easydict import EasyDict
# cfg = EasyDict(cfg)





# # tiny net

# cfg = {}

# cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
#                     # 'vocab_size': 512, # TODO: for atari
#                     # 'embed_dim': 512,
#                     # 'vocab_size': 128,  # TODO: for atari debug
#                     # 'embed_dim': 128,
#                     'vocab_size': 32,  # TODO: for cartpole
#                     'embed_dim': 32,
#                     'encoder':
#                         {'resolution': 1, 'in_channels': 4, 'z_channels': 32, 'ch': 64,
#                          'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
#                          'out_ch': 4, 'dropout': 0.0},# TODO: for cartpole
#                     'decoder':
#                         {'resolution': 1, 'in_channels': 4, 'z_channels': 32, 'ch': 64,
#                          'ch_mult': [1, 1, 1, 1, 1], 'num_res_blocks': 2, 'attn_resolutions': [8, 16],
#                          'out_ch': 4, 'dropout': 0.0}} # TODO: for cartpole

# cfg['world_model'] = {
#                       'tokens_per_block': 17,
#                       # 'max_blocks': 20,
#                       #   "max_tokens": 17 * 20,  # TODO： horizon
#                       'max_blocks': 5,
#                         "max_tokens": 17 * 5,  # TODO： horizon
#                       'attention': 'causal',
#                       'num_heads': 2,
#                       # 'num_layers': 10,# TODO：for atari
#                       'num_layers': 2, # TODO：for debug
#                       # 'embed_dim': 128, # TODO: for cartpole
#                       'embed_dim': 32, # TODO: for cartpole
#                       'embed_pdrop': 0.1,
#                       'resid_pdrop': 0.1,
#                       'attn_pdrop': 0.1,
#                       "device": 'cuda:0',
#                       # "device": 'cpu',
#                       # 'support_size': 601,
#                       'support_size': 21,
#                       'action_shape': 2,# TODO: for cartpole
#                       'max_cache_size': 500,
#                       # 'max_cache_size':25,
#                       "env_num":8,
#                       }

# from easydict import EasyDict

# cfg = EasyDict(cfg)
