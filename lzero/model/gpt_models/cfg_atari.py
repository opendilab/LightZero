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
                      # 'tokens_per_block': 17,
                      # 'max_blocks': 20,
                      #   "max_tokens": 17 * 20,  # TODO： horizon
                      # 'max_blocks': 5,
                      # "max_tokens": 17 * 5,  # TODO： horizon
                      # 'embed_dim': 128, # TODO：for atari

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 50,
                      # "max_tokens": 2 * 50,  # TODO： horizon

                      'tokens_per_block': 2,
                      'max_blocks': 5,
                      "max_tokens": 2 * 5,  # TODO： horizon

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 10,
                      # "max_tokens": 2 * 10,  # TODO： horizon

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 6,
                      # "max_tokens": 2 * 6,  # TODO： horizon

                      # 'embed_dim':512, # TODO：for atari
                      # 'embed_dim':256, # TODO：for atari
                      # 'embed_dim':1024, # TODO：for atari
                      'embed_dim': 768, # TODO：for atari
                      'group_size': 8,  # NOTE

                      'attention': 'causal',

                      # 'num_layers': 2, # TODO：for atari debug
                      'num_layers': 4, # TODO：for atari debug
                      # 'num_layers': 6, # TODO：for atari debug
                      # 'num_layers': 12, # TODO：for atari debug
                      'num_heads': 8,

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:4',
                    #   "device": 'cpu',
                      'support_size': 601,

                      # 'action_shape': 18,# TODO：for multi-task

                      # 'action_shape': 18,# TODO：for Seaquest boxing Frostbite
                      # 'action_shape': 9,# TODO：for mspacman
                      # 'action_shape': 4,# TODO：for breakout
                      'action_shape': 6, # TODO：for pong qbert 

                      'max_cache_size':5000,
                      # 'max_cache_size':50000,
                      # 'max_cache_size':500,
                      "env_num": 8,
                      # "env_num":16, # TODO
                      # "env_num":1, # TODO

                      'latent_recon_loss_weight':0.05,
                      'perceptual_loss_weight':0.05, # for stack1 rgb obs
                      # 'perceptual_loss_weight':0., # for stack4 gray obs

                      # 'latent_recon_loss_weight':0.,
                      # 'perceptual_loss_weight':0.,

                      # 'policy_entropy_weight': 0,
                      'policy_entropy_weight': 1e-4,

                      # 'predict_latent_loss_type': 'group_kl',
                      'predict_latent_loss_type': 'mse',
                      'obs_type': 'image', # 'vector', 'image'

                      }
from easydict import EasyDict
cfg = EasyDict(cfg)