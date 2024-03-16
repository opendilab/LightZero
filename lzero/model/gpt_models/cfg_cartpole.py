# mediumnet
cfg = {}
cfg['tokenizer'] = {'_target_': 'models.tokenizer.Tokenizer',
                    'vocab_size': 64,  # TODO: for cartpole
                    'embed_dim': 128,
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
                      'max_blocks': 5,
                      "max_tokens": 2 * 5,  # TODO： horizon

                    # 'tokens_per_block': 2,
                    #   'max_blocks': 2,
                    #   "max_tokens": 2 * 2,  # TODO： horizon

                      'embed_dim':128, # TODO：for cartpole
                      # 'embed_dim':64, # TODO：for cartpole
                      'group_size': 8,  # NOTE

                      'attention': 'causal',
                      # 'num_layers': 2, 
                      # 'num_heads': 8,  # 128/8=16

                      'num_layers': 2, 
                      'num_heads': 1,  # 128/8=16

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:0',
                      # 'support_size': 601,
                      'support_size': 21,

                      'action_shape': 2, # TODO：for cartpole

                      'max_cache_size':5000,
                      "env_num": 8,

                      'latent_recon_loss_weight':0.05,
                      'perceptual_loss_weight':0.05, # for stack1 rgb obs
                      # 'perceptual_loss_weight':0., # for stack4 gray obs

                      # 'latent_recon_loss_weight':0.,
                      # 'perceptual_loss_weight':0.,

                      'policy_entropy_weight': 0,
                      # 'policy_entropy_weight': 1e-4,

                      # 'predict_latent_loss_type': 'group_kl', # 'mse'
                      'predict_latent_loss_type': 'mse', # 'mse'

                      'obs_type': 'vector', # 'vector', 'image'
                      }
from easydict import EasyDict
cfg = EasyDict(cfg)