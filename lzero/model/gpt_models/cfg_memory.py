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

                    # 'max_blocks': 5,
                    #   "max_tokens": 2 * 5,  # H=5

                      'max_blocks': 16,
                      "max_tokens": 2 * 16,  # memory_length = 0

                      # 'max_blocks': 18,
                      # "max_tokens": 2 * 18,  # memory_length = 2

                      # 'max_blocks': 32,
                      # "max_tokens": 2 * 32,  # memory_length = 2

                      # 'max_blocks': 60,  # memory_length = 30 
                      # "max_tokens": 2 * 60,  

                      # 'max_blocks': 80, # memory_length = 50 
                      # "max_tokens": 2 * 80, 

                      #   'max_blocks': 90, # memory_length = 60 
                      # "max_tokens": 2 * 90, 

                      #   'max_blocks': 130, # memory_length = 100
                      # "max_tokens": 2 * 130, 

                      #   'max_blocks': 150, # memory_length = 120 
                      # "max_tokens": 2 * 150, 

                      #   'max_blocks': 280, # memory_length = 250 
                      # "max_tokens": 2 * 280, 

                      # 'max_blocks': 130, # memory_length = 100
                      # "max_tokens": 2 * 130, 

                      # 'max_blocks': 280, # memory_length = 250
                      # "max_tokens": 2 * 280, 

                    #    'max_blocks': 530, #  memory_length = 500
                    #   "max_tokens": 2 * 530, 

                      #   'max_blocks': 780, #  memory_length = 750
                      # "max_tokens": 2 * 780, 

                      #  'max_blocks': 1030, #  memory_length = 1000
                      # "max_tokens": 2 * 1030, 

                        'embed_dim': 64,  # TODO：for memory # same as <Transformer shine in RL> paper
                        # 'embed_dim': 100,  # TODO：for memory # same as <Transformer shine in RL> paper
                      'group_size': 8,  # NOTE

                      'attention': 'causal',
                      'num_layers': 2, 
                      'num_heads': 2, # same as <Transformer shine in RL> paper
                      
                      # 'num_layers': 4, 
                      # 'num_heads': 8, 

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                      "device": 'cuda:0',
                      'support_size': 21,
                      'action_shape': 4, # NOTE：for memory
                      'max_cache_size': 5000,
                      "env_num": 8,
                      #   "env_num": 20,

                      'latent_recon_loss_weight':0.05,
                      # 'latent_recon_loss_weight':0.5,
                      # 'latent_recon_loss_weight':1,


                      'perceptual_loss_weight': 0.,
                      'policy_entropy_weight': 1e-4,  # NOTE：for key_to_door
                      # 'policy_entropy_weight': 1e-1,  # NOTE：for visual_match

                      'predict_latent_loss_type': 'group_kl',
                      # 'predict_latent_loss_type': 'mse',

                      'obs_type': 'image_memory',  # 'vector', 'image'

}
from easydict import EasyDict
cfg = EasyDict(cfg)