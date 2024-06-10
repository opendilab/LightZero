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

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 5,
                      # "max_tokens": 2 * 5,  # TODO： horizon:4
                      # # "context_length": 2 * 5,
                      # # "context_length_for_recurrent":2 * 5,
                      # "context_length": 2 * 4,
                      # "context_length_for_recurrent":2 * 4,
                      # "gru_gating": False,
                      # # "gru_gating": True,


                      # 'tokens_per_block': 2,
                      # 'max_blocks': 8,
                      # "max_tokens": 2 * 8,  # TODO： horizon:8
                      # # "context_length": 2 * 8,
                      # # "context_length_for_recurrent":2 * 8,
                      # "context_length": 2 * 4,
                      # "context_length_for_recurrent":2 * 4,
                      # "gru_gating": False,
                      # # "gru_gating": True,

                      'tokens_per_block': 2,
                      'max_blocks': 10,
                      "max_tokens": 2 * 10,  # TODO： horizon:8
                      # "context_length": 20,
                      # "context_length_for_recurrent": 20,
                      "context_length": 2 * 4, # TODO
                      "context_length_for_recurrent": 2 * 4,
                      "recurrent_keep_deepth": 100,
                      "gru_gating": False,
                      # "gru_gating": True,

                    #  'tokens_per_block': 2,
                    #   'max_blocks': 20,
                    #   "max_tokens": 2 * 20,  # TODO： horizon:8
                    #   # "context_length": 2*20,
                    #   # "context_length_for_recurrent": 2*20,
                    #   "context_length": 2 * 4, # TODO
                    #   "context_length_for_recurrent": 2 * 4,
                    #   "recurrent_keep_deepth": 100,
                    #   "gru_gating": False,

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 30,
                      # "max_tokens": 2 * 30,  # TODO： horizon:8
                      # # "context_length": 2*20,
                      # # "context_length_for_recurrent": 2*20,
                      # "context_length": 2 * 4, # TODO
                      # "context_length_for_recurrent": 2 * 4,
                      # "recurrent_keep_deepth": 100,
                      # "gru_gating": False,

                      # 'tokens_per_block': 2,
                      # 'max_blocks': 40,
                      # "max_tokens": 2 * 40,  # TODO： horizon:8
                      # # "context_length": 2*20,
                      # # "context_length_for_recurrent": 2*20,
                      # "context_length": 2 * 4, # TODO
                      # "context_length_for_recurrent": 2 * 4,
                      # "recurrent_keep_deepth": 100,
                      # "gru_gating": False,


                      "device": 'cuda:1',
                      "task_num": 4,
                      # "task_num": 2,

                      'analysis_sim_norm': False,
                      'analysis_dormant_ratio': False,
                      
                      'action_shape': 18, # TODO：for multi-task
                      
                      # 'action_shape': 6, # TODO：for pong qbert 
                      # 'action_shape': 9,# TODO：for mspacman
                      # 'action_shape': 18,# TODO：for Seaquest boxing Frostbite
                      # 'action_shape': 4,# TODO：for breakout

                      # 'embed_dim':512, # TODO：for atari
                      # 'embed_dim':1024, # TODO：for atari
                      'group_size': 8,  # NOTE: sim_norm
                      # 'group_size': 768,  # NOTE: softmax

                      'attention': 'causal',

                      # 'num_layers': 1, # TODO：for atari debug
                      'num_layers': 2, # TODO：for atari debug

                      # 'num_layers': 4, # TODO：for atari debug
                      
                      # 'num_layers': 6, # TODO：for atari debug
                      # 'num_layers': 8, # TODO：for atari debug
                      # 'num_layers': 12, # TODO：for atari debug
                      'num_heads': 8,
                      'embed_dim': 768, # TODO：for atari

                      # for concat setting
                      # 'obs_embed_dim': 672,
                      # 'task_embed_dim': 96,
                      
                      'obs_embed_dim': 768,
                      'task_embed_dim': 768,

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

                      'embed_pdrop': 0.1,
                      'resid_pdrop': 0.1,
                      'attn_pdrop': 0.1,
                    #   "device": 'cpu',
                      # 'support_size': 601,
                      'support_size': 101,  # TODO


                      'max_cache_size': 5000,
                      # 'max_cache_size':50000,
                      # 'max_cache_size':500,
                      # "env_num": 8,
                      "env_num": [2,8,8,2],
                      "eval_env_num": 3,
                      "batch_size":[32,32,32,32],

                      # 'latent_recon_loss_weight': 0.05,
                      # 'perceptual_loss_weight': 0.05, # for stack1 rgb obs
                      # # 'perceptual_loss_weight':0., # for stack4 gray obs

                      'latent_recon_loss_weight': 0.,
                      'perceptual_loss_weight': 0., # for stack1 rgb obs

                      # 'latent_recon_loss_weight':0.,
                      # 'perceptual_loss_weight':0.,

                      'policy_entropy_weight': 0,
                      # 'policy_entropy_weight': 1e-4,

                      'predict_latent_loss_type': 'group_kl',
                      # 'predict_latent_loss_type': 'mse',
                      'obs_type': 'image', # 'vector', 'image'

                      # 'gamma': 1.5, # 0.5, 0.9, 0.99, 0.999
                      'gamma': 1, # 0.5, 0.9, 0.99, 0.999
                      'dormant_threshold': 0.025,
                      }
from easydict import EasyDict
cfg = EasyDict(cfg)