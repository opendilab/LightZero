from easydict import EasyDict

dmc_state_env_action_space_map = EasyDict({
    'cartpole-swingup': 1,
    'hopper-hop': 4,
    'cheetah-run': 6,
    'walker-walk': 6,
    'humanoid-run': 21,

})

dmc_state_env_obs_space_map = EasyDict({
    'cartpole-swingup': 5,
    'hopper-hop': 15,
    'cheetah-run': 17,
    'walker-walk': 24,
    'humanoid-run': 67,
})