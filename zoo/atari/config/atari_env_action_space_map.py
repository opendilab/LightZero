from easydict import EasyDict

# ale-py==0.10.1, gymnasium==1.2.1
atari_env_action_space_map = EasyDict({
    'ALE/Alien-v5': 18,
    'ALE/Amidar-v5': 10,
    'ALE/Assault-v5': 7,
    'ALE/Asterix-v5': 9,
    'ALE/BankHeist-v5': 18,
    'ALE/BattleZone-v5': 18,
    'ALE/ChopperCommand-v5': 18,
    'ALE/CrazyClimber-v5': 9,
    'ALE/DemonAttack-v5': 6,
    'ALE/Freeway-v5': 3,
    'ALE/Frostbite-v5': 18,
    'ALE/Gopher-v5': 8,
    'ALE/Hero-v5': 18,
    'ALE/Jamesbond-v5': 18,
    'ALE/Kangaroo-v5': 18,
    'ALE/Krull-v5': 18,
    'ALE/KungFuMaster-v5': 14,
    'ALE/PrivateEye-v5': 18,
    'ALE/RoadRunner-v5': 18,
    'ALE/UpNDown-v5': 6,
    'ALE/Pong-v5': 6,
    'ALE/MsPacman-v5': 9,
    'ALE/Qbert-v5': 6,
    'ALE/Seaquest-v5': 18,
    'ALE/Boxing-v5': 18,
    'ALE/Breakout-v5': 4,
    'ALE/SpaceInvaders-v5': 6,
    'ALE/BeamRider-v5': 9,
    'ALE/Gravitar-v5': 18,
})

for _env_id, _action_space_size in list(atari_env_action_space_map.items()):
    if _env_id.startswith('ALE/') and _env_id.endswith('-v5'):
        _legacy_env_id = _env_id[len('ALE/'):-len('-v5')] + 'NoFrameskip-v4'
        atari_env_action_space_map[_legacy_env_id] = _action_space_size
