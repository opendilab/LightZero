"""
Atari Action Space Mapping

Maps integer action indices to semantic action names for better VL understanding.
"""

# Atari action space mappings
# Source: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
ATARI_ACTION_MEANINGS = {
    'PongNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'RIGHTFIRE',
        5: 'LEFTFIRE',
    },
    'BreakoutNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT',
    },
    'SpaceInvadersNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'RIGHTFIRE',
        5: 'LEFTFIRE',
    },
    'QbertNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'UP',
        3: 'RIGHT',
        4: 'LEFT',
        5: 'DOWN',
    },
    'MsPacmanNoFrameskip-v4': {
        0: 'NOOP',
        1: 'UP',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'DOWN',
        5: 'UPRIGHT',
        6: 'UPLEFT',
        7: 'DOWNRIGHT',
        8: 'DOWNLEFT',
    },
    'SeaquestNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'UP',
        3: 'RIGHT',
        4: 'LEFT',
        5: 'DOWN',
        6: 'UPRIGHT',
        7: 'UPLEFT',
        8: 'DOWNRIGHT',
        9: 'DOWNLEFT',
        10: 'UPFIRE',
        11: 'RIGHTFIRE',
        12: 'LEFTFIRE',
        13: 'DOWNFIRE',
        14: 'UPRIGHTFIRE',
        15: 'UPLEFTFIRE',
        16: 'DOWNRIGHTFIRE',
        17: 'DOWNLEFTFIRE',
    },
    'MontezumaRevengeNoFrameskip-v4': {
        0: 'NOOP',
        1: 'FIRE',
        2: 'UP',
        3: 'RIGHT',
        4: 'LEFT',
        5: 'DOWN',
        6: 'UPRIGHT',
        7: 'UPLEFT',
        8: 'DOWNRIGHT',
        9: 'DOWNLEFT',
        10: 'UPFIRE',
        11: 'RIGHTFIRE',
        12: 'LEFTFIRE',
        13: 'DOWNFIRE',
        14: 'UPRIGHTFIRE',
        15: 'UPLEFTFIRE',
        16: 'DOWNRIGHTFIRE',
        17: 'DOWNLEFTFIRE',
    },
    'LunarLander-v2': {
        0: 'NOOP',
        1: 'LEFT_ENGINE',
        2: 'MAIN_ENGINE',
        3: 'RIGHT_ENGINE',
    },
}


def get_action_meanings(env_id: str, action_space_size: int) -> dict:
    """
    Get action meanings for a given Atari environment.

    Args:
        env_id: Environment ID (e.g., 'PongNoFrameskip-v4')
        action_space_size: Number of actions in the action space

    Returns:
        Dictionary mapping action indices to semantic names
    """
    if env_id in ATARI_ACTION_MEANINGS:
        return ATARI_ACTION_MEANINGS[env_id]

    # Fallback: generic action names
    return {i: f'ACTION_{i}' for i in range(action_space_size)}


def action_index_to_name(env_id: str, action_index: int, action_space_size: int) -> str:
    """
    Convert action index to semantic name.

    Args:
        env_id: Environment ID
        action_index: Action index (0, 1, 2, ...)
        action_space_size: Total number of actions

    Returns:
        Semantic action name (e.g., 'FIRE', 'RIGHT')
    """
    meanings = get_action_meanings(env_id, action_space_size)
    return meanings.get(action_index, f'ACTION_{action_index}')


def action_name_to_index(env_id: str, action_name: str, action_space_size: int) -> int:
    """
    Convert semantic action name to index.

    Args:
        env_id: Environment ID
        action_name: Semantic action name (e.g., 'FIRE', 'RIGHT')
        action_space_size: Total number of actions

    Returns:
        Action index (0, 1, 2, ...)
    """
    meanings = get_action_meanings(env_id, action_space_size)

    # Create reverse mapping
    name_to_idx = {name: idx for idx, name in meanings.items()}

    # Try exact match first
    if action_name in name_to_idx:
        return name_to_idx[action_name]

    # Try case-insensitive match
    action_name_upper = action_name.upper()
    if action_name_upper in name_to_idx:
        return name_to_idx[action_name_upper]

    # Try parsing "ACTION_X" format
    if action_name.startswith('ACTION_'):
        try:
            return int(action_name.split('_')[1])
        except (IndexError, ValueError):
            pass

    # Fallback: return 0 (NOOP)
    return 0


if __name__ == '__main__':
    # Test
    print("Testing Atari action mappings:")
    print("\nPong actions:")
    for i in range(6):
        name = action_index_to_name('PongNoFrameskip-v4', i, 6)
        print(f"  {i} -> {name}")

    print("\nBreakout actions:")
    for i in range(4):
        name = action_index_to_name('BreakoutNoFrameskip-v4', i, 4)
        print(f"  {i} -> {name}")

    print("\nReverse mapping (Pong):")
    for name in ['NOOP', 'FIRE', 'RIGHT', 'LEFT']:
        idx = action_name_to_index('PongNoFrameskip-v4', name, 6)
        print(f"  {name} -> {idx}")
