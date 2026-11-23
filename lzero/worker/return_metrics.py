"""Name measured Atari returns by both reward scale and episode boundary."""


RETURN_METRICS = tuple(f'{scale}_{unit}_return' for scale in ('raw', 'clipped') for unit in ('life', 'game'))


def explicit_return_metrics(info):
    # Other environments need not implement these Atari annotations. Do not
    # manufacture raw scores from clipped rewards or infer the episode unit.
    if 'episode_return_is_life' not in info:
        return {}
    unit = 'life' if info['episode_return_is_life'] else 'game'
    return {f'{scale}_{unit}_return': float(info[f'{scale}_episode_return'])
            for scale in ('raw', 'clipped') if f'{scale}_episode_return' in info}
