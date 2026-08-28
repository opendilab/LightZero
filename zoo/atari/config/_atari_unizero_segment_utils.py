import re


class _Tee:
    """Mirror output to all configured streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')


def _safe_run_name(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-_.')
    if not value:
        raise ValueError('run_name must contain at least one letter or number')
    return value


def _resolve_grad_clip_mode(use_augmentation, override=None):
    """Use isolated encoder clipping for augmentation unless explicitly overridden."""
    mode = ('separate_encoder' if use_augmentation else 'global') if override is None else str(override)
    if mode not in {'global', 'separate_encoder'}:
        raise ValueError(f'Unsupported grad_clip_mode: {mode}')
    return mode
