import warnings

try:
    from .smac_env_ace import SMACACEEnv
except ImportError:
    warnings.warn("not found pysc2 env, please install it")
    SMACEnv = None
