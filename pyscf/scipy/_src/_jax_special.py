try:
    from jax.scipy.special import *
except ImportError:
    raise("Unable to import jax.scipy.special")

from jax import config as _config
_config.update("jax_enable_x64", True)
