try:
    from jax.scipy.linalg import *
except ImportError:
    raise("Unable to import jax.scipy.linalg")

from jax import config as _config
_config.update("jax_enable_x64", True)
