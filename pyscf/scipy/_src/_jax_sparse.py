try:
    from jax.scipy.sparse import *
    import jax.scipy.sparse.linalg as linalg
except ImportError:
    raise("Unable to import jax.scipy.sparse")

from jax import config as _config
_config.update("jax_enable_x64", True)
