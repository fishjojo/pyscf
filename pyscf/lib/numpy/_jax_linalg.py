try:
    from jax.numpy.linalg import *
except ImportError:
    raise("Unable to import jax.numpy.linalg")

from jax import numpy as _jnp
from jax.numpy import linalg as _linalg
from jax import config as _config
_config.update("jax_enable_x64", True)


# Custom functions will replace the ones from jax
def cond(x, p=None):
    '''Compute the condition number of a matrix or a list of matrices'''
    if getattr(x, "ndim", None) == 2:
        return _linalg.cond(x, p)
    else:
        return _jnp.asarray([_linalg.cond(xi, p) for xi in x])
