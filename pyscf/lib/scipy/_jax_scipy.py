'''
Scipy functions from Jax
'''

try:
    from jax.scipy import *
    from jax.config import config as _config
    _config.update("jax_enable_x64", True)
    globals().pop('linalg', None)
except ImportError:
    raise ("Unable to import jax.scipy")
