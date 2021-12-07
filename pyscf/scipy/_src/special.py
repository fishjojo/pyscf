'''
Scipy linalg APIs
'''
from pyscf import __config__

BACKEND = getattr(__config__, "pyscf_scipy_backend", "pyscf")

if BACKEND.upper() in ("JAX", "PYSCFAD"):
    from ._jax_special import *
else:
    #from scipy.special import *
    pass

del BACKEND
