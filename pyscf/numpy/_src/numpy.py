'''
Numpy APIs
'''
from pyscf import __config__

BACKEND = getattr(__config__, "pyscf_numpy_backend", "pyscf")

if BACKEND.upper() == "PYSCF":
    from ._pyscf_numpy import *
elif BACKEND.upper() in ("JAX", "PYSCFAD"):
    from ._jax_numpy import *
elif BACKEND.upper() == "CUPY":
    raise NotImplementedError
else:
    raise KeyError(f"Numpy backend {BACKEND} is not available.")

del BACKEND
