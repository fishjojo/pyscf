'''
Numpy linalg APIs
'''
from pyscf import __config__

BACKEND = getattr(__config__, "pyscf_numpy_backend", "pyscf")

if BACKEND.upper() == "PYSCF":
    from pyscf.lib.numpy._pyscf_linalg import *
elif BACKEND.upper() == "JAX":
    from pyscf.lib.numpy._jax_linalg import *
elif BACKEND.upper() == "CUPY":
    raise NotImplementedError
else:
    raise KeyError(f"Numpy backend {BACKEND} is not available.")

del BACKEND
