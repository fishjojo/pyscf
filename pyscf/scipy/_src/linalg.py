'''
Scipy linalg APIs
'''
from pyscf import __config__

BACKEND = getattr(__config__, "pyscf_scipy_backend", "pyscf")

if BACKEND.upper() == "PYSCF":
    from ._pyscf_linalg import *
elif BACKEND.upper() == "JAX":
    from ._jax_linalg import *
elif BACKEND.upper() == "PYSCFAD":
    from ._pyscfad_linalg import *
elif BACKEND.upper() == "CUPY":
    raise NotImplementedError
else:
    raise KeyError(f"Scipy backend {BACKEND} is not available.")

del BACKEND
