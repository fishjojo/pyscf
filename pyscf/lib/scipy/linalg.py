'''
Scipy linalg APIs
'''
from pyscf import __config__

BACKEND = getattr(__config__, "pyscf_scipy_backend", "pyscf")

if BACKEND.upper() == "PYSCF":
    from pyscf.lib.scipy._pyscf_linalg import *
elif BACKEND.upper() == "JAX":
    from pyscf.lib.scipy._jax_linalg import *
elif BACKEND.upper() == "PYSCFAD":
    from pyscf.lib.scipy._pyscfad_linalg import *
elif BACKEND.upper() == "CUPY":
    raise NotImplementedError
else:
    raise KeyError(f"Numpy backend {BACKEND} is not available.")

del BACKEND
