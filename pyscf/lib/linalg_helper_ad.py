import scipy.linalg
from pyscf import __config__

PYSCFAD = getattr(__config__, 'pyscfad', False)
if PYSCFAD:
    from pyscfad.lib.linalg_helper import eigh
else:
    eigh = scipy.linalg.eigh
