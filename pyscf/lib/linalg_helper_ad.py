import scipy.linalg
from pyscf import __config__

JAXNUMPY = getattr(__config__, 'jaxnumpy', False)
if JAXNUMPY:
    from pyscfad.lib.linalg_helper import eigh
else:
    eigh = scipy.linalg.eigh
