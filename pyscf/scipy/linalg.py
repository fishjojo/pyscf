'''
PySCF Scipy linalg APIs
'''
from pyscf.lib.scipy.linalg import *

# add default scipy functions
import scipy.linalg as lin
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(lin, globals())
del lin, add_functions_from_module
