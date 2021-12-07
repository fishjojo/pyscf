'''
PySCF Scipy linalg APIs
'''
from ._src.linalg import *

# add default scipy functions
from scipy import linalg as lin
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(lin, globals())
del lin, add_functions_from_module
