'''
PySCF Scipy APIs
'''

from pyscf.lib.scipy import *
from pyscf.scipy import linalg

# add default scipy functions
import scipy as sp
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(sp, globals())
del sp, add_functions_from_module
