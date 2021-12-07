'''
PySCF Numpy linalg APIs
'''

from pyscf.lib.numpy.linalg import *

# add default numpy functions
import numpy.linalg as lin
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(lin, globals())
del lin, add_functions_from_module
