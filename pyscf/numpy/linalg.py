'''
PySCF Numpy linalg APIs
'''

from ._src.linalg import *

# add default numpy functions
import numpy.linalg as lin
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(lin, globals())
del lin, add_functions_from_module
