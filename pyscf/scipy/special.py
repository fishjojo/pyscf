'''
PySCF Scipy special APIs
'''
from ._src.special import *

# add default scipy functions
import scipy.special as spe
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(spe, globals())
del spe, add_functions_from_module
