'''
PySCF Numpy APIs
'''

from ._src.numpy import *
from . import linalg

# add default numpy functions
import numpy as np
from pyscf.lib.misc import add_functions_from_module
add_functions_from_module(np, globals())
del np, add_functions_from_module
