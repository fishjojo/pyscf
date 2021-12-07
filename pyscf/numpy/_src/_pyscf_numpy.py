'''
PySCF native optimized Numpy functions
'''
from pyscf.lib import numpy_helper

dot = numpy_helper.dot
einsum = numpy_helper.einsum

del numpy_helper
