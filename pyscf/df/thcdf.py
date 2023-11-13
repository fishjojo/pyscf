#!/usr/bin/env python
# Copyright 2023- The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy as np
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf.df import df
from pyscf.df import addons
from pyscf.df import isdf
from pyscf.df.incore import MAX_MEMORY, LINEAR_DEP_THR
from pyscf.df.incore import _eig_decompose

libcgto = lib.load_library('libcgto')
libdf = lib.load_library('libdf')


def incore_(mydf, auxbasis='weigend+etb', auxmol=None,
            int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
            max_memory=MAX_MEMORY, decompose_j2c='cd',
            lindep=LINEAR_DEP_THR, verbose=0):
    assert comp == 1
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mydf, verbose)

    mol = mydf.mol
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    if decompose_j2c == 'eig':
        low = _eig_decompose(mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
            decompose_j2c = 'cd'
        except scipy.linalg.LinAlgError:
            low = _eig_decompose(mol, j2c, lindep)
            decompose_j2c = 'eig'
    j2c = None
    naux, naoaux = low.shape
    log.debug('size of aux basis %d', naux)
    log.timer_debug1('2c2e', *t0)

    if mydf.grids is None:
        mydf.grids = grids = isdf.build_isdf_grids(mol, mydf.c_isdf)
    else:
        grids = mydf.grids

    t1 = (logger.process_clock(), logger.perf_counter())
    rc_coords = grids.coords
    rc_weights = grids.weights

    non0tab = gen_grid.make_mask(mol, rc_coords)
    non0tab = np.asarray(non0tab, order='C')
    aoRg = numint.eval_ao(mol, rc_coords, non0tab=non0tab)

    X = aoRg * (rc_weights**.25)[:,None]
    X = np.asarray(X, order='C')
    ngrids = X.shape[0]
    aoRg = None

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    atm = np.asarray(atm, order='C')
    bas = np.asarray(bas, order='C')
    env = np.asarray(env, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

    #: int3c_aoRg2 = einsum('uvP,Lu,Lv->PL', int3c, X, X)
    int3c_aoRg2 = np.zeros((naoaux,ngrids))
    drv = libdf.THCDFctr_int3c_aoRg2
    fill = getattr(libdf, 'THCDFctr_' + aosym)
    drv(getattr(libcgto, int3c), fill,
        int3c_aoRg2.ctypes.data_as(ctypes.c_void_p),
        X.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrids), 
        non0tab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        ctypes.c_int(nao), ctypes.c_int(comp),
        (ctypes.c_int*6)(*(shls_slice[:6])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    log.timer_debug1('contract_int3c_aoRg2', *t1)

    S = lib.dot(X, X.T) ** 2
    S_inv = scipy.linalg.pinv(S, atol=lindep, rtol=0)
    B = lib.dot(int3c_aoRg2, S_inv)

    int3c_aoRg2 = None

    if decompose_j2c == 'cd':
        D = scipy.linalg.solve_triangular(low, B, lower=True,
                                          overwrite_b=True, check_finite=False)
    else:
        D = lib.dot(low, B)

    log.timer('incore THCDF', *t0)
    return D, X


class THCDF(df.DF):
    '''
    Tensor hypercontraction (THC) of Gaussian density fitting.
    See also :class:`df.DF`.

    Attributes:
        grids : :class:`dft.gen_grid.Grids`
            Real-space grid for representing the pair density.
            The default grid is constructed from the interpolative
            separable density fitting (ISDF) scheme. Custom grid
            is also allowed.
        c_isdf : float
            The parameter to control the number of ISDF grid points.
            The total number of grid points is equal to
            :math:`N * c_{isdf}`, where `N` is the number of basis functions.
            For convenience, this parameter is exposed to the :class:`THCDF` class API.
            To specify more options for building the ISDF grids,
            refer to the `df.isdf` module. The default value is 10.
    '''
    _keys = {'grids', 'c_isdf'}

    def __init__(self, mol, auxbasis=None, grids=None, c_isdf=10):
        super().__init__(mol, auxbasis=auxbasis)
        self.grids = grids
        self.c_isdf = c_isdf

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        super().dump_flags(verbose=log)
        if self.grids is None:
            log.info('ISDF grid with c_isdf = %.1f', self.c_isdf)
        return self

    def build(self):
        log = logger.Logger(self.stdout, self.verbose)

        self.check_sanity()
        self.dump_flags()
        if self._cderi is not None and self.auxmol is None:
            log.info('Skip THCDF.build(). Tensor _cderi will be used.')
            return self

        mol = self.mol
        auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        if self.grids is None: 
            ngrids = int(nao * self.c_isdf)
        else:
            ngrids = self.grids.weights.size

        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if ((naux+nao+4)*ngrids*8/1e6 < .9*max_memory and not is_custom_storage):
            self._cderi = incore_(self, auxmol=auxmol,
                                  int3c=int3c, int2c=int2c,
                                  max_memory=max_memory, verbose=log)
        else:
            raise NotImplementedError("outcore THCDF is not implemented")
        return self
