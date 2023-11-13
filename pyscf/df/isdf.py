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

import numpy as np
from pyscf.lib import logger
from pyscf.scf.hf import init_guess_by_atom
from pyscf.dft import gen_grid
from pyscf.dft import numint

def _kmeans(coords, weights, nc, tol=1e-4, maxiter=300, verbose=0):
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        #TODO add native kmeans implementation
        raise ImportError('Unable to import scikit-learn') from e

    kmeans_float = KMeans(n_clusters=nc,
                          max_iter=maxiter,
                          tol=tol,
                          n_init='auto',
                          verbose=verbose)
    kmeans_float.fit(coords, sample_weight=weights)
    rc = kmeans_float.cluster_centers_

    idx = np.empty((nc,), dtype=int)
    for i in range(nc):
        dist = np.linalg.norm(coords - rc[i][None,:], axis=1)
        idx[i] = np.argmin(dist)
    idx = np.sort(idx)
    return idx

def build_isdf_grids(mol,
                     c_isdf,
                     grid_level=1,
                     prune=gen_grid.sg1_prune,
                     verbose=None,
                     kmeans_tol=1e-4,
                     kmeans_maxiter=300,
                     kmeans_verbose=0):
    '''
    Build ISDF grids.
    Currently, only the K-means method is applied to construct the grid.
    The weighting function is defined as the atomic density multiplied by
    the square root of the DFT quadrature weight, which works best for THCDF.

    Returns:
        grids : :class:`dft.gen_grid.Grids`
            ISDF grid
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)

    #TODO custom grid type
    grids = gen_grid.Grids(mol)
    grids.alignment = 0
    grids.level = grid_level
    grids.prune = prune

    atom_grids_tab = grids.gen_atomic_grids(mol,
                                            grids.atom_grid,
                                            grids.radi_method,
                                            grids.level,
                                            grids.prune)
    coords, weights = grids.get_partition(mol,
                                          atom_grids_tab,
                                          grids.radii_adjust,
                                          grids.atomic_radii,
                                          grids.becke_scheme,
                                          concat=False)

    #TODO custom weighting method
    dm0 = init_guess_by_atom(mol)
    ao_slice = mol.aoslice_by_atom()[:,2:]
    nao_by_atom = ao_slice[:,1] - ao_slice[:,0]

    rc_coords = []
    rc_weights = []
    for ia in range(mol.natm):
        nc = int(nao_by_atom[ia] * c_isdf)
        rho0 = numint.eval_rho(mol, numint.eval_ao(mol, coords[ia]), dm0)
        rc_idx = _kmeans(coords[ia],
                         rho0 * np.sqrt(weights[ia]),
                         nc,
                         tol=kmeans_tol,
                         maxiter=kmeans_maxiter,
                         verbose=kmeans_verbose)
        rc_coords.append(coords[ia][rc_idx])
        rc_weights.append(weights[ia][rc_idx])

    grids.coords = np.vstack(rc_coords)
    grids.weights = np.hstack(rc_weights)
    log.timer('build_isdf_grids', *t0)
    return grids
