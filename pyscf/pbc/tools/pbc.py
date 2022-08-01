#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import warnings
import copy
import numpy
import scipy.linalg
from pyscf import numpy as np
from pyscf import lib
from pyscf.lib import ops, stop_grad
from pyscf.lib import logger
from pyscf.gto import ATM_SLOTS, BAS_SLOTS, ATOM_OF, PTR_COORD
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3  # noqa
from pyscf import __config__

PYSCFAD = getattr(__config__, 'pyscfad', False)
FFT_ENGINE = getattr(__config__, 'pbc_tools_pbc_fft_engine', 'BLAS')

def _fftn_blas(f, mesh):
    Gx = numpy.fft.fftfreq(mesh[0])
    Gy = numpy.fft.fftfreq(mesh[1])
    Gz = numpy.fft.fftfreq(mesh[2])
    expRGx = numpy.exp(numpy.einsum('x,k->xk', -2j*numpy.pi*numpy.arange(mesh[0]), Gx))
    expRGy = numpy.exp(numpy.einsum('x,k->xk', -2j*numpy.pi*numpy.arange(mesh[1]), Gy))
    expRGz = numpy.exp(numpy.einsum('x,k->xk', -2j*numpy.pi*numpy.arange(mesh[2]), Gz))
    out = numpy.empty(f.shape, dtype=numpy.complex128)
    buf = numpy.empty(mesh, dtype=numpy.complex128)
    for i, fi in enumerate(f):
        buf[:] = fi.reshape(mesh)
        g = lib.dot(buf.reshape(mesh[0],-1).T, expRGx, c=out[i].reshape(-1,mesh[0]))
        g = lib.dot(g.reshape(mesh[1],-1).T, expRGy, c=buf.reshape(-1,mesh[1]))
        g = lib.dot(g.reshape(mesh[2],-1).T, expRGz, c=out[i].reshape(-1,mesh[2]))
    return out.reshape(-1, *mesh)

def _ifftn_blas(g, mesh):
    Gx = numpy.fft.fftfreq(mesh[0])
    Gy = numpy.fft.fftfreq(mesh[1])
    Gz = numpy.fft.fftfreq(mesh[2])
    expRGx = numpy.exp(numpy.einsum('x,k->xk', 2j*numpy.pi*numpy.arange(mesh[0]), Gx))
    expRGy = numpy.exp(numpy.einsum('x,k->xk', 2j*numpy.pi*numpy.arange(mesh[1]), Gy))
    expRGz = numpy.exp(numpy.einsum('x,k->xk', 2j*numpy.pi*numpy.arange(mesh[2]), Gz))
    out = numpy.empty(g.shape, dtype=numpy.complex128)
    buf = numpy.empty(mesh, dtype=numpy.complex128)
    for i, gi in enumerate(g):
        buf[:] = gi.reshape(mesh)
        f = lib.dot(buf.reshape(mesh[0],-1).T, expRGx, alpha=1./mesh[0], c=out[i].reshape(-1,mesh[0]))
        f = lib.dot(f.reshape(mesh[1],-1).T, expRGy, alpha=1./mesh[1], c=buf.reshape(-1,mesh[1]))
        f = lib.dot(f.reshape(mesh[2],-1).T, expRGz, alpha=1./mesh[2], c=out[i].reshape(-1,mesh[2]))
    return out.reshape(-1, *mesh)

if FFT_ENGINE == 'FFTW':
    # pyfftw is slower than numpy.fft in most cases
    try:
        import pyfftw
        pyfftw.interfaces.cache.enable()
        numpyroc = lib.num_threads()
        def _fftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.fftn(a, axes=(1,2,3), threads=numpyroc)
        def _ifftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.ifftn(a, axes=(1,2,3), threads=numpyroc)
    except ImportError:
        def _fftn_wrapper(a):
            return numpy.fft.fftn(a, axes=(1,2,3))
        def _ifftn_wrapper(a):
            return numpy.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY':
    def _fftn_wrapper(a):
        return numpy.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        return numpy.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY+BLAS':
    _EXCLUDE = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97,101,103,107,109,113,127,131,137,139,149,151,157,163,
                167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,
                257,263,269,271,277,281,283,293]
    _EXCLUDE = set(_EXCLUDE + [n*2 for n in _EXCLUDE] + [n*3 for n in _EXCLUDE])
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _fftn_blas(a, mesh)
        else:
            return numpy.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _ifftn_blas(a, mesh)
        else:
            return numpy.fft.ifftn(a, axes=(1,2,3))

#?elif:  # 'FFTW+BLAS'
else:  # 'BLAS'
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        return _fftn_blas(a, mesh)
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        return _ifftn_blas(a, mesh)


def fft(f, mesh):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    if f.size == 0:
        return numpy.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = _fftn_wrapper(f3d)
    ngrids = numpy.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

def ifft(g, mesh):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    if g.size == 0:
        return numpy.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = _ifftn_wrapper(g3d)
    ngrids = numpy.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)


def fftk(f, mesh, expmikr):
    r'''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, mesh)


def ifftk(g, mesh, expikr):
    r'''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, mesh) * expikr


def get_coulG(cell, k=numpy.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        k : (3,) ndarray
            k-point
        exx : bool or str
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngrids,) ndarray
            The Coulomb kernel.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.
        omega : float
            Enable Coulomb kernel erf(|omega|*r12)/r12 if omega > 0
            and erfc(|omega|*r12)/r12 if omega < 0.
            Note this parameter is slightly different to setting cell.omega
            for the treatment of exxdiv (at G0).  cell.omega affects Ewald
            probe charge at G0. It is used mostly with RSH functional for
            the long-range part of HF exchange. This parameter is used by
            real-space JK builder which requires Ewald probe charge to be
            computed with regular Coulomb interaction (1/r12) while the rest
            coulG is scaled as long-range Coulomb kernel.
    '''
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
        exxdiv = mf.exxdiv

    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    if abs(k).sum() > 1e-9:
        kG = k + Gv
    else:
        kG = Gv

    equal2boundary = numpy.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', numpy.asarray(mesh)//2+0.5, b)
        box_edge_T = stop_grad(box_edge).T
        assert(all(numpy.linalg.solve(box_edge_T, stop_grad(k)).round(9).astype(int)==0))
        reduced_coords = numpy.linalg.solve(box_edge_T, stop_grad(kG).T).T.round(9)
        on_edge = reduced_coords.astype(int)
        if cell.dimension >= 1:
            equal2boundary |= reduced_coords[:,0] == 1
            equal2boundary |= reduced_coords[:,0] ==-1
            vx = 2 * box_edge[0]
            kG = ops.index_add(kG, ops.index[on_edge[:,0]== 1], -vx)
            kG = ops.index_add(kG, ops.index[on_edge[:,0]==-1],  vx)
        if cell.dimension >= 2:
            equal2boundary |= reduced_coords[:,1] == 1
            equal2boundary |= reduced_coords[:,1] ==-1
            vy = 2 * box_edge[1]
            kG = ops.index_add(kG, ops.index[on_edge[:,1]== 1], -vy)
            kG = ops.index_add(kG, ops.index[on_edge[:,1]==-1],  vy)
        if cell.dimension == 3:
            equal2boundary |= reduced_coords[:,2] == 1
            equal2boundary |= reduced_coords[:,2] ==-1
            vz = 2 * box_edge[2]
            kG = ops.index_add(kG, ops.index[on_edge[:,2]== 1], -vz)
            kG = ops.index_add(kG, ops.index[on_edge[:,2]==-1],  vz)

    absG2 = np.einsum('gi,gi->g', kG, kG)

    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*numpy.pi))**(1./3)
        with numpy.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*numpy.pi/absG2*(1.0 - numpy.cos(numpy.sqrt(absG2)*Rc))
        coulG[absG2==0] = 4*numpy.pi*0.5*Rc**2

        if cell.dimension < 3:
            raise NotImplementedError

    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        assert(cell.dimension == 3)
        if not getattr(mf, '_ws_exx', None):
            mf._ws_exx = precompute_exx(cell, kpts)
        exx_alpha = mf._ws_exx['alpha']
        exx_kcell = mf._ws_exx['kcell']
        exx_q = mf._ws_exx['q']
        exx_vq = mf._ws_exx['vq']

        with numpy.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*numpy.pi/absG2*(1.0 - numpy.exp(-absG2/(4*exx_alpha**2)))
        coulG[absG2==0] = numpy.pi / exx_alpha**2
        # Index k+Gv into the precomputed vq and add on
        gxyz = numpy.dot(kG, exx_kcell.lattice_vectors().T)/(2*numpy.pi)
        gxyz = gxyz.round(decimals=6).astype(int)
        mesh = numpy.asarray(exx_kcell.mesh)
        gxyz = (gxyz + mesh)%mesh
        qidx = (gxyz[:,0]*mesh[1] + gxyz[:,1])*mesh[2] + gxyz[:,2]
        #qidx = [numpy.linalg.norm(exx_q-kGi,axis=1).argmin() for kGi in kG]
        maxqv = abs(exx_q).max(axis=0)
        is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
        coulG = coulG.astype(exx_vq.dtype)
        coulG[is_lt_maxqv] += exx_vq[qidx[is_lt_maxqv]]

        if cell.dimension < 3:
            raise NotImplementedError

    else:
        # Ewald probe charge method to get the leading term of the finite size
        # error in exchange integrals

        G0_idx = np.where(absG2==0)[0]
        if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
            #with numpy.errstate(divide='ignore'):
            #    coulG = 4*numpy.pi/absG2
            #    coulG = ops.index_update(coulG, ops.index[G0_idx], 0)
            coulG = 4*numpy.pi/np.where(absG2>1e-16, absG2, 1e200)

        elif cell.dimension == 2:
            # The following 2D analytical fourier transform is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            b = cell.reciprocal_vectors()
            Ld2 = numpy.pi/numpy.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = numpy.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - numpy.cos(Gz*Ld2) * numpy.exp(-Gp*Ld2)
            with numpy.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*numpy.pi/absG2
            if len(G0_idx) > 0:
                coulG[G0_idx] = -2*numpy.pi*Ld2**2 #-pi*L_z^2/2

        elif cell.dimension == 1:
            logger.warn(cell, 'No method for PBC dimension 1, dim-type %s.'
                        '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                        cell.low_dim_ft_type)
            raise NotImplementedError

            # Carlo A. Rozzi, PRB 73, 205119 (2006)
            a = cell.lattice_vectors()
            # Rc is the cylindrical radius
            Rc = numpy.sqrt(cell.vol / numpy.linalg.norm(a[0])) / 2
            Gx = abs(kG[:,0])
            Gp = numpy.linalg.norm(kG[:,1:], axis=1)
            with numpy.errstate(divide='ignore', invalid='ignore'):
                weights = 1 + Gp*Rc * scipy.special.j1(Gp*Rc) * scipy.special.k0(Gx*Rc)
                weights -= Gx*Rc * scipy.special.j0(Gp*Rc) * scipy.special.k1(Gx*Rc)
                coulG = 4*numpy.pi/absG2 * weights
                # TODO: numerical integation
                # coulG[Gx==0] = -4*numpy.pi * (dr * r * scipy.special.j0(Gp*r) * numpy.log(r)).sum()
            if len(G0_idx) > 0:
                coulG[G0_idx] = -numpy.pi*Rc**2 * (2*numpy.log(Rc) - 1)

        # The divergent part of periodic summation of (ii|ii) integrals in
        # Coulomb integrals were cancelled out by electron-nucleus
        # interaction. The periodic part of (ii|ii) in exchange cannot be
        # cancelled out by Coulomb integrals. Its leading term is calculated
        # using Ewald probe charge (the function madelung below)
        if cell.dimension > 0 and exxdiv == 'ewald' and len(G0_idx) > 0:
            coulG[G0_idx] += Nk*cell.vol*madelung(cell, kpts)

    if equal2boundary.sum() > 1e-9:
        coulG = ops.index_update(coulG, ops.index[equal2boundary], 0)

    # Scale the coulG kernel for attenuated Coulomb integrals.
    # * omega is used by RealSpaceJKBuilder which requires ewald probe charge
    # being evaluated with regular Coulomb interaction (1/r12).
    # * cell.omega, which affects the ewald probe charge, is often set by
    # DFT-RSH functionals to build long-range HF-exchange for erf(omega*r12)/r12
    if omega is not None:
        if omega > 0:
            coulG *= np.exp(-.25/omega**2 * absG2)
        else:
            coulG *= (1 - np.exp(-.25/omega**2 * absG2))
    elif cell.omega != 0:
        coulG *= np.exp(-.25/cell.omega**2 * absG2)

    return coulG

def precompute_exx(cell, kpts):
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.dft import gen_grid
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    log.debug("# Precomputing Wigner-Seitz EXX kernel")
    Nk = get_monkhorst_pack_size(cell, kpts)
    log.debug("# Nk = %s", Nk)

    kcell = pbcgto.Cell()
    kcell.atom = 'H 0. 0. 0.'
    kcell.spin = 1
    kcell.unit = 'B'
    kcell.verbose = 0
    kcell.a = cell.lattice_vectors() * Nk
    Lc = 1.0/lib.norm(numpy.linalg.inv(kcell.a), axis=0)
    log.debug("# Lc = %s", Lc)
    Rin = Lc.min() / 2.0
    log.debug("# Rin = %s", Rin)
    # ASE:
    alpha = 5./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
    log.info("WS alpha = %s", alpha)
    kcell.mesh = numpy.array([4*int(L*alpha*3.0) for L in Lc])  # ~ [120,120,120]
    # QE:
    #alpha = 3./Rin * numpy.sqrt(0.5)
    #kcell.mesh = (4*alpha*numpy.linalg.norm(kcell.a,axis=1)).astype(int)
    log.debug("# kcell.mesh FFT = %s", kcell.mesh)
    rs = gen_grid.gen_uniform_grids(kcell)
    kngs = len(rs)
    log.debug("# kcell kngs = %d", kngs)
    corners_coord = lib.cartesian_prod(([0, 1], [0, 1], [0, 1]))
    corners = numpy.dot(corners_coord, kcell.a)
    #vR = numpy.empty(kngs)
    #for i, rv in enumerate(rs):
    #    # Minimum image convention to corners of kcell parallelepiped
    #    r = lib.norm(rv-corners, axis=1).min()
    #    if numpy.isclose(r, 0.):
    #        vR[i] = 2*alpha / numpy.sqrt(numpy.pi)
    #    else:
    #        vR[i] = scipy.special.erf(alpha*r) / r
    r = numpy.min([lib.norm(rs-c, axis=1) for c in corners], axis=0)
    vR = scipy.special.erf(alpha*r) / (r+1e-200)
    vR[r<1e-9] = 2*alpha / numpy.sqrt(numpy.pi)
    vG = (kcell.vol/kngs) * fft(vR, kcell.mesh)

    if abs(vG.imag).max() > 1e-6:
        # vG should be real in regular lattice. If imaginary part is observed,
        # this probably means a ws cell was built from a unconventional
        # lattice. The SR potential erfc(alpha*r) for the charge in the center
        # of ws cell decays to the region out of ws cell. The Ewald-sum based
        # on the minimum image convention cannot be used to build the kernel
        # Eq (12) of PRB 87, 165122
        raise RuntimeError('Unconventional lattice was found')

    ws_exx = {'alpha': alpha,
              'kcell': kcell,
              'q'    : kcell.Gv,
              'vq'   : vG.real.copy()}
    log.debug("# Finished precomputing")
    return ws_exx


def madelung(cell, kpts):
    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = copy.copy(cell)
    if hasattr(ecell, "coords"):
        ecell.coords = None
    ecell._atm = numpy.array([[1, cell._env.size, 0, 0, 0, 0]])
    ecell._env = numpy.append(cell._env, [0., 0., 0.])
    ecell.unit = 'B'
    #ecell.verbose = 0
    ecell.a = numpy.einsum('xi,x->xi', cell.lattice_vectors(), Nk)
    ecell.mesh = numpy.asarray(cell.mesh) * Nk

    if cell.omega == 0:
        ew_eta, ew_cut = ecell.get_ewald_params(cell.precision, ecell.mesh)
        lib.logger.debug1(cell, 'Monkhorst pack size %s ew_eta %s ew_cut %s',
                          Nk, ew_eta, ew_cut)
        return -2*ecell.ewald(ew_eta, ew_cut)

    else:
        # cell.ewald function does not use the Coulomb kernel function
        # get_coulG. When computing the nuclear interactions with attenuated
        # Coulomb operator, the Ewald summation technique is not needed
        # because the Coulomb kernel 4pi/G^2*exp(-G^2/4/omega**2) decays
        # quickly.
        Gv, Gvbase, weights = ecell.get_Gv_weights(ecell.mesh)
        coulG = get_coulG(ecell, Gv=Gv)
        ZSI = numpy.einsum("i,ij->j", ecell.atom_charges(), ecell.get_SI(Gv))
        return -numpy.einsum('i,i,i->', ZSI.conj(), ZSI, coulG*weights).real


def get_monkhorst_pack_size(cell, kpts):
    skpts = cell.get_scaled_kpts(kpts).round(decimals=6)
    Nk = numpy.array([len(numpy.unique(ki)) for ki in skpts.T])
    return Nk


def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.
    The translation vectors can be used for the lattice summation.'''
    a = cell.lattice_vectors()
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = lib.norm(stop_grad(b), axis=1)

    if nimgs is None:
        if rcut is None:
            rcut = cell.rcut
        # For atoms outside the cell, distance between certain basis of nearby
        # images may be smaller than rcut threshold even the corresponding Ls is
        # larger than rcut. The boundary penalty ensures that Ls would be able to
        # cover the basis that sitting out of the cell.
        # See issue https://github.com/pyscf/pyscf/issues/1017
        scaled_atom_coords = stop_grad(cell.atom_coords()).dot(stop_grad(b).T)
        boundary_penalty = numpy.max([abs(scaled_atom_coords).max(axis=0),
                                      abs(1 - scaled_atom_coords).max(axis=0)], axis=0)
        nimgs = numpy.ceil(rcut * heights_inv + boundary_penalty).astype(int)
    else:
        rcut = max((numpy.asarray(nimgs))/heights_inv)

    if dimension is None:
        dimension = cell.dimension
    if dimension == 0:
        nimgs = [0, 0, 0]
    elif dimension == 1:
        nimgs = [nimgs[0], 0, 0]
    elif dimension == 2:
        nimgs = [nimgs[0], nimgs[1], 0]

    Ts = lib.cartesian_prod((numpy.arange(-nimgs[0], nimgs[0]+1),
                             numpy.arange(-nimgs[1], nimgs[1]+1),
                             numpy.arange(-nimgs[2], nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    if discard:
        Ls = _discard_edge_images(cell, Ls, rcut)
    if PYSCFAD:
        return np.asarray(Ls)
    else:
        return numpy.asarray(Ls, order='C')

def _discard_edge_images(cell, Ls, rcut):
    '''
    Discard images if no basis in the image would contribute to lattice sum.
    '''
    if rcut <= 0:
        return numpy.zeros((1, 3))

    a = stop_grad(cell.lattice_vectors())
    scaled_atom_coords = numpy.linalg.solve(a.T, stop_grad(cell.atom_coords()).T).T
    atom_boundary_max = scaled_atom_coords.max(axis=0)
    atom_boundary_min = scaled_atom_coords.min(axis=0)
    # ovlp_penalty ensures the overlap integrals for atoms in the adjcent
    # images are converged.
    ovlp_penalty = atom_boundary_max - atom_boundary_min
    # atom_boundary_min-1 ensures the values of basis at the grids on the edge
    # of the primitive cell converged
    boundary_max = numpy.ceil(numpy.max([atom_boundary_max  ,  ovlp_penalty], axis=0)).astype(int)
    boundary_min = numpy.floor(numpy.min([atom_boundary_min-1, -ovlp_penalty], axis=0)).astype(int)
    penalty_x = numpy.arange(boundary_min[0], boundary_max[0]+1)
    penalty_y = numpy.arange(boundary_min[1], boundary_max[1]+1)
    penalty_z = numpy.arange(boundary_min[2], boundary_max[2]+1)
    shifts = lib.cartesian_prod([penalty_x, penalty_y, penalty_z]).dot(a)
    Ls_mask = (numpy.linalg.norm(stop_grad(Ls) + shifts[:,None,:], axis=2) < rcut).any(axis=0)
    # cell0 (Ls == 0) should always be included.
    Ls_mask[len(Ls)//2] = True
    return Ls[Ls_mask]


def super_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] supercell of the inumpyut cell
    Note this function differs from :fun:`cell_plus_imgs` that cell_plus_imgs
    creates images in both +/- direction.

    Args:
        cell : instance of :class:`Cell`
        ncopy : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    a = cell.lattice_vectors()
    #:supcell.atom = []
    #:for Lx in range(ncopy[0]):
    #:    for Ly in range(ncopy[1]):
    #:        for Lz in range(ncopy[2]):
    #:            # Using cell._atom guarantees coord is in Bohr
    #:            for atom, coord in cell._atom:
    #:                L = numpy.dot([Lx, Ly, Lz], a)
    #:                supcell.atom.append([atom, coord + L])
    Ts = lib.cartesian_prod((numpy.arange(ncopy[0]),
                             numpy.arange(ncopy[1]),
                             numpy.arange(ncopy[2])))
    Ls = numpy.dot(Ts, a)
    supcell = cell.copy()
    supcell.a = numpy.einsum('i,ij->ij', ncopy, a)
    supcell.mesh = numpy.array([ncopy[0]*cell.mesh[0],
                             ncopy[1]*cell.mesh[1],
                             ncopy[2]*cell.mesh[2]])
    return _build_supcell_(supcell, cell, Ls)


def cell_plus_imgs(cell, nimgs):
    '''Create a supercell via nimgs[i] in each +/- direction, as in get_lattice_Ls().
    Note this function differs from :fun:`super_cell` that super_cell only
    stacks the images in + direction.

    Args:
        cell : instance of :class:`Cell`
        nimgs : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    a = cell.lattice_vectors()
    Ts = lib.cartesian_prod((numpy.arange(-nimgs[0], nimgs[0]+1),
                             numpy.arange(-nimgs[1], nimgs[1]+1),
                             numpy.arange(-nimgs[2], nimgs[2]+1)))
    Ls = numpy.dot(Ts, a)
    supcell = cell.copy()
    supcell.a = numpy.einsum('i,ij->ij', nimgs, a)
    supcell.mesh = numpy.array([(nimgs[0]*2+1)*cell.mesh[0],
                             (nimgs[1]*2+1)*cell.mesh[1],
                             (nimgs[2]*2+1)*cell.mesh[2]])
    return _build_supcell_(supcell, cell, Ls)

def _build_supcell_(supcell, cell, Ls):
    '''
    Construct supcell ._env directly without calling supcell.build() method.
    This reserves the basis contraction coefficients defined in cell
    '''
    nimgs = len(Ls)
    symbs = [atom[0] for atom in cell._atom] * nimgs
    coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    supcell.atom = supcell._atom = list(zip(symbs, coords.reshape(-1,3).tolist()))
    supcell.unit = 'B'

    # Do not call supcell.build() since it may normalize the basis contraction
    # coefficients
    _env = numpy.append(cell._env, coords.ravel())
    _atm = numpy.repeat(cell._atm[None,:,:], nimgs, axis=0)
    _atm = _atm.reshape(-1, ATM_SLOTS)
    # Point to the corrdinates appended to _env
    _atm[:,PTR_COORD] = cell._env.size + numpy.arange(nimgs * cell.natm) * 3

    _bas = numpy.repeat(cell._bas[None,:,:], nimgs, axis=0)
    # For atom pointers in each image, shift natm*image_id
    _bas[:,:,ATOM_OF] += numpy.arange(nimgs)[:,None] * cell.natm

    supcell._atm = numpy.asarray(_atm, dtype=numpy.int32)
    supcell._bas = numpy.asarray(_bas.reshape(-1, BAS_SLOTS), dtype=numpy.int32)
    supcell._env = _env
    return supcell


def cutoff_to_mesh(a, cutoff):
    r'''
    Convert KE cutoff to FFT-mesh

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args:
        a : (3,3) ndarray
            The real-space unit cell lattice vectors. Each row represents a
            lattice vector.
        cutoff : float
            KE energy cutoff in a.u.

    Returns:
        mesh : (3,) array
    '''
    b = 2 * numpy.pi * numpy.linalg.inv(a.T)
    cutoff = cutoff * _cubic2nonorth_factor(a)
    mesh = numpy.ceil(numpy.sqrt(2*cutoff)/lib.norm(b, axis=1) * 2).astype(int)
    return mesh

def mesh_to_cutoff(a, mesh):
    '''
    Convert #grid points to KE cutoff
    '''
    b = 2 * numpy.pi * numpy.linalg.inv(a.T)
    Gmax = lib.norm(b, axis=1) * numpy.asarray(mesh) * .5
    ke_cutoff = Gmax**2/2
    # scale down Gmax to get the real energy cutoff for non-orthogonal lattice
    return ke_cutoff / _cubic2nonorth_factor(a)

def _cubic2nonorth_factor(a):
    '''The factors to transform the energy cutoff from cubic lattice to
    non-orthogonal lattice. Energy cutoff is estimated based on cubic lattice.
    It needs to be rescaled for the non-orthogonal lattice to ensure that the
    minimal Gv vector in the reciprocal space is larger than the required
    energy cutoff.
    '''
    # Using ke_cutoff to set up a sphere, the sphere needs to be completely
    # inside the box defined by Gv vectors
    abase = a / numpy.linalg.norm(a, axis=1)[:,None]
    bbase = numpy.linalg.inv(abase.T)
    overlap = numpy.einsum('ix,ix->i', abase, bbase)
    return 1./overlap**2

def cutoff_to_gs(a, cutoff):
    '''Deprecated.  Replaced by function cutoff_to_mesh.'''
    return [n//2 for n in cutoff_to_mesh(a, cutoff)]

def gs_to_cutoff(a, gs):
    '''Deprecated.  Replaced by function mesh_to_cutoff.'''
    return mesh_to_cutoff(a, [2*n+1 for n in gs])
