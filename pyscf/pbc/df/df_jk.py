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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Density fitting with Gaussian basis
Ref:
J. Chem. Phys. 147, 164119 (2017)
'''


import copy
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member

def density_fit(mf, auxbasis=None, mesh=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : DF object
    '''
    from pyscf.pbc.df import df
    if with_df is None:
        if getattr(mf, 'kpts', None) is not None:
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))

        if isinstance(kpts, KPoints):
            kpts = kpts.kpts
        with_df = df.DF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh

    mf = copy.copy(mf)
    mf.with_df = with_df
    mf._eri = None
    return mf


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               kderiv=0):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t1 = log.timer_debug1('Init get_j_kpts', *t1)

    kcomp = (kderiv+1)*(kderiv+2)//2
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        # If mydf._cderi is the file that generated from another calculation,
        # guess naux based on the contents of the integral file.
        naux = mydf.get_naoaux()
    else:
        naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band) and not numpy.iscomplexobj(dms) and kderiv == 0

    dmsR = dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    rhoR = numpy.zeros((nset,naux))
    rhoI = numpy.zeros((nset,naux))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    for k, kpt in enumerate(kpts):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).reshape(-1,nao,nao)
            #:rhoR[:,p0:p1] += numpy.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).real
            #:rhoI[:,p0:p1] += numpy.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).imag
            rhoR[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoI[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            if LpqI is not None:
                rhoR[:,p0:p1] -= sign * numpy.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
                rhoI[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            LpqR = LpqI = None
    t1 = log.timer_debug1('get_j pass 1', *t1)

    weight = 1./nkpts
    rhoR *= weight
    rhoI *= weight
    if kderiv > 0:
        vjR = numpy.zeros((nset,nband,kcomp,nao_pair))
        vjI = numpy.zeros((nset,nband,kcomp,nao_pair))
    else:
        vjR = numpy.zeros((nset,nband,nao_pair))
        vjI = numpy.zeros((nset,nband,nao_pair))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, True, kderiv=kderiv):
            if kderiv > 0:
                p0, p1 = p1, p1+LpqR.shape[1]
                vjR[:,k] += numpy.einsum('nL,xLp->nxp', rhoR[:,p0:p1], LpqR)
                if not j_real:
                    vjI[:,k] += numpy.einsum('nL,xLp->nxp', rhoI[:,p0:p1], LpqR)
                    if LpqI is not None:
                        vjR[:,k] -= numpy.einsum('nL,xLp->nxp', rhoI[:,p0:p1], LpqI)
                        vjI[:,k] += numpy.einsum('nL,xLp->nxp', rhoR[:,p0:p1], LpqI)
            else:
                p0, p1 = p1, p1+LpqR.shape[0]
                #:Lpq = (LpqR + LpqI*1j)#.reshape(-1,nao,nao)
                #:vjR[:,k] += numpy.dot(rho[:,p0:p1], Lpq).real
                #:vjI[:,k] += numpy.dot(rho[:,p0:p1], Lpq).imag
                vjR[:,k] += numpy.dot(rhoR[:,p0:p1], LpqR)
                if not j_real:
                    vjI[:,k] += numpy.dot(rhoI[:,p0:p1], LpqR)
                    if LpqI is not None:
                        vjR[:,k] -= numpy.dot(rhoI[:,p0:p1], LpqI)
                        vjI[:,k] += numpy.dot(rhoR[:,p0:p1], LpqI)
            LpqR = LpqI = None
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))
    if kderiv > 0:
        vj_kpts += vj_kpts.transpose(0,2,1).conj()  #FIXME check if correct
        vj_kpts = vj_kpts.reshape(nset,nband,kcomp,nao,nao)
    else:
        vj_kpts = vj_kpts.reshape(nset,nband,nao,nao)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts, kderiv=kderiv)


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None, kderiv=0):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    kcomp = (kderiv+1)*(kderiv+2)//2

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t1 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t1 = log.timer_debug1('Init get_k_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if kderiv > 0:
        vkR = numpy.zeros((nset,nband,kcomp,nao,nao))
        vkI = numpy.zeros((nset,nband,kcomp,nao,nao))
    else:
        vkR = numpy.zeros((nset,nband,nao,nao))
        vkI = numpy.zeros((nset,nband,nao,nao))
    dmsR = numpy.asarray(dms.real, order='C')
    dmsI = numpy.asarray(dms.imag, order='C')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    bufR = numpy.empty((mydf.blockdim*nao**2))
    bufI = numpy.empty((mydf.blockdim*nao**2))
    bufR1 = bufI1 = None
    if kderiv > 0:
        bufR1 = numpy.empty((mydf.blockdim*nao**2))
        bufI1 = numpy.empty((mydf.blockdim*nao**2))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    def make_kpt(ki, kj, swap_2e, inverse_idx=None):
        kpti = kpts[ki]
        kptj = kpts_band[kj]

        for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
            pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
            tmpR = numpy.ndarray((nao,nrow*nao), buffer=LpqR)
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=LpqI)
            pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
            pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            for i in range(nset):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       sign, vkR[i,kj], vkI[i,kj], 1)

            if swap_2e:
                tmpR = tmpR.reshape(nao*nrow,nao)
                tmpI = tmpI.reshape(nao*nrow,nao)
                ki_tmp = ki
                kj_tmp = kj
                if inverse_idx:
                    ki_tmp = inverse_idx[0]
                    kj_tmp = inverse_idx[1]
                for i in range(nset):
                    zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                           dmsR[i,kj_tmp], dmsI[i,kj_tmp], 1, tmpR, tmpI)
                    zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           sign, vkR[i,ki_tmp], vkI[i,ki_tmp], 1)

    def make_kpt_kderiv(ki, kj):
        kpti = kpts[ki]
        kptj = kpts_band[kj]

        blksize = max_memory*1e6/16/((kcomp+1)*nao**2*2)
        blksize /= 2  # For prefetch
        blksize = max(16, min(int(blksize), mydf.blockdim))

        for (LpqR, LpqI, sign), (LpqR_kderiv, LpqI_kderiv, _) in \
            zip(mydf.sr_loop((kpti,kptj), compact=False, blksize=blksize),
                mydf.sr_loop((kpti,kptj), compact=False, blksize=blksize, kderiv=kderiv)):
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
            pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
            pLqR_kc = numpy.ndarray((nao,nrow,nao), buffer=bufR1)
            pLqI_kc = numpy.ndarray((nao,nrow,nao), buffer=bufI1)
            tmpR = numpy.ndarray((nao,nrow*nao), buffer=LpqR)
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=LpqI)
            pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
            pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)


            for kc in range(kcomp):
                pLqR_kc[:] = LpqR_kderiv[kc].reshape(-1,nao,nao).transpose(1,0,2)
                pLqI_kc[:] = LpqI_kderiv[kc].reshape(-1,nao,nao).transpose(1,0,2)
                for i in range(nset):
                    zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR_kc.reshape(nao,-1),
                           pLqI_kc.reshape(nao,-1), 1, tmpR, tmpI)
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                           sign, vkR[i,kj,kc], vkI[i,kj,kc], 1)

    if kderiv > 0:
        for ki in range(nkpts):
            for kj in range(nband):
                make_kpt_kderiv(ki, kj)
                t1 = log.timer_debug1('get_k_kpts: make_kpt ki, kj (%d,%d)'%(ki, kj), *t1)
    else:
        if kpts_band is kpts:  # normal k-points HF/DFT
            for ki in range(nkpts):
                for kj in range(ki):
                    make_kpt(ki, kj, True)
                make_kpt(ki, ki, False)
                t1 = log.timer_debug1('get_k_kpts: make_kpt ki>=kj (%d,*)'%ki, *t1)
        else:
            idx_in_kpts = []
            for kpt in kpts_band:
                idx = member(kpt, kpts)
                if len(idx) > 0:
                    idx_in_kpts.append(idx[0])
                else:
                    idx_in_kpts.append(-1)
            idx_in_kpts_band = []
            for kpt in kpts:
                idx = member(kpt, kpts_band)
                if len(idx) > 0:
                    idx_in_kpts_band.append(idx[0])
                else:
                    idx_in_kpts_band.append(-1)

            for ki in range(nkpts):
                for kj in range(nband):
                    if idx_in_kpts[kj] == -1 or idx_in_kpts[kj] == ki:
                        make_kpt(ki, kj, False)
                    elif idx_in_kpts[kj] < ki:
                        if idx_in_kpts_band[ki] == -1:
                            make_kpt(ki, kj, False)
                        else:
                            make_kpt(ki, kj, True, (idx_in_kpts_band[ki], idx_in_kpts[kj]))
                    else:
                        if idx_in_kpts_band[ki] == -1:
                            make_kpt(ki, kj, False)
                t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts) and kderiv == 0):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    if kderiv > 0:
        vk_kpts += vk_kpts.transpose(0,1,2,4,3).conj()  #FIXME check if correct

    if exxdiv == 'ewald' and kderiv == 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts, kderiv=kderiv)


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    vj = vk = None
    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpt, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpt, kpts_band)
        return vj, vk

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t1 = log.timer_debug1('Init get_jk', *t1)

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)
    kptii = numpy.asarray((kpt,kpt))
    dmsR = dms.real.reshape(nset,nao,nao)
    dmsI = dms.imag.reshape(nset,nao,nao)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))
    if with_j:
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
        buf1R = numpy.empty((mydf.blockdim*nao**2))
        buf2R = numpy.empty((mydf.blockdim*nao**2))
        buf1I = numpy.zeros((mydf.blockdim*nao**2))
        buf2I = numpy.empty((mydf.blockdim*nao**2))
        max_memory *= .5
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    def contract_k(pLqR, pLqI, sign):
        # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
        #:pLq = (LpqR + LpqI.reshape(-1,nao,nao)*1j).transpose(1,0,2)
        #:tmp = numpy.dot(dm, pLq.reshape(nao,-1))
        #:vk += numpy.dot(pLq.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
        nrow = pLqR.shape[1]
        tmpR = numpy.ndarray((nao,nrow*nao), buffer=buf2R)
        if k_real:
            for i in range(nset):
                lib.ddot(dmsR[i], pLqR.reshape(nao,-1), 1, tmpR)
                lib.ddot(pLqR.reshape(-1,nao).T, tmpR.reshape(-1,nao), sign, vkR[i], 1)
        else:
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=buf2I)
            for i in range(nset):
                zdotNN(dmsR[i], dmsI[i], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI, 0)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       sign, vkR[i], vkI[i], 1)
    pLqI = None
    thread_k = None
    for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, False):
        LpqR = LpqR.reshape(-1,nao,nao)
        t1 = log.timer_debug1('        load', *t1)
        if thread_k is not None:
            thread_k.join()
        if with_j:
            #:rho_coeff = numpy.einsum('Lpq,xqp->xL', Lpq, dms)
            #:vj += numpy.dot(rho_coeff, Lpq.reshape(-1,nao**2))
            rhoR  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsR)
            if not j_real:
                LpqI = LpqI.reshape(-1,nao,nao)
                rhoR -= numpy.einsum('Lpq,xpq->xL', LpqI, dmsI)
                rhoI  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsI)
                rhoI += numpy.einsum('Lpq,xpq->xL', LpqI, dmsR)
            vjR += sign * numpy.einsum('xL,Lpq->xpq', rhoR, LpqR)
            if not j_real:
                vjR -= sign * numpy.einsum('xL,Lpq->xpq', rhoI, LpqI)
                vjI += sign * numpy.einsum('xL,Lpq->xpq', rhoR, LpqI)
                vjI += sign * numpy.einsum('xL,Lpq->xpq', rhoI, LpqR)

        t1 = log.timer_debug1('        with_j', *t1)
        if with_k:
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=buf1R)
            pLqR[:] = LpqR.transpose(1,0,2)
            if not k_real:
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=buf1I)
                if LpqI is not None:
                    pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            thread_k = lib.background_thread(contract_k, pLqR, pLqI, sign)
            t1 = log.timer_debug1('        with_k', *t1)
        LpqR = LpqI = pLqR = pLqI = None
    if thread_k is not None:
        thread_k.join()
    thread_k = None

    if with_j:
        if j_real:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj = vj.reshape(dm.shape)
    if with_k:
        if k_real:
            vk = vkR
        else:
            vk = vkR + vkI * 1j
        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        vk = vk.reshape(dm.shape)

    t1 = log.timer('sr jk', *t1)
    return vj, vk


def _format_dms(dm_kpts, kpts):
    nkpts = len(kpts)
    nao = dm_kpts.shape[-1]
    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    if dms.dtype not in (numpy.double, numpy.complex128):
        dms = numpy.asarray(dms, dtype=numpy.double)
    return dms

def _format_kpts_band(kpts_band, kpts):
    if kpts_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpts_band, (-1,3))
    return kpts_band

def _format_jks(v_kpts, dm_kpts, kpts_band, kpts, kderiv=0):
    kcomp = (kderiv+1)*(kderiv+2)//2
    if kpts_band is kpts or kpts_band is None:
        shape = dm_kpts.shape
        if kderiv > 0:
            shape = list(shape)
            shape.insert(-2, kcomp)
        return v_kpts.reshape(shape)
    else:
        if getattr(kpts_band, 'ndim', None) == 1:
            v_kpts = v_kpts[:,0]
# A temporary solution for issue 242. Looking for better ways to sort out the
# dimension of the output
# dm_kpts.shape     kpts.shape     nset
# (Nao,Nao)         (1 ,3)         None
# (Ndm,Nao,Nao)     (1 ,3)         Ndm
# (Nk,Nao,Nao)      (Nk,3)         None
# (Ndm,Nk,Nao,Nao)  (Nk,3)         Ndm
        if dm_kpts.ndim < 3:     # nset=None
            return v_kpts[0]
        elif dm_kpts.ndim == 3 and dm_kpts.shape[0] == kpts.shape[0]:
            return v_kpts[0]
        else:  # dm_kpts.ndim == 4 or kpts.shape[0] == 1:  # nset=Ndm
            return v_kpts

def _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band=None):
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    madelung = tools.pbc.madelung(cell, kpts)
    if kpts is None:
        for i,dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))
    elif numpy.shape(kpts) == (3,):
        if kpts_band is None or is_zero(kpts_band-kpts):
            for i,dm in enumerate(dms):
                vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    elif kpts_band is None or numpy.array_equal(kpts, kpts_band):
        for k in range(len(kpts)):
            for i,dm in enumerate(dms):
                vk[i,k] += madelung * reduce(numpy.dot, (s[k], dm[k], s[k]))
    else:
        for k, kpt in enumerate(kpts):
            for kp in member(kpt, kpts_band.reshape(-1,3)):
                for i,dm in enumerate(dms):
                    vk[i,kp] += madelung * reduce(numpy.dot, (s[k], dm[k], s[k]))


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf

    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    mf = pscf.RHF(cell)
    dm = mf.get_init_guess()
    auxbasis = 'weigend'
    #from pyscf import df
    #auxbasis = df.addons.aug_etb_for_dfbasis(cell, beta=1.5, start_at=0)
    #from pyscf.pbc.df import mdf
    #mf.with_df = mdf.MDF(cell)
    #mf.auxbasis = auxbasis
    mf = density_fit(mf, auxbasis)
    mf.with_df.mesh = (n,) * 3
    vj = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv, with_k=False)[0]
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    vj, vk = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.348163681114187')
    print(numpy.einsum('ij,ji->', mf.get_hcore(cell), dm), 'ref=-75.5758086593503')

    kpts = cell.make_kpts([2]*3)[:4]
    from pyscf.pbc.df import DF
    with_df = DF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.mesh = [n] * 3
    dms = numpy.array([dm]*len(kpts))
    vj, vk = with_df.get_jk(dms, exxdiv=mf.exxdiv, kpts=kpts)
    print(numpy.einsum('ij,ji->', vj[0], dms[0]) - 46.69784067248350)
    print(numpy.einsum('ij,ji->', vj[1], dms[1]) - 46.69814992718212)
    print(numpy.einsum('ij,ji->', vj[2], dms[2]) - 46.69526120279135)
    print(numpy.einsum('ij,ji->', vj[3], dms[3]) - 46.69570739526301)
    print(numpy.einsum('ij,ji->', vk[0], dms[0]) - 37.26974254415191)
    print(numpy.einsum('ij,ji->', vk[1], dms[1]) - 37.27001407288309)
    print(numpy.einsum('ij,ji->', vk[2], dms[2]) - 37.27000643285160)
    print(numpy.einsum('ij,ji->', vk[3], dms[3]) - 37.27010299675364)
