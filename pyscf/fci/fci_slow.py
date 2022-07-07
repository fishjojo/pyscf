#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

from pyscf import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import ops
from pyscf.fci import cistring

def contract_1e(f1e, fcivec, norb, nelec):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1 = np.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * ci0[:,str0]
    fcinew = np.dot(f1e.reshape(-1), t1.reshape(-1,na*nb))
    return fcinew.reshape(fcivec.shape)


def contract_2e(eri, fcivec, norb, nelec, opt=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1 = np.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1 = ops.index_add(t1, ops.index[a,i,str1], sign * ci0[str0])
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1 = ops.index_add(t1, ops.index[a,i,:,str1], sign * ci0[:,str0])

    t1 = np.einsum('bjai,aiAB->bjAB', eri.reshape([norb]*4), t1)

    fcinew = np.zeros_like(ci0)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew = ops.index_add(fcinew, ops.index[str1], sign * t1[a,i,str0])
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew = ops.index_add(fcinew, ops.index[:,str1], sign * t1[a,i,:,str0])
    return fcinew.reshape(fcivec.shape)

def contract_2e_hubbard(u, fcivec, norb, nelec, opt=None):
    if isinstance(nelec, (int, np.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    u_aa, u_ab, u_bb = u

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    t1a = np.zeros((norb,na,nb))
    t1b = np.zeros((norb,na,nb))
    fcinew = np.zeros_like(fcivec)

    for addr, s in enumerate(strsa):
        for i in range(norb):
            if s & (1 << i):
                t1a[i,addr] += fcivec[addr]
    for addr, s in enumerate(strsb):
        for i in range(norb):
            if s & (1 << i):
                t1b[i,:,addr] += fcivec[:,addr]

    if u_aa != 0:
        # u * n_alpha^+ n_alpha
        for addr, s in enumerate(strsa):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[addr] += t1a[i,addr] * u_aa
    if u_ab != 0:
        # u * n_alpha^+ n_beta
        for addr, s in enumerate(strsa):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[addr] += t1b[i,addr] * u_ab
        # u * n_beta^+ n_alpha
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[:,addr] += t1a[i,:,addr] * u_ab
    if u_bb != 0:
        # u * n_beta^+ n_beta
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[:,addr] += t1b[i,:,addr] * u_bb
    return fcinew


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, np.integer)):
        nelec = sum(nelec)
    if eri.size != norb**4:
        h2e = ao2mo.restore(1, eri.copy(), norb)
    else:
        h2e = eri
    f1e = h1e - np.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e = ops.index_add(h2e, ops.index[k,k,:,:], f1e)
        h2e = ops.index_add(h2e, ops.index[:,:,k,k], f1e)
    return h2e * fac


def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    occslista = cistring._gen_occslst(range(norb), neleca)
    occslistb = cistring._gen_occslst(range(norb), nelecb)
    if eri.size != norb**4:
        eri = ao2mo.restore(1, eri, norb)
    diagj = np.einsum('iijj->ij', eri)
    diagk = np.einsum('ijji->ij', eri)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5)
    return np.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = np.zeros((na,na))
    ci0 = ops.index_update(ci0, ops.index[0,0], 1.)

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore, c


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)
    rdm1 = np.zeros((norb,norb))
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * np.dot(fcivec[str1],fcivec[str0])
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * np.dot(fcivec[:,str1],fcivec[:,str0])
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)

    rdm1 = np.zeros((norb,norb))
    rdm2 = np.zeros((norb,norb,norb,norb))
    for str0, tab in enumerate(link_index):
        t1 = np.zeros((na,norb,norb))
        for a, i, str1, sign in link_index[str0]:
            t1[:,i,a] += sign * fcivec[str1,:]

        for k, tab in enumerate(link_index):
            for a, i, str1, sign in tab:
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += np.einsum('m,mij->ij', fcivec[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += np.einsum('mij,mkl->jikl', t1, t1)
    return reorder_rdm(rdm1, rdm2)


def reorder_rdm(rdm1, rdm2, inplace=True):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the true 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    if inplace:
        rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    else:
        rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.9766331504361414)
