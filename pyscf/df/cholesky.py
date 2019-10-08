import numpy as np
from pyscf import lib
from pyscf.lib import logger
import ctypes
import math
import time
from pyscf import __config__
from functools import reduce

libcgto = lib.load_library('libcgto')

LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)

class ShlPair():

    def __init__(self):

        self.di = 0
        self.dj = 0
        self.ish = -1
        self.jsh = -1

        self.eri_diag = None
        self.Dmask_ao = None  #for ao pair
        self.Dmask = 0        #for this shell pair

        self.Bmask = 0
        self.Bmask_ao = None

        self.eri_off_ioff = 0

        self.Lpq = None
        self.max_diag = None

    def init_shlpair(self, ish, jsh, ao_loc, shlpair_sym = 's2'):

        self.ish = ish
        self.jsh = jsh
        self.di = ao_loc[ish+1] - ao_loc[ish]
        self.dj = ao_loc[jsh+1] - ao_loc[jsh]

        dim = self.di*self.dj
        if shlpair_sym == 's2' and ish == jsh:
            dim = self.di*(self.di+1)//2

        self.eri_diag = np.empty([dim], dtype=np.double)
        self.Dmask_ao = np.zeros([dim], dtype=int)
        self.Bmask_ao = np.zeros([dim], dtype=int)


    @property
    def shl_ind(self):
        '''Shell indices in the mol object
        '''
        return self.ish, self.jsh



class ShlPairs():

    def __init__(self, mol, shls_slice, shlpair_sym = 's2'):

        from pyscf.gto.moleintor import make_loc

        self.mol = mol
        self.nbas = mol.nao_nr()
        self.ish0 = shls_slice[0]
        self.ish1 = shls_slice[1]
        self.jsh0 = shls_slice[2]
        self.jsh1 = shls_slice[3]
        self.n_ish = self.ish1 - self.ish0
        self.n_jsh = self.jsh1 - self.jsh0
        self.shlpair_sym =  shlpair_sym
        if self.shlpair_sym not in ('s1','s2'):
            raise ValueError('Not supported shlpair_sym: %s.' % shlpair_sym)

        if self.shlpair_sym == 's2' and (self.ish0 != self.jsh0 or self.ish1 != self.jsh1):
            self.shlpair_sym =  's1'
            print("Warning: symmetry lowered due to non-identical shells.")

        self.n_shlpairs = self.n_ish * self.n_jsh
        if self.shlpair_sym == 's2':
            self.n_shlpairs = self.n_ish*(self.n_ish+1)//2

        self.data = None
        self.ao_loc = make_loc(mol._bas, mol._add_suffix('int2e'))
        self.eri_diag_size = 0
        self.sorted_ind = None

    def get_shlpair_ind(self, shl_pair):
        '''
        Input:  shell pair
        Output: indices for shell pairs in the self.data array
        '''
        ish, jsh = shl_pair.shl_ind
        return ish-self.ish0, jsh-self.jsh0


    def init_shlpairs(self, ao_loc=None):

        ish0 = self.ish0
        jsh0 = self.jsh0
        ni = self.n_ish
        nj = self.n_jsh
        if ao_loc is None: ao_loc = self.ao_loc

        if self.shlpair_sym == 's1':
            self.data = np.empty([ni,nj], dtype = object)
            for i,j in lib.mat_loop_2d(ni,nj):
                    ish = i+ish0
                    jsh = j+jsh0
                    self.data[i,j] = ShlPair()
                    self.data[i,j].init_shlpair(ish,jsh,ao_loc,self.shlpair_sym)
                    self.eri_diag_size += self.data[i,j].eri_diag.size

        elif self.shlpair_sym == 's2':
            self.data = np.empty([ni*(ni+1)//2], dtype = object)
            ind = 0
            for i,j in lib.tril_loop(ni):
                    ish = i+ish0
                    jsh = j+jsh0
                    self.data[ind] = ShlPair()
                    self.data[ind].init_shlpair(ish,jsh,ao_loc,self.shlpair_sym)
                    self.eri_diag_size += self.data[ind].eri_diag.size
                    ind += 1

    def get_shlpair(self, i, j):

        if self.shlpair_sym == 's1':
            return self.data[i,j]
        elif self.shlpair_sym == 's2':
            ind = i*(i+1)//2+j
            if (i<j):
                ind = j*(j+1)//2+i 
            return self.data[ind]

    def get_eri_diag(self):

        out = get_eri_diag(self.mol, self)

        ioff = 0
        for i,j in self.ijloop():
            shl_pair = self.get_shlpair(i,j)
            size = shl_pair.eri_diag.size
            shl_pair.eri_diag[:] = out[ioff:ioff+size]
            ioff += size

        return out

    def reorder_shlpairs(self):

        max_diag_values = np.zeros([self.n_shlpairs])
        ind = 0
        for i,j in self.ijloop():
            shl_pair = self.get_shlpair(i,j)
            max_diag_values[ind] = np.amax(shl_pair.eri_diag)
            ind += 1
        self.sorted_ind = np.argsort(-max_diag_values) 

    def ijloop(self):

        if self.shlpair_sym == 's1':
            for i,j in lib.mat_loop_2d(self.n_ish, self.n_jsh):
                yield i,j
        elif self.shlpair_sym == 's2':
            for i,j in lib.tril_loop(self.n_ish):
                yield i,j

    def sorted_ijloop(self):

        if self.sorted_ind is None:
            self.reorder_shlpairs()

        nj = self.n_jsh
        for ind in self.sorted_ind:
            if self.shlpair_sym == 's1':
                i = ind//nj
                j = ind % nj
            elif self.shlpair_sym == 's2': 
                #should return ind directly
                '''
                for ii in range(nj):
                    last = (ii+1)*(ii+2)//2-1
                    if ind <= last:
                        i = ii
                        break
                '''
                i = int(math.ceil((math.sqrt(9+8*ind)-3)/2))
                j = ind - i*(i+1)//2 
            yield i,j


    def sorted_shlpairs(self):

        for i, j in self.sorted_ijloop(): 
            yield self.get_shlpair(i,j)



    def make_eri_offdiag(self, ij_ind, kl_ind):

        nao_r = 0
        for ind in range(kl_ind.shape[0]):
            k = kl_ind[ind,0]
            l = kl_ind[ind,1]
            kl_pair = self.get_shlpair(k,l)
            nao_kl = kl_pair.eri_diag.size
            nao_r += nao_kl

        nao_l = 0
        eri_size = 0
        ioff = np.zeros([ij_ind.shape[0]],dtype = np.int32)
        for ind in range(ij_ind.shape[0]):
            ioff[ind] = eri_size
            i = ij_ind[ind,0]
            j = ij_ind[ind,1]
            ij_pair = self.get_shlpair(i,j)
            nao_ij = ij_pair.eri_diag.size
            nao_l += nao_ij
            eri_size += nao_ij * nao_r

        ijshl = np.asarray(ij_ind[:,2:4], dtype=np.int32, order="C")
        klshl = np.asarray(kl_ind[:,2:4], dtype=np.int32, order="C")
        eri = get_eri_offdiag(self.mol, self, ijshl, klshl, nao_r, ioff, eri_size)

        return eri, nao_l, nao_r

    def make_eri_shlslice_shlpair(self, ij_shls_slice, kl_ind):

        nao_r = 0
        for ind in range(kl_ind.shape[0]):
            k = kl_ind[ind,0]
            l = kl_ind[ind,1]
            kl_pair = self.get_shlpair(k,l)
            nao_kl = kl_pair.eri_diag.size
            nao_r += nao_kl

        ish0 = ij_shls_slice[0]
        ish1 = ij_shls_slice[1]
        jsh0 = ij_shls_slice[0]
        jsh1 = ij_shls_slice[1]
        di = self.ao_loc[ish1] - self.ao_loc[ish0]
        dj = self.ao_loc[jsh1] - self.ao_loc[jsh0]

        if self.shlpair_sym == 's1':
            nao_l = di*dj
        elif self.shlpair_sym == 's2':
            nao_l = di*(di+1)//2

        eri_size = nao_l * nao_r

        klshl = np.asarray(kl_ind[:,2:4], dtype=np.int32, order="C")
        eri = get_eri_shlslice_shlpair(self.mol, self, ij_shls_slice, klshl, nao_r, eri_size) 

        return eri, nao_l, nao_r 


def get_eri_diag(mol, shl_pairs):

    from pyscf.gto.moleintor import make_cintopt

    cput0 = (time.clock(), time.time())
 
    atm = np.asarray(mol._atm, dtype=np.int32, order='C')
    bas = np.asarray(mol._bas, dtype=np.int32, order='C')
    env = np.asarray(mol._env, dtype=np.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    intor_name = mol._add_suffix('int2e')
    ao_loc = shl_pairs.ao_loc

    comp = 1
    ish0 = shl_pairs.ish0
    ish1 = shl_pairs.ish1
    jsh0 = shl_pairs.jsh0
    jsh1 = shl_pairs.jsh1
    shls_slice = (ish0, ish1, jsh0, jsh1, ish0, ish1, jsh0, jsh1)

    drv = libcgto.GTOnr2e_fill_drv
    fill = getattr(libcgto, 'GTOnr2e_fill_diag_'+ shl_pairs.shlpair_sym)

    nelem = shl_pairs.eri_diag_size
    out = np.ndarray([nelem],dtype=np.double) 

    cintopt = make_cintopt(atm, bas, env, intor_name)
    prescreen = lib.c_null_ptr()
    drv(getattr(libcgto, intor_name), fill, prescreen,\
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),\
        (ctypes.c_int*8)(*shls_slice),\
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,\
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

    cput1 = logger.timer(mol, 'eri_diag', *cput0)

    return out


def get_eri_offdiag(mol, shl_pairs, p_shlpair_ind, q_shlpair_ind, nao_pair_kl, ioff, eri_size):

    from pyscf.gto.moleintor import make_cintopt

    cput0 = (time.clock(), time.time())

    atm = np.asarray(mol._atm, dtype=np.int32, order='C')
    bas = np.asarray(mol._bas, dtype=np.int32, order='C')
    env = np.asarray(mol._env, dtype=np.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    c_p_shlpair_ind = p_shlpair_ind.ctypes.data_as(ctypes.c_void_p)
    c_q_shlpair_ind = q_shlpair_ind.ctypes.data_as(ctypes.c_void_p)
    c_ioff = ioff.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    intor_name = mol._add_suffix('int2e')
    ao_loc = shl_pairs.ao_loc

    sym = shl_pairs.shlpair_sym
    has_s2_sym = 0
    if sym == 's2': has_s2_sym = 1

    drv = libcgto.GTOnr2e_fill_shl
    fill = getattr(libcgto, 'GTOnr2e_fill_offdiag_'+ shl_pairs.shlpair_sym)

    comp = 1
    nij = p_shlpair_ind.shape[0]
    nkl = q_shlpair_ind.shape[0]

    eri = np.ndarray([eri_size],dtype = np.double)
    cintopt = make_cintopt(atm, bas, env, intor_name) #need recode
    prescreen = lib.c_null_ptr()
    drv(getattr(libcgto, intor_name), fill, prescreen,\
        eri.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp), ctypes.c_int(has_s2_sym),\
        c_p_shlpair_ind, ctypes.c_int(nij),\
        c_q_shlpair_ind, ctypes.c_int(nkl),\
        ctypes.c_int(nao_pair_kl), c_ioff,\
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,\
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

    cput1 = logger.timer(mol, 'eri_offdiag', *cput0)

    return eri


def get_eri_shlslice_shlpair(mol, shl_pairs, ij_shls_slice, kl_shlpair_ind, nao_pair_kl, eri_size):

    from pyscf.gto.moleintor import make_cintopt

    cput0 = (time.clock(), time.time())

    atm = np.asarray(mol._atm, dtype=np.int32, order='C')
    bas = np.asarray(mol._bas, dtype=np.int32, order='C')
    env = np.asarray(mol._env, dtype=np.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = shl_pairs.ao_loc
    intor_name = mol._add_suffix('int2e')
    comp = 1

    c_kl_shlpair_ind = kl_shlpair_ind.ctypes.data_as(ctypes.c_void_p)
    nkl = kl_shlpair_ind.shape[0]

    drv = libcgto.GTOnr2e_fill_shlslice_shlpair
    fill = getattr(libcgto, 'GTOnr2e_fill_shlslice_shlpair_' + shl_pairs.shlpair_sym)

    prescreen = lib.c_null_ptr()
    cintopt = make_cintopt(atm, bas, env, intor_name)
    eri = np.ndarray([eri_size],dtype = np.double)

    ij_shls_slice = tuple(ij_shls_slice)
    drv(getattr(libcgto, intor_name), fill, prescreen,\
        eri.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),\
        (ctypes.c_int*4)(*ij_shls_slice),\
        c_kl_shlpair_ind, ctypes.c_int(nkl), ctypes.c_int(nao_pair_kl),\
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,\
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)


    cput1 = logger.timer(mol, 'eri_shlslice_shlpair', *cput0)

    return eri


class df_cholesky(lib.StreamObject):

    '''
    Performs Cholesky decomposition for two electron integrals
    Reference: JCP, 150, 194112 (2019)
    '''

    def __init__(self, mol, shls_slice, shlpair_sym='s2', tau=1e-8, sigma=1e-2, dimQ=10):

        self.mol = mol
        self.tau = tau
        self.sigma = sigma
        self.dimQ = dimQ
        self.shlpair_sym =  shlpair_sym
        self.shl_pairs = ShlPairs(mol, shls_slice, shlpair_sym = self.shlpair_sym)
        self.shl_pairs.init_shlpairs()
        self.ij_shls_slice = shls_slice
        self.verbose = mol.verbose

    def kernel(self):

        self.step1()
        return self.step2()

    def step1(self):

        self.shl_pairs.get_eri_diag()

        while True:

            cput0 = (time.clock(), time.time())

            Dmax = 0.0
            D = []
            Q = []
            nq = 0
            ioff = 0
            self.shl_pairs.sorted_ind = None
            for ind, shl_pair in enumerate(self.shl_pairs.sorted_shlpairs()):

                max_diag = np.amax(shl_pair.eri_diag)
                if shl_pair.max_diag is None: shl_pair.max_diag = max_diag
                if max_diag < self.tau:
                    shl_pair.Dmask = 0
                    shl_pair.Dmask_ao[:] = 0
                    shl_pair.Lpq = None
                    continue

                if ind == 0: Dmax = max_diag

                ish, jsh = shl_pair.shl_ind
                i, j = self.shl_pairs.get_shlpair_ind(shl_pair)

                shl_pair.Dmask = 1
                shl_pair.Dmask_ao[:] = 0
                p = shl_pair.eri_diag > self.tau
                shl_pair.Dmask_ao[p] = 1

                D.append((i,j,ish,jsh))

                shl_pair.eri_off_ioff = ioff
                ioff += shl_pair.eri_diag.size

                if nq < self.dimQ:
                    q = (shl_pair.eri_diag > self.sigma * Dmax) * (shl_pair.Dmask_ao == 1)
                    if q.any() == True:
                        shl_pair.Dmask == 2
                        shl_pair.Dmask_ao[q] = 2
                        Q.append((i,j,ish,jsh))
                        nq += np.sum(shl_pair.Dmask_ao == 2)

            if nq == 0: break

            p_shlpair_ind = np.asarray(D)
            q_shlpair_ind = np.asarray(Q)

            cput1 = logger.timer(self, 'before eri_offdiag', *cput0)

            eri_offdiag, nao_ij, nao_kl = self.shl_pairs.make_eri_offdiag(p_shlpair_ind, q_shlpair_ind)
            eri_offdiag = eri_offdiag.reshape((nao_ij, nao_kl))

            cput0 = (time.clock(), time.time())

            tmp = []
            for q in Q:
                i = q[0]
                j = q[1]
                shl_pair = self.shl_pairs.get_shlpair(i,j)

                for ind, val in enumerate(shl_pair.Dmask_ao):
                    Dq = -999999.0
                    if val == 2: Dq = shl_pair.eri_diag[ind]
                    tmp.append((i, j, ind, Dq))


            tmp1=np.asarray(tmp)[:,-1]
            sorted_q = np.argsort(-tmp1)


            for p in D:
                shl_pair = self.shl_pairs.get_shlpair(p[0],p[1])
                if shl_pair.Lpq is None: continue
                ioff = shl_pair.eri_off_ioff
                size = shl_pair.eri_diag.size

                LL = np.zeros([size, nao_kl], dtype=np.double)
                for ind, q in enumerate(tmp):
                    if q[-1] < -9999: continue
                    shl_pair_q = self.shl_pairs.get_shlpair(q[0],q[1])
                    if shl_pair_q.Lpq is None: continue

                    LL[:,ind:ind+1] = np.dot(shl_pair.Lpq, shl_pair_q.Lpq[q[2]:q[2]+1,:].T)
                eri_offdiag[ioff:ioff+size,:] -= LL


            Lpq = np.empty([nao_ij, nq], dtype=np.double)
            iq = 0
            for ind in sorted_q:
                q = tmp[ind]
                if q[-1] < -9999: break

                Mpq = eri_offdiag[:,ind]
                Mqq = -1.0
                shl_pair = self.shl_pairs.get_shlpair(q[0],q[1])
                ioff = shl_pair.eri_off_ioff
                q_left = ioff + q[2]
                Mqq = eri_offdiag[q_left,ind]

                Ltmp = np.zeros([nao_ij], dtype=np.double)
                for J in range(0,iq):
                    Ltmp += Lpq[:,J] * Lpq[q_left, J]

                Lpq[:,iq] = (Mpq - Ltmp)/math.sqrt(Mqq)


                shl_pair.Dmask_ao[q[2]] = 1 #remove q from Q

                shl_pair.Bmask = 1
                shl_pair.Bmask_ao[q[2]] = 1


                for p in D:
                    shl_pair = self.shl_pairs.get_shlpair(p[0],p[1])
                    ioff = shl_pair.eri_off_ioff
                    size = shl_pair.eri_diag.size
                    shl_pair.eri_diag -= Lpq[ioff:ioff+size,iq]**2

                iq += 1


            for p in D:
                shl_pair = self.shl_pairs.get_shlpair(p[0],p[1])
                ioff = shl_pair.eri_off_ioff
                size = shl_pair.eri_diag.size
                if shl_pair.Lpq is None:
                    shl_pair.Lpq = Lpq[ioff:ioff+size,:].copy() 
                else:
                    shl_pair.Lpq = np.append(shl_pair.Lpq, Lpq[ioff:ioff+size,:], axis=1)

            cput1 = logger.timer(self, 'after eri_offdiag', *cput0)



    def step2(self):

        shlpair_ind = []
        for i, j in self.shl_pairs.ijloop():
            shl_pair = self.shl_pairs.get_shlpair(i,j)
            ish = shl_pair.ish
            jsh = shl_pair.jsh

            if shl_pair.Bmask == 1:
                shlpair_ind.append((i,j,ish,jsh))

        shlpair_ind = np.asarray(shlpair_ind)

        eri_S, nao_ij, nao_kl = self.shl_pairs.make_eri_offdiag(shlpair_ind, shlpair_ind)
        eri_S = eri_S.reshape(nao_ij,nao_kl)

        mask = np.ndarray([nao_ij,nao_kl],dtype=bool)
        mask[:,:] = True
        ioff = 0
        izero = 0
        insert_loc = []
        for i, j in self.shl_pairs.ijloop():
            shl_pair = self.shl_pairs.get_shlpair(i,j)
            if shl_pair.Bmask == 1:
                for ind in range(shl_pair.Bmask_ao.size):
                    if shl_pair.Bmask_ao[ind] == 0:
                        izero += 1
                        insert_loc.append(ioff+ind-izero+1)
                        mask[ioff + ind,:]=False
                        mask[:,ioff + ind]=False
                ioff += shl_pair.eri_diag.size

        eri_S_ao = eri_S[mask].reshape((nao_ij-izero,nao_ij-izero))

        try:
            L = np.linalg.cholesky(eri_S_ao)
            tag = 'cd'
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(eri_S_ao)
            idx = w > LINEAR_DEP_THR
            L = v[:,idx]/np.sqrt(w[idx])
            tag = 'eig'
            #print(w)

        if tag == 'cd':
            L = np.linalg.inv(L).T

        print("L.shape = ", L.shape)
        #L = np.insert(L, insert_loc, 0.0, axis = 0)
        #print(L.shape)

        eri, nao_ij, nao_kl = self.shl_pairs.make_eri_shlslice_shlpair(self.ij_shls_slice, shlpair_ind)
        eri = eri.reshape(nao_ij, nao_kl)

        mask = np.ndarray([nao_ij,nao_kl],dtype=bool)
        mask[:,:] = True
        ioff = 0
        izero = 0
        insert_loc = []
        for i, j in self.shl_pairs.ijloop():
            shl_pair = self.shl_pairs.get_shlpair(i,j)
            if shl_pair.Bmask == 1:
                for ind in range(shl_pair.Bmask_ao.size):
                    if shl_pair.Bmask_ao[ind] == 0:
                        izero += 1
                        insert_loc.append(ioff+ind-izero+1)
                        mask[:,ioff + ind]=False
                ioff += shl_pair.eri_diag.size
        eri = eri[mask].reshape((nao_ij, nao_kl-izero))

        cderi = np.dot(eri,L).T

        return cderi
        
