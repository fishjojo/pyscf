/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        int di = 0;
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        for (i = i0; i < i1; i++) {
                di = MAX(di, ao_loc[i+1]-ao_loc[i]);
        }
        return di;
}
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, n;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        int shls[4];
        int cache_size = 0;
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                shls[2] = i;
                shls[3] = i;
                n = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

int GTOmax_dim_shell(int *ao_loc, int* ij_shlp, int nij, int* kl_shlp, int nkl)
{// for non-consecutive shells
    int ish,jsh,ksh,lsh,di=0;
    int max_shl=0;
    for (int ij = 0; ij < nij; ij++){
        ish = ij_shlp[ij*2];
        jsh = ij_shlp[ij*2+1];
        int itmp = ao_loc[ish+1]-ao_loc[ish];
        int jtmp = ao_loc[jsh+1]-ao_loc[jsh];
        if (itmp > di){
            max_shl = ish;
            di = itmp; 
        }
        if (jtmp > di){
            max_shl = jsh;
            di = jtmp;
        }
    }

    for (int kl = 0; kl < nkl; kl++){
        ksh = kl_shlp[kl*2];
        lsh = kl_shlp[kl*2+1];
        int ktmp = ao_loc[ksh+1]-ao_loc[ksh];
        int ltmp = ao_loc[lsh+1]-ao_loc[lsh];
        if (ktmp > di){
            max_shl = ksh;
            di = ktmp;
        }
        if (ltmp > di){
            max_shl = lsh;
            di = ltmp;
        }
    }
    return max_shl;
}
int GTO_cache_size(int (*intor)(), int ish,
                      int *atm, int natm, int *bas, int nbas, double *env)
{// know the shell
    int shls[4];
    shls[0] = ish;
    shls[1] = ish;
    shls[2] = ish;
    shls[3] = ish;

    int cache_size = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
    return cache_size;
}
int GTO_num_bas_pairs_s1(int *ao_loc, int* ij_shlp, int nij)
{
    int n_bas_pairs = 0;
    for (int ij = 0; ij < nij; ij++){
        int ish = ij_shlp[ij*2];
        int jsh = ij_shlp[ij*2+1];
        int di = ao_loc[ish+1]-ao_loc[ish];
        int dj = ao_loc[jsh+1]-ao_loc[jsh];
        n_bas_pairs += di*dj;
    }
    return n_bas_pairs;
}
int GTO_num_bas_pairs_s2(int *ao_loc, int* ij_shlp, int nij)
{
    int n_bas_pairs = 0;
    for (int ij = 0; ij < nij; ij++){
        int ish = ij_shlp[ij*2];
        int jsh = ij_shlp[ij*2+1];
        int di = ao_loc[ish+1]-ao_loc[ish];
        int dj = ao_loc[jsh+1]-ao_loc[jsh];
        if(ish == jsh) 
            n_bas_pairs += di*(di+1)/2;
        else 
            n_bas_pairs += di*dj;
    }
    return n_bas_pairs;
}

/*
 *************************************************
 * 2e AO integrals in s4, s2ij, s2kl, s1
 */

void GTOnr2e_fill_s1(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf, int comp, int ishp, int jshp,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * nj;
        size_t nkl = nk * nl;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0 * nj + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh;
        int shls[4];
        double *eri0, *peri, *buf0, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*nl+l0;
                        for (icomp = 0; icomp < comp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++) {
                                                for (l = 0; l < dl; l++) {
                                                        peri[k*nl+l] = 0;
                                                }
                                        }
                                } }
                                eri0 += neri;
                        }
                }
        } }
}

void GTOnr2e_fill_s2ij(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf, int comp, int ishp, int jshp,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        if (ishp < jshp) {
                return;
        }

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        //int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        //int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * (ni+1) / 2;
        size_t nkl = nk * nl;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0*(i0+1)/2 + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh;
        int shls[4];
        double *eri0, *peri0, *peri, *buf0, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
                                peri0 = eri0;
                                if (ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*nl+l0;
                        for (icomp = 0; icomp < comp; icomp++) {
                                peri0 = eri0;
                                if (ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nl+l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nl+l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
}

void GTOnr2e_fill_s2kl(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf, int comp, int ishp, int jshp,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        //int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        //int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * nj;
        size_t nkl = nk * (nk+1) / 2;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0 * nj + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh, kshp, lshp;
        int shls[4];
        double *eri0, *peri, *buf0, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;

        for (kshp = 0; kshp < ksh1-ksh0; kshp++) {
        for (lshp = 0; lshp <= kshp; lshp++) {
                ksh = kshp + ksh0;
                lsh = lshp + lsh0;
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
                                if (kshp > lshp) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        for (icomp = 0; icomp < comp; icomp++) {
                                if (kshp > lshp) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
}

void GTOnr2e_fill_s4(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf, int comp, int ishp, int jshp,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        if (ishp < jshp) {
                return;
        }

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        //int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        //int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        //int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        //int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * (ni+1) / 2;
        size_t nkl = nk * (nk+1) / 2;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0*(i0+1)/2 + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh, kshp, lshp;
        int shls[4];
        double *eri0, *peri0, *peri, *buf0, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;

        for (kshp = 0; kshp < ksh1-ksh0; kshp++) {
        for (lshp = 0; lshp <= kshp; lshp++) {
                ksh = kshp + ksh0;
                lsh = lshp + lsh0;
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
                                peri0 = eri0;
                                if (kshp > lshp && ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else if (ish > jsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else if (ksh > lsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
                                peri0 = eri0;
                                if (kshp > lshp && ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else if (ish > jsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else if (ksh > lsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
}


void GTOnr2e_fill_diag_s1(int (*intor)(), int (*fprescreen)(),
                          double *eri, double *buf, int comp, int ishp, int jshp,
                          int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;

		int ioff = 0;
		for (int ii=ish0; ii<ish1; ii++){
		for (int jj=jsh0; jj<jsh1; jj++){
			if(ii==ish && jj==jsh) {ii=ish1; break;}
			int di = ao_loc[ii+1] - ao_loc[ii];
			int dj = ao_loc[jj+1] - ao_loc[jj];
			ioff += di*dj;
		}
		}
        eri += ioff;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dijk, dijkl;
        int i, j, k, l;
        int shls[4];
        double *peri, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;
        shls[2] = ish;
        shls[3] = jsh;

		dijk = dij*di;
		dijkl = dij*dij;
	    cache = buf + dijkl * comp;
		if ((*fprescreen)(shls, atm, bas, env) &&
            (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
			peri = eri;
			pbuf = buf;
			for(i=0; i<di; i++){
				    for(j=0; j<dj; j++){
					    k=i; l=j;
					    peri[i*dj+j] = pbuf[i+j*di+k*dij+l*dijk];
				    }
			}
        }
		else{
			for(i=0; i<di; i++){
                    for(j=0; j<dj; j++){
					    peri[i*dj+j] = 0.0;
				    }
			}
		}

        return;
}

void GTOnr2e_fill_diag_s2(int (*intor)(), int (*fprescreen)(),
                          double *eri, double *buf, int comp, int ishp, int jshp,
                          int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{

        if (ishp < jshp) {
                return;
        }

        int ish0 = shls_slice[0];
        int jsh0 = shls_slice[2];

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;

		int ioff = 0;
		for (int ii=ish0; ii<=ish; ii++){
		for (int jj=jsh0; jj<=ii; jj++){
			if(ii==ish && jj==jsh) break;
			int di = ao_loc[ii+1] - ao_loc[ii];
			int dj = ao_loc[jj+1] - ao_loc[jj];
            if(ii==jj) {ioff += di*(di+1)/2;}
			else {ioff += di*dj;}
		}
		}
        eri += ioff;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dijk, dijkl;
        int i, j, k, l;
        int shls[4];
        double *peri, *pbuf, *cache;

        shls[0] = ish;
        shls[1] = jsh;
        shls[2] = ish;
        shls[3] = jsh;

		dijk = dij*di;
		dijkl = dij*dij;
	    cache = buf + dijkl * comp;
		if ((*fprescreen)(shls, atm, bas, env) &&
            (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
			peri = eri;
			pbuf = buf;
            if(ish != jsh){
			    for(i=0; i<di; i++){
				    for(j=0; j<dj; j++){
					    k=i; l=j;
					    peri[i*dj+j] = pbuf[i+j*di+k*dij+l*dijk];
				    }
			    }
            }
            else{
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<=i; j++){
                        k=i; l=j;
                        peri[ind] = pbuf[i+j*di+k*dij+l*dijk];
                        ind++;
                    }
                }
            }
        }
		else{
            if(ish != jsh){
			    for(i=0; i<di; i++){
                    for(j=0; j<dj; j++){
					    peri[i*dj+j] = 0.0;
				    }
			    }
            }
            else{
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<=i; j++){
                        peri[ind] = 0.0;
                        ind++;
                    }
                }
            }
		}

        return;
}


void GTOnr2e_fill_offdiag_s2(int (*intor)(), int (*fprescreen)(),
                          double *eri, double *buf, int comp, int ish, int jsh,
                          int* kl_shlp, int nkl, int nao_pair_kl, int *ao_loc, CINTOpt *cintopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{

    int ksh,lsh,shls[4];
    shls[0] = ish;
    shls[1] = jsh;

    int di,dj,dk,dl;
    di = ao_loc[ish+1] - ao_loc[ish];
    dj = ao_loc[jsh+1] - ao_loc[jsh];

    int i,j,k,l;
    int dij, dijk, dijkl;
    dij = di*dj;

    double *cache, *eri0, *peri, *pbuf;
    int kl_off = 0;
    for (int kl = 0; kl < nkl; kl++){
        ksh = kl_shlp[kl*2];
        lsh = kl_shlp[kl*2+1];
        shls[2] = ksh;
        shls[3] = lsh;

        dk = ao_loc[ksh+1] - ao_loc[ksh];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        
        dijk = dij*dk;
        dijkl = dijk*dl;
       
        cache = buf + dijkl*comp;

        if ((*fprescreen)(shls, atm, bas, env) &&
            (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
            if(ish != jsh){
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<dj; j++){
                        eri0 = eri + ind * nao_pair_kl + kl_off;
                        peri = eri0;
                        pbuf = buf;
                        if(ksh != lsh){
                            for(k=0; k<dk; k++){
                                for(l=0; l<dl; l++){
                                    peri[k*dl+l] = pbuf[i+j*di+k*dij+l*dijk];
                                }
                            }
                        }
                        else{
                            int ind1 = 0;
                            for(k=0; k<dk; k++){
                                for(l=0; l<=k; l++){
                                    peri[ind1] = pbuf[i+j*di+k*dij+l*dijk];
                                    ind1++;
                                }
                            }
                        }
                        ind++;
                    }
                }
            }
            else{
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<=i; j++){
                        eri0 = eri + ind * nao_pair_kl + kl_off;
                        peri = eri0;
                        pbuf = buf;
                        if(ksh != lsh){
                            for(k=0; k<dk; k++){
                                for(l=0; l<dl; l++){
                                    peri[k*dl+l] = pbuf[i+j*di+k*dij+l*dijk];
                                }
                            }
                        }
                        else{
                            int ind1 = 0;
                            for(k=0; k<dk; k++){
                                for(l=0; l<=k; l++){
                                    peri[ind1] = pbuf[i+j*di+k*dij+l*dijk];
                                    ind1++;
                                }
                            }
                        }
                        ind++;
                    }
                }
            }
        }
        else{
            if(ish != jsh){
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<dj; j++){
                        eri0 = eri + ind * nao_pair_kl + kl_off;
                        peri = eri0;
                        int n0 = dk*dl;
                        if(ksh == lsh) n0 = dk*(dk+1)/2;
                        for(k=0; k<n0; k++){
                            peri[k] = 0.0;
                        }
                        ind++;
                    }
                }
            }
            else{
                int ind = 0;
                for(i=0; i<di; i++){
                    for(j=0; j<=i; j++){ 
                        eri0 = eri + ind * nao_pair_kl + kl_off;
                        peri = eri0;
                        int n0 = dk*dl;
                        if(ksh == lsh) n0 = dk*(dk+1)/2;
                        for(k=0; k<n0; k++){
                            peri[k] = 0.0;
                        }
                        ind++;
                    }
                }
            }
        }

        if(ksh == lsh) kl_off += dk*(dk+1)/2;
        else kl_off += dk*dl;

    }
    return;
}



static int no_prescreen()
{
        return 1;
}

void GTOnr2e_fill_drv(int (*intor)(), void (*fill)(), int (*fprescreen)(),
                      double *eri, int comp,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        if (fprescreen == NULL) {
                fprescreen = no_prescreen;
        }

        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int ij, i, j;
        double *buf = malloc(sizeof(double) * (di*di*di*di*comp + cache_size));
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                (*fill)(intor, fprescreen, eri, buf, comp, i, j, shls_slice,
                        ao_loc, cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}



void GTOnr2e_fill_shl(int (*intor)(), void (*fill)(), int (*fprescreen)(),
                      double* eri, int comp, int has_s2_sym, 
                      int* ij_shlp, int nij, int* kl_shlp, int nkl, int nao_pair_kl, int* ioff, 
                      int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    if (fprescreen == NULL) {
        fprescreen = no_prescreen;
    }

    int max_shl = GTOmax_dim_shell(ao_loc,ij_shlp,nij,kl_shlp,nkl);
    const int dmax = ao_loc[max_shl+1] - ao_loc[max_shl];
    const int cache_size = GTO_cache_size(intor, max_shl, atm, natm, bas, nbas, env);

    /*
    int nao_pair_kl;
    if (has_s2_sym == 1) 
        nao_pair_kl = GTO_num_bas_pairs_s2(ao_loc, kl_shlp, nkl);
    else
        nao_pair_kl = GTO_num_bas_pairs_s1(ao_loc, kl_shlp, nkl);
    */

    /*
    int icount = 0;
    int ioff[nij];
    for (int ij = 0; ij < nij; ij++){
        ioff[ij] = icount;
        int ish = ij_shlp[ij*2];
        int jsh = ij_shlp[ij*2+1];
        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di*dj;
        if (ish == jsh && has_s2_sym == 1) dij = di*(di+1)/2;
        icount += dij * nao_pair_kl;
    }
    */


#pragma omp parallel
{
    double *buf = malloc(sizeof(double) * (dmax*dmax*dmax*dmax*comp + cache_size));
    #pragma omp for nowait schedule(dynamic)
    for (int ij = 0; ij < nij; ij++){
        int ish = ij_shlp[ij*2];
        int jsh = ij_shlp[ij*2+1];

        (*fill)(intor, fprescreen,
                eri+ioff[ij], buf, comp, ish, jsh,
                kl_shlp, nkl, nao_pair_kl, ao_loc, cintopt,
                atm, natm, bas, nbas, env);
    }
    free(buf);
}
}

