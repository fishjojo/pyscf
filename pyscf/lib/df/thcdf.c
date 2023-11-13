/* Copyright 2023- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"


static void fill_s2(double *out, double *eri, double *aoRg,
                    double fac, double *cache,
                    int i0, int j0, int di, int dj, int dk,
                    int ng, int ngrids, int nao)
{
    const int dij = di * dj;
    const int I1 = 1;
    const double D1 = 1.0;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    int ig;
    double *pao_i, *pao_j, *pcache;
    pao_i = aoRg + i0;
    pao_j = aoRg + j0;
    pcache = cache;

    memset(pcache, 0, dij*ng*sizeof(double));
    for (ig = 0; ig < ng; ig++) {
        dger_(&di, &dj, &D1, pao_i, &I1, pao_j, &I1, pcache, &di);
        pcache += dij;
        pao_i += nao;
        pao_j += nao;
    }

    dgemm_(&TRANS_T, &TRANS_N, &ng, &dk, &dij,
           &fac, cache, &dij, eri, &dij,
           &D1, out, &ngrids);
}


void THCDFctr_s2ij(int (*intor)(), double *out, double *aoRg, int ngrids,
                  uint8_t *non0table, int non0tab_ncol,
                  int nao, int comp, int ksh,
                  int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                  int *atm, int natm, int *bas, int nbas, double *env, double *buf)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];

    const int dk = ao_loc[ksh+1] - ao_loc[ksh];
    const int k0 = ao_loc[ksh] - ao_loc[ksh0];
    out += ngrids * k0;

    const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
    const int ng_last = BLKSIZE - (nblk * BLKSIZE - ngrids);

    int iblk, ng;
    int ish, jsh, i0, j0, di, dj;
    int shls[3] = {0, 0, ksh};
    double fac;

    di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
    double *cache = buf + di * di * dk * comp;
    double *pout, *pao;

    for (ish = ish0; ish < ish1; ish++) {
    for (jsh = jsh0; jsh < jsh1; jsh++) {
        if (ish < jsh) {
            continue;
        }
        shls[0] = ish;
        shls[1] = jsh;
        i0 = ao_loc[ish];
        j0 = ao_loc[jsh];
        di = ao_loc[ish+1] - ao_loc[ish];
        dj = ao_loc[jsh+1] - ao_loc[jsh];

        fac = (ish == jsh) ? 1.0 : 2.0;
        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
            pout = out;
            pao = aoRg;
            for (iblk = 0; iblk < nblk; iblk++) {
                ng = (iblk < nblk-1) ? BLKSIZE : ng_last;
                if (non0table[iblk*non0tab_ncol+ish] > 0 && non0table[iblk*non0tab_ncol+jsh] > 0) {
                    fill_s2(pout, buf, pao, fac, cache, i0, j0, di, dj, dk, ng, ngrids, nao);
                }
                pout += ng;
                pao += nao * ng;
            }
        }
    } }
}


void THCDFctr_int3c_aoRg2(int (*intor)(), void (*fill)(),
                         double *out, double *aoRg, int ngrids,
                         uint8_t *non0table, int non0tab_ncol,
                         int nao, int comp,
                         int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int di = GTOmax_shell_dim(ao_loc, shls_slice, 3);
    const int cache_size = MAX(di*di*BLKSIZE,
                               GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env));

    #pragma omp parallel
    {
        int ksh;
        double *buf = malloc(sizeof(double) * (di*di*di*comp + cache_size));
        #pragma omp for nowait schedule(dynamic)
        for (ksh = ksh0; ksh < ksh1; ksh++) {
            (*fill)(intor, out, aoRg, ngrids,
                    non0table, non0tab_ncol,
                    nao, comp, ksh,
                    shls_slice, ao_loc, cintopt,
                    atm, natm, bas, nbas, env, buf);
        }
        free(buf);
    }
}
